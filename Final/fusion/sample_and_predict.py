"""
sample_and_predict.py — Correct temporal fusion inference
=========================================================
Demonstrates how the Temporal Attention Fusion model works:

  ONE prediction = ONE window of W=8 synchronized (image, sensor) pairs
                   feeding through cross-attention together.

The W=8 pairs represent the same activity captured across 8 consecutive
timesteps by the same person.  The cross-attention layer fuses all 8
pairs jointly before producing a single class prediction.

Two modes
---------
  window  (default): sample W images + W sensor readings of the SAME class,
                     build one temporal window, get ONE prediction.
                     Run this N times for N independent activity windows.

  sliding           : sample a longer sequence (e.g. 24 images + 24 sensor
                     readings) and slide a window of W=8 across it, showing
                     how the prediction evolves as more context is seen.

Usage
-----
  # 4 independent activity windows (one prediction each)
  python sample_and_predict.py --mode window --n_windows 4

  # Sliding window over a 24-step sequence
  python sample_and_predict.py --mode sliding --sequence_len 24

  # Fix the class (otherwise random per window)
  python sample_and_predict.py --class_name walking --n_windows 6

  # Image-only (zero out sensor stream)
  python sample_and_predict.py --image_only

Output
------
  predictions.png  — for each window: W images side-by-side with the
                     single probability bar chart and one prediction label.
"""

import argparse, pathlib, random, sys, os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms
from PIL import Image

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))

from phase1_ontology import UNIFIED_CLASSES, NUM_UNIFIED, PROJ_DIM
from phase2_encoders import ImageEncoder, MLPSensorEncoder, get_device
from phase3_dataset  import WINDOW
from phase4_model    import TemporalFusionTransformer
from phase5_train    import load_checkpoint, project_batch
from run_all         import load_your_swin_model, load_your_mlp_model

# ── Constants ─────────────────────────────────────────────────────────────────

_UNIFIED_TO_IMAGE_FOLDERS = {
    0: ["walking", "running"],
    3: ["sitting"],
    4: ["standing"],
    5: ["sleeping"],
}
_UCI_LABEL_TO_UNIFIED = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Dataset helpers ───────────────────────────────────────────────────────────

def get_image_paths_for_class(kaggle_root: str, unified_class: int) -> list[pathlib.Path]:
    """Return all image paths that belong to a given unified class."""
    paths = []
    for folder_name in _UNIFIED_TO_IMAGE_FOLDERS.get(unified_class, []):
        folder = pathlib.Path(kaggle_root) / folder_name
        if not folder.exists():
            # case-insensitive fallback
            for d in pathlib.Path(kaggle_root).iterdir():
                if d.is_dir() and d.name.lower() == folder_name.lower():
                    folder = d; break
        if folder.exists():
            paths += [p for p in folder.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    return paths


def load_ucihar(ucihar_root: str):
    """Load UCI-HAR train split. Returns (X [N,561], y_unified [N])."""
    X_path = pathlib.Path(ucihar_root) / "train" / "X_train.txt"
    y_path = pathlib.Path(ucihar_root) / "train" / "y_train.txt"
    if not X_path.exists():
        raise FileNotFoundError(f"UCI-HAR not found: {X_path}")
    X = np.loadtxt(X_path, dtype=np.float32)
    y = np.loadtxt(y_path, dtype=int)
    y_unified = np.array([_UCI_LABEL_TO_UNIFIED[l] for l in y])
    return X, y_unified


def sample_window_for_class(
    kaggle_root:   str,
    ucihar_X:      np.ndarray,
    ucihar_y:      np.ndarray,
    unified_class: int,
    window:        int,
    rng:           random.Random,
    np_rng:        np.random.Generator,
    image_only:    bool = False,
) -> dict:
    """
    Build ONE temporal window: W images + W sensor readings of the same class.

    Returns
    -------
    {
        "image_paths":   list[Path]  length W  — the W images in this window
        "sensor_window": np.ndarray  [W, 561]  — W consecutive sensor rows
        "true_class":    str
    }
    """
    # ── Sample W images from the class folder ─────────────────────────────────
    all_img_paths = get_image_paths_for_class(kaggle_root, unified_class)
    if len(all_img_paths) < window:
        raise ValueError(
            f"Not enough images for class '{UNIFIED_CLASSES[unified_class]}'. "
            f"Found {len(all_img_paths)}, need {window}.")

    # Sample W distinct images to represent different moments in the activity
    img_paths = rng.sample(all_img_paths, k=window)

    # ── Sample W consecutive sensor rows from the same class ──────────────────
    if image_only:
        sen_win = np.zeros((window, 561), dtype=np.float32)
    else:
        idx   = np.where(ucihar_y == unified_class)[0]
        if len(idx) < window:
            raise ValueError(
                f"Not enough sensor rows for class "
                f"'{UNIFIED_CLASSES[unified_class]}'.")
        # Pick a random contiguous block of W rows (simulates a time segment)
        start  = np_rng.integers(0, len(idx) - window + 1)
        sen_win = ucihar_X[idx[start : start + window]]    # [W, 561]

    return {
        "image_paths":   img_paths,
        "sensor_window": sen_win,
        "true_class":    UNIFIED_CLASSES[unified_class],
    }


def sample_sequence_for_class(
    kaggle_root:   str,
    ucihar_X:      np.ndarray,
    ucihar_y:      np.ndarray,
    unified_class: int,
    seq_len:       int,
    rng:           random.Random,
    np_rng:        np.random.Generator,
    image_only:    bool = False,
) -> dict:
    """
    Build a LONGER sequence of seq_len (image, sensor) pairs for sliding window.

    Returns
    -------
    {
        "image_paths":   list[Path]  length seq_len
        "sensor_seq":    np.ndarray  [seq_len, 561]
        "true_class":    str
    }
    """
    all_img_paths = get_image_paths_for_class(kaggle_root, unified_class)
    if len(all_img_paths) < seq_len:
        raise ValueError(
            f"Not enough images for class '{UNIFIED_CLASSES[unified_class]}'.")

    img_paths = rng.sample(all_img_paths, k=seq_len)

    if image_only:
        sen_seq = np.zeros((seq_len, 561), dtype=np.float32)
    else:
        idx   = np.where(ucihar_y == unified_class)[0]
        start = np_rng.integers(0, max(len(idx) - seq_len + 1, 1))
        sen_seq = ucihar_X[idx[start : start + seq_len]]

    return {
        "image_paths": img_paths,
        "sensor_seq":  sen_seq,
        "true_class":  UNIFIED_CLASSES[unified_class],
    }


# ── Backbone encoding helpers ─────────────────────────────────────────────────

def encode_images(img_enc, image_paths, device):
    """
    Encode a list of image paths through the frozen backbone only.
    Returns raw feature tensor [len(image_paths), raw_img_dim].
    """
    feats = []
    img_enc.backbone.eval()
    for p in image_paths:
        img = IMG_TRANSFORM(Image.open(p).convert("RGB"))
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            f = img_enc.backbone(img)          # [1, 1024]
        if f.dim() > 2:
            f = f.mean(dim=list(range(1, f.dim() - 1)))
        feats.append(f.squeeze(0))
    return torch.stack(feats, dim=0)           # [N, 1024]


def encode_sensors(sen_enc, sensor_array, device):
    """
    Encode sensor rows through the frozen MLP backbone only.
    sensor_array: np.ndarray [N, 561]
    Returns raw feature tensor [N, raw_sen_dim].
    """
    sen_t = torch.from_numpy(sensor_array).to(device)
    sen_enc.backbone.eval()
    with torch.no_grad():
        feats = sen_enc.backbone(sen_t)        # [N, 64]
    return feats


# ── Core inference ────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_window(
    img_enc:      ImageEncoder,
    sen_enc:      MLPSensorEncoder,
    fusion_model: TemporalFusionTransformer,
    img_raw:      torch.Tensor,    # [W, raw_img_dim]  e.g. [8, 1024]
    sen_raw:      torch.Tensor,    # [W, raw_sen_dim]  e.g. [8, 64]
    device:       torch.device,
    image_only:   bool = False,
) -> tuple[str, float, np.ndarray]:
    """
    Feed ONE temporal window of W (image, sensor) pairs through the fusion model.
    Returns (predicted_class, confidence, probs[NUM_UNIFIED]).

    This is what temporal attention fusion does:
    All W pairs are processed jointly by cross-attention → ONE prediction.
    """
    fusion_model.eval()

    # Add batch dimension → [1, W, raw_dim]
    img_seq = img_raw.unsqueeze(0).to(device)
    sen_seq = sen_raw.unsqueeze(0).to(device) if not image_only else None

    # Project live to PROJ_DIM (same as training loop)
    img_p = project_batch(img_enc.proj, img_seq)               # [1, W, 256]

    if image_only or sen_seq is None:
        sen_p          = torch.zeros_like(img_p)
        force_sen_zero = True
    else:
        sen_p          = project_batch(sen_enc.proj, sen_seq)  # [1, W, 256]
        force_sen_zero = False

    # Cross-attention fusion over all W pairs → ONE prediction
    logits = fusion_model(img_p, sen_p, force_sen_zero=force_sen_zero)
    probs  = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    idx    = int(probs.argmax())
    return UNIFIED_CLASSES[idx], float(probs[idx]), probs


# ── Visualisation ─────────────────────────────────────────────────────────────

def _bar_chart(ax, probs, pred_class, true_class):
    """Draw a horizontal probability bar chart on ax."""
    pred_idx   = UNIFIED_CLASSES.index(pred_class)
    bar_colors = ["#3498db" if i == pred_idx else "#bdc3c7"
                  for i in range(NUM_UNIFIED)]
    y_pos      = list(range(NUM_UNIFIED - 1, -1, -1))
    ax.barh(y_pos, probs[::-1], color=list(reversed(bar_colors)),
            edgecolor="white", linewidth=0.5, height=0.65)
    ax.set_xlim(0, 1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.replace("_", " ") for c in reversed(UNIFIED_CLASSES)],
                       fontsize=7.5)
    ax.set_xlabel("Probability", fontsize=7)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for i, p in enumerate(reversed(probs)):
        if p > 0.03:
            ax.text(min(p + 0.01, 0.95), y_pos[i], f"{p:.1%}",
                    va="center", ha="left", fontsize=6.5)
    correct     = pred_class == true_class
    result_txt  = "✓ Correct" if correct else "✗ Incorrect"
    result_col  = "#27ae60" if correct else "#e74c3c"
    ax.set_title(f"{result_txt} — {pred_class}  ({probs[pred_idx]:.1%})",
                 fontsize=8, color=result_col, fontweight="bold")


def visualise_window_mode(window_results, output_path, title):
    """
    One row per activity window.
    Columns: [img_t1] [img_t2] ... [img_tW] | [probability bar chart]

    All W images feed into the model together and produce the single bar
    chart shown on the right — making it visually clear that all images
    contributed to one prediction.
    """
    n_windows = len(window_results)
    n_img_cols = WINDOW
    n_cols     = n_img_cols + 1          # images + bar chart
    fig, axes  = plt.subplots(
        n_windows, n_cols,
        figsize=(2.2 * n_img_cols + 4, 2.8 * n_windows),
        gridspec_kw={"width_ratios": [1] * n_img_cols + [2.5]},
    )
    if n_windows == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for row_i, res in enumerate(window_results):
        ax_row   = axes[row_i]
        img_axes = ax_row[:n_img_cols]
        ax_bar   = ax_row[n_img_cols]

        correct     = res["pred_class"] == res["true_class"]
        border_col  = "#27ae60" if correct else "#e74c3c"

        # ── Image strip ───────────────────────────────────────────────────────
        for t, (ax_img, img_path) in enumerate(
                zip(img_axes, res["image_paths"])):
            img = Image.open(img_path).convert("RGB").resize((112, 112))
            ax_img.imshow(img)
            ax_img.axis("off")
            ax_img.set_title(f"t={t+1}", fontsize=6.5, pad=2)
            # Coloured border on each image cell
            for spine in ax_img.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_col)
                spine.set_linewidth(1.5)

        # Label left of image strip
        ax_row[0].set_ylabel(
            f"True: {res['true_class']}", fontsize=7.5,
            rotation=90, labelpad=4, color=border_col, fontweight="bold")

        # ── Bar chart (one per window = one prediction) ───────────────────────
        _bar_chart(ax_bar, res["probs"], res["pred_class"], res["true_class"])

        # Annotation arrow from last image to bar chart
        fig.text(
            0.5, 0.5,   # placeholder — tight_layout moves things
            "", fontsize=1)

    # Add a shared annotation explaining the fusion
    fig.text(
        0.5, -0.02,
        f"← {WINDOW} images + {WINDOW} sensor readings feed jointly into "
        f"cross-attention → ONE prediction per row →",
        ha="center", fontsize=9, style="italic", color="#555")

    plt.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


def visualise_sliding_mode(sliding_results, output_path, title, window):
    """
    Show how the prediction changes as the sliding window moves through a sequence.
    X-axis = time step, Y-axis = probability for each class.
    """
    n_steps      = len(sliding_results)
    probs_matrix = np.array([r["probs"] for r in sliding_results])  # [n_steps, 6]
    time_steps   = [r["window_end"] for r in sliding_results]
    true_class   = sliding_results[0]["true_class"]

    fig, (ax_prob, ax_pred) = plt.subplots(
        2, 1, figsize=(max(10, n_steps * 0.6), 7),
        gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Probability lines per class
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_UNIFIED))
    for c_idx, (cls, col) in enumerate(zip(UNIFIED_CLASSES, colors)):
        ax_prob.plot(time_steps, probs_matrix[:, c_idx],
                     label=cls.replace("_", " "), color=col,
                     linewidth=2, marker="o", markersize=4)

    ax_prob.set_ylabel("Probability", fontsize=9)
    ax_prob.set_ylim(-0.02, 1.05)
    ax_prob.axhline(0.5, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
    ax_prob.legend(loc="upper right", fontsize=7.5, ncol=2)
    ax_prob.grid(alpha=0.25)
    ax_prob.set_title(f"True class: {true_class}", fontsize=9)

    # Predicted class per step
    pred_indices = [UNIFIED_CLASSES.index(r["pred_class"])
                    for r in sliding_results]
    correct_mask = [r["pred_class"] == true_class for r in sliding_results]
    point_colors = ["#27ae60" if c else "#e74c3c" for c in correct_mask]
    ax_pred.scatter(time_steps, pred_indices, c=point_colors, s=50, zorder=3)
    ax_pred.plot(time_steps, pred_indices, color="grey",
                 linewidth=1, linestyle="--", zorder=2)
    ax_pred.set_yticks(range(NUM_UNIFIED))
    ax_pred.set_yticklabels([c.replace("_", " ") for c in UNIFIED_CLASSES],
                             fontsize=7.5)
    ax_pred.set_xlabel(f"Window end timestep  (window size = {window})",
                       fontsize=9)
    ax_pred.set_ylabel("Predicted class", fontsize=9)
    ax_pred.grid(alpha=0.2)

    acc = sum(correct_mask) / len(correct_mask)
    fig.text(0.5, -0.02,
             f"Accuracy over {n_steps} windows: {acc:.1%}  "
             f"(green = correct, red = wrong)",
             ha="center", fontsize=9, color="#555")

    plt.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ── Console table ─────────────────────────────────────────────────────────────

def print_table(results, mode="window", image_only=False):
    modality = "IMAGE-ONLY" if image_only else "image + sensor"
    print(f"\n{'='*72}")
    print(f"  Mode: {mode}   Modality: {modality}")
    print(f"  Each row = ONE window of {WINDOW} (image, sensor) pairs "
          f"→ ONE prediction")
    print(f"{'='*72}")
    hdr_classes = "  ".join(f"{c[:10]:>10}" for c in UNIFIED_CLASSES)
    print(f"  {'#':>3}  {'True':^18}  {'Predicted':^18}  {'Conf':>6}  "
          + hdr_classes)
    print(f"  {'─'*3}  {'─'*18}  {'─'*18}  {'─'*6}  "
          + "  ".join("─"*10 for _ in UNIFIED_CLASSES))

    correct_count = 0
    for i, r in enumerate(results):
        ok   = r["pred_class"] == r["true_class"]
        correct_count += int(ok)
        mark = "✓" if ok else "✗"
        prob_str = "  ".join(f"{p:>10.1%}" for p in r["probs"])
        print(f"{mark} {i+1:>3}  {r['true_class']:^18}  "
              f"{r['pred_class']:^18}  {r['pred_conf']:>5.1%}  {prob_str}")

    acc = correct_count / len(results) if results else 0.0
    print(f"\n  {correct_count}/{len(results)} correct  ({acc:.1%})")
    print(f"{'='*72}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = get_device()
    print(f"Device : {device}")
    print(f"Mode   : {args.mode}  |  "
          f"Modality: {'image-only' if args.image_only else 'image + sensor'}")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\nLoading pretrained models ...")
    swin     = load_your_swin_model()
    mlp_full = load_your_mlp_model()

    img_enc = ImageEncoder(swin, swin_feat_dim=args.swin_feat_dim).to(device)
    sen_enc = MLPSensorEncoder(
        mlp_full.net, mlp_feat_dim=args.mlp_feat_dim,
        is_sequential=True).to(device)

    ckpt_path = pathlib.Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print("  Train first:  python run_all.py ...")
        sys.exit(1)

    fusion_model, img_enc, sen_enc = load_checkpoint(
        str(ckpt_path), img_enc, sen_enc, device=device)
    fusion_model = fusion_model.to(device).eval()

    # ── Load UCI-HAR ──────────────────────────────────────────────────────────
    if not args.image_only:
        print("Loading UCI-HAR sensor data ...")
        ucihar_X, ucihar_y = load_ucihar(args.ucihar_root)
    else:
        ucihar_X = ucihar_y = None

    # ── Resolve which classes to use ──────────────────────────────────────────
    image_classes = list(_UNIFIED_TO_IMAGE_FOLDERS.keys())  # [0, 3, 4, 5]
    if args.class_name:
        try:
            cls_idx = UNIFIED_CLASSES.index(args.class_name.lower())
        except ValueError:
            print(f"[ERROR] Unknown class '{args.class_name}'. "
                  f"Options: {UNIFIED_CLASSES}")
            sys.exit(1)
        if cls_idx not in image_classes:
            print(f"[ERROR] '{args.class_name}' has no image data. "
                  f"Image classes: {[UNIFIED_CLASSES[c] for c in image_classes]}")
            sys.exit(1)
        fixed_cls = cls_idx
    else:
        fixed_cls = None

    rng    = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    # ═════════════════════════════════════════════════════════════════════════
    # MODE A — WINDOW
    # Each call samples W images + W sensor readings → one prediction
    # ═════════════════════════════════════════════════════════════════════════
    if args.mode == "window":
        print(f"\nSampling {args.n_windows} activity windows "
              f"(W={WINDOW} pairs each) ...")
        window_results = []

        for win_i in range(args.n_windows):
            cls = fixed_cls if fixed_cls is not None \
                  else rng.choice(image_classes)

            win = sample_window_for_class(
                args.kaggle_root, ucihar_X, ucihar_y,
                cls, WINDOW, rng, np_rng, args.image_only)

            # Encode all W images through backbone
            img_raw = encode_images(img_enc, win["image_paths"], device)
            # Encode all W sensor rows through backbone
            if not args.image_only:
                sen_raw = encode_sensors(
                    sen_enc, win["sensor_window"], device)
            else:
                sen_raw = torch.zeros(WINDOW, 64, device=device)

            # Feed the whole window → ONE prediction
            pred_cls, pred_conf, probs = predict_window(
                img_enc, sen_enc, fusion_model,
                img_raw, sen_raw, device, args.image_only)

            window_results.append({
                "image_paths": win["image_paths"],
                "true_class":  win["true_class"],
                "pred_class":  pred_cls,
                "pred_conf":   pred_conf,
                "probs":       probs,
            })
            correct = "✓" if pred_cls == win["true_class"] else "✗"
            print(f"  Window {win_i+1:>2}: true={win['true_class']:<18} "
                  f"pred={pred_cls:<18} conf={pred_conf:.1%}  {correct}")

        print_table(window_results, mode="window", image_only=args.image_only)
        visualise_window_mode(
            window_results, args.output,
            title=(f"Temporal Fusion — Window Mode  "
                   f"({'image-only' if args.image_only else 'image + sensor'})\n"
                   f"Each row: {WINDOW} pairs feed jointly → one prediction"))

    # ═════════════════════════════════════════════════════════════════════════
    # MODE B — SLIDING WINDOW
    # Build a longer sequence, slide a window of W across it
    # ═════════════════════════════════════════════════════════════════════════
    else:
        cls = fixed_cls if fixed_cls is not None else rng.choice(image_classes)
        seq_len = args.sequence_len
        print(f"\nBuilding sequence of {seq_len} pairs for class "
              f"'{UNIFIED_CLASSES[cls]}' ...")

        seq = sample_sequence_for_class(
            args.kaggle_root, ucihar_X, ucihar_y,
            cls, seq_len, rng, np_rng, args.image_only)

        # Encode full sequence
        all_img_raw = encode_images(img_enc, seq["image_paths"], device)
        if not args.image_only:
            all_sen_raw = encode_sensors(sen_enc, seq["sensor_seq"], device)
        else:
            all_sen_raw = torch.zeros(seq_len, 64, device=device)

        # Slide window of W across the sequence
        sliding_results = []
        n_windows = seq_len - WINDOW + 1
        print(f"Sliding W={WINDOW} across {seq_len} steps → {n_windows} predictions")

        for start in range(n_windows):
            end     = start + WINDOW
            img_win = all_img_raw[start:end]   # [W, 1024]
            sen_win = all_sen_raw[start:end]   # [W, 64]

            pred_cls, pred_conf, probs = predict_window(
                img_enc, sen_enc, fusion_model,
                img_win, sen_win, device, args.image_only)

            correct = "✓" if pred_cls == seq["true_class"] else "✗"
            print(f"  t={start+1:>2}→{end:>2}: pred={pred_cls:<18} "
                  f"conf={pred_conf:.1%}  {correct}")
            sliding_results.append({
                "window_start": start + 1,
                "window_end":   end,
                "true_class":   seq["true_class"],
                "pred_class":   pred_cls,
                "pred_conf":    pred_conf,
                "probs":        probs,
            })

        acc = sum(r["pred_class"] == seq["true_class"]
                  for r in sliding_results) / len(sliding_results)
        print(f"\nOverall accuracy on {n_windows} windows: {acc:.1%}")

        visualise_sliding_mode(
            sliding_results, args.output,
            title=(f"Temporal Fusion — Sliding Window  "
                   f"[class: {seq['true_class']}  "
                   f"seq_len={seq_len}  W={WINDOW}]"),
            window=WINDOW)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Temporal fusion inference on image+sensor windows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--mode",         choices=["window", "sliding"],
                   default="window",
                   help="window: N independent W-pair windows, each → one pred. "
                        "sliding: slide W across a longer sequence.")
    p.add_argument("--kaggle_root",  default="../HAR_dataset/train")
    p.add_argument("--ucihar_root",  default="../ucihar")
    p.add_argument("--checkpoint",   default="checkpoints/best_model.pt")
    p.add_argument("--class_name",   default="",
                   help="Fix to one class. Empty = random per window.")
    p.add_argument("--n_windows",    type=int, default=4,
                   help="[window mode] How many independent windows to run")
    p.add_argument("--sequence_len", type=int, default=24,
                   help="[sliding mode] Length of sequence to slide over")
    p.add_argument("--image_only",   action="store_true",
                   help="Zero out sensor stream")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--output",       default="predictions.png")
    p.add_argument("--swin_feat_dim",type=int, default=1024)
    p.add_argument("--mlp_feat_dim", type=int, default=64)

    main(p.parse_args())