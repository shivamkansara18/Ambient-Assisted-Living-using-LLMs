"""
run_all.py — End-to-end training pipeline (Phases 1-5)
=======================================================
Loads your actual pretrained models and runs the full fusion pipeline.

Models used
-----------
SwinTransformer : timm  swin_base_patch4_window7_224
                  checkpoint : ../outputs_v5/best_swinb_har_v5.pth
                  saved with : torch.save({"model_state_dict": ..., ...})

MLP sensor model: PyTorch  Classifier(input_dim=561)
                  checkpoint : ../classifier_model.pth
                  saved with : torch.save(classifier.state_dict(), ...)
                  last hidden : 64 units  →  mlp_feat_dim = 64

Device priority : CUDA > MPS (Apple Silicon) > CPU
"""

import os
import pathlib
import argparse
import numpy as np
import torch
import torch.nn as nn

from phase1_ontology import UNIFIED_CLASSES, NUM_UNIFIED
from phase2_encoders import (ImageEncoder, MLPSensorEncoder,
                              build_and_cache_all, get_device, DEVICE)
from phase3_dataset  import WINDOW
from phase5_train    import train as run_training, CFG

# Paths relative to this file
_HERE     = pathlib.Path(__file__).parent
SWIN_CKPT = _HERE / ".." / "outputs_v5" / "best_swinb_har_v5.pth"
MLP_CKPT  = _HERE / ".." / "classifier_model.pth"


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — SwinTransformer  (timm, PyTorch checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

def load_your_swin_model(ckpt_path: pathlib.Path = SWIN_CKPT) -> nn.Module:
    """
    Loads best_swinb_har_v5.pth.

    The checkpoint was saved as:
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_acc":              vl_acc,
            "class_names":          final_classes,
            "history":              history,
        }, ...)

    Loading strategy
    ----------------
    1. Read class_names from the checkpoint to determine num_classes.
    2. Recreate the model with the same num_classes so the state dict
       shapes match exactly.
    3. Load model_state_dict.
    4. ImageEncoder will call reset_classifier(0) to remove the head
       while keeping global average pooling — yielding [B, 1024] features.
    """
    try:
        import timm
    except ImportError:
        raise ImportError(
            "timm is required.  Install with:  pip install timm")

    ckpt_path = pathlib.Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"SwinTransformer checkpoint not found: {ckpt_path}\n"
            f"Expected (relative to run_all.py): "
            f"../outputs_v5/best_swinb_har_v5.pth")

    print(f"Loading SwinTransformer from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Determine num_classes from saved class list
    class_names = ckpt.get("class_names", None)
    if class_names is not None:
        num_classes = len(class_names)
        print(f"  class_names ({num_classes}): {class_names}")
    else:
        # Fallback: read from head fc weight in state dict
        sd = ckpt["model_state_dict"]
        head_key = next((k for k in sd if k.endswith("head.fc.weight")), None)
        if head_key:
            num_classes = sd[head_key].shape[0]
            print(f"  Inferred num_classes={num_classes} from {head_key}")
        else:
            raise KeyError(
                "Cannot determine num_classes — 'class_names' missing from "
                "checkpoint and no 'head.fc.weight' found in state dict.")

    # Recreate model with original num_classes so state dict loads cleanly
    model = timm.create_model(
        "swin_base_patch4_window7_224",
        pretrained=False,
        num_classes=num_classes,
    )
    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"], strict=True)
    if missing:
        print(f"  [warn] missing keys: {missing[:5]}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:5]}")

    model.eval()

    val_acc = ckpt.get("val_acc", "n/a")
    epoch   = ckpt.get("epoch",   "n/a")
    print(f"  Loaded OK — epoch={epoch}  val_acc={val_acc}")
    print(f"  Feature dim: 1024  (reset_classifier(0) applied inside ImageEncoder)")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — PyTorch MLP sensor classifier  (.pth state dict)
# ─────────────────────────────────────────────────────────────────────────────

class Classifier(nn.Module):
    """
    Exact replica of the MLP class used during sensor model training.

    Architecture
    ------------
    Linear(561→512) → ReLU → BatchNorm1d(512) → Dropout(0.4)
    Linear(512→256) → ReLU → BatchNorm1d(256) → Dropout(0.3)
    Linear(256→128) → ReLU → BatchNorm1d(128) → Dropout(0.2)
    Linear(128→64)  → ReLU → BatchNorm1d(64)
    Linear(64→6)                                         ← head (stripped later)

    Last hidden layer before the head: 64 units
    → mlp_feat_dim = 64
    """
    def __init__(self, input_dim: int = 561):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 6),   # classification head — stripped by MLPSensorEncoder
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_your_mlp_model(ckpt_path: pathlib.Path = MLP_CKPT) -> nn.Module:
    """
    Loads classifier_model.pth.

    The checkpoint was saved as a bare state dict:
        torch.save(classifier.state_dict(), "classifier_model.pth")

    Loading strategy
    ----------------
    1. Instantiate Classifier(input_dim=561) — the exact class used in training.
    2. Load the state dict with strict=True to catch any shape mismatches.
    3. Set to eval() mode (disables Dropout and puts BatchNorm in inference mode).
    4. Return the full model including the final Linear(64→6) head.
       MLPSensorEncoder will strip that head automatically via is_sequential=True,
       exposing the BatchNorm1d(64) output as the 64-dim feature vector.
    """
    ckpt_path = pathlib.Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"MLP checkpoint not found: {ckpt_path}\n"
            f"Expected (relative to run_all.py): ../classifier_model.pth")

    print(f"Loading MLP sensor classifier from: {ckpt_path}")

    model = Classifier(input_dim=561)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=True)

    if missing:
        raise RuntimeError(
            f"Missing keys when loading MLP state dict: {missing}\n"
            f"This usually means the Classifier class definition here does not "
            f"match the one used during training. Check layer sizes.")
    if unexpected:
        print(f"  [warn] unexpected keys (ignored): {unexpected}")

    model.eval()
    print(f"  Loaded OK — architecture: 561→512→256→128→64→6")
    print(f"  Feature dim: 64  (final Linear(64→6) stripped by MLPSensorEncoder)")
    return model


# ─────────────────────────────────────────────────────────────────────────────


def main(args):
    device = get_device()
    print("HAR Fusion Training Pipeline")
    print(f"Device      : {device}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch size  : {args.batch_size}")

    # ── Load your pretrained models ───────────────────────────────────────────
    swin     = load_your_swin_model()
    mlp_full = load_your_mlp_model()

    # Classifier wraps a plain nn.Sequential in .net.
    # We pass .net directly so MLPSensorEncoder(is_sequential=True)
    # can correctly drop its last element (Linear 64→6).
    mlp = mlp_full.net

    # ── Phase 2: extract and cache features ───────────────────────────────────
    print("\n" + "=" * 55)
    print("Phase 2 — Feature extraction & caching")
    print("=" * 55)
    cache = build_and_cache_all(
        swin_model        = swin,
        mlp_model         = mlp,
        swin_feat_dim     = args.swin_feat_dim,
        mlp_feat_dim      = args.mlp_feat_dim,
        mlp_head_attr     = args.mlp_head_attr,
        mlp_is_seq        = args.mlp_is_seq,
        kaggle_root       = args.kaggle_root,
        kaggle_pre_split  = args.kaggle_pre_split,
        ucihar_root       = args.ucihar_root,
        cache_dir         = args.cache_dir,
        device            = device,
    )

    img_enc = cache["img_enc"]
    sen_enc = cache["sen_enc"]
    img_tr_f, img_tr_l = cache["img_train"]
    img_va_f, img_va_l = cache["img_val"]
    sen_tr_f, sen_tr_l = cache["sen_train"]
    sen_va_f, sen_va_l = cache["sen_val"]

    # ── Dataset size report ───────────────────────────────────────────────────
    from collections import Counter
    print("\nDataset overview after ontology filtering:")
    for name, labels in [("img_train", img_tr_l), ("img_val",   img_va_l),
                          ("sen_train", sen_tr_l), ("sen_val",   sen_va_l)]:
        dist = {UNIFIED_CLASSES[k]: v
                for k, v in sorted(Counter(labels.tolist()).items())}
        print(f"  {name:<12}: {len(labels):>6} samples  {dist}")

    # ── Phase 5: train fusion model ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Phase 5 — Training temporal fusion model")
    print("=" * 55)
    cfg = {
        **CFG,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "num_workers":    args.num_workers,
        "checkpoint_dir": args.checkpoint_dir,
        "log_dir":        args.log_dir,
        "device":         str(device),
    }

    fusion_model, img_enc, sen_enc = run_training(
        img_enc, sen_enc,
        img_tr_f, img_tr_l,
        sen_tr_f, sen_tr_l,
        img_va_f, img_va_l,
        sen_va_f, sen_va_l,
        cfg=cfg,
    )

    print("\nPipeline complete.")
    print(f"Best model  : {args.checkpoint_dir}/best_model.pt")
    print(f"Training log: {args.log_dir}/train_log.csv")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="HAR Temporal Fusion — phases 1-5 training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data paths ──────────────────────────────────────────────────────────
    p.add_argument("--kaggle_root",      default="data/kaggle_har",
                   help="Root of Kaggle image HAR dataset (ImageFolder layout)")
    p.add_argument("--kaggle_pre_split", action="store_true",
                   help="Set if kaggle_root already has train/ and val/ "
                        "sub-folders; otherwise an 80/20 split is applied")
    p.add_argument("--ucihar_root",      default="data/ucihar",
                   help="Root of UCI-HAR sensor dataset")
    p.add_argument("--cache_dir",        default="cache",
                   help="Directory for cached feature .npy files")

    # ── Model paths (override defaults if your files are elsewhere) ──────────
    p.add_argument("--swin_ckpt", default=str(SWIN_CKPT),
                   help="Path to best_swinb_har_v5.pth")
    p.add_argument("--mlp_ckpt",  default=str(MLP_CKPT),
                   help="Path to classifier_model.pth")

    # ── Model dimensions ────────────────────────────────────────────────────
    p.add_argument("--swin_feat_dim", type=int, default=1024,
                   help="SwinTransformer-Base feature dim before head "
                        "(swin_base_patch4_window7_224 = 1024)")
    p.add_argument("--mlp_feat_dim",  type=int, default=64,
                   help="MLP last hidden layer dim — 64 for your Classifier model "
                        "(the BatchNorm1d(64) output before the final Linear(64→6))")
    p.add_argument("--mlp_head_attr", type=str, default="",
                   help="Unused — model is always wrapped as nn.Sequential")
    p.add_argument("--mlp_is_seq",    action="store_true", default=True,
                   help="Always True — Classifier.net is nn.Sequential; "
                        "MLPSensorEncoder drops the last layer (Linear 64→6)")

    # ── Training hyper-parameters ────────────────────────────────────────────
    p.add_argument("--epochs",         type=int, default=60)
    p.add_argument("--batch_size",     type=int, default=128)
    p.add_argument("--num_workers",    type=int, default=4,
                   help="DataLoader workers (auto-set to 0 on MPS)")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_dir",        default="logs")

    args = p.parse_args()

    # Override checkpoint paths if CLI flags were given
    if args.swin_ckpt != str(SWIN_CKPT):
        SWIN_CKPT_OVERRIDE = pathlib.Path(args.swin_ckpt)
        load_your_swin_model.__defaults__ = (SWIN_CKPT_OVERRIDE,)
    if args.mlp_ckpt != str(MLP_CKPT):
        MLP_CKPT_OVERRIDE = pathlib.Path(args.mlp_ckpt)
        load_your_mlp_model.__defaults__ = (MLP_CKPT_OVERRIDE,)

    # The reconstructed Keras MLP is always nn.Sequential
    args.mlp_is_seq = True

    main(args)