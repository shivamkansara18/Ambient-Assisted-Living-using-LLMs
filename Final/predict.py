"""
Inference Script — Swin-B HAR
==============================
Supports single image, flat image folder, or CSV + image folder (your dataset layout).

Usage:
    # Single image
    python predict.py --checkpoint outputs/best_swinb_har.pth --image path/to/img.jpg

    # Flat folder (no labels, just predictions)
    python predict.py --checkpoint outputs/best_swinb_har.pth --image_dir ./test/

    # CSV + folder (evaluates accuracy if labels are present)
    python predict.py --checkpoint outputs/best_swinb_har.pth \
                      --csv Testing_set.csv --image_dir ./test/
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


CLASSES = [
    "calling", "clapping", "cycling", "dancing", "drinking",
    "eating", "fighting", "hugging", "laughing", "standing",
    "running", "sitting", "sleeping", "walking", "using_laptop"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}


# ─── Folder-based Dataset ─────────────────────────────────────────────────────
class InferenceDataset(Dataset):
    """
    Folder-based dataset:
    image_dir/
        class1/
            img1.jpg
        class2/
            img2.jpg
    """

    def __init__(self, image_dir, csv_path=None, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.samples = []
        self.has_labels = True  # folder-based always has labels

        for class_name in sorted(os.listdir(self.image_dir)):
            class_path = self.image_dir / class_name

            if not class_path.is_dir():
                continue

            if class_name not in CLASS_TO_IDX:
                continue  # skip unknown classes

            label = CLASS_TO_IDX[class_name]

            for img_path in class_path.iterdir():
                if img_path.suffix.lower() in IMG_EXTS:
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label, img_path.name


# ─── Model ────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path, num_classes=15, device="cpu"):
    model = timm.create_model(
        "swin_base_patch4_window7_224",
        pretrained=False,
        num_classes=num_classes
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


# ─── Single image helper ──────────────────────────────────────────────────────
@torch.no_grad()
def predict_single(model, image_path, device, top_k=3):
    img    = Image.open(image_path).convert("RGB")
    tensor = VAL_TRANSFORM(img).unsqueeze(0).to(device)
    probs  = F.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    top_i  = np.argsort(probs)[::-1][:top_k]
    return [(CLASSES[i], float(probs[i]) * 100) for i in top_i]


# ─── Batch evaluation ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_batch(model, dataset, device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4,
                        pin_memory=torch.cuda.is_available())
    all_preds, all_labels, all_files = [], [], []

    for imgs, labels, fnames in loader:
        imgs = imgs.to(device)
        probs = F.softmax(model(imgs), dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_files.extend(fnames)

    return all_preds, all_labels, all_files


def plot_confusion(labels, preds, save_path):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix — Swin-B HAR")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Confusion matrix saved → {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(args):
    device = get_device()
    model  = load_model(args.checkpoint, device=device)
    print(f"✅ Model loaded | device={device}\n")

    # ── Mode 1: single image ──
    if args.image:
        results = predict_single(model, args.image, device, top_k=args.top_k)
        print(f"🖼  {Path(args.image).name}")
        for rank, (cls, conf) in enumerate(results, 1):
            bar = "█" * int(conf / 5)
            print(f"   {rank}. {cls:<22} {conf:5.1f}%  {bar}")
        return

    # ── Mode 2: CSV + folder  OR  plain folder ──
    if not args.image_dir:
        raise ValueError("Provide --image, or --image_dir (with optional --csv)")

    dataset = InferenceDataset(
        image_dir=args.image_dir,
        csv_path=args.csv,
        transform=VAL_TRANSFORM
    )
    print(f"📂 {len(dataset)} images found")

    preds, labels, filenames = evaluate_batch(model, dataset, device, args.batch_size)

    # ── Save predictions CSV ──
    out_df = pd.DataFrame({
        "filename": filenames,
        "predicted": [CLASSES[p] for p in preds],
    })
    if dataset.has_labels:
        out_df["true_label"] = [CLASSES[l] if l >= 0 else "unknown" for l in labels]
        out_df["correct"]    = (out_df["predicted"] == out_df["true_label"])

    out_csv = Path(args.output_csv)
    out_df.to_csv(out_csv, index=False)
    print(f"💾 Predictions saved → {out_csv}")

    # ── Accuracy report (only if labels available) ──
    if dataset.has_labels:
        valid = [(l, p) for l, p in zip(labels, preds) if l >= 0]
        true_labels = [v[0] for v in valid]
        pred_labels = [v[1] for v in valid]
        acc = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
        print(f"\n🎯 Accuracy: {acc * 100:.2f}%\n")
        print(classification_report(true_labels, pred_labels,
                                    target_names=CLASSES, zero_division=0))
        plot_confusion(true_labels, pred_labels, Path(args.output_csv).parent / "confusion_matrix.png")
    else:
        print("\n(No labels in CSV — skipping accuracy report)")
        print(out_df["predicted"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swin-B HAR Inference")
    parser.add_argument("--checkpoint",   type=str, required=True,
                        help="Path to best_swinb_har.pth")
    parser.add_argument("--image",        type=str, default=None,
                        help="Single image path (overrides other modes)")
    parser.add_argument("--image_dir",    type=str, default=None,
                        help="Flat folder of images (e.g. ./test/)")
    parser.add_argument("--csv",          type=str, default=None,
                        help="CSV with filename & label columns (e.g. Testing_set.csv)")
    parser.add_argument("--output_csv",   type=str, default="predictions.csv",
                        help="Where to save prediction results")
    parser.add_argument("--top_k",        type=int, default=3,
                        help="Top-K classes shown in single-image mode")
    parser.add_argument("--batch_size",   type=int, default=32)
    main(parser.parse_args())