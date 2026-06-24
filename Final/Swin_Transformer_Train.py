"""
Swin-B Training Script for Human Action Recognition (HAR) Dataset
================================================================
Dataset: https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset
Classes: 15 (calling, clapping, cycling, dancing, drinking, eating,
         fighting, hugging, laughing, listeningtomusic, running,
         sitting, sleeping, texting, using_laptop)

Usage:
    python train.py --data_dir /path/to/HAR --epochs 30 --batch_size 32
"""

import os
import argparse
import time
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import timm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ─── Constants ────────────────────────────────────────────────────────────────
CLASSES = [
    "calling", "clapping", "cycling", "dancing", "drinking",
    "eating", "fighting", "hugging", "laughing", "standing",
    "running", "sitting", "sleeping", "walking", "using_laptop"
]
NUM_CLASSES = 15
IMG_SIZE = 224           # Swin-B default; use 384 for higher accuracy if VRAM allows


# ─── Device Detection (CUDA → MPS → CPU) ─────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def get_amp_context(device: torch.device):
    """
    Returns (scaler, autocast_ctx_fn).

    - CUDA  : full AMP with GradScaler + autocast("cuda")
    - MPS   : autocast("cpu") only — MPS doesn't support GradScaler yet
    - CPU   : no-op scaler + autocast("cpu")

    Usage in training loop:
        scaler, amp_ctx = get_amp_context(device)
        with amp_ctx():
            outputs = model(imgs)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
    """
    if device.type == "cuda":
        scaler    = GradScaler("cuda")
        amp_ctx   = lambda: autocast("cuda")          # noqa: E731
    elif device.type == "mps":
        # MPS supports autocast from PyTorch 2.3+ but not GradScaler
        scaler    = _NoOpScaler()
        amp_ctx   = lambda: autocast("cpu")            # noqa: E731
    else:
        scaler    = _NoOpScaler()
        amp_ctx   = lambda: autocast("cpu")            # noqa: E731
    return scaler, amp_ctx


class _NoOpScaler:
    """Drop-in GradScaler replacement for MPS / CPU that does nothing."""
    def scale(self, loss):        return loss
    def unscale_(self, optimizer): pass
    def step(self, optimizer):    optimizer.step()
    def update(self):             pass


# ─── Custom CSV Dataset ───────────────────────────────────────────────────────
class HARDataset(Dataset):
    """
    Reads images from a flat folder using a CSV that maps filename → label.

    Expected CSV columns (auto-detected, case-insensitive):
        filename / file / image   →  image filename  (e.g. "img_001.jpg")
        label / action / class    →  class name      (e.g. "running")

    The image files must live inside `img_dir`.
    """

    # Column name aliases — handles different CSV conventions
    FILENAME_COLS = ["filename", "file", "image", "img", "image_id"]
    LABEL_COLS    = ["label", "action", "class", "activity", "category"]

    def __init__(self, csv_path: Path, img_dir: Path,
                 class_to_idx: dict, transform=None):
        self.img_dir      = Path(img_dir)
        self.transform    = transform
        self.class_to_idx = class_to_idx

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()

        # ── auto-detect filename column ──
        fn_col = next((c for c in self.FILENAME_COLS if c in df.columns), None)
        if fn_col is None:
            raise ValueError(
                f"Cannot find a filename column in {csv_path}.\n"
                f"Found columns: {list(df.columns)}\n"
                f"Expected one of: {self.FILENAME_COLS}"
            )

        # ── auto-detect label column ──
        lbl_col = next((c for c in self.LABEL_COLS if c in df.columns), None)
        if lbl_col is None:
            raise ValueError(
                f"Cannot find a label column in {csv_path}.\n"
                f"Found columns: {list(df.columns)}\n"
                f"Expected one of: {self.LABEL_COLS}"
            )

        df[lbl_col] = df[lbl_col].str.strip()
        # Drop rows whose label isn't in our class list
        valid_mask  = df[lbl_col].isin(class_to_idx)
        dropped     = (~valid_mask).sum()
        if dropped:
            print(f"   ⚠️  Dropped {dropped} rows with unknown labels")
        df = df[valid_mask].reset_index(drop=True)

        self.filenames = df[fn_col].tolist()
        self.labels    = [class_to_idx[l] for l in df[lbl_col]]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.img_dir / self.filenames[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ─── Data Transforms ──────────────────────────────────────────────────────────
def get_transforms(img_size=IMG_SIZE):
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf


# ─── Dataset & Sampler ────────────────────────────────────────────────────────
def build_dataloaders(data_dir, batch_size, img_size, num_workers=4):
    """
    Expected layout:
        <data_dir>/
        ├── train/                  ← flat folder of training images
        ├── test/                   ← flat folder of test images
        ├── Training_set.csv        ← filename, label
        └── Testing_set.csv         ← filename, label  (optional — used for val)
    """
    data_dir  = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir  = data_dir / "test"

    train_csv = data_dir / "Training_set.csv"
    test_csv  = data_dir / "Testing.csv"

    # Validate paths
    for p in [train_dir, train_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Required path not found: {p}")

    class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
    train_tf, val_tf = get_transforms(img_size)

    train_ds = HARDataset(train_csv, train_dir, class_to_idx, transform=train_tf)

    # Val dataset: use Testing_set.csv + test/ if available, else 20% split of train
    if test_csv.exists() and test_dir.exists():
        val_ds = HARDataset(test_csv, test_dir, class_to_idx, transform=val_tf)
        print(f"\n📂 Dataset loaded from CSVs:")
    else:
        # Fallback: random 80/20 split on training set
        print(f"\n⚠️  Testing_set.csv or test/ not found — using 80/20 split of train")
        n     = len(train_ds)
        n_val = int(0.2 * n)
        train_ds, val_ds = torch.utils.data.random_split(
            train_ds, [n - n_val, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        # Re-apply val transform to val split
        val_ds.dataset.transform = val_tf

    # Balanced sampler — handles class imbalance automatically
    targets = torch.tensor(
        train_ds.labels if hasattr(train_ds, "labels")
        else [train_ds.dataset.labels[i] for i in train_ds.indices]
    )
    class_counts  = torch.bincount(targets, minlength=NUM_CLASSES)
    class_weights = 1.0 / class_counts.float().clamp(min=1)
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # pin_memory only works correctly with CUDA; disable for MPS/CPU
    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=use_pin, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin
    )

    print(f"   Train : {len(train_ds)} images | Val: {len(val_ds)} images")
    print(f"   Classes ({NUM_CLASSES}): {CLASSES}\n")

    return train_loader, val_loader, CLASSES


# ─── Model ────────────────────────────────────────────────────────────────────
def build_model(num_classes, pretrained=True):
    """
    Swin-B pretrained on ImageNet-22k → fine-tuned for HAR.
    timm model name: swin_base_patch4_window7_224
    For 384px use:  swin_base_patch4_window12_384
    """
    model = timm.create_model(
        "swin_base_patch4_window7_224",
        pretrained=pretrained,
        num_classes=num_classes
    )

    # Freeze early stages for first few epochs (optional — improves stability)
    for name, param in model.named_parameters():
        if "layers.0" in name or "layers.1" in name:
            param.requires_grad = False

    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Swin-B loaded | Total params: {total/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M")
    return model


def unfreeze_all(model):
    """Call after warm-up epochs to unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔓 All layers unfrozen | Trainable: {trainable/1e6:.1f}M params")


# ─── MixUp Augmentation ───────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.4):
    """MixUp: blends two samples for better generalization."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─── Training & Validation ────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, amp_ctx, device,
                    use_mixup=True, mixup_alpha=0.4, epoch=0, unfreeze_epoch=5):
    if epoch == unfreeze_epoch:
        unfreeze_all(model)

    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        if use_mixup:
            imgs, la, lb, lam = mixup_data(imgs, labels, mixup_alpha)

        optimizer.zero_grad()
        with amp_ctx():
            outputs = model(imgs)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, la, lb, lam)
            else:
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        if use_mixup:
            correct += (lam * preds.eq(la).sum().item()
                        + (1 - lam) * preds.eq(lb).sum().item())
        else:
            correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(f"  Step [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f}", end="\r")

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, amp_ctx):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with amp_ctx():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ─── Plotting ─────────────────────────────────────────────────────────────────
def plot_history(history, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(history["val_loss"],   label="Val Loss",   marker="s")
    ax1.set_title("Loss Curve"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True)

    ax2.plot([a * 100 for a in history["train_acc"]], label="Train Acc %", marker="o")
    ax2.plot([a * 100 for a in history["val_acc"]],   label="Val Acc %",   marker="s")
    ax2.set_title("Accuracy Curve"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"📈 Saved training curves → {save_dir / 'training_curves.png'}")


def plot_confusion_matrix(labels, preds, class_names, save_dir):
    all_label_ids = list(range(len(class_names)))
    cm = confusion_matrix(labels, preds, labels=all_label_ids)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix — Swin-B HAR")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"📊 Saved confusion matrix → {save_dir / 'confusion_matrix.png'}")


# ─── Checkpoint Resume ────────────────────────────────────────────────────────
def load_checkpoint(path, model, optimizer, scheduler, device):
    """
    Loads a saved checkpoint and restores model weights, optimizer state,
    scheduler state, best_acc, start_epoch, and training history.
    Returns (start_epoch, best_acc, history).
    """
    print(f"\n📂 Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])

    if "optimizer_state_dict" in ckpt and optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Move optimizer state tensors to the right device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    if "scheduler_state_dict" in ckpt and scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = ckpt.get("epoch", -1) + 1   # resume from NEXT epoch
    best_acc    = ckpt.get("val_acc", 0.0)
    history     = ckpt.get("history", {"train_loss": [], "val_loss": [],
                                        "train_acc":  [], "val_acc":  []})

    print(f"   ✅ Restored — start epoch: {start_epoch} | best val acc so far: {best_acc*100:.2f}%")
    return start_epoch, best_acc, history


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(args):
    device = get_device()
    print(f"⚡ Device: {device}")
    if device.type == "mps":
        print("   🍎 Apple Metal (MPS) detected — using CPU autocast (GradScaler not supported on MPS)")
    elif device.type == "cpu":
        print("   ⚠️  No GPU found — training on CPU (will be slow)")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, class_names = build_dataloaders(
        args.data_dir, args.batch_size, args.img_size, args.num_workers
    )

    # Model — always build fresh architecture first (weights loaded below if resuming)
    model = build_model(NUM_CLASSES, pretrained=not bool(args.resume)).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer — differential LR
    backbone_params = [p for n, p in model.named_parameters()
                       if "head" not in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if "head" in n and p.requires_grad]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=0.05)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler, amp_ctx = get_amp_context(device)

    # ── Resume from checkpoint if provided ──
    start_epoch = 0
    best_acc    = 0.0
    history     = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    if args.resume:
        start_epoch, best_acc, history = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        # Unfreeze all layers when resuming — warm-up already done
        unfreeze_all(model)

    remaining = args.epochs - start_epoch
    if remaining <= 0:
        print(f"\n⚠️  Checkpoint already completed {start_epoch} epochs "
              f"and --epochs={args.epochs}. "
              f"Pass a larger --epochs value to keep training.")
        return

    print(f"\n{'='*60}")
    if args.resume:
        print(f"  Resuming Swin-B | epochs {start_epoch+1}→{args.epochs} "
              f"({remaining} remaining) | batch={args.batch_size}")
    else:
        print(f"  Training Swin-B | {args.epochs} epochs | batch={args.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, amp_ctx, device,
            use_mixup=args.mixup, mixup_alpha=args.mixup_alpha,
            epoch=epoch, unfreeze_epoch=args.unfreeze_epoch
        )
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device, amp_ctx
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch+1:02d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        # Save best checkpoint (now includes scheduler state + history)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc":    val_acc,
                "class_names": class_names,
                "history":    history,
            }, save_dir / "best_swinb_har.pth")
            print(f"  ✅ Best model saved (val_acc={val_acc*100:.2f}%)")

        # Save last checkpoint (always overwritten — safe recovery point)
        torch.save({
            "epoch": epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_acc":    val_acc,
            "class_names": class_names,
            "history":    history,
        }, save_dir / "last_swinb_har.pth")

    # ── Final evaluation ──
    print(f"\n{'='*60}")
    print(f"  Best Val Accuracy: {best_acc*100:.2f}%")
    print(f"{'='*60}\n")

    all_label_ids = list(range(NUM_CLASSES))
    print(classification_report(
        val_labels, val_preds,
        labels=all_label_ids,
        target_names=class_names,
        zero_division=0,
    ))

    present = set(val_labels)
    missing = [class_names[i] for i in all_label_ids if i not in present]
    if missing:
        print(f"Warning: these classes had no val samples: {missing}")

    plot_history(history, save_dir)
    plot_confusion_matrix(val_labels, val_preds, class_names, save_dir)

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n🎉 Done! All outputs saved to: {save_dir}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swin-B HAR Trainer")
    parser.add_argument("--resume",        type=str,   default=None,
                        help="Path to checkpoint to resume from "
                             "(e.g. outputs/best_swinb_har.pth or outputs/last_swinb_har.pth)")
    parser.add_argument("--data_dir",      type=str,   default="./HAR_dataset",
                        help="Root folder with train/ and val/ (or test/) subfolders")
    parser.add_argument("--save_dir",      type=str,   default="./outputs",
                        help="Where to save checkpoints and plots")
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--img_size",      type=int,   default=224,
                        help="224 (default) or 384 for higher accuracy")
    parser.add_argument("--lr",            type=float, default=2e-4,
                        help="Base learning rate for the classifier head")
    parser.add_argument("--mixup",         action="store_true", default=True)
    parser.add_argument("--mixup_alpha",   type=float, default=0.4)
    parser.add_argument("--unfreeze_epoch",type=int,   default=5,
                        help="Epoch at which to unfreeze all backbone layers")
    parser.add_argument("--num_workers",   type=int,   default=4)
    args = parser.parse_args()
    main(args)