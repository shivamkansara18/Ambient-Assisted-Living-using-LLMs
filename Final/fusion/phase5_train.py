"""
Phase 5 — Training Loop
========================
Trains the fusion module + projection heads on pseudo-paired data.
SwinTransformer and MLP backbones remain fully frozen throughout.

Why projection happens here, not in the dataset
-------------------------------------------------
The .npy cache files store RAW backbone features (1024-dim for Swin,
64-dim for MLP).  Projection to PROJ_DIM=256 is applied LIVE on every
training batch using the current projection head weights.

This is critical for correctness: if projection were applied once at
dataset-build time, updating the projection head weights during training
would have zero effect on the cached features — the model would be
learning to classify features that never change, causing artificially
inflated accuracy from the very first epoch.

Live projection ensures that as the projection heads learn a better
shared embedding space, every subsequent batch sees the improved projections.

Training outputs
----------------
  checkpoints/best_model.pt   — best validation accuracy
  checkpoints/last_model.pt   — final epoch
  logs/train_log.csv          — per-epoch metrics
"""

import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from phase1_ontology import UNIFIED_CLASSES, NUM_UNIFIED, PROJ_DIM
from phase2_encoders import (ImageEncoder, MLPSensorEncoder,
                              get_device, dataloader_kwargs, DEVICE)
from phase3_dataset  import (
    PseudoPairDataset, build_val_pair_dataset,
    UnimodalWindowDataset, make_dataloaders,
    N_TRAIN, N_VAL, WINDOW,
)
from phase4_model    import TemporalFusionTransformer, count_parameters


# ── Hyper-parameters ──────────────────────────────────────────────────────────

CFG = dict(
    epochs          = 60,
    batch_size      = 128,
    lr              = 3e-4,
    weight_decay    = 0.01,
    label_smoothing = 0.1,
    grad_clip       = 1.0,
    n_heads         = 4,
    dropout         = 0.2,
    p_modal_drop    = 0.20,
    window          = WINDOW,
    feat_dim        = PROJ_DIM,
    n_train         = N_TRAIN,
    n_val           = N_VAL,
    checkpoint_dir  = "checkpoints",
    log_dir         = "logs",
    num_workers     = 4,
    device          = str(DEVICE),
)


# ── Live projection of a raw-feature batch ────────────────────────────────────

def project_batch(proj_layer, raw_window_batch, proj_dim=PROJ_DIM):
    """
    Apply a projection head to a [B, W, raw_dim] batch live.
    Returns [B, W, proj_dim].
    proj_layer is nn.Sequential(Linear, LayerNorm) — trainable.
    """
    B, W, D = raw_window_batch.shape
    flat = raw_window_batch.reshape(B * W, D)
    out  = proj_layer(flat)
    return out.reshape(B, W, proj_dim)


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(fusion_model, img_proj, sen_proj,
             val_loader, img_uni_ds, sen_uni_ds, device):
    """
    Evaluates fused accuracy, image-only accuracy, and sensor-only accuracy.
    Raw features from val_loader are projected live using current proj weights.
    UnimodalWindowDataset was built with a snapshot of current proj weights.
    """
    fusion_model.eval()
    criterion = nn.CrossEntropyLoss()

    # ── Fused ────────────────────────────────────────────────────────────────
    total_loss = correct = n = 0
    for img_w, sen_w, labels in val_loader:
        img_w  = img_w.to(device)
        sen_w  = sen_w.to(device)
        labels = labels.to(device)
        img_p  = project_batch(img_proj, img_w)
        sen_p  = project_batch(sen_proj, sen_w)
        logits = fusion_model(img_p, sen_p)
        total_loss += criterion(logits, labels).item()
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)

    # ── Image-only ───────────────────────────────────────────────────────────
    img_correct = img_n = 0
    uni_kw = dataloader_kwargs(device, 256, shuffle=False, num_workers=0)
    uni_kw.pop("drop_last")
    for win, labels in DataLoader(img_uni_ds, **uni_kw):
        win, labels = win.to(device), labels.to(device)
        logits = fusion_model(win, torch.zeros_like(win), force_sen_zero=True)
        img_correct += (logits.argmax(1) == labels).sum().item()
        img_n       += len(labels)

    # ── Sensor-only ──────────────────────────────────────────────────────────
    sen_correct = sen_n = 0
    for win, labels in DataLoader(sen_uni_ds, **uni_kw):
        win, labels = win.to(device), labels.to(device)
        logits = fusion_model(torch.zeros_like(win), win, force_img_zero=True)
        sen_correct += (logits.argmax(1) == labels).sum().item()
        sen_n       += len(labels)

    return dict(
        fused_loss = total_loss / max(len(val_loader), 1),
        fused_acc  = correct    / max(n,      1),
        img_acc    = img_correct / max(img_n, 1),
        sen_acc    = sen_correct / max(sen_n, 1),
    )


# ── Main training function ────────────────────────────────────────────────────

def train(
    img_enc,
    sen_enc,
    img_feats_tr, img_labels_tr,   # raw backbone features [N, 1024]
    sen_feats_tr, sen_labels_tr,   # raw backbone features [M, 64]
    img_feats_va, img_labels_va,
    sen_feats_va, sen_labels_va,
    cfg: dict = CFG,
):
    """
    Trains TemporalFusionTransformer on pseudo-paired raw feature windows.

    Trainable parameters (everything else frozen):
        fusion_model — all layers
        img_enc.proj — Linear(1024→256) + LayerNorm
        sen_enc.proj — Linear(64→256)   + LayerNorm

    Raw features are projected live on every batch so that the projection
    head update is reflected immediately in the next batch's features.
    """
    device  = torch.device(cfg["device"])
    nw      = cfg.get("num_workers", 4)
    use_amp = (device.type == "cuda")

    print(f"\nDevice       : {device}")
    print(f"AMP          : {'enabled' if use_amp else 'disabled'}")
    print(f"num_workers  : {0 if device.type == 'mps' else nw}")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"],        exist_ok=True)

    # ── Build pseudo-pair datasets from raw cached features ──────────────────
    print("\nBuilding pseudo-pair datasets (raw features, no projection yet) ...")
    train_ds = PseudoPairDataset.from_cache(
        img_feats_tr, img_labels_tr,
        sen_feats_tr, sen_labels_tr,
        n_samples = cfg["n_train"],
        window    = cfg["window"],
    )
    val_ds = build_val_pair_dataset(
        img_feats_va, img_labels_va,
        sen_feats_va, sen_labels_va,
        n_samples = cfg["n_val"],
        window    = cfg["window"],
    )

    train_loader, val_loader = make_dataloaders(
        train_ds, val_ds,
        batch_size  = cfg["batch_size"],
        num_workers = nw,
        device      = device,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    fusion_model = TemporalFusionTransformer(
        feat_dim     = cfg["feat_dim"],
        n_heads      = cfg["n_heads"],
        window       = cfg["window"],
        n_classes    = NUM_UNIFIED,
        dropout      = cfg["dropout"],
        p_modal_drop = cfg["p_modal_drop"],
    ).to(device)

    img_enc = img_enc.to(device)
    sen_enc = sen_enc.to(device)

    total, trainable = count_parameters(fusion_model)
    print(f"\nFusion model : {total:,} params ({trainable:,} trainable)")
    print(f"img proj     : {img_enc.proj[0].in_features} → "
          f"{img_enc.proj[0].out_features} (trainable)")
    print(f"sen proj     : {sen_enc.proj[0].in_features} → "
          f"{sen_enc.proj[0].out_features} (trainable)")

    # ── Optimiser: fusion + projection heads only ─────────────────────────────
    trainable_params = (
        list(fusion_model.parameters()) +
        list(img_enc.proj.parameters()) +
        list(sen_enc.proj.parameters())
    )
    optimizer  = AdamW(trainable_params, lr=cfg["lr"],
                        weight_decay=cfg["weight_decay"])
    scheduler  = CosineAnnealingLR(optimizer, T_max=cfg["epochs"],
                                    eta_min=1e-6)
    criterion  = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler     = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Build unimodal val datasets once (snapshot of initial proj weights) ──
    # These are rebuilt each epoch so they reflect current projection weights.
    def _build_uni_datasets():
        img_uni = UnimodalWindowDataset(
            img_feats_va, img_labels_va,
            window=cfg["window"],
            proj=img_enc.proj, proj_device=device)
        sen_uni = UnimodalWindowDataset(
            sen_feats_va, sen_labels_va,
            window=cfg["window"],
            proj=sen_enc.proj, proj_device=device)
        return img_uni, sen_uni

    # ── CSV logger ────────────────────────────────────────────────────────────
    log_path = os.path.join(cfg["log_dir"], "train_log.csv")
    log_f    = open(log_path, "w", newline="")
    writer   = csv.DictWriter(log_f, fieldnames=[
        "epoch", "train_loss", "train_acc",
        "val_fused_loss", "val_fused_acc",
        "val_img_acc", "val_sen_acc", "lr", "epoch_time_s",
    ])
    writer.writeheader()

    best_val_acc = 0.0
    header = (f"{'Epoch':>6} {'TrLoss':>8} {'TrAcc':>7} "
              f"{'VaLoss':>8} {'VaAcc':>7} "
              f"{'ImgAcc':>7} {'SenAcc':>7} {'LR':>9}")
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        fusion_model.train()
        ep_loss = ep_correct = ep_n = 0

        for img_w, sen_w, labels in train_loader:
            # img_w: [B, W, img_raw_dim]  e.g. [128, 8, 1024]
            # sen_w: [B, W, sen_raw_dim]  e.g. [128, 8, 64]
            img_w  = img_w.to(device)
            sen_w  = sen_w.to(device)
            labels = labels.to(device)

            # ── Project raw features live with current head weights ────────
            img_p = project_batch(img_enc.proj, img_w)  # [B, W, 256]
            sen_p = project_batch(sen_enc.proj, sen_w)  # [B, W, 256]

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = fusion_model(img_p, sen_p)
                    loss   = criterion(logits, labels)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = fusion_model(img_p, sen_p)
                loss   = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg["grad_clip"])
                optimizer.step()

            ep_loss    += loss.item()
            ep_correct += (logits.argmax(1) == labels).sum().item()
            ep_n       += len(labels)

        scheduler.step()
        train_acc  = ep_correct / ep_n
        train_loss = ep_loss / len(train_loader)
        ep_time    = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        # Rebuild unimodal datasets with current proj weights for honest eval
        img_uni, sen_uni = _build_uni_datasets()

        vm = validate(fusion_model, img_enc.proj, sen_enc.proj,
                      val_loader, img_uni, sen_uni, device)

        print(f"{epoch:>6} {train_loss:>8.4f} {train_acc*100:>6.1f}% "
              f"{vm['fused_loss']:>8.4f} {vm['fused_acc']*100:>6.1f}% "
              f"{vm['img_acc']*100:>6.1f}% {vm['sen_acc']*100:>6.1f}% "
              f"{current_lr:>9.2e}  [{ep_time:.0f}s]")

        writer.writerow({
            "epoch":          epoch,
            "train_loss":     round(train_loss,         5),
            "train_acc":      round(train_acc,           5),
            "val_fused_loss": round(vm["fused_loss"],   5),
            "val_fused_acc":  round(vm["fused_acc"],    5),
            "val_img_acc":    round(vm["img_acc"],      5),
            "val_sen_acc":    round(vm["sen_acc"],      5),
            "lr":             current_lr,
            "epoch_time_s":   round(ep_time, 1),
        })
        log_f.flush()

        ckpt = dict(
            epoch         = epoch,
            fusion_model  = fusion_model.state_dict(),
            img_proj      = img_enc.proj.state_dict(),
            sen_proj      = sen_enc.proj.state_dict(),
            optimizer     = optimizer.state_dict(),
            val_fused_acc = vm["fused_acc"],
            cfg           = cfg,
        )
        torch.save(ckpt, os.path.join(cfg["checkpoint_dir"], "last_model.pt"))
        if vm["fused_acc"] > best_val_acc:
            best_val_acc = vm["fused_acc"]
            torch.save(ckpt, os.path.join(cfg["checkpoint_dir"], "best_model.pt"))
            print(f"           *** new best: {best_val_acc*100:.2f}% ***")

    log_f.close()
    print(f"\nTraining complete. Best val fused acc: {best_val_acc*100:.2f}%")
    print(f"Checkpoints saved to: {cfg['checkpoint_dir']}/")
    return fusion_model, img_enc, sen_enc


# ── Load checkpoint ───────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path, img_enc, sen_enc,
                    device: torch.device = DEVICE):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["cfg"]
    fm   = TemporalFusionTransformer(
        feat_dim     = cfg["feat_dim"],
        n_heads      = cfg["n_heads"],
        window       = cfg["window"],
        n_classes    = NUM_UNIFIED,
        dropout      = cfg["dropout"],
        p_modal_drop = cfg["p_modal_drop"],
    )
    fm.load_state_dict(ckpt["fusion_model"])
    img_enc.proj.load_state_dict(ckpt["img_proj"])
    sen_enc.proj.load_state_dict(ckpt["sen_proj"])
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}  "
          f"val_acc={ckpt['val_fused_acc']*100:.2f}%")
    return fm, img_enc, sen_enc


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    class DummySwin(nn.Module):
        def __init__(self): super().__init__(); self.head = nn.Linear(1024, 6)
        def forward(self, x): return torch.randn(x.shape[0], 1024)

    class DummyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(561, 512), nn.ReLU(), nn.BatchNorm1d(512),
                nn.Linear(512, 64),  nn.ReLU(), nn.BatchNorm1d(64),
                nn.Linear(64, 6),
            )
            self.classifier = self.net[-1]
        def forward(self, x): return self.net(x)

    ie = ImageEncoder(DummySwin(), 1024)
    se = MLPSensorEncoder(DummyMLP().net, mlp_feat_dim=64, is_sequential=True)

    # Simulate raw cached features at correct backbone dims
    N = 400
    img_f = np.random.randn(N,          1024).astype(np.float32)  # Swin raw
    img_l = np.repeat([0, 3, 4, 5],     N // 4)
    sen_f = np.random.randn(N * 6 // 4, 64).astype(np.float32)    # MLP raw
    sen_l = np.repeat(np.arange(6),     N // 4)

    test_cfg = {**CFG,
                "epochs": 3, "n_train": 200, "n_val": 60,
                "batch_size": 32, "num_workers": 0,
                "checkpoint_dir": "/tmp/ckpt_test",
                "log_dir": "/tmp/log_test",
                "device": str(get_device())}

    train(ie, se,
          img_f, img_l, sen_f, sen_l,
          img_f, img_l, sen_f, sen_l,
          cfg=test_cfg)
    print("\nPhase 5 smoke test passed.")