"""
Phase 3 — Pseudo-Pairing & Temporal Sequence Dataset
=====================================================
Bridges the domain gap between unsynced image and sensor datasets.

Key design: features stored in the dataset are RAW backbone outputs
(e.g. 1024-dim for Swin, 64-dim for MLP). Projection to PROJ_DIM=256
happens LIVE in the training loop on each batch, so that as projection
head weights update during training, those updates actually affect what
the fusion model sees.

If raw features were projected at dataset-build time, the projection
would be frozen at initial random weights and training the projection
head would have no effect — causing artificially inflated accuracy.

Pairing strategy (all 6 classes get real image partners)
---------------------------------------------------------
  sensor class          image class used for pairing
  ──────────────────    ──────────────────────────────
  walking           →   walking images
  walking_upstairs  →   walking images  (stairs ≈ walking in images)
  walking_downstrs  →   walking images  (stairs ≈ walking in images)
  sitting           →   sitting images
  standing          →   standing images
  laying            →   sleeping/laying images
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from phase1_ontology import (
    UNIFIED_CLASSES, NUM_UNIFIED, PROJ_DIM,
    IMAGE_PAIRING_MAP,
    OVERLAP_UNIFIED,
    SENSOR_ONLY_UNIFIED,
)
from phase2_encoders import get_device, dataloader_kwargs

WINDOW     = 8
N_TRAIN    = 30_000
N_VAL      = 6_000
FEAT_NOISE = 0.015


# ── Window sampler ────────────────────────────────────────────────────────────

def _sample_window(feats, valid_idx, noise_std=FEAT_NOISE):
    """Sample WINDOW rows (with replacement + noise) → [W, raw_dim]."""
    chosen = np.random.choice(valid_idx, size=WINDOW, replace=True)
    win    = feats[chosen].copy()
    if noise_std > 0:
        win += np.random.randn(*win.shape).astype(np.float32) * noise_std
    return win


# ── Pseudo-pair builder ───────────────────────────────────────────────────────

def build_pseudo_pair_dataset(
    img_feats,   img_labels,    # raw backbone features [N, img_raw_dim]
    sen_feats,   sen_labels,    # raw backbone features [M, sen_raw_dim]
    n_samples  = N_TRAIN,
    window     = WINDOW,
    noise_std  = FEAT_NOISE,
    seed       = 42,
):
    """
    Returns list of (img_win [W, img_raw_dim], sen_win [W, sen_raw_dim], label int).

    Features are stored at RAW backbone dimensions.
    Projection to PROJ_DIM happens in the training loop per-batch.

    For each sensor class, IMAGE_PAIRING_MAP defines which image class to
    sample from.  All 6 classes get real image feature partners.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    img_by = {c: np.where(img_labels == c)[0] for c in np.unique(img_labels)}
    sen_by = {c: np.where(sen_labels == c)[0] for c in np.unique(sen_labels)}

    active_sen_classes = [c for c in range(NUM_UNIFIED) if c in sen_by]
    per_class = n_samples // max(len(active_sen_classes), 1)

    samples      = []
    pairing_log  = {}

    for sen_cls in active_sen_classes:
        img_partner_cls = IMAGE_PAIRING_MAP[sen_cls]

        if img_partner_cls not in img_by:
            # Fallback: no image data for this partner class
            img_raw_dim = img_feats.shape[1]
            for _ in range(per_class):
                iw = np.zeros((window, img_raw_dim), dtype=np.float32)
                sw = _sample_window(sen_feats, sen_by[sen_cls], noise_std)
                samples.append((iw, sw.astype(np.float32), int(sen_cls)))
            pairing_log[UNIFIED_CLASSES[sen_cls]] = "zero-image (fallback)"
            continue

        for _ in range(per_class):
            iw = _sample_window(img_feats, img_by[img_partner_cls], noise_std)
            sw = _sample_window(sen_feats, sen_by[sen_cls],          noise_std)
            samples.append((iw.astype(np.float32),
                             sw.astype(np.float32), int(sen_cls)))

        pairing_log[UNIFIED_CLASSES[sen_cls]] = (
            f"→ image class '{UNIFIED_CLASSES[img_partner_cls]}'"
            + (" (shared pool)" if img_partner_cls != sen_cls else ""))

    rng.shuffle(samples)

    print(f"\nPseudo-pair dataset: {len(samples)} samples")
    print(f"  {'Sensor class':<22}  Paired with")
    for cls_name, note in pairing_log.items():
        print(f"  {cls_name:<22}  {note}")
    return samples


# ── Dataset ───────────────────────────────────────────────────────────────────

class PseudoPairDataset(Dataset):
    """
    Each item: (img_window [W, img_raw_dim], sen_window [W, sen_raw_dim], label).
    Raw dimensions — projection happens in the training loop.
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        iw, sw, lbl = self.samples[i]
        return (torch.from_numpy(iw),
                torch.from_numpy(sw),
                torch.tensor(lbl, dtype=torch.long))

    @staticmethod
    def from_cache(img_feats, img_labels, sen_feats, sen_labels,
                   n_samples=N_TRAIN, window=WINDOW, seed=42):
        s = build_pseudo_pair_dataset(
            img_feats, img_labels, sen_feats, sen_labels,
            n_samples=n_samples, window=window, seed=seed)
        return PseudoPairDataset(s)


def build_val_pair_dataset(img_feats, img_labels, sen_feats, sen_labels,
                            n_samples=N_VAL, window=WINDOW):
    return PseudoPairDataset.from_cache(
        img_feats, img_labels, sen_feats, sen_labels,
        n_samples=n_samples, window=window, seed=999)


class UnimodalWindowDataset(Dataset):
    """
    Single-modality windows for per-modality validation.
    Stores RAW backbone features; proj_layer is applied once at build time
    since this dataset is only used for evaluation (proj weights don't need
    to be live here — we just want a snapshot of current accuracy).
    """
    def __init__(self, feats, labels, window=WINDOW, n_samples=2000, seed=77,
                 proj=None, proj_device="cpu"):
        # Apply projection once for eval snapshot
        if proj is not None:
            proj = proj.to(proj_device).eval()
            chunks = []
            bs = 512
            with torch.no_grad():
                for start in range(0, len(feats), bs):
                    x   = torch.from_numpy(feats[start:start+bs]).to(proj_device)
                    out = proj(x).cpu().numpy()
                    chunks.append(out)
            feats_proj = np.concatenate(chunks, axis=0)
        else:
            assert feats.shape[1] == PROJ_DIM, \
                "Pass proj= if features are not already at PROJ_DIM"
            feats_proj = feats

        rng     = np.random.default_rng(seed)
        by_cls  = {c: np.where(labels == c)[0] for c in np.unique(labels)}
        per_cls = max(n_samples // max(len(by_cls), 1), 1)
        self.samples = []
        for cls, idx in by_cls.items():
            for _ in range(per_cls):
                chosen = np.random.choice(idx, size=window, replace=True)
                win    = feats_proj[chosen].astype(np.float32)
                self.samples.append((win, int(cls)))
        rng.shuffle(self.samples)

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        win, lbl = self.samples[i]
        return torch.from_numpy(win), torch.tensor(lbl, dtype=torch.long)


def make_dataloaders(train_ds, val_ds, batch_size=128, num_workers=4,
                     device=None):
    if device is None:
        device = get_device()
    tr_kw = dataloader_kwargs(device, batch_size, shuffle=True,
                               num_workers=num_workers)
    va_kw = dataloader_kwargs(device, batch_size, shuffle=False,
                               num_workers=num_workers)
    va_kw["drop_last"] = False
    return DataLoader(train_ds, **tr_kw), DataLoader(val_ds, **va_kw)


def report_class_distribution(dataset, name="dataset"):
    from collections import Counter
    counts = Counter(s[2] for s in dataset.samples)
    print(f"\nClass distribution — {name}")
    total = sum(counts.values())
    for idx in sorted(counts):
        pct = 100 * counts[idx] / total
        print(f"  {UNIFIED_CLASSES[idx]:<22}  {counts[idx]:>6}  {pct:>5.1f}%")


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch.nn as nn

    IMG_RAW, SEN_RAW, N = 1024, 64, 500

    sen_f = np.random.randn(N * 6, SEN_RAW).astype(np.float32)
    sen_l = np.repeat(np.arange(6), N)
    img_f = np.random.randn(N * 4, IMG_RAW).astype(np.float32)
    img_l = np.repeat([0, 3, 4, 5], N)

    train_ds = PseudoPairDataset.from_cache(
        img_f, img_l, sen_f, sen_l, n_samples=1200)
    val_ds   = build_val_pair_dataset(
        img_f, img_l, sen_f, sen_l, n_samples=240)

    report_class_distribution(train_ds, "train")

    loader, _ = make_dataloaders(train_ds, val_ds, batch_size=16, num_workers=0)
    bi, bs, bl = next(iter(loader))
    print(f"\nBatch — img:{bi.shape}  sen:{bs.shape}  lbl:{bl.shape}")
    assert bi.shape == (16, WINDOW, IMG_RAW), f"Wrong: {bi.shape}"
    assert bs.shape == (16, WINDOW, SEN_RAW), f"Wrong: {bs.shape}"

    # Verify stairs (1,2) have real image features (non-zero)
    for iw, sw, lbl in train_ds.samples[:200]:
        if lbl in (1, 2):
            assert np.any(iw != 0), f"Class {lbl} has zero image window"
    print("Stairs classes verified: real walking image features  OK")
    print("Phase 3 smoke test passed.")