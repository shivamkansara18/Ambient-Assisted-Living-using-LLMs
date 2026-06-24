"""
Phase 2 — Data Loading & Feature Extraction
============================================
Loads UCI-HAR sensor data (flat .txt format) and Kaggle image HAR data
(folder-based / ImageFolder format).  Builds encoder wrappers, freezes
pretrained backbones, adds projection heads, and caches all features.

Mac GPU (MPS) support
---------------------
Apple Silicon GPUs are exposed via torch.backends.mps.  This file detects
them automatically:  CUDA > MPS > CPU.  All DataLoaders are configured
to be compatible with MPS (pin_memory=False, num_workers=0 on MPS).

Kaggle image dataset — two folder layouts supported
-----------------------------------------------------
Layout A — pre-split (train/val already separated):
    kaggle_har/
        train/
            walking/  running/  sitting/  ...
        val/
            walking/  running/  ...

Layout B — single root (code does the 80/20 split):
    kaggle_har/
        walking/  running/  sitting/  sleeping/  ...

Set `pre_split=True` in KaggleHARImageDataset for Layout A.

UCI-HAR sensor dataset layout (unchanged):
    ucihar/
        train/
            X_train.txt   (7352 × 561)
            y_train.txt   (7352,)  labels 1–6
        test/
            X_test.txt    (2947 × 561)
            y_test.txt    (2947,)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from phase1_ontology import (
    remap_sensor_labels, remap_image_labels,
    filter_image_dataset, NUM_UNIFIED, PROJ_DIM
)

SWIN_FEAT_DIM = 768    # SwinTransformer-Base/Small/Tiny output dim
MLP_FEAT_DIM  = 256    # YOUR MLP last hidden layer dim — adjust to match yours
BATCH_SIZE    = 64


# ── Device detection: CUDA > MPS (Apple Silicon) > CPU ───────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()


def dataloader_kwargs(device: torch.device, batch_size: int,
                      shuffle: bool, num_workers: int = 4) -> dict:
    """
    Returns DataLoader keyword arguments compatible with the target device.
    MPS does not support pin_memory, and multiprocessing can be unstable
    on some macOS setups — so we force num_workers=0 and pin_memory=False
    when running on MPS.
    """
    on_mps = (device.type == "mps")
    return dict(
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = 0 if on_mps else num_workers,
        pin_memory  = False if on_mps else True,
        drop_last   = False,
    )


# ── UCI-HAR sensor dataset ────────────────────────────────────────────────────

class UCIHARDataset(Dataset):
    """
    Loads flat UCI-HAR feature files.
    X_train.txt : (7352, 561) — 561 features per 2.56-sec window
    y_train.txt : (7352,)     — integer labels 1–6
    """
    def __init__(self, split="train", data_root="data/ucihar"):
        assert split in ("train", "test"), \
            f"split must be 'train' or 'test', got '{split}'"
        X_path = os.path.join(data_root, split, f"X_{split}.txt")
        y_path = os.path.join(data_root, split, f"y_{split}.txt")

        if not os.path.exists(X_path):
            raise FileNotFoundError(
                f"UCI-HAR feature file not found: {X_path}\n"
                f"Expected structure: {data_root}/train/X_train.txt  "
                f"and {data_root}/test/X_test.txt")

        X     = np.loadtxt(X_path, dtype=np.float32)
        y_raw = np.loadtxt(y_path, dtype=int)
        self.X = torch.from_numpy(X)
        self.y = torch.LongTensor(remap_sensor_labels(y_raw, one_indexed=True))
        print(f"UCI-HAR {split}: {len(self.X)} samples, "
              f"classes={sorted(set(self.y.tolist()))}")

    def __len__(self):          return len(self.X)
    def __getitem__(self, i):   return self.X[i], self.y[i]


# ── Kaggle image HAR dataset ──────────────────────────────────────────────────

def _img_transform(train: bool) -> transforms.Compose:
    """Standard ImageNet-normalised transforms matching SwinTransformer training."""
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])


class KaggleHARImageDataset(Dataset):
    """
    Folder-based image HAR dataset with unified label remapping.

    Supports two layouts:

    Layout A — pre-split folders (pre_split=True):
        root/
            train/
                walking/  *.jpg ...
                running/  *.jpg ...
                sitting/  *.jpg ...
                ...
            val/
                walking/  ...

    Layout B — single root, code splits 80/20 (pre_split=False, default):
        root/
            walking/  *.jpg ...
            running/  *.jpg ...
            ...

    In both cases:
      - Classes not in IMAGE_STR_TO_UNIFIED, or mapped to -1, are silently excluded.
      - JPEG, PNG, and most PIL-readable formats are supported.
    """

    def __init__(
        self,
        root:       str,
        train:      bool  = True,
        pre_split:  bool  = False,   # True if root/train/ and root/val/ exist
        val_split:  float = 0.2,     # ignored when pre_split=True
        seed:       int   = 42,
    ):
        self.transform = _img_transform(train)

        if pre_split:
            split_name = "train" if train else "val"
            split_root = os.path.join(root, split_name)
            if not os.path.isdir(split_root):
                raise FileNotFoundError(
                    f"pre_split=True but folder not found: {split_root}\n"
                    f"Expected: {root}/train/ and {root}/val/")
            base = ImageFolder(root=split_root)
            all_samples = base.samples
            chosen_idx  = list(range(len(all_samples)))
        else:
            if not os.path.isdir(root):
                raise FileNotFoundError(f"Image root not found: {root}")
            base       = ImageFolder(root=root)
            rng        = np.random.default_rng(seed)
            all_idx    = np.arange(len(base))
            rng.shuffle(all_idx)
            split_pt   = int(len(all_idx) * (1 - val_split))
            chosen_idx = (all_idx[:split_pt] if train
                          else all_idx[split_pt:]).tolist()
            all_samples = base.samples

        # Build unified-label mapping for this ImageFolder's class list
        class_to_unified = {
            base.class_to_idx[c]: int(remap_image_labels([c])[0])
            for c in base.classes
        }

        # Filter: keep only samples whose class maps to a valid unified label
        self.samples: list[tuple[str, int]] = []
        skipped: dict[str, int] = {}
        for i in chosen_idx:
            path, orig_lbl = all_samples[i]
            unified = class_to_unified[orig_lbl]
            if unified == -1:
                cls_name = base.classes[orig_lbl]
                skipped[cls_name] = skipped.get(cls_name, 0) + 1
            else:
                self.samples.append((path, unified))

        if skipped:
            print(f"  [excluded] "
                  + ", ".join(f"{k}({v})" for k, v in sorted(skipped.items())))
        print(f"KaggleHAR {'train' if train else 'val'} "
              f"({'pre-split' if pre_split else 'auto-split'}): "
              f"{len(self.samples)} samples  "
              f"classes={sorted(set(s[1] for s in self.samples))}")

    def __len__(self):  return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ── Image encoder (SwinTransformer) ──────────────────────────────────────────

class ImageEncoder(nn.Module):
    """
    Wraps a pretrained SwinTransformer (timm or torchvision).

    For timm models, we call model.reset_classifier(0) which keeps the
    global average pooling but replaces the final linear layer with Identity,
    yielding a flat [B, feat_dim] feature vector.

    For models where reset_classifier is not available (e.g. torchvision),
    we fall back to setting model.head = nn.Identity().  In that case make
    sure your model's forward() already applies global average pooling before
    the head, otherwise set swin_feat_dim to match the spatial output.
    """

    def __init__(self, swin_model, swin_feat_dim: int = SWIN_FEAT_DIM,
                 proj_dim: int = PROJ_DIM):
        super().__init__()

        # Strip classifier but keep pooling.
        # timm models expose reset_classifier(num_classes) for this.
        if hasattr(swin_model, "reset_classifier"):
            swin_model.reset_classifier(0)   # keeps global pool, removes fc
        else:
            # Fallback for torchvision / custom models
            swin_model.head = nn.Identity()

        self.backbone = swin_model
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(swin_feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()               # keep frozen backbone in eval always
        with torch.no_grad():
            feat = self.backbone(x)        # [B, swin_feat_dim]
        if feat.dim() > 2:
            # Safety: if backbone returns spatial map, apply global avg pool
            feat = feat.mean(dim=list(range(1, feat.dim() - 1)))
        return self.proj(feat)             # [B, proj_dim]


# ── Sensor encoder (MLP) ──────────────────────────────────────────────────────

class MLPSensorEncoder(nn.Module):
    """
    Wraps a pretrained MLP sensor classifier.

    How to determine mlp_feat_dim
    -------------------------------
    Run:  print(your_mlp)
    Find the last Linear layer BEFORE the final classification Linear.
    Its output size is mlp_feat_dim.

    Example:
        MLP(
          (layers): Sequential(
            Linear(561→512), ReLU,
            Linear(512→256), ReLU,   ← mlp_feat_dim = 256
            Linear(256→6)            ← head (removed)
          )
        )

    Usage variants
    ---------------
    # Named head attribute (most common):
    sen_enc = MLPSensorEncoder(your_mlp, mlp_feat_dim=256,
                                head_attr="classifier")

    # Plain nn.Sequential — drops last layer automatically:
    sen_enc = MLPSensorEncoder(your_mlp, mlp_feat_dim=256,
                                is_sequential=True)
    """

    def __init__(
        self,
        mlp_model,
        mlp_feat_dim:   int  = MLP_FEAT_DIM,
        proj_dim:       int  = PROJ_DIM,
        head_attr:      str  = "classifier",
        is_sequential:  bool = False,
    ):
        super().__init__()

        if is_sequential:
            layers   = list(mlp_model.children())
            backbone = nn.Sequential(*layers[:-1])
        else:
            if not hasattr(mlp_model, head_attr):
                available = [n for n, _ in mlp_model.named_children()]
                raise ValueError(
                    f"Attribute '{head_attr}' not found on MLP. "
                    f"Available children: {available}")
            setattr(mlp_model, head_attr, nn.Identity())
            backbone = mlp_model

        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False            # freeze MLP backbone weights
        self.backbone.eval()                   # BatchNorm1d → use running stats
                                               # Dropout → disabled at feature time

        self.proj = nn.Sequential(
            nn.Linear(mlp_feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self._expected_in = mlp_feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Re-apply eval every forward call: the training loop calls
        # fusion_model.train() which can recursively set child modules
        # (including this encoder) back to train mode, which would make
        # BatchNorm1d use batch statistics instead of running statistics.
        self.backbone.eval()
        with torch.no_grad():
            feat = self.backbone(x)            # [B, mlp_feat_dim]
        if feat.shape[-1] != self._expected_in:
            raise RuntimeError(
                f"MLP output dim {feat.shape[-1]} ≠ mlp_feat_dim="
                f"{self._expected_in}. Check head_attr / mlp_feat_dim.")
        return self.proj(feat)                 # [B, proj_dim]


# ── Backbone-only wrapper for caching ────────────────────────────────────────

class _BackboneOnly(nn.Module):
    """
    Thin wrapper that runs ONLY the frozen backbone (no projection head).
    Used exclusively during feature caching so that the stored .npy files
    contain raw backbone outputs (e.g. 1024-dim for Swin, 64-dim for MLP).

    The projection heads are intentionally excluded from caching because they
    are TRAINABLE — applying them at cache time would freeze their effect into
    the .npy files, meaning weight updates during training would have no effect
    on the cached features.  Instead, projection is applied live on every
    training batch so it always reflects the current head weights.
    """
    def __init__(self, encoder: nn.Module):
        super().__init__()
        # encoder is either ImageEncoder or MLPSensorEncoder
        self.backbone = encoder.backbone
        self._is_image = isinstance(encoder, ImageEncoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            feat = self.backbone(x)
        if feat.dim() > 2:          # spatial map from Swin → global avg pool
            feat = feat.mean(dim=list(range(1, feat.dim() - 1)))
        return feat                 # [B, raw_feat_dim]  e.g. [B, 1024] or [B, 64]


# ── Feature caching (backbone only — raw dims) ───────────────────────────────

# Bump this string any time the caching logic changes so stale caches are
# automatically rebuilt on the next run.
_CACHE_VERSION = "v2_raw"


def extract_and_cache(
    encoder:     nn.Module,
    dataset:     Dataset,
    save_prefix: str,
    device:      torch.device = DEVICE,
    batch_size:  int          = BATCH_SIZE,
    num_workers: int          = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the BACKBONE ONLY over the entire dataset, cache raw features to disk.
    The projection head is intentionally skipped — projection happens live
    during training so updating projection weights actually affects the model.

    Cache files are invalidated automatically when _CACHE_VERSION changes or
    when the stored feature dimension does not match the expected backbone dim.

    Returns
    -------
    features : np.ndarray [N, raw_backbone_dim]   e.g. (7352, 64) or (3200, 1024)
    labels   : np.ndarray [N]
    """
    os.makedirs(os.path.dirname(save_prefix) or ".", exist_ok=True)
    feat_path    = save_prefix + "_features.npy"
    label_path   = save_prefix + "_labels.npy"
    version_path = save_prefix + "_version.txt"

    # ── Cache hit: validate version stamp ────────────────────────────────────
    if (os.path.exists(feat_path) and
            os.path.exists(label_path) and
            os.path.exists(version_path)):
        saved_version = open(version_path).read().strip()
        if saved_version == _CACHE_VERSION:
            feats  = np.load(feat_path)
            labels = np.load(label_path)
            print(f"[cache hit]  {save_prefix}  shape={feats.shape}")
            return feats, labels
        else:
            print(f"[cache stale — version {saved_version} → {_CACHE_VERSION}]"
                  f"  rebuilding {save_prefix}")

    # ── Cache miss: extract backbone features ─────────────────────────────────
    wrapper = _BackboneOnly(encoder).to(device)
    wrapper.eval()

    kw = dataloader_kwargs(device, batch_size, shuffle=False,
                           num_workers=num_workers)
    kw.pop("drop_last")
    loader = DataLoader(dataset, **kw)

    all_feats, all_labels = [], []
    n_batches = len(loader)

    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(loader, 1):
            feats = wrapper(batch_x.to(device)).cpu().numpy()
            all_feats.append(feats)
            all_labels.append(batch_y.numpy())
            if i % 20 == 0 or i == n_batches:
                print(f"  extracting ... {i}/{n_batches}", end="\r", flush=True)

    print()
    features = np.concatenate(all_feats,  axis=0)
    labels   = np.concatenate(all_labels, axis=0)
    np.save(feat_path,  features)
    np.save(label_path, labels)
    open(version_path, "w").write(_CACHE_VERSION)
    print(f"[cached]  {save_prefix}  shape={features.shape}  "
          f"(raw backbone dim, no projection)")
    return features, labels


# ── Main pipeline entry: build encoders + cache everything ────────────────────

def build_and_cache_all(
    swin_model,
    mlp_model,
    swin_feat_dim:  int   = SWIN_FEAT_DIM,
    mlp_feat_dim:   int   = MLP_FEAT_DIM,
    mlp_head_attr:  str   = "classifier",
    mlp_is_seq:     bool  = False,
    kaggle_root:    str   = "data/kaggle_har",
    kaggle_pre_split: bool = False,    # True if kaggle_root/train/ and /val/ exist
    ucihar_root:    str   = "data/ucihar",
    cache_dir:      str   = "cache",
    device:         torch.device = DEVICE,
) -> dict:
    """
    Call this once before training.
    Builds both encoders, extracts all features, caches to `cache_dir/`.

    Returns
    -------
    dict with keys:
        img_enc, sen_enc  — encoder nn.Modules (proj heads trainable)
        img_train, img_val, sen_train, sen_val  — (features, labels) tuples
    """
    print(f"\nDevice: {device}")
    print(f"Building encoders ...")

    img_enc = ImageEncoder(swin_model, swin_feat_dim).to(device)
    sen_enc = MLPSensorEncoder(
        mlp_model, mlp_feat_dim,
        head_attr     = mlp_head_attr,
        is_sequential = mlp_is_seq,
    ).to(device)

    print(f"  ImageEncoder   proj: {swin_feat_dim} → {PROJ_DIM}")
    print(f"  SensorEncoder  proj: {mlp_feat_dim}  → {PROJ_DIM}")

    # ── Image datasets (Kaggle, folder-based) ────────────────────────────────
    print(f"\nLoading Kaggle image dataset from: {kaggle_root}")
    img_train_ds = KaggleHARImageDataset(
        kaggle_root, train=True,  pre_split=kaggle_pre_split)
    img_val_ds   = KaggleHARImageDataset(
        kaggle_root, train=False, pre_split=kaggle_pre_split)

    print("\nExtracting image features (train) ...")
    img_tr_f, img_tr_l = extract_and_cache(
        img_enc, img_train_ds, f"{cache_dir}/img_train", device=device)
    print("Extracting image features (val) ...")
    img_va_f, img_va_l = extract_and_cache(
        img_enc, img_val_ds,   f"{cache_dir}/img_val",   device=device)

    # ── Sensor datasets (UCI-HAR, flat .txt) ─────────────────────────────────
    print(f"\nLoading UCI-HAR sensor dataset from: {ucihar_root}")
    sen_train_ds = UCIHARDataset("train", ucihar_root)
    sen_val_ds   = UCIHARDataset("test",  ucihar_root)

    print("\nExtracting sensor features (train) ...")
    sen_tr_f, sen_tr_l = extract_and_cache(
        sen_enc, sen_train_ds, f"{cache_dir}/sen_train", device=device)
    print("Extracting sensor features (test) ...")
    sen_va_f, sen_va_l = extract_and_cache(
        sen_enc, sen_val_ds,   f"{cache_dir}/sen_val",   device=device)

    print(f"\nAll features cached to: {cache_dir}/")
    print(f"  img_train {img_tr_f.shape}  img_val {img_va_f.shape}")
    print(f"  sen_train {sen_tr_f.shape}  sen_val {sen_va_f.shape}")

    return {
        "img_enc":   img_enc,
        "sen_enc":   sen_enc,
        "img_train": (img_tr_f, img_tr_l),
        "img_val":   (img_va_f, img_va_l),
        "sen_train": (sen_tr_f, sen_tr_l),
        "sen_val":   (sen_va_f, sen_va_l),
    }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Detected device: {DEVICE}")

    # ── Test MLPSensorEncoder: named head ─────────────────────────────────────
    class DummyMLP_named(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers     = nn.Sequential(
                nn.Linear(561, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
            )
            self.classifier = nn.Linear(256, 6)
        def forward(self, x): return self.classifier(self.layers(x))

    se1 = MLPSensorEncoder(DummyMLP_named(), mlp_feat_dim=256,
                            head_attr="classifier")
    x   = torch.randn(4, 561)
    out = se1(x)
    assert out.shape == (4, PROJ_DIM)
    print(f"MLPSensorEncoder (named head)  → {out.shape}  OK")

    # ── Test MLPSensorEncoder: sequential ─────────────────────────────────────
    seq_mlp = nn.Sequential(
        nn.Linear(561, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 6),
    )
    se2  = MLPSensorEncoder(seq_mlp, mlp_feat_dim=128, is_sequential=True)
    out2 = se2(x)
    assert out2.shape == (4, PROJ_DIM)
    print(f"MLPSensorEncoder (sequential)  → {out2.shape}  OK")

    # ── Test ImageEncoder ─────────────────────────────────────────────────────
    class DummySwin(nn.Module):
        def __init__(self): super().__init__(); self.head = nn.Linear(768, 15)
        def forward(self, x): return torch.randn(x.shape[0], 768)

    ie   = ImageEncoder(DummySwin(), 768)
    img  = torch.randn(4, 3, 224, 224)
    out3 = ie(img)
    assert out3.shape == (4, PROJ_DIM)
    print(f"ImageEncoder                   → {out3.shape}  OK")

    # Freeze / trainable checks
    assert all(not p.requires_grad for p in se1.backbone.parameters())
    assert all(not p.requires_grad for p in ie.backbone.parameters())
    assert all(p.requires_grad for p in se1.proj.parameters())
    assert all(p.requires_grad for p in ie.proj.parameters())
    print("\nAll encoder smoke tests passed.")
    print(f"\nSet mlp_feat_dim to match your MLP's penultimate layer output size.")