"""
Phase 1 — Ontology Bridge
=========================
Defines the exact unified class space and all cross-dataset mappings.

Unified classes (6 total):
  0 = walking
  1 = walking_upstairs
  2 = walking_downstairs
  3 = sitting
  4 = standing
  5 = laying

Key insight
-----------
In the Kaggle image dataset, stairs activity is indistinguishable from
flat walking — cameras cannot see incline from a side/front view.
Therefore:
  • walking_upstairs   sensor windows  →  pair with  walking/running  images
  • walking_downstairs sensor windows  →  pair with  walking/running  images

This means ALL 6 unified classes now have image partners for pseudo-pairing.
None are "sensor-only" anymore — every class gets real cross-modal pairs.

IMAGE_PAIRING_MAP[sensor_unified_class] = image_unified_class_to_use_for_pairing
"""

import numpy as np

# ── Unified label space ──────────────────────────────────────────────────────

UNIFIED_CLASSES = [
    "walking",            # 0
    "walking_upstairs",   # 1
    "walking_downstairs", # 2
    "sitting",            # 3
    "standing",           # 4
    "laying",             # 5
]
NUM_UNIFIED = len(UNIFIED_CLASSES)   # 6
PROJ_DIM    = 256                    # shared projection dim for both encoders

# ── Cross-dataset pairing map ─────────────────────────────────────────────────
# For each sensor unified class, which image unified class should be
# used when constructing pseudo-pairs?
#
#  sensor class          image class used for pairing     rationale
#  ─────────────────     ─────────────────────────────    ──────────────────────
#  0 walking          →  0 walking                        direct match
#  1 walking_upstairs →  0 walking  (+ running)           stairs ≈ walking in images
#  2 walking_downstrs →  0 walking  (+ running)           stairs ≈ walking in images
#  3 sitting          →  3 sitting                        direct match
#  4 standing         →  4 standing                       direct match
#  5 laying           →  5 laying   (sleeping in images)  direct match

IMAGE_PAIRING_MAP = {
    0: 0,   # walking            pairs with  walking images
    1: 0,   # walking_upstairs   pairs with  walking images (incl. running)
    2: 0,   # walking_downstairs pairs with  walking images (incl. running)
    3: 3,   # sitting            pairs with  sitting images
    4: 4,   # standing           pairs with  standing images
    5: 5,   # laying             pairs with  laying/sleeping images
}

# All 6 classes participate in cross-modal pairing — none are image-absent.
OVERLAP_UNIFIED     = set(IMAGE_PAIRING_MAP.keys())   # {0,1,2,3,4,5}
SENSOR_ONLY_UNIFIED = set()                           # empty — all have image partners

# ── UCI-HAR sensor label mapping ─────────────────────────────────────────────
# Raw labels are 1-indexed integers (1–6) in the standard UCI-HAR files.

SENSOR_RAW_TO_UNIFIED = {
    1: 0,   # WALKING            → walking
    2: 1,   # WALKING_UPSTAIRS   → walking_upstairs
    3: 2,   # WALKING_DOWNSTAIRS → walking_downstairs
    4: 3,   # SITTING            → sitting
    5: 4,   # STANDING           → standing
    6: 5,   # LAYING             → laying
}

# Some UCI-HAR versions ship 0-indexed labels (0–5).
SENSOR_0IDX_TO_UNIFIED = {i: i for i in range(6)}

# ── Kaggle image HAR label mapping ───────────────────────────────────────────
# Maps folder/class name strings → unified index.
# -1 means excluded (class not used in this project).

IMAGE_STR_TO_UNIFIED = {
    # ── Included classes ─────────────────────────────────────────────────────
    "walking":  0,   # walking  → walking
    "running":  0,   # running  → walking  (stairs look like walking in images)
    "sitting":  3,   # sitting  → sitting
    "standing": 4,   # standing → standing
    "sleeping": 5,   # sleeping → laying

    # ── Excluded classes ─────────────────────────────────────────────────────
    "archery":           -1,
    "boxing":            -1,
    "cricket_batting":   -1,
    "cricket_bowling":   -1,
    "cycling":           -1,
    "golf_swing":        -1,
    "jump_rope":         -1,
    "jumping_jack":      -1,
    "kayaking":          -1,
    "pull_ups":          -1,
    "push_ups":          -1,
    "rowing":            -1,
    "swimming":          -1,
    "table_tennis_shot": -1,
    "tennis_swing":      -1,
    "using_laptop":      -1,
    "weightlifting":     -1,
}

# ── Convenience functions ─────────────────────────────────────────────────────

def remap_sensor_labels(raw_labels, one_indexed=True):
    """Convert raw UCI-HAR integer labels → unified indices (all 0–5, no -1)."""
    mapping = SENSOR_RAW_TO_UNIFIED if one_indexed else SENSOR_0IDX_TO_UNIFIED
    return np.array([mapping[int(l)] for l in raw_labels])


def remap_image_labels(str_labels):
    """
    Convert Kaggle folder-name strings → unified indices.
    Returns -1 for excluded classes.
    """
    return np.array([
        IMAGE_STR_TO_UNIFIED.get(str(l).lower().strip(), -1)
        for l in str_labels
    ])


def filter_image_dataset(features, labels_unified):
    """Drop rows where unified label == -1 (excluded image classes)."""
    mask = labels_unified != -1
    return features[mask], labels_unified[mask]


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Unified class space")
    print("=" * 60)
    print(f"  {'Idx':<4}  {'Unified class':<22}  {'Image partner used'}")
    print("-" * 60)
    for i, name in enumerate(UNIFIED_CLASSES):
        img_partner_idx  = IMAGE_PAIRING_MAP[i]
        img_partner_name = UNIFIED_CLASSES[img_partner_idx]
        note = " (via walking images)" if i in {1, 2} else ""
        print(f"  {i:<4}  {name:<22}  {img_partner_name}{note}")

    print(f"\nOVERLAP_UNIFIED     = {sorted(OVERLAP_UNIFIED)}   (all 6 classes)")
    print(f"SENSOR_ONLY_UNIFIED = {sorted(SENSOR_ONLY_UNIFIED)}  (none — all classes have image partners)")

    print("\nSensor label mapping (UCI-HAR 1-indexed → unified):")
    for k, v in SENSOR_RAW_TO_UNIFIED.items():
        print(f"  {k} → {v}  ({UNIFIED_CLASSES[v]})")

    print("\nImage label mapping (Kaggle string → unified):")
    for k, v in IMAGE_STR_TO_UNIFIED.items():
        tag = UNIFIED_CLASSES[v] if v != -1 else "EXCLUDED"
        print(f"  {k:<25} → {v:>2}  ({tag})")

    # Assertions
    raw    = np.array([1, 2, 3, 4, 5, 6])
    mapped = remap_sensor_labels(raw, one_indexed=True)
    assert list(mapped) == [0, 1, 2, 3, 4, 5], "Sensor mapping failed"

    img_strs  = ["walking", "running", "sitting", "standing",
                 "sleeping", "boxing", "cycling"]
    img_mapped = remap_image_labels(img_strs)
    assert list(img_mapped) == [0, 0, 3, 4, 5, -1, -1], "Image mapping failed"

    # Verify stairs classes map to walking images
    assert IMAGE_PAIRING_MAP[1] == 0, "upstairs should pair with walking(0)"
    assert IMAGE_PAIRING_MAP[2] == 0, "downstairs should pair with walking(0)"

    print("\nAll assertions passed.")