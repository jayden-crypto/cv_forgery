

import argparse
import os
import sys
import pickle

import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix)

from feature_extraction import (
    extract_lbp_histogram,
    extract_gabor_features,
    extract_dwt_features,
    compute_ela_map,
    extract_ela_features_from_map,
    extract_noise_residual_features,
    build_gabor_bank,
)

# ────────────────────────────────────────────────────────────────
#  Settings
# ────────────────────────────────────────────────────────────────

BLOCK_SIZE = 64
BLOCK_STRIDE = 32
PCA_COMPONENTS = 30
# Max tampered blocks sampled per forged image (all within the mask).
# Authentic blocks per forged image are matched 1:1 to this count.
MAX_TAMPERED_PER_IMAGE = 30
# Authentic blocks sampled per authentic image.
SAMPLES_PER_AUTH_IMAGE = 20
# Fraction of a block's pixels that must overlap the tamper mask
# for the block to be labeled as tampered (class 1).
TAMPER_OVERLAP_THRESHOLD = 0.25
# ELA percentile above which a pixel is considered part of the tampered
# region when no ground-truth mask is available.
ELA_MASK_PERCENTILE = 80
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────

def _list_images(directory: str) -> list:
    import random
    paths = []
    for fname in sorted(os.listdir(directory)):
        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
            paths.append(os.path.join(directory, fname))
    random.seed(42)
    random.shuffle(paths)
    return paths

def _extract_block_features(block_gray: np.ndarray,
                            gabor_bank: list,
                            ela_patch: np.ndarray = None) -> np.ndarray:
    lbp   = extract_lbp_histogram(block_gray, radius=1, n_points=8)
    gabor = extract_gabor_features(block_gray, gabor_bank)
    dwt   = extract_dwt_features(block_gray, wavelet="haar", level=2)
    ela   = (extract_ela_features_from_map(ela_patch)
             if ela_patch is not None
             else extract_ela_features_from_map(compute_ela_map(block_gray)))
    noise = extract_noise_residual_features(block_gray)
    return np.concatenate([lbp, gabor, dwt, ela, noise])

def _load_gt_mask(masks_dir: str, image_path: str,
                  target_shape: tuple) -> np.ndarray:
    """
    Load the ground-truth binary mask for a tampered image.
    Tries both plain stem and CASIA-style stem_gt naming conventions.
    Returns a boolean array (True = tampered pixel), or None if not found.
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    candidates = []
    for suffix in ("", "_gt"):
        for ext in (".png", ".jpg", ".bmp", ".tif"):
            candidates.append(stem + suffix + ext)
    for fname in candidates:
        mask_path = os.path.join(masks_dir, fname)
        if os.path.isfile(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (target_shape[1], target_shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
                return mask > 127
    return None

def _ela_pseudo_mask(gray: np.ndarray,
                     quality: int = 90,
                     percentile: float = ELA_MASK_PERCENTILE) -> np.ndarray:
    """Fallback when no ground-truth mask is available."""
    ok, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return np.ones(gray.shape, dtype=bool)
    recompressed = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    ela = np.abs(gray.astype(np.float64) - recompressed.astype(np.float64))
    return ela >= np.percentile(ela, percentile)

def _sample_blocks(gray: np.ndarray, n_samples: int,
                   gabor_bank: list) -> list:
    """Sample n_samples random blocks; all labeled class 0 (authentic)."""
    h, w = gray.shape[:2]
    if h < BLOCK_SIZE or w < BLOCK_SIZE:
        return []
    ela_map  = compute_ela_map(gray)   # computed once per image
    features = []
    for _ in range(n_samples):
        r = np.random.randint(0, h - BLOCK_SIZE + 1)
        c = np.random.randint(0, w - BLOCK_SIZE + 1)
        block     = gray[r:r + BLOCK_SIZE, c:c + BLOCK_SIZE]
        ela_patch = ela_map[r:r + BLOCK_SIZE, c:c + BLOCK_SIZE]
        features.append(_extract_block_features(block, gabor_bank, ela_patch))
    return features

def _sample_blocks_mask_aware(gray: np.ndarray, gabor_bank: list,
                               mask: np.ndarray) -> tuple:
    """
    Enumerate all block positions on a regular grid, split into tampered
    (≥TAMPER_OVERLAP_THRESHOLD mask coverage) and authentic, then sample
    deliberately from each group:
      - Up to MAX_TAMPERED_PER_IMAGE tampered blocks  → label 1
      - Same count of authentic blocks from this image → label 0

    Guarantees every forged image contributes class-1 samples regardless of
    how small the tampered region is, and keeps within-image class balance.
    Returns (features, labels).
    """
    import random as _random
    h, w = gray.shape[:2]
    if h < BLOCK_SIZE or w < BLOCK_SIZE:
        return [], []

    ela_map = compute_ela_map(gray)   # computed once per image

    tampered_pos, authentic_pos = [], []
    for r in range(0, h - BLOCK_SIZE + 1, BLOCK_STRIDE):
        for c in range(0, w - BLOCK_SIZE + 1, BLOCK_STRIDE):
            block_mask = mask[r:r + BLOCK_SIZE, c:c + BLOCK_SIZE]
            if block_mask.sum() / block_mask.size >= TAMPER_OVERLAP_THRESHOLD:
                tampered_pos.append((r, c))
            else:
                authentic_pos.append((r, c))

    n_take = min(MAX_TAMPERED_PER_IMAGE, len(tampered_pos))
    if n_take == 0:
        return [], []

    chosen_t = _random.sample(tampered_pos, n_take)
    n_auth   = min(n_take, len(authentic_pos))
    chosen_a = _random.sample(authentic_pos, n_auth) if n_auth else []

    features, labels = [], []
    for r, c in chosen_t:
        ela_patch = ela_map[r:r + BLOCK_SIZE, c:c + BLOCK_SIZE]
        features.append(_extract_block_features(
            gray[r:r + BLOCK_SIZE, c:c + BLOCK_SIZE], gabor_bank, ela_patch))
        labels.append(1)
    for r, c in chosen_a:
        ela_patch = ela_map[r:r + BLOCK_SIZE, c:c + BLOCK_SIZE]
        features.append(_extract_block_features(
            gray[r:r + BLOCK_SIZE, c:c + BLOCK_SIZE], gabor_bank, ela_patch))
        labels.append(0)

    return features, labels

# ────────────────────────────────────────────────────────────────
#  Main Training Pipeline
# ────────────────────────────────────────────────────────────────

def _collect_authentic(auth_dirs: list, gabor_bank: list) -> tuple:
    X, y = [], []
    for auth_dir in auth_dirs:
        images = _list_images(auth_dir)
        print(f"[TRAIN]   {auth_dir}  ({len(images)} images)")
        for i, path in enumerate(images):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            feats = _sample_blocks(img, SAMPLES_PER_AUTH_IMAGE, gabor_bank)
            X.extend(feats)
            y.extend([0] * len(feats))
            if (i + 1) % 50 == 0:
                print(f"             {i + 1}/{len(images)}")
    return X, y


def _collect_tampered(spliced_dirs: list, masks_dirs: list,
                      gabor_bank: list,
                      image_level: bool = False) -> tuple:
    X, y = [], []
    for spliced_dir, masks_dir in zip(spliced_dirs, masks_dirs):
        images = _list_images(spliced_dir)
        if image_level:
            print(f"[TRAIN]   {spliced_dir}  ({len(images)} images)  [image-level labels]")
        else:
            using_gt = masks_dir is not None and os.path.isdir(masks_dir)
            label_src = f"GT masks: {masks_dir}" if using_gt else "ELA pseudo-masks"
            print(f"[TRAIN]   {spliced_dir}  ({len(images)} images)  [{label_src}]")
        gt_found = gt_missing = 0
        for i, path in enumerate(images):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if image_level:
                # All blocks from forged images → class 1, no mask needed
                feats = _sample_blocks(img, SAMPLES_PER_AUTH_IMAGE, gabor_bank)
                X.extend(feats)
                y.extend([1] * len(feats))
            else:
                using_gt = masks_dir is not None and os.path.isdir(masks_dir)
                if using_gt:
                    mask = _load_gt_mask(masks_dir, path, img.shape)
                    if mask is not None:
                        gt_found += 1
                    else:
                        gt_missing += 1
                        mask = _ela_pseudo_mask(img)
                else:
                    mask = _ela_pseudo_mask(img)
                feats, labels = _sample_blocks_mask_aware(img, gabor_bank, mask)
                X.extend(feats)
                y.extend(labels)
            if (i + 1) % 50 == 0:
                print(f"             {i + 1}/{len(images)}")
        if not image_level and (masks_dir is not None and os.path.isdir(masks_dir)):
            print(f"             GT found: {gt_found}  ELA fallback: {gt_missing}")
    return X, y


def train(authentic_dirs: list, spliced_dirs: list,
          masks_dirs: list = None,
          image_level: bool = False,
          output_path: str = "splicing_svm_model.pkl") -> None:

    if masks_dirs is None:
        masks_dirs = [None] * len(spliced_dirs)
    while len(masks_dirs) < len(spliced_dirs):
        masks_dirs.append(None)

    gabor_bank = build_gabor_bank()

    print("[TRAIN] ── Authentic images ──────────────────────────")
    X_auth, y_auth = _collect_authentic(authentic_dirs, gabor_bank)
    print(f"         Total authentic blocks: {len(X_auth)}")

    print("[TRAIN] ── Tampered images ───────────────────────────")
    X_spl, y_spl = _collect_tampered(spliced_dirs, masks_dirs, gabor_bank, image_level)
    n_tampered         = sum(y_spl)
    n_auth_from_forged = len(y_spl) - n_tampered
    print(f"         Tampered blocks:                  {n_tampered}")
    if not image_level:
        print(f"         Authentic blocks (forged images): {n_auth_from_forged}")

    X = np.array(X_auth + X_spl)
    y = np.array(y_auth + y_spl)

    n_class0 = int((y == 0).sum())
    n_class1 = int((y == 1).sum())
    print(f"[TRAIN] Class distribution — Authentic: {n_class0}  Tampered: {n_class1}")

    if n_class0 < 10 or n_class1 < 10:
        print("[ERROR] Not enough samples. Need at least 10 blocks per class.")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[TRAIN] Standardising features…")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    n_comp = min(PCA_COMPONENTS, X_train_s.shape[0] - 1, X_train_s.shape[1])
    print(f"[TRAIN] Applying PCA ({n_comp} components)…")
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca  = pca.transform(X_test_s)

    print("[TRAIN] Grid-searching SVM hyperparameters…")
    svm = GridSearchCV(
        LinearSVC(dual=False, max_iter=2000, class_weight="balanced"),
        {"C": [0.01, 0.1, 1, 10, 100]},
        cv=3, scoring="f1", n_jobs=-1, verbose=1,
    )
    svm.fit(X_train_pca, y_train)

    print(f"[TRAIN] Best params: {svm.best_params_}")
    print(f"[TRAIN] Best CV F1:  {svm.best_score_:.4f}")

    y_pred = svm.predict(X_test_pca)
    print("\n[TRAIN] ─── Test Set Results ─────────────────────")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Authentic", "Spliced"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    pipeline = {"scaler": scaler, "pca": pca, "svm": svm.best_estimator_}
    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n[TRAIN] Model saved → {output_path}")

# ────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train splicing detection SVM on authentic + spliced images. "
                    "Pass multiple --authentic and --spliced dirs to combine datasets. "
                    "Optionally pair each --spliced dir with a --masks dir.")
    parser.add_argument("--authentic", required=True, nargs="+",
                        help="One or more directories of authentic images")
    parser.add_argument("--spliced", required=True, nargs="+",
                        help="One or more directories of tampered images")
    parser.add_argument("--masks", nargs="*", default=None,
                        help="Ground-truth mask directories, one per --spliced dir "
                             "(use '' to skip masks for a specific dataset). "
                             "Falls back to ELA pseudo-masks where omitted.")
    parser.add_argument("--image-level", action="store_true",
                        help="Label ALL blocks from spliced images as class 1 "
                             "(no mask logic). Fastest option; use when no masks "
                             "are available and ELA pseudo-masks are unreliable.")
    parser.add_argument("--output", default="splicing_svm_model.pkl",
                        help="Output model file path")
    args = parser.parse_args()

    masks = [m if m else None for m in args.masks] if args.masks else None
    train(args.authentic, args.spliced, masks, args.image_level, args.output)
