

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
    build_gabor_bank,
)

# ────────────────────────────────────────────────────────────────
#  Settings
# ────────────────────────────────────────────────────────────────

BLOCK_SIZE = 64
BLOCK_STRIDE = 32
PCA_COMPONENTS = 30
SAMPLES_PER_IMAGE = 20
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
                            gabor_bank: list) -> np.ndarray:
    
    lbp = extract_lbp_histogram(block_gray, radius=1, n_points=8)
    gabor = extract_gabor_features(block_gray, gabor_bank)
    dwt = extract_dwt_features(block_gray, wavelet="haar", level=2)
    return np.concatenate([lbp, gabor, dwt])

def _sample_blocks(gray: np.ndarray, n_samples: int,
                   gabor_bank: list) -> list:
    
    h, w = gray.shape[:2]
    if h < BLOCK_SIZE or w < BLOCK_SIZE:
        return []

    features = []
    for _ in range(n_samples):
        r = np.random.randint(0, h - BLOCK_SIZE + 1)
        c = np.random.randint(0, w - BLOCK_SIZE + 1)
        block = gray[r:r + BLOCK_SIZE, c:c + BLOCK_SIZE]
        feat = _extract_block_features(block, gabor_bank)
        features.append(feat)
    return features

# ────────────────────────────────────────────────────────────────
#  Main Training Pipeline
# ────────────────────────────────────────────────────────────────

def train(authentic_dir: str, spliced_dir: str,
          output_path: str = "splicing_svm_model.pkl") -> None:
    
    gabor_bank = build_gabor_bank()

    print("[TRAIN] Extracting features from authentic images…")
    auth_images = _list_images(authentic_dir)
    print(f"         Found {len(auth_images)} authentic images")

    X_auth = []
    for i, path in enumerate(auth_images):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        feats = _sample_blocks(img, SAMPLES_PER_IMAGE, gabor_bank)
        X_auth.extend(feats)
        if (i + 1) % 50 == 0:
            print(f"         Processed {i + 1}/{len(auth_images)}")

    print(f"         Total authentic blocks: {len(X_auth)}")

    print("[TRAIN] Extracting features from spliced images…")
    spl_images = _list_images(spliced_dir)
    print(f"         Found {len(spl_images)} spliced images")

    X_spl = []
    for i, path in enumerate(spl_images):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        feats = _sample_blocks(img, SAMPLES_PER_IMAGE, gabor_bank)
        X_spl.extend(feats)
        if (i + 1) % 50 == 0:
            print(f"         Processed {i + 1}/{len(spl_images)}")

    print(f"         Total spliced blocks: {len(X_spl)}")

    if len(X_auth) < 10 or len(X_spl) < 10:
        print("[ERROR] Not enough samples.  Need at least 10 blocks per class.")
        sys.exit(1)

    X = np.array(X_auth + X_spl)
    y = np.array([0] * len(X_auth) + [1] * len(X_spl))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[TRAIN] Standardising features…")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_comp = min(PCA_COMPONENTS, X_train_s.shape[0] - 1, X_train_s.shape[1])
    print(f"[TRAIN] Applying PCA ({n_comp} components)…")
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)

    print("[TRAIN] Grid-searching SVM hyperparameters…")
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100]
    }
    svm = GridSearchCV(
        LinearSVC(dual=False, max_iter=2000),
        param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=3,
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

    pipeline = {
        "scaler": scaler,
        "pca": pca,
        "svm": svm.best_estimator_,
    }
    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n[TRAIN] Model saved → {output_path}")

# ────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train splicing detection SVM on authentic + spliced images")
    parser.add_argument("--authentic", required=True,
                        help="Directory of authentic (unmodified) images")
    parser.add_argument("--spliced", required=True,
                        help="Directory of spliced (forged) images")
    parser.add_argument("--output", default="splicing_svm_model.pkl",
                        help="Output model file path")
    args = parser.parse_args()

    train(args.authentic, args.spliced, args.output)