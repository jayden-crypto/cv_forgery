

import os
import pickle
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Optional

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
#  Configuration
# ────────────────────────────────────────────────────────────────

BLOCK_SIZE = 64
BLOCK_STRIDE = 32
ANOMALY_PERCENTILE = 85
MIN_FLAGGED_RATIO = 0.08
MAX_FLAGGED_RATIO = 0.85
PCA_COMPONENTS = 30

# ────────────────────────────────────────────────────────────────
#  Block-Level Feature Extraction
# ────────────────────────────────────────────────────────────────

def _extract_block_features(block_gray: np.ndarray,
                            gabor_bank: list,
                            ela_patch: np.ndarray = None) -> np.ndarray:

    lbp_hist   = extract_lbp_histogram(block_gray, radius=1, n_points=8)
    gabor_feat = extract_gabor_features(block_gray, gabor_bank)
    dwt_feat   = extract_dwt_features(block_gray, wavelet="haar", level=2)
    ela_feat   = (extract_ela_features_from_map(ela_patch)
                  if ela_patch is not None
                  else extract_ela_features_from_map(compute_ela_map(block_gray)))
    noise_feat = extract_noise_residual_features(block_gray)

    return np.concatenate([lbp_hist, gabor_feat, dwt_feat, ela_feat, noise_feat])

def extract_block_features_grid(gray: np.ndarray,
                                block_size: int = BLOCK_SIZE,
                                stride: int = BLOCK_STRIDE) -> tuple:
    
    h, w = gray.shape[:2]
    gabor_bank = build_gabor_bank()
    ela_map    = compute_ela_map(gray)   # computed once for the whole image

    features_list = []
    positions = []

    for r in range(0, h - block_size + 1, stride):
        for c in range(0, w - block_size + 1, stride):
            block     = gray[r:r + block_size, c:c + block_size]
            ela_patch = ela_map[r:r + block_size, c:c + block_size]
            feat = _extract_block_features(block, gabor_bank, ela_patch)
            features_list.append(feat)
            positions.append((r, c))

    if not features_list:
        return np.array([]), []

    return np.array(features_list), positions

# ────────────────────────────────────────────────────────────────
#  Unsupervised Anomaly Detection (Mahalanobis)
# ────────────────────────────────────────────────────────────────

def _mahalanobis_anomaly_scores(features: np.ndarray) -> np.ndarray:
    
    if len(features) < 5:
        return np.zeros(len(features))

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    n_comp = min(PCA_COMPONENTS, X.shape[0] - 1, X.shape[1])
    if n_comp < 2:
        return np.zeros(len(features))
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)

    mean = X_pca.mean(axis=0)
    cov = np.cov(X_pca, rowvar=False)
    try:
        cov_inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return np.zeros(len(features))

    diff = X_pca - mean
    left = diff @ cov_inv
    dists = np.sqrt(np.sum(left * diff, axis=1))

    return dists

# ────────────────────────────────────────────────────────────────
#  SVM-Based Classification (if model available)
# ────────────────────────────────────────────────────────────────

def _svm_classify_blocks(features: np.ndarray,
                         model_path: str) -> tuple:
    
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    scaler = pipeline["scaler"]
    pca = pipeline["pca"]
    svm = pipeline["svm"]

    X = scaler.transform(features)
    X_pca = pca.transform(X)

    labels = svm.predict(X_pca)
    distances = svm.decision_function(X_pca)

    return labels, distances

# ────────────────────────────────────────────────────────────────
#  Heatmap Construction
# ────────────────────────────────────────────────────────────────

def _build_heatmap(image_shape: tuple,
                   positions: list,
                   scores: np.ndarray,
                   block_size: int = BLOCK_SIZE) -> np.ndarray:
    
    h, w = image_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.float64)

    for (r, c), score in zip(positions, scores):
        heatmap[r:r + block_size, c:c + block_size] += score
        count[r:r + block_size, c:c + block_size] += 1

    count = np.maximum(count, 1)
    heatmap /= count

    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = (heatmap * 255).astype(np.uint8)

    return heatmap

# ────────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────────

def detect_splicing(preprocessed: dict,
                    model_path: Optional[str] = None) -> dict:
    
    gray = preprocessed["y_eq"]
    h, w = gray.shape[:2]

    result = {
        "detected": False,
        "confidence": 0.0,
        "heatmap": np.zeros((h, w), dtype=np.uint8),
        "method": "unsupervised",
        "n_blocks": 0,
        "n_flagged": 0,
    }

    # ── Step 1: Extract block-level features ──────────────────
    print("    [Splicing] Extracting block features (LBP + Gabor + DWT)…")
    features, positions = extract_block_features_grid(gray)

    if len(features) == 0:
        return result

    result["n_blocks"] = len(features)

    # ── Step 2: Classify / Score blocks ───────────────────────
    use_svm = (model_path is not None and os.path.isfile(model_path))

    if use_svm:
        print("    [Splicing] Classifying blocks with SVM model…")
        result["method"] = "svm"
        labels, distances = _svm_classify_blocks(features, model_path)
        scores = distances
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        n_flagged = int(labels.sum())
    else:
        print("    [Splicing] Using unsupervised anomaly detection (Mahalanobis)…")
        scores = _mahalanobis_anomaly_scores(features)

        if scores.max() > 0:
            threshold = np.percentile(scores, ANOMALY_PERCENTILE)
            flagged = scores > threshold
            n_flagged = int(flagged.sum())
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        else:
            n_flagged = 0
            scores_norm = scores

    result["n_flagged"] = n_flagged

    # ── Step 2.5: Fuzzy C-Means Clustering (FCM) ──────────────
    if n_flagged > 0 and len(scores_norm) >= 2 and scores_norm.max() > 0:
        print("    [Splicing] Applying Fuzzy C-Means to refine heatmap…")
        try:
            import skfuzzy as fuzz
            
            pos_array = np.array(positions, dtype=float)
            pos_array[:, 0] /= max(1.0, float(h))
            pos_array[:, 1] /= max(1.0, float(w))
            
            # Feature matrix (3, n_blocks). We weight the anomaly score strongly (3.0) 
            # while using spatial coordinates to enforce cohesive contiguous clusters.
            data = np.vstack((scores_norm * 3.0, pos_array[:, 0], pos_array[:, 1]))
            
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                data, c=2, m=2.0, error=0.005, maxiter=1000, init=None
            )
            
            c0_avg = np.average(scores_norm, weights=u[0])
            c1_avg = np.average(scores_norm, weights=u[1])
            tampered_c = 0 if c0_avg > c1_avg else 1
            
            fcm_scores = u[tampered_c]
            
            # Only accept refinement if the split separates anomalous blocks properly
            if max(c0_avg, c1_avg) > min(c0_avg, c1_avg) * 1.5:
                # We update scores_norm for a smoother heatmap visual, 
                # but we DO NOT overwrite n_flagged to prevent massive false positives 
                # on mostly-authentic images with only a few noisy blocks.
                scores_norm = fcm_scores
                
        except ImportError:
            print("    [Splicing] WARNING: scikit-fuzzy not installed. Skipping FCM.")

    # ── Step 3: Build heatmap ─────────────────────────────────
    result["heatmap"] = _build_heatmap((h, w), positions, scores_norm)

    # ── Step 4: Decision + confidence ─────────────────────────
    flagged_ratio = n_flagged / max(1, len(features))

    if use_svm:
        in_range = MIN_FLAGGED_RATIO <= flagged_ratio <= MAX_FLAGGED_RATIO
        result["detected"] = n_flagged > 0 and in_range
        mean_dist = np.mean(scores[scores > 0]) if np.any(scores > 0) else 0
        conf = min(100.0, (
            50.0 * min(1.0, flagged_ratio / 0.3) +
            50.0 * min(1.0, mean_dist / 2.0)
        ))
    else:
        if scores.max() > 0:
            top_scores = scores[scores > np.percentile(scores, ANOMALY_PERCENTILE)]
            score_strength = np.mean(top_scores) / (np.mean(scores) + 1e-8)
            score_std_ratio = np.std(scores) / (np.mean(scores) + 1e-8)
            
            is_significant = (score_strength > 1.6) and (score_std_ratio > 0.45)
            result["detected"] = is_significant
            
            conf = min(100.0, (
                40.0 * min(1.0, flagged_ratio / 0.20) +
                35.0 * min(1.0, score_strength / 4.0) +
                25.0 * min(1.0, score_std_ratio / 1.5)
            ))
        else:
            result["detected"] = False
            conf = 0.0

    result["confidence"] = round(conf, 1)

    return result