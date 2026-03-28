

import cv2
import numpy as np
from typing import Optional
from sklearn.cluster import DBSCAN

from feature_extraction import extract_sift_features

# ────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────

FLANN_INDEX_KDTREE = 1
LOWE_RATIO = 0.7
MIN_SPATIAL_DIST = 50
MIN_INLIERS_HOMOGRAPHY = 12
MIN_GOOD_MATCHES = 15

# ────────────────────────────────────────────────────────────────
#  Self-Matching via FLANN
# ────────────────────────────────────────────────────────────────

def _self_match_descriptors(descriptors: np.ndarray,
                            keypoints: list,
                            ratio: float = LOWE_RATIO,
                            min_dist: float = MIN_SPATIAL_DIST) -> list:
    
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    raw_matches = flann.knnMatch(descriptors, descriptors, k=3)

    good = []
    for match_group in raw_matches:
        candidates = [m for m in match_group
                      if m.queryIdx != m.trainIdx]
        if len(candidates) < 2:
            continue
        m1, m2 = candidates[0], candidates[1]

        if m1.distance < ratio * m2.distance:
            qi, ti = m1.queryIdx, m1.trainIdx
            pt1 = np.array(keypoints[qi].pt)
            pt2 = np.array(keypoints[ti].pt)
            if np.linalg.norm(pt1 - pt2) > min_dist:
                good.append((qi, ti))

    return good

# ────────────────────────────────────────────────────────────────
#  Geometric Verification (Homography)
# ────────────────────────────────────────────────────────────────

def _verify_homography(keypoints: list,
                       matches: list,
                       min_inliers: int = MIN_INLIERS_HOMOGRAPHY):
    
    if len(matches) < 4:
        return None, None, None, None

    src_pts = np.float32([keypoints[m[0]].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints[m[1]].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if mask is None:
        return None, None, None, None

    n_inliers = int(mask.sum())
    if n_inliers >= min_inliers:
        return H, mask, src_pts, dst_pts
    return None, None, None, None

# ────────────────────────────────────────────────────────────────
#  Forgery Mask Generation
# ────────────────────────────────────────────────────────────────

def _build_forgery_mask(image_shape: tuple,
                        keypoints: list,
                        matches: list,
                        inlier_mask: np.ndarray) -> np.ndarray:
    
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    inlier_indices = [i for i, v in enumerate(inlier_mask.flatten()) if v]
    if len(inlier_indices) < 3:
        return mask

    src_points = np.array([keypoints[matches[i][0]].pt
                           for i in inlier_indices], dtype=np.int32)
    dst_points = np.array([keypoints[matches[i][1]].pt
                           for i in inlier_indices], dtype=np.int32)

    if len(src_points) >= 3:
        hull_src = cv2.convexHull(src_points)
        cv2.fillConvexPoly(mask, hull_src, 255)
    if len(dst_points) >= 3:
        hull_dst = cv2.convexHull(dst_points)
        cv2.fillConvexPoly(mask, hull_dst, 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    return mask

# ────────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────────

def detect_copy_move(preprocessed: dict,
                     n_features: int = 1500) -> dict:
    
    gray = preprocessed["y_eq"]
    h, w = gray.shape[:2]

    keypoints, descriptors = extract_sift_features(gray, n_features)

    result = {
        "detected": False,
        "confidence": 0.0,
        "mask": np.zeros((h, w), dtype=np.uint8),
        "n_matches": 0,
        "n_inliers": 0,
        "matches": [],
        "keypoints": keypoints,
    }

    if descriptors is None or len(keypoints) < 10:
        return result

    good_matches = _self_match_descriptors(descriptors, keypoints)
    result["n_matches"] = len(good_matches)
    result["matches"] = good_matches

    if len(good_matches) < MIN_GOOD_MATCHES:
        return result

    H, inlier_mask, src_pts, dst_pts = _verify_homography(
        keypoints, good_matches
    )

    if H is None:
        result["confidence"] = min(30.0, len(good_matches) * 1.5)
        return result

    # --- ADVANCED FILTERING: DBSCAN Spatial Clustering ---
    # A true copy-move forgery will have matched points geometrically concentrated in at least 2 dense regions (source and destination).
    inlier_idx = [i for i, val in enumerate(inlier_mask.flatten()) if val]
    pts_src = np.array([keypoints[good_matches[i][0]].pt for i in inlier_idx])
    pts_dst = np.array([keypoints[good_matches[i][1]].pt for i in inlier_idx])
    
    all_pts = np.vstack((pts_src, pts_dst))
    clustering = DBSCAN(eps=40, min_samples=4).fit(all_pts)
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    
    if n_clusters < 2:
        # Rejected by Spatial Filter: Matches do not form distinct source/destination geometrical clusters
        result["confidence"] = 5.0
        result["detected"] = False
        return result

    n_inliers = int(inlier_mask.sum())
    result["n_inliers"] = n_inliers
    result["detected"] = True

    result["mask"] = _build_forgery_mask(
        (h, w), keypoints, good_matches, inlier_mask
    )

    inlier_ratio = n_inliers / len(good_matches) if good_matches else 0
    conf = min(100.0, (
        40.0 * inlier_ratio +
        30.0 * min(1.0, n_inliers / 50) +
        30.0 * min(1.0, len(good_matches) / 80)
    ))
    result["confidence"] = round(conf, 1)

    return result