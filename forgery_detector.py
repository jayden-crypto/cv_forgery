#!/usr/bin/env python3

import argparse
import os
import sys
import time

import cv2
import numpy as np

# ── Project modules ──────────────────────────────────────────
from preprocessing import preprocess_image
from copy_move_detector import detect_copy_move
from splicing_detector import detect_splicing
from visualization import save_report, create_heatmap_overlay, create_mask_overlay

# ────────────────────────────────────────────────────────────────
#  Verdict Logic
# ────────────────────────────────────────────────────────────────

def _decide_verdict(cm_result: dict, sp_result: dict) -> tuple:
    
    cm_det = cm_result["detected"]
    sp_det = sp_result["detected"]
    cm_conf = cm_result["confidence"]
    sp_conf = sp_result["confidence"]

    details = {
        "Copy-Move matches": cm_result["n_matches"],
        "Copy-Move inliers": cm_result["n_inliers"],
        "Splicing method": sp_result["method"],
        "Splicing blocks flagged": f"{sp_result['n_flagged']}/{sp_result['n_blocks']}",
    }

    if cm_det and sp_det:
        cm_has_inliers = cm_result["n_inliers"] >= 10
        sp_is_supervised = sp_result["method"] == "svm"
        if cm_has_inliers and (not sp_is_supervised or cm_conf >= sp_conf * 0.7):
            return "COPY-MOVE FORGERY", cm_conf, cm_result["mask"], details
        elif sp_conf > cm_conf:
            return "SPLICING FORGERY", sp_conf, sp_result["heatmap"], details
        else:
            return "COPY-MOVE FORGERY", cm_conf, cm_result["mask"], details

    if cm_det:
        return "COPY-MOVE FORGERY", cm_conf, cm_result["mask"], details

    if sp_det:
        return "SPLICING FORGERY", sp_conf, sp_result["heatmap"], details

    auth_conf = max(5.0, 100.0 - max(cm_conf, sp_conf))
    return "AUTHENTIC", auth_conf, np.zeros_like(cm_result["mask"]), details

# ────────────────────────────────────────────────────────────────
#  Main Pipeline
# ────────────────────────────────────────────────────────────────

def run_detection(image_path: str,
                  model_path: str = "splicing_svm_model.pkl",
                  output_path: str = "forgery_report.png") -> dict:
    
    # ── Load Image ────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  IMAGE FORGERY DETECTOR")
    print(f"{'═' * 60}")
    print(f"  Input: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"  Size:  {w}×{h} px")
    print(f"{'─' * 60}")

    start = time.time()

    # ── Step 1: Preprocessing ─────────────────────────────────
    print("\n[1/4] Preprocessing (YCbCr → CLAHE → Canny → Fourier)…")
    preprocessed = preprocess_image(image)
    print("       ✓ Done")

    # ── Step 2: Copy-Move Detection ───────────────────────────
    print("\n[2/4] Copy-Move Detection (SIFT → FLANN → Homography)…")
    cm_result = detect_copy_move(preprocessed)
    cm_status = "DETECTED" if cm_result["detected"] else "not detected"
    print(f"       ✓ {cm_status}  "
          f"(matches={cm_result['n_matches']}, "
          f"inliers={cm_result['n_inliers']}, "
          f"conf={cm_result['confidence']:.1f}%)")

    # ── Step 3: Splicing Detection ────────────────────────────
    print("\n[3/4] Splicing Detection (LBP + Gabor + DWT → SVM)…")
    sp_result = detect_splicing(preprocessed, model_path)
    sp_status = "DETECTED" if sp_result["detected"] else "not detected"
    print(f"       ✓ {sp_status}  "
          f"(method={sp_result['method']}, "
          f"flagged={sp_result['n_flagged']}/{sp_result['n_blocks']}, "
          f"conf={sp_result['confidence']:.1f}%)")

    # ── Step 4: Verdict + Report ──────────────────────────────
    print("\n[4/4] Generating forensic report…")
    verdict, confidence, forgery_map, details = _decide_verdict(
        cm_result, sp_result
    )

    combined_heatmap = np.maximum(
        cm_result["mask"],
        sp_result["heatmap"]
    )

    report_path = save_report(
        image_bgr=image,
        heatmap=combined_heatmap,
        mask=forgery_map,
        verdict=verdict,
        confidence=confidence,
        details=details,
        output_path=output_path,
        canny=preprocessed["canny"],
        fourier_mag=preprocessed["fourier_mag"],
    )

    elapsed = time.time() - start

    # ── Final Output ──────────────────────────────────────────
    print(f"\n{'═' * 60}")
    if "AUTHENTIC" in verdict:
        print(f"  ✅ VERDICT:    {verdict}")
    else:
        print(f"  🚨 VERDICT:    {verdict}")
    print(f"  📊 CONFIDENCE: {confidence:.1f}%")
    print(f"  📄 REPORT:     {report_path}")
    print(f"  ⏱  TIME:       {elapsed:.2f}s")
    print(f"{'═' * 60}\n")

    return {
        "verdict": verdict,
        "confidence": confidence,
        "report_path": report_path,
        "copy_move": cm_result,
        "splicing": sp_result,
        "details": details,
    }

# ────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Image Forgery Detector — Copy-Move & Splicing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", default="test.jpg",
                        help="Path to input image (default: test.jpg)")
    parser.add_argument("--model", default="splicing_svm_model.pkl",
                        help="Path to trained SVM model (.pkl) for splicing "
                             "detection. If not provided, uses unsupervised "
                             "anomaly detection.")
    parser.add_argument("--output", default="forgery_report.png",
                        help="Output report image path (default: forgery_report.png)")

    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"[ERROR] File not found: {args.image}")
        sys.exit(1)

    run_detection(args.image, args.model, args.output)

if __name__ == "__main__":
    main()