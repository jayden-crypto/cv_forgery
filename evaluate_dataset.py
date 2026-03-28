#!/usr/env/bin python3
import os
import sys
import time
import glob
import random
import argparse
from contextlib import redirect_stdout
from forgery_detector import run_detection

def evaluate(authentic_dir: str, forged_dir: str, limit: int = 0):
    y_true = []
    y_pred = []
    times = []
    
    categories = [
        {"name": "Original (Authentic)", "path": authentic_dir, "label": 0},
        {"name": "Forged", "path": forged_dir, "label": 1}
    ]
    
    print(f"\n{'═' * 60}")
    print(f"  DATASET EVALUATION TOOL")
    print(f"{'═' * 60}")
    
    for cat in categories:
        if not os.path.isdir(cat["path"]):
            print(f"[ERROR] Directory not found: {cat['path']}")
            continue
            
        img_paths = glob.glob(os.path.join(cat["path"], "*.*"))
        img_paths = [p for p in img_paths if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
        
        print(f"\n[INFO] Found {len(img_paths)} images in {cat['name']}")
        
        if limit > 0 and len(img_paths) > limit:
            random.seed(42)
            random.shuffle(img_paths)
            img_paths = img_paths[:limit]
            print(f"       Sampling {limit} images due to --limit...")
            
        for i, p in enumerate(img_paths):
            label = cat["label"]
            y_true.append(label)
            
            start = time.time()
            # Suppress prints from the cv pipeline
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                try:
                    res = run_detection(p, model_path="splicing_svm_model.pkl", output_path="forgery_report.png")
                except Exception as e:
                    res = {"verdict": "ERROR"}
            t = time.time() - start
            times.append(t)
            
            pred_label = 0 if "AUTHENTIC" in res.get("verdict", "") else 1
            y_pred.append(pred_label)
            
            correct = "✓" if pred_label == label else "✗"
            print(f" [{correct}] {os.path.basename(p)[:20]:<20} | Time: {t:.2f}s | Pred: {'Forged' if pred_label else 'Authentic'}")
                
    if not y_true:
        print("\n[ERROR] No images evaluated.")
        return
        
    accuracy = sum([1 for i, j in zip(y_true, y_pred) if i == j]) / len(y_true)
    
    tp = sum([1 for i, j in zip(y_true, y_pred) if i == 1 and j == 1])
    fp = sum([1 for i, j in zip(y_true, y_pred) if i == 0 and j == 1])
    fn = sum([1 for i, j in zip(y_true, y_pred) if i == 1 and j == 0])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_time = sum(times) / len(times)
    
    print("\n" + "═"*60)
    print("  FINAL METRICS".center(60))
    print("═"*60)
    print(f"  Total Evaluated: {len(y_true)} images")
    print(f"  Accuracy:        {accuracy * 100:.2f}%")
    print(f"  Precision:       {precision * 100:.2f}%")
    print(f"  Recall:          {recall * 100:.2f}%")
    print(f"  F1-Score:        {f1 * 100:.2f}%")
    print(f"  Average Time:    {avg_time:.2f}s / image")
    print("═"*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Forgery Detector on a Dataset")
    parser.add_argument("--authentic", default="dataset/Original", help="Path to authentic images directory")
    parser.add_argument("--forged", default="dataset/Forged", help="Path to forged images directory")
    parser.add_argument("--limit", type=int, default=50, help="Max images to test per class (0 = Unlimited. Default: 50)")
    args = parser.parse_args()
    
    evaluate(args.authentic, args.forged, args.limit)
