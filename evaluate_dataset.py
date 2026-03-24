import os
import sys
import time
import glob
from contextlib import redirect_stdout
from forgery_detector import run_detection

def evaluate():
    dataset_dir = "dataset"
    categories = ["Forged", "Original"]
    
    y_true = []
    y_pred = []
    times = []
    
    print(f"Starting evaluation on dataset: {dataset_dir}...")
    
    import random
    
    for cat in categories:
        img_paths = glob.glob(os.path.join(dataset_dir, cat, "*.*"))
        random.seed(42)  # Fixed seed for consistent baselining
        random.shuffle(img_paths)
        img_paths = img_paths[:50]
        print(f"Sampling {len(img_paths)} images in {cat}...")
        for p in img_paths:
            if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                label = 1 if cat == "Forged" else 0
                y_true.append(label)
                
                start = time.time()
                # Suppress prints from the pipeline
                with open(os.devnull, 'w') as f, redirect_stdout(f):
                    try:
                        res = run_detection(p, model_path="splicing_svm_model.pkl", output_path="forgery_report.png")
                    except Exception as e:
                        res = {"verdict": "ERROR"}
                t = time.time() - start
                times.append(t)
                
                pred_label = 0 if "AUTHENTIC" in res.get("verdict", "") else 1
                y_pred.append(pred_label)
                
                print(f"Tested {os.path.basename(p)} | True: {label} | Pred: {pred_label} | Time: {t:.2f}s")
                
    if not y_true:
        print("No images found to evaluate.")
        return
        
    accuracy = sum([1 for i, j in zip(y_true, y_pred) if i == j]) / len(y_true)
    
    true_positives = sum([1 for i, j in zip(y_true, y_pred) if i == 1 and j == 1])
    false_positives = sum([1 for i, j in zip(y_true, y_pred) if i == 0 and j == 1])
    false_negatives = sum([1 for i, j in zip(y_true, y_pred) if i == 1 and j == 0])
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_time = sum(times) / len(times)
    
    print("\n" + "="*40)
    print(" BASELINE METRICS ".center(40, "="))
    print("="*40)
    print(f"Total Images Evaluated: {len(y_true)}")
    print(f"Accuracy:               {accuracy * 100:.2f}%")
    print(f"Precision (Forgery):    {precision * 100:.2f}%")
    print(f"Recall (Forgery):       {recall * 100:.2f}%")
    print(f"F1-Score:               {f1 * 100:.2f}%")
    print(f"Avg Time per Image:     {avg_time:.2f}s")
    print("="*40)

if __name__ == "__main__":
    evaluate()
