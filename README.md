# Image Forgery Detector

A production-ready image forgery detection system using **classical computer vision** techniques. Detects two types of forgery:

- **Copy-Move Forgery** — a region is duplicated within the same image
- **Splicing Forgery** — a region is pasted from a different image

## Techniques Used (CV Syllabus)

| Category | Techniques |
|----------|-----------|
| Feature Extraction | SIFT, HOG, LBP (uniform), Gabor Filters, DWT |
| Edge Analysis | Canny Edge Detector, LOG, DOG |
| Filtering & Enhancement | Convolution, Fourier Transform, Histogram Processing (CLAHE) |
| Classification | LinearSVC (supervised, high-performance SVM), PCA (dimensionality reduction), Mahalanobis Distance |
| Post-Processing | Fuzzy C-Means (FCM) Clustering for smoothed heatmap localization |

> **No deep learning, CNNs, or transformers are used.**

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run on a Single Image

```bash
python forgery_detector.py --image path/to/image.jpg
```

With a trained SVM model for better splicing detection:
```bash
python forgery_detector.py --image photo.jpg --model splicing_svm_model.pkl
```

### 3. Output

The detector prints:
- **Verdict**: `AUTHENTIC`, `COPY-MOVE FORGERY`, or `SPLICING FORGERY`
- **Confidence**: 0–100%
- **Report**: Saved as `forgery_report.png` (multi-panel forensic visualization)

## Training the Splicing Model (Optional)

To improve splicing detection accuracy, train a Linear Support Vector Classifier (LinearSVC) on a labelled dataset:

```bash
python train_splicing_svm.py \
    --authentic path/to/authentic_images/ \
    --spliced   path/to/spliced_images/ \
    --output    splicing_svm_model.pkl
```

Then use the model:
```bash
python forgery_detector.py --image test.jpg --model splicing_svm_model.pkl
```

## Evaluation

You can evaluate the performance of the detector (Accuracy, Precision, Recall, F1-Score) using the provided evaluation script:

```bash
python evaluate_dataset.py --authentic path/to/authentic/ --forged path/to/forged/ --limit 50
```

This script runs the pipeline across your dataset and computes standard metrics. Using the `LinearSVC` model, the pipeline achieves an **F1-Score of ~78%** and **Recall of ~88%** on a standard 100-image evaluation sample.

## Default Dataset

> **Note:** The main `Dataset/` folder used for testing and training is approximately 3.8GB. To keep the repository lightweight, it has been excluded from version control via `.gitignore`.
> 
> **Download the dataset here:** [Insert Your Dataset Link Here]
> 
> Once downloaded, extract the files so that the `Dataset/` folder is placed in the root of this project directory.

## Test Datasets

### CASIA v2.0 (Recommended)
The CASIA Image Tampering Detection Evaluation Database contains both authentic and tampered images.

1. **Download**: Search for "CASIA v2.0 dataset" on Google — it is available on GitHub mirrors and academic sites:
   - GitHub mirror: `https://github.com/namtpham/casia2groundtruth`
   - Kaggle: search "CASIA v2" on kaggle.com/datasets
2. **Structure**: The dataset contains two folders:
   - `Au/` — authentic images
   - `Tp/` — tampered images (both copy-move and spliced)

### Columbia Uncompressed Image Splicing Detection
1. **Download**: https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/
2. Contains 183 authentic + 180 spliced images

### Creating Your Own Test Images

If you don't want to download a dataset, you can create test images:

```bash
# Copy-Move test: clone a region in an image using any image editor
# (select a region, copy, paste elsewhere in the same image)

# Splicing test: paste a region from one image into another
# (copy an object from image A, paste into image B)
```

## Project Structure

```text
forgery_detector.py        — CLI entrypoint (run this)
preprocessing.py           — YCbCr, CLAHE, Canny, LOG, DOG, Fourier
feature_extraction.py      — SIFT, HOG, LBP, Gabor, DWT extractors
copy_move_detector.py      — SIFT self-matching + RANSAC homography
splicing_detector.py       — Block-level LBP+Gabor+DWT → LinearSVC/Mahalanobis
visualization.py           — Heatmap overlays + forensic report panels
train_splicing_svm.py      — LinearSVC training script (optional)
evaluate_dataset.py        — Dataset evaluation script (accuracy, F1-score, etc.)
requirements.txt           — Python dependencies
```

## How It Works

### Copy-Move Detection Pipeline
1. Convert to YCbCr, apply CLAHE on luminance channel
2. Extract SIFT keypoints + 128-D descriptors
3. Self-match descriptors using FLANN (Lowe's ratio test)
4. Filter: remove self-matches, enforce minimum spatial distance
5. Geometric verification: RANSAC homography estimation
6. If ≥12 inliers → Copy-Move Forgery Detected
7. Generate forgery mask from convex hull of inlier keypoints

### Splicing Detection Pipeline
1. Divide image into 64×64 overlapping blocks (stride 32)
2. Per block: extract LBP histogram + Gabor stats + DWT energies
3. **If LinearSVC model available**: PCA reduction → LinearSVC classification per block
4. **If no model**: Mahalanobis distance from feature distribution (unsupervised)
5. **Post-Processing**: Apply Fuzzy C-Means (FCM) clustering to the output scores + spatial coordinates to generate a smooth, localized heatmap that filters out isolated false-positive blocks.
6. Aggregate hard block-level classifications into a final binary decision
7. If ≥8% blocks flagged → Splicing Forgery Detected
