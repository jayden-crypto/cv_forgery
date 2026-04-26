# Image Forgery Detector

A production-ready image forgery detection system using **classical computer vision** techniques. Detects two types of forgery:

- **Copy-Move Forgery** — a region is duplicated within the same image
- **Splicing Forgery** — a region is pasted from a different image

## Techniques Used (CV Syllabus)

| Category | Techniques |
|----------|-----------|
| Feature Extraction | SIFT, HOG, LBP (uniform), Gabor Filters, DWT, ELA, Noise Residual |
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

Multiple directories are supported — useful for combining datasets:

```bash
python train_splicing_svm.py \
    --authentic dir1/Au dir2/Original \
    --spliced   dir1/Tp dir2/Forged \
    --output    splicing_svm_model.pkl
```

If ground-truth tamper masks are available, pass them with `--masks` (one directory per spliced directory). Where masks are missing, ELA pseudo-masks are used as a fallback:

```bash
python train_splicing_svm.py \
    --authentic path/to/Au \
    --spliced   path/to/Tp \
    --masks     path/to/Groundtruth \
    --output    splicing_svm_model.pkl
```

Use `--image-level` when no masks are available and you want the fastest, most reliable training — all blocks from spliced images are labelled as class 1:

```bash
python train_splicing_svm.py \
    --authentic path/to/Original \
    --spliced   path/to/Forged \
    --image-level \
    --output    splicing_svm_model.pkl
```

Then use the model:
```bash
python forgery_detector.py --image test.jpg --model splicing_svm_model.pkl
```

## Evaluation

You can evaluate the performance of the detector (Accuracy, Precision, Recall, F1-Score) using the provided evaluation script:

```bash
python evaluate_dataset.py --authentic path/to/authentic/ --forged path/to/forged/ --limit 100
```

This script runs the full pipeline across your dataset and computes standard metrics. The `--limit N` flag samples N images **per class**. Using the LinearSVC model trained on the Kaggle dataset with image-level labels, the pipeline achieves:

| Metric | Score |
|--------|-------|
| Accuracy | 89.8% |
| Precision | 93% |
| Recall | 86% |
| F1-Score | **89.3%** |

## Default Dataset

> **Note:** The `Dataset/` folder used for testing and training is approximately 3.8 GB. To keep the repository lightweight, it has been excluded from version control via `.gitignore`.
>
> **Download the dataset here:** [Kaggle — Image Forgery Detection Dataset](https://www.kaggle.com/datasets/labid93/image-forgery-detection)
>
> Once downloaded, extract the files so that the `Dataset/` folder is placed in the root of this project directory.

## Project Structure

```text
forgery_detector.py        — CLI entrypoint (run this)
preprocessing.py           — YCbCr, CLAHE, Canny, LOG, DOG, Fourier
feature_extraction.py      — SIFT, HOG, LBP, Gabor, DWT, ELA, Noise Residual extractors
copy_move_detector.py      — SIFT self-matching + RANSAC homography
splicing_detector.py       — Block-level features → LinearSVC / Mahalanobis anomaly scoring
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
2. Per block: extract a **71-dimensional feature vector**:
   - LBP histogram (10 features)
   - Gabor filter bank, 4 orientations × 3 scales (24 features)
   - DWT subband energies, Haar level 2 (21 features)
   - ELA statistics — mean, std, max, 75th/90th/95th percentile (6 features)
   - Noise residual at σ = 0.5 / 1.0 / 2.0 + cross-scale std (10 features)
3. **If LinearSVC model available**: StandardScaler → PCA (30 components) → LinearSVC classification per block
4. **If no model**: Mahalanobis distance from feature distribution (unsupervised fallback)
5. **Post-Processing**: Fuzzy C-Means (FCM) clustering on scores + spatial coordinates for a smooth, localized heatmap
6. Aggregate block-level decisions into a final binary verdict
7. If 8%–85% of blocks are flagged → Splicing Forgery Detected

> **ELA optimisation:** the full-image ELA map is computed once per image via a single JPEG encode/decode cycle; per-block ELA features are then obtained by slicing the pre-computed map, avoiding ~991k redundant JPEG operations per training epoch.
