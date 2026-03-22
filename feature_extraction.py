

import cv2
import numpy as np
import pywt
from skimage.feature import local_binary_pattern

# ────────────────────────────────────────────────────────────────
#  SIFT Features
# ────────────────────────────────────────────────────────────────

def extract_sift_features(gray: np.ndarray,
                          n_features: int = 2000) -> tuple:
    
    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

# ────────────────────────────────────────────────────────────────
#  HOG Features
# ────────────────────────────────────────────────────────────────

def extract_hog_features(gray: np.ndarray,
                         cell_size: int = 16,
                         block_size: int = 2,
                         n_bins: int = 9) -> np.ndarray:
    
    win_size = (
        (gray.shape[1] // cell_size) * cell_size,
        (gray.shape[0] // cell_size) * cell_size,
    )
    hog = cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=(cell_size * block_size, cell_size * block_size),
        _blockStride=(cell_size, cell_size),
        _cellSize=(cell_size, cell_size),
        _nbins=n_bins,
    )
    resized = cv2.resize(gray, win_size)
    descriptor = hog.compute(resized)
    return descriptor.flatten()

# ────────────────────────────────────────────────────────────────
#  LBP Features
# ────────────────────────────────────────────────────────────────

def extract_lbp_histogram(gray: np.ndarray,
                          radius: int = 1,
                          n_points: int = 8,
                          method: str = "uniform") -> np.ndarray:
    
    lbp = local_binary_pattern(gray, n_points, radius, method=method)
    if method == "uniform":
        n_bins = n_points + 2
    else:
        n_bins = 2 ** n_points
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins),
                           density=True)
    return hist

def extract_lbp_image(gray: np.ndarray,
                      radius: int = 1,
                      n_points: int = 8,
                      method: str = "uniform") -> np.ndarray:
    
    return local_binary_pattern(gray, n_points, radius, method=method)

# ────────────────────────────────────────────────────────────────
#  Gabor Filter Features
# ────────────────────────────────────────────────────────────────

def build_gabor_bank(ksize: int = 31,
                     n_orientations: int = 4,
                     n_scales: int = 3,
                     sigma: float = 4.0,
                     lambd_start: float = 5.0,
                     gamma: float = 0.5,
                     psi: float = 0.0) -> list:
    
    bank = []
    for t in range(n_orientations):
        theta = t * np.pi / n_orientations
        for s in range(n_scales):
            lambd = lambd_start * (2 ** s)
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambd, gamma, psi,
                ktype=cv2.CV_64F,
            )
            bank.append(kernel)
    return bank

def extract_gabor_features(gray: np.ndarray,
                           gabor_bank: list = None) -> np.ndarray:
    
    if gabor_bank is None:
        gabor_bank = build_gabor_bank()

    features = []
    for kernel in gabor_bank:
        filtered = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, kernel)
        features.append(filtered.mean())
        features.append(filtered.std())
    return np.array(features, dtype=np.float64)

def extract_gabor_response_maps(gray: np.ndarray,
                                gabor_bank: list = None) -> list:
    
    if gabor_bank is None:
        gabor_bank = build_gabor_bank()
    maps = []
    for kernel in gabor_bank:
        filtered = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, kernel)
        maps.append(np.abs(filtered))
    return maps

# ────────────────────────────────────────────────────────────────
#  DWT Features
# ────────────────────────────────────────────────────────────────

def extract_dwt_features(gray: np.ndarray,
                         wavelet: str = "haar",
                         level: int = 2) -> np.ndarray:
    
    img = gray.astype(np.float64) / 255.0
    coeffs = pywt.wavedec2(img, wavelet, level=level)

    features = []
    ll = coeffs[0]
    features.extend([np.mean(np.abs(ll)), np.std(ll),
                     np.sum(ll ** 2) / ll.size])

    for detail in coeffs[1:]:
        for subband in detail:
            features.extend([
                np.mean(np.abs(subband)),
                np.std(subband),
                np.sum(subband ** 2) / subband.size,
            ])

    return np.array(features, dtype=np.float64)

def extract_dwt_subbands(gray: np.ndarray,
                         wavelet: str = "haar",
                         level: int = 2) -> list:
    
    img = gray.astype(np.float64) / 255.0
    return pywt.wavedec2(img, wavelet, level=level)