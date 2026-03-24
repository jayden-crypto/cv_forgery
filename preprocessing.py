

import cv2
import numpy as np

def convert_to_ycbcr(image_bgr: np.ndarray) -> np.ndarray:
    
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)

def apply_clahe(y_channel: np.ndarray, clip_limit: float = 2.0,
                tile_size: int = 8) -> np.ndarray:
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(tile_size, tile_size))
    return clahe.apply(y_channel)

def compute_canny_edges(gray: np.ndarray, low: int = 50,
                        high: int = 150) -> np.ndarray:
    
    return cv2.Canny(gray, low, high)

def compute_log_edges(gray: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    log_norm = np.uint8(np.clip(np.abs(log) / np.abs(log).max() * 255, 0, 255))
    return log_norm

def compute_dog_edges(gray: np.ndarray, sigma1: float = 1.0,
                      sigma2: float = 2.0) -> np.ndarray:
    
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma1).astype(np.float64)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma2).astype(np.float64)
    dog = g1 - g2
    dog_norm = np.uint8(np.clip(np.abs(dog) / (np.abs(dog).max() + 1e-8) * 255,
                                0, 255))
    return dog_norm

def compute_fourier_magnitude(gray: np.ndarray) -> np.ndarray:
    
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    magnitude = np.uint8(magnitude / magnitude.max() * 255)
    return magnitude

def gaussian_blur(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

# ────────────────────────────────────────────────────────────────
#  Main preprocessing entry point
# ────────────────────────────────────────────────────────────────

def resize_if_large(image_bgr: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image_bgr

def preprocess_image(image_bgr: np.ndarray, max_dim: int = 1024) -> dict:
    image_bgr = resize_if_large(image_bgr, max_dim)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_smooth = gaussian_blur(gray, ksize=3)

    ycbcr = convert_to_ycbcr(image_bgr)
    y_ch, cr_ch, cb_ch = cv2.split(ycbcr)
    y_eq = apply_clahe(y_ch)

    canny = compute_canny_edges(gray_smooth)
    log_edges = compute_log_edges(gray_smooth)
    dog_edges = compute_dog_edges(gray_smooth)
    fourier_mag = compute_fourier_magnitude(gray)

    return {
        "original":    image_bgr,
        "gray":        gray,
        "gray_smooth": gray_smooth,
        "ycbcr":       ycbcr,
        "y_eq":        y_eq,
        "cb":          cb_ch,
        "cr":          cr_ch,
        "canny":       canny,
        "log_edges":   log_edges,
        "dog_edges":   dog_edges,
        "fourier_mag": fourier_mag,
    }