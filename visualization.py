

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ────────────────────────────────────────────────────────────────
#  Heatmap Overlay
# ────────────────────────────────────────────────────────────────

def create_heatmap_overlay(image_bgr: np.ndarray,
                           heatmap: np.ndarray,
                           alpha: float = 0.5,
                           colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    
    heatmap_resized = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap_resized, colormap)
    blended = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return blended

# ────────────────────────────────────────────────────────────────
#  Mask Contour Overlay
# ────────────────────────────────────────────────────────────────

def create_mask_overlay(image_bgr: np.ndarray,
                        mask: np.ndarray,
                        color: tuple = (0, 0, 255),
                        thickness: int = 2,
                        label: str = "FORGED") -> np.ndarray:
    
    annotated = image_bgr.copy()
    mask_resized = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]))

    _, binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        cv2.drawContours(annotated, [cnt], -1, color, thickness)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(annotated, label, (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return annotated

# ────────────────────────────────────────────────────────────────
#  Forensic Report Panel
# ────────────────────────────────────────────────────────────────

def save_report(image_bgr: np.ndarray,
                heatmap: np.ndarray,
                mask: np.ndarray,
                verdict: str,
                confidence: float,
                details: dict,
                output_path: str = "forgery_report.png",
                canny: np.ndarray = None,
                fourier_mag: np.ndarray = None) -> str:
    
    fig = plt.figure(figsize=(22, 12))
    fig.patch.set_facecolor("#1a1a2e")
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)

    # ── Panel 1: Original Image ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image", color="white", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # ── Panel 2: Canny Edge Map ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if canny is not None:
        ax2.imshow(canny, cmap="gray")
        ax2.set_title("Canny Edge Map", color="white", fontsize=12, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "N/A", ha="center", va="center",
                 color="white", fontsize=14, transform=ax2.transAxes)
        ax2.set_title("Edge Map", color="white", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # ── Panel 3: Fourier Magnitude Spectrum ───────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if fourier_mag is not None:
        ax3.imshow(fourier_mag, cmap="magma")
        ax3.set_title("Fourier Magnitude Spectrum",
                      color="white", fontsize=12, fontweight="bold")
    else:
        ax3.text(0.5, 0.5, "N/A", ha="center", va="center",
                 color="white", fontsize=14, transform=ax3.transAxes)
        ax3.set_title("Frequency Domain", color="white", fontsize=12, fontweight="bold")
    ax3.axis("off")

    # ── Panel 4: Heatmap Overlay ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    overlay = create_heatmap_overlay(image_bgr, heatmap, alpha=0.55)
    ax4.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax4.set_title("Forgery Heatmap", color="white", fontsize=12, fontweight="bold")
    ax4.axis("off")

    # ── Panel 5: Annotated Result ─────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    annotated = create_mask_overlay(image_bgr, mask, label=verdict)
    ax5.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    ax5.set_title("Detected Regions", color="white", fontsize=12, fontweight="bold")
    ax5.axis("off")

    # ── Panel 6: Verdict Summary ──────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#16213e")
    ax6.axis("off")

    if "AUTHENTIC" in verdict:
        verdict_color = "#00ff88"
    elif "COPY-MOVE" in verdict:
        verdict_color = "#ff4444"
    else:
        verdict_color = "#ffaa00"

    lines = [
        f"VERDICT:  {verdict}",
        f"CONFIDENCE:  {confidence:.1f}%",
        "",
    ]
    for key, val in details.items():
        lines.append(f"{key}:  {val}")

    summary_text = "\n".join(lines)

    ax6.text(0.5, 0.7, verdict, ha="center", va="center",
             fontsize=20, fontweight="bold", color=verdict_color,
             transform=ax6.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f3460",
                       edgecolor=verdict_color, linewidth=2))

    ax6.text(0.5, 0.45, f"Confidence: {confidence:.1f}%",
             ha="center", va="center", fontsize=14, color="white",
             transform=ax6.transAxes)

    detail_text = "\n".join(f"{k}: {v}" for k, v in details.items())
    ax6.text(0.5, 0.15, detail_text,
             ha="center", va="center", fontsize=10, color="#cccccc",
             transform=ax6.transAxes, family="monospace")

    fig.suptitle("Image Forgery Detection — Forensic Report",
                 fontsize=16, fontweight="bold", color="white", y=0.98)

    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_path