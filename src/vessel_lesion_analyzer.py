"""
Retinal Vasculature & Lesion Segmentation Analyzer
Diabetic Retinopathy Screening Using Deep Learning

Features:
- Blood vessel segmentation overlay and vessel density quantification
- Multi-lesion semantic overlay:
  * MA (Microaneurysms) -> Crimson Red
  * HE (Hemorrhages) -> Deep Ruby Red
  * EX (Hard Exudates) -> Bright Yellow
  * SE / CW (Cotton Wool Spots / Soft Exudates) -> Cyan
  * NV (Neovascularization) -> Vivid Violet/Purple
  * OD (Optic Disc) -> Gold Border
- Automated morphological vessel extractor for novel uploaded images
- Lesion burden percentage & ICDR criteria mapping
"""

import os
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

LESION_PALETTE = {
    "MA": {"name": "Microaneurysms (MA)", "color_rgb": (255, 30, 30), "severity": "Mild (Grade 1)", "description": "Focal capillary outpouchings; earliest recognizable sign of DR."},
    "HE": {"name": "Hemorrhages (HE)", "color_rgb": (180, 0, 40), "severity": "Moderate/Severe (Grade 2-3)", "description": "Intraretinal dot-blot and flame-shaped microvascular hemorrhages."},
    "EX": {"name": "Hard Exudates (EX)", "color_rgb": (255, 235, 20), "severity": "Moderate (Grade 2)", "description": "Lipid and lipoprotein precipitates leaked from compromised capillaries."},
    "SE": {"name": "Soft Exudates / Cotton-Wool Spots (SE)", "color_rgb": (0, 220, 255), "severity": "Severe (Grade 3)", "description": "Micro-infarctions of the retinal nerve fiber layer (axoplasmic stasis); medically identical to Cotton-Wool Spots (CW)."},
    "CW": {"name": "Cotton Wool Spots / Soft Exudates (CW)", "color_rgb": (50, 200, 255), "severity": "Severe (Grade 3)", "description": "Retinal nerve fiber layer infarctions; clinical synonym for Soft Exudates (SE)."},
    "NV": {"name": "Neovascularization (NV)", "color_rgb": (200, 50, 255), "severity": "Proliferative (Grade 4)", "description": "Abnormal fragile new vessel growth (NVD/NVE) signaling proliferative diabetic retinopathy."},
    "OD": {"name": "Optic Disc (OD)", "color_rgb": (255, 180, 0), "severity": "Anatomical Landmark", "description": "Exit site for retinal ganglion cell axons and entrance of retinal blood vessels."}
}

def extract_vessels_morphology(img_rgb: np.ndarray) -> np.ndarray:
    """
    Automated Retinal Vessel Segmentation for novel fundus images without ground-truth masks:
    Uses green-channel extraction, CLAHE contrast enhancement, and Top-Hat morphological filtering.
    """
    # Green channel has the best contrast for retinal blood vessels
    green = img_rgb[:, :, 1]
    
    # 1. CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    
    # 2. Morphological Top-Hat and Black-Hat filtering to detect linear vessel tracts
    kernel_sizes = [3, 5, 7]
    combined_vessels = np.zeros_like(green, dtype=np.float32)
    
    for k_size in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
        combined_vessels += blackhat.astype(np.float32)
        
    combined_vessels = cv2.normalize(combined_vessels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 3. Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        combined_vessels, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, -2
    )
    
    # Clean noise with morphological opening
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel)
    
    # Mask out non-retinal perimeter
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    retina_mask = cv2.erode((gray > 20).astype(np.uint8), np.ones((15, 15), np.uint8))
    vessel_mask = cv2.bitwise_and(cleaned, cleaned, mask=retina_mask)
    return vessel_mask

def compute_vessel_density(vessel_mask: np.ndarray, fov_mask: Optional[np.ndarray] = None) -> float:
    """
    Computes vascular density percentage: (vessel area / total retinal field area) * 100.
    """
    if vessel_mask is None:
        return 0.0
    vessel_pixels = np.count_nonzero(vessel_mask > 0)
    
    if fov_mask is not None:
        total_pixels = np.count_nonzero(fov_mask > 0)
    else:
        total_pixels = vessel_mask.shape[0] * vessel_mask.shape[1]
        
    if total_pixels == 0:
        return 0.0
    return float((vessel_pixels / total_pixels) * 100.0)

def overlay_vessels(
    img_rgb: np.ndarray,
    vessel_mask: np.ndarray,
    vessel_color: Tuple[int, int, int] = (0, 255, 160),
    alpha: float = 0.55
) -> np.ndarray:
    """
    Overlays detected blood vessel network in neon teal/green onto the fundus image.
    """
    h, w = img_rgb.shape[:2]
    if vessel_mask.shape[:2] != (h, w):
        vessel_mask = cv2.resize(vessel_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
    overlay = img_rgb.copy()
    vessel_indices = vessel_mask > 10
    
    color_layer = np.zeros_like(img_rgb)
    color_layer[vessel_indices] = vessel_color
    
    blended = img_rgb.copy()
    blended[vessel_indices] = cv2.addWeighted(
        img_rgb[vessel_indices], 1.0 - alpha,
        color_layer[vessel_indices], alpha, 0
    )
    return blended

def load_lesion_masks_for_image(
    base_name: str,
    lesion_dir: str = "lesion_masks"
) -> Dict[str, np.ndarray]:
    """
    Discovers and loads all individual lesion masks for a given base image name.
    """
    lesions = {}
    
    # 1. IDRID masks
    for ltype in ["MA", "HE", "EX", "SE", "OD"]:
        p = os.path.join(lesion_dir, ltype, f"{base_name}_{ltype}.tif")
        if os.path.exists(p):
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is not None and np.any(m > 10):
                lesions[ltype] = m

    # 2. MAPLES masks
    for ltype in ["CW", "NV"]:
        p = os.path.join(lesion_dir, ltype, f"{base_name}.png")
        if os.path.exists(p):
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is not None and np.any(m > 10):
                lesions[ltype] = m
                
    # 3. Merged mask
    merged_p = os.path.join(lesion_dir, "merged", f"{base_name}.png")
    if os.path.exists(merged_p):
        m = cv2.imread(merged_p, cv2.IMREAD_GRAYSCALE)
        if m is not None and np.any(m > 10):
            lesions["merged"] = m

    return lesions

def create_clinical_lesion_overlay(
    img_rgb: np.ndarray,
    lesion_dict: Dict[str, np.ndarray],
    alpha: float = 0.65
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Creates a multi-color composite visualization with all detected lesion types overlaid
    onto the fundus image, plus statistical burden summary.
    """
    h, w = img_rgb.shape[:2]
    composite = img_rgb.copy()
    stats = {}
    total_lesion_pixels = 0
    
    for ltype, palette in LESION_PALETTE.items():
        if ltype in lesion_dict and lesion_dict[ltype] is not None:
            mask = lesion_dict[ltype]
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
            px_count = int(np.count_nonzero(mask > 10))
            if px_count > 0:
                stats[ltype] = {
                    "name": palette["name"],
                    "pixel_count": px_count,
                    "color_rgb": palette["color_rgb"],
                    "clinical_significance": palette["severity"],
                    "description": palette.get("description", "")
                }
                total_lesion_pixels += px_count
                
                # Overlay
                indices = mask > 10
                color_arr = np.zeros_like(img_rgb)
                color_arr[indices] = palette["color_rgb"]
                
                composite[indices] = cv2.addWeighted(
                    composite[indices], 1.0 - alpha,
                    color_arr[indices], alpha, 0
                )
                
                # Optional contour outline for better visibility of small lesions like MA
                contours, _ = cv2.findContours((mask > 10).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(composite, contours, -1, palette["color_rgb"], 1)

    retinal_area = h * w
    burden_pct = (total_lesion_pixels / max(1, retinal_area)) * 100.0

    summary = {
        "detected_lesions": stats,
        "total_lesion_pixels": total_lesion_pixels,
        "retinal_burden_pct": round(burden_pct, 3),
        "has_lesions": len(stats) > 0
    }
    return composite, summary
