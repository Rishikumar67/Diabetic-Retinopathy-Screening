"""
Image Preprocessing and Quality Assessment Module
Diabetic Retinopathy Screening Using Deep Learning

Features:
- Automated Image Quality Assessment (blurriness via Laplacian variance, exposure/brightness check)
- Retinal Field-of-View (FOV) circular cropping (Ben Graham method)
- CLAHE (Contrast Limited Adaptive Histogram Equalization) on green channel
- Normalization and preparation for EfficientNetB0 (224x224 RGB)
"""

import os
from typing import Union, Tuple, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image

def load_image(image_input: Union[str, bytes, np.ndarray, Image.Image]) -> Optional[np.ndarray]:
    """
    Safely loads and converts any image format into a standard RGB NumPy array (uint8).
    Supports file paths, raw bytes, PIL Images, or OpenCV arrays.
    """
    try:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return None
            img = cv2.imread(image_input)
            if img is None:
                # Try reading with PIL for TIFF or other special formats
                pil_img = Image.open(image_input)
                img = np.array(pil_img.convert("RGB"))
                return img
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.open(io.BytesIO(image_input))
            return np.array(pil_img.convert("RGB"))
            
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))
            
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.ndim == 3:
                if image_input.shape[2] == 4:
                    return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
                return image_input.copy()
        return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def check_image_quality(
    img_rgb: np.ndarray,
    blur_threshold: float = 60.0,
    dark_threshold: float = 25.0,
    bright_threshold: float = 225.0
) -> Dict[str, Any]:
    """
    Performs clinical-grade image quality validation on fundus images:
    1. Blurriness Detection: Laplacian variance on grayscale.
    2. Exposure & Brightness: Evaluates mean pixel intensity inside retinal field.
    3. Retinal Contrast: Standard deviation of pixel values.
    """
    if img_rgb is None or not isinstance(img_rgb, np.ndarray):
        return {
            "is_valid": False,
            "status": "Invalid Image",
            "message": "Image data could not be decoded.",
            "blur_score": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "issues": ["Unreadable image file"]
        }

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Calculate blur score using Laplacian operator variance
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Create mask to analyze only retinal tissue (excluding extreme black corners)
    retina_mask = gray > 15
    if np.count_nonzero(retina_mask) > 100:
        retinal_pixels = gray[retina_mask]
        mean_brightness = float(np.mean(retinal_pixels))
        contrast_score = float(np.std(retinal_pixels))
    else:
        mean_brightness = float(np.mean(gray))
        contrast_score = float(np.std(gray))

    issues = []
    if laplacian_var < blur_threshold:
        issues.append(f"Focus score is low ({laplacian_var:.1f} < {blur_threshold:.0f}): Image may be blurry or out-of-focus.")
        
    if mean_brightness < dark_threshold:
        issues.append(f"Retinal illumination is low ({mean_brightness:.1f}/255): Image is underexposed/dark.")
    elif mean_brightness > bright_threshold:
        issues.append(f"Retinal illumination is high ({mean_brightness:.1f}/255): Image is overexposed/glared.")

    if contrast_score < 15.0:
        issues.append(f"Low retinal contrast ({contrast_score:.1f}): Fine microvasculature may not be clearly visible.")

    is_good = len(issues) == 0

    return {
        "is_valid": True,
        "is_good_quality": is_good,
        "status": "Sufficient Clinical Quality" if is_good else "Suboptimal Quality (Proceed with Caution)",
        "blur_score": laplacian_var,
        "brightness": mean_brightness,
        "contrast": contrast_score,
        "issues": issues,
        "message": (
            "Retinal fundus image meets diagnostic clarity thresholds."
            if is_good else
            "Artifacts or suboptimal illumination detected. The AI will still analyze the fundus, but verified review is recommended."
        )
    }

def crop_retina_circle(img_rgb: np.ndarray, tolerance: int = 15) -> np.ndarray:
    """
    Crops out excess black borders to focus specifically on the circular retinal field.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > tolerance
    if not np.any(mask):
        return img_rgb

    # Find bounding box of non-black retinal pixels
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Crop
    cropped = img_rgb[y0:y1, x0:x1]
    return cropped

def apply_ben_graham_enhancement(img_rgb: np.ndarray, sigma: int = 15) -> np.ndarray:
    """
    Ben Graham's Retinal Illumination Normalization:
    Subtracts the local color average (via Gaussian blur) and scales contrast.
    Eliminates variations in camera lighting and pupil dilation across datasets.
    """
    try:
        blurred = cv2.GaussianBlur(img_rgb, (0, 0), sigma)
        enhanced = cv2.addWeighted(img_rgb, 4.0, blurred, -4.0, 128)
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    except Exception:
        return img_rgb

def apply_clahe_green(img_rgb: np.ndarray, clip_limit: float = 2.5, tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Extracts the Green channel (which has the highest contrast for blood vessels and hemorrhages in fundus)
    and applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    """
    try:
        green_channel = img_rgb[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        enhanced_green = clahe.apply(green_channel)
        
        # Merge back as a 3-channel visual enhancement
        enhanced_rgb = img_rgb.copy()
        enhanced_rgb[:, :, 1] = enhanced_green
        return enhanced_rgb
    except Exception:
        return img_rgb

def preprocess_for_model(
    img_rgb: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    apply_crop: bool = True
) -> np.ndarray:
    """
    Preprocesses RGB fundus image for EfficientNetB0 input:
    1. Optional circular border crop.
    2. Resizes smoothly using area-averaging interpolation to target_size (224x224).
    3. Normalizes pixel values to [0.0, 1.0] float32.
    """
    if apply_crop:
        img_rgb = crop_retina_circle(img_rgb)
        
    resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 255.0
    return norm
