"""
Dataset Preparation and Catalog Generation Module
Diabetic Retinopathy Screening Using Deep Learning

Scans:
- images/ (585 images: IDRID, MAPLES, RETINOMIX, VESSEL)
- lesion_masks/ (MA, HE, EX, SE, CW, NV, OD, merged)
- vessel_masks/ (vessel segmentation masks)

Produces:
- data/dataset_catalog.csv: Comprehensive metadata with ground truth DR grading,
  lesion pixel burdens, vessel mask paths, and clinical stage classifications.
"""

import os
import cv2
import numpy as np
import pandas as pd

def get_mask_nonzero_count(path: str) -> int:
    """Returns number of positive mask pixels or 0 if file doesn't exist."""
    if not path or not os.path.exists(path):
        return 0
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0
        return int(np.count_nonzero(img > 10))
    except Exception:
        return 0

def generate_catalog(base_dir: str = ".") -> pd.DataFrame:
    images_dir = os.path.join(base_dir, "images")
    vessel_dir = os.path.join(base_dir, "vessel_masks")
    lesion_dir = os.path.join(base_dir, "lesion_masks")
    output_dir = os.path.join(base_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found at: {images_dir}")

    all_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
    ])
    
    records = []
    print(f"Indexing {len(all_files)} fundus images from {images_dir}...")
    
    for f in all_files:
        base_name = os.path.splitext(f)[0]
        full_img_path = os.path.join(images_dir, f).replace("\\", "/")
        
        # Identify dataset source
        if f.startswith("IDRID"):
            source = "IDRID"
        elif f.startswith("MAPLES"):
            source = "MAPLES"
        elif f.startswith("RETINOMIX"):
            source = "RETINOMIX"
        elif f.startswith("VESSEL"):
            source = "VESSEL"
        else:
            source = "OTHER"
            
        # Match vessel mask
        vessel_mask_path = ""
        for ext in ['.png', '.tif', '.tiff', '.jpg']:
            cand = os.path.join(vessel_dir, f"{base_name}{ext}")
            if os.path.exists(cand):
                vessel_mask_path = cand.replace("\\", "/")
                break

        # Match lesion masks
        # 1. IDRID masks
        ma_path = os.path.join(lesion_dir, "MA", f"{base_name}_MA.tif")
        he_path = os.path.join(lesion_dir, "HE", f"{base_name}_HE.tif")
        ex_path = os.path.join(lesion_dir, "EX", f"{base_name}_EX.tif")
        se_path = os.path.join(lesion_dir, "SE", f"{base_name}_SE.tif")
        od_path = os.path.join(lesion_dir, "OD", f"{base_name}_OD.tif")
        
        # 2. MAPLES masks
        cw_path = os.path.join(lesion_dir, "CW", f"{base_name}.png")
        nv_path = os.path.join(lesion_dir, "NV", f"{base_name}.png")
        
        # 3. Merged mask
        merged_path = os.path.join(lesion_dir, "merged", f"{base_name}.png")
        if not os.path.exists(merged_path):
            merged_path = ""
            
        ma_px = get_mask_nonzero_count(ma_path)
        he_px = get_mask_nonzero_count(he_path)
        ex_px = get_mask_nonzero_count(ex_path)
        se_px = get_mask_nonzero_count(se_path)
        od_px = get_mask_nonzero_count(od_path)
        cw_px = get_mask_nonzero_count(cw_path)
        nv_px = get_mask_nonzero_count(nv_path)
        
        # Determine HRF specific condition
        is_hrf_dr = "_dr_HRF" in f or "_dr_" in f
        
        # Ground Truth Clinical DR Severity Grading (ICDR Scale):
        # Grade 4: Proliferative DR (Neovascularization NV > 20 OR massive hemorrhage > 400,000 px)
        # Grade 3: Severe DR (Cotton Wool spots / Soft Exudates SE > 500 or CW > 500 OR heavy hemorrhages > 80,000 px)
        # Grade 2: Moderate DR (Hard Exudates EX > 500 OR moderate hemorrhages > 2,000 px)
        # Grade 1: Mild DR (Microaneurysms MA > 50 OR mild lesions OR HRF DR)
        # Grade 0: No DR (Healthy retina, no lesions)
        if nv_px > 20 or he_px > 400000:
            dr_grade = 4
            dr_stage = "Proliferative DR"
            clinical_description = "Neovascularization and/or severe preretinal hemorrhage detected. High proliferative risk."
        elif se_px > 500 or cw_px > 500 or he_px > 80000:
            dr_grade = 3
            dr_stage = "Severe DR"
            clinical_description = "Extensive intraretinal hemorrhages and/or cotton-wool soft exudates present."
        elif ex_px > 500 or he_px > 2000:
            dr_grade = 2
            dr_stage = "Moderate DR"
            clinical_description = "More than microaneurysms: hard exudates and moderate hemorrhages detected."
        elif ma_px > 50 or is_hrf_dr or (he_px + ex_px + se_px + cw_px + nv_px > 0):
            dr_grade = 1
            dr_stage = "Mild DR"
            clinical_description = "Isolated microaneurysms present. Early diabetic retinal changes."
        else:
            dr_grade = 0
            dr_stage = "No DR"
            clinical_description = "Healthy retinal fundus with normal vasculature and no detectable diabetic lesions."
            
        # Clinical Screening Distinctions:
        # 1. is_dr (Any DR): Grade >= 1 (Disease Present vs Absent)
        # 2. referable_dr (rDR): Grade >= 2 (Moderate/Severe/Proliferative DR requiring ophthalmologist clinic referral)
        # 3. urgent_referral: Grade >= 3 (Immediate specialist triage)
        is_dr = 1 if dr_grade >= 1 else 0
        referable_dr = 1 if dr_grade >= 2 else 0
        urgent_referral = 1 if dr_grade >= 3 else 0

        records.append({
            "image_file": f,
            "image_path": full_img_path,
            "source": source,
            "dr_grade": dr_grade,
            "dr_stage": dr_stage,
            "is_dr": 0 if dr_grade == 0 else 1,
            "referable_dr": referable_dr,
            "urgent_referral": urgent_referral,
            "has_vessel_mask": int(bool(vessel_mask_path)),
            "vessel_mask_path": vessel_mask_path,
            "ma_px": ma_px,
            "he_px": he_px,
            "ex_px": ex_px,
            "se_px": se_px,
            "cw_px": cw_px,
            "nv_px": nv_px,
            "od_px": od_px,
            "has_merged_mask": int(bool(merged_path and os.path.exists(merged_path))),
            "merged_mask_path": merged_path.replace("\\", "/"),
            "clinical_description": clinical_description
        })

    df = pd.DataFrame(records)
    csv_out = os.path.join(output_dir, "dataset_catalog.csv")
    df.to_csv(csv_out, index=False)
    print(f"Successfully generated catalog: {csv_out}")
    print(f"Total Images: {len(df)}")
    print("Class Distribution:\n", df['dr_grade'].value_counts().sort_index())
    return df

if __name__ == "__main__":
    generate_catalog(".")
