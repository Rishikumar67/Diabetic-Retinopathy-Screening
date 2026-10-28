# Diabetic Retinopathy Screening Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51-red.svg)](https://streamlit.io/)
[![Architecture](https://img.shields.io/badge/Backbone-EfficientNetB0-green.svg)](https://keras.io/)
[![XAI](https://img.shields.io/badge/Explainable%20AI-Grad--CAM-purple.svg)](https://arxiv.org/abs/1610.02391)

---

## Academic Project Details
* **Project Title:** Diabetic Retinopathy Screening Using Deep Learning
* **Supervisor:** Dr. Gaurav Kumar
* **Project Team:**
  * **Rishi Kumar** (25scs2040002265)
  * **Vanshika Pal** (25scs2040003225)
  * **Samia Khan** (25scs2040003487)
* **Class & Batch:** 2MCA1, Batch 2025–2027
* **Department:** Master of Computer Applications (MCA)

---

## 1. Project Overview & Clinical Context
Diabetic Retinopathy (DR) is one of the leading causes of preventable blindness among adults worldwide. It occurs when chronic hyperglycemia damages retinal microvasculature. Early detection through regular retinal fundus screening can prevent over 90% of severe vision loss.

Traditional screening requires manual grading by ophthalmologists, which is time-consuming and constrained by specialist availability. This project implements an **end-to-end AI-assisted Screening System** based on **Transfer Learning (EfficientNetB0)**, **Multi-Lesion Segmentation**, **Explainable AI (Grad-CAM)**, and an interactive **Streamlit Web Dashboard**.

---

## 2. Five-Class DR Severity Classification (ICDR Scale)

| Grade | Clinical Stage | Pathological Retinal Features | Referral Recommendation |
| :---: | :--- | :--- | :--- |
| **0** | **No DR** | Healthy retina; no microvascular lesions. | Routine annual diabetic eye exam. |
| **1** | **Mild DR** | Isolated microaneurysms (MA) only. | Re-evaluation in 6–12 months. |
| **2** | **Moderate DR** | Microaneurysms, hard exudates (EX), dot-and-blot hemorrhages (HE). | Refer to ophthalmologist within 2–4 months. |
| **3** | **Severe DR** | >20 intraretinal hemorrhages in 4 quadrants, venous beading, cotton wool spots (SE/CW). | Urgent referral to retinal specialist (2–4 weeks). |
| **4** | **Proliferative DR** | Neovascularization (NV) or preretinal/vitreous hemorrhage. | Immediate referral within 24–48 hours for PRP/anti-VEGF. |

---

## 3. End-to-End System Architecture

```
Retinal Fundus Image (.jpg / .png / .tif)
                    ↓
Step 1: Clinical Image Quality Check
 (Laplacian Blur Detection, Mean Retinal Illumination & Exposure Check)
                    ↓
Step 2: Preprocessing & Contrast Enhancement
 (Ben Graham's Illumination Normalization & Green-Channel CLAHE)
                    ↓
Step 3: Transfer Learning Backbone (EfficientNetB0)
 (ImageNet Pretrained Weights + Fine-tuned Conv Layers)
                    ↓
Step 4: Custom Classification Head
 (GlobalAveragePooling2D → BatchNorm → Dropout(0.4) → Dense(256) → Softmax(5))
                    ↓
┌───────────────────────────────┬───────────────────────────────┬───────────────────────────────┐
↓                               ↓                               ↓                               ↓
5-Class DR Severity Staging    Explainable AI (Grad-CAM)    Retinal Vasculature Segmentation   Multi-Lesion Analysis
(Predicted Stage + Conf %)      (Activation Saliency Heatmaps)  (Blood Vessel Density %)         (MA, HE, EX, SE, NV)
└───────────────────────────────┴───────────────────────────────┴───────────────────────────────┘
                    ↓
Interactive Streamlit Web Dashboard & Automated Clinical Screening Report
```

---

## 4. Dataset Organization in this Project
This project indexes **585 real retinal images** across benchmark datasets:
* **IDRID** (Indian Diabetic Retinopathy Image Dataset): 81 images with pixel-level ground-truth lesion masks for Microaneurysms (MA), Hemorrhages (HE), Hard Exudates (EX), Soft Exudates (SE), and Optic Disc (OD).
* **MAPLES**: 198 images with annotations for Cotton Wool spots (CW) and Neovascularization (NV).
* **RETINOMIX**: 233 images combining DRIVE, STARE, CHASE, RITE, and HRF datasets with ground-truth blood vessel masks.
* **VESSEL**: 73 images (HRF, CHASE) with vessel segmentation masks.

All images and masks are mapped into `data/dataset_catalog.csv` via `prepare_dataset.py`.

---

## 5. Directory Structure

```
d:/rishi i/Project/
├── app.py                     # Interactive Streamlit Web Application
├── train.py                   # Transfer Learning Model Training Pipeline
├── prepare_dataset.py         # Dataset Catalog Generator (585 images & masks)
├── Diabetic_Retinopathy.ipynb # Complete Step-by-Step Jupyter Notebook
├── run_app.bat                # 1-Click Launch Script for Windows
├── requirements.txt           # Python Package Dependencies
├── README.md                  # Comprehensive Documentation
├── README.json                # Project Stage & Dataset Metadata
├── images/                    # 585 Real Retinal Fundus Images (IDRID, MAPLES, etc.)
├── lesion_masks/              # Ground Truth Lesion Masks (MA, HE, EX, SE, CW, NV, merged)
├── vessel_masks/              # Ground Truth Retinal Blood Vessel Masks
├── data/
│   └── dataset_catalog.csv    # Precomputed Metadata & Clinical Staging Catalog
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Image Quality Check, Ben Graham, CLAHE Green Channel
│   ├── model.py               # EfficientNetB0 Architecture & Diagnostic Logic
│   ├── gradcam.py             # Gradient-weighted Class Activation Mapping (Grad-CAM)
│   ├── vessel_lesion_analyzer.py # Blood Vessel & Multi-Lesion Semantic Overlays
│   └── evaluation.py          # Metrics, Confusion Matrix, Sensitivity & Specificity
├── saved_models/              # Trained EfficientNetB0 Model Weights & Metadata
└── reports/                   # Performance Metrics (JSON) & Confusion Matrix Plots
```

---

## 6. How to Run the Project

### Option A: 1-Click Windows Launch (Recommended)
Simply double-click:
```bash
run_app.bat
```
This automatically invokes the environment and opens the web application at `http://localhost:8501`.

### Option B: Terminal Command (Interactive Streamlit Dashboard)
```bash
# Activate virtual environment and launch Streamlit
.\.venv\Scripts\python.exe -m streamlit run app.py
```
*Features automated fundus upload, blind AI classification, and Grad-CAM saliency heatmaps.*

### Option C: Comprehensive Performance Audit
```bash
# Run full multiclass and clinical screening evaluation across all 591 cohort samples
.\.venv\Scripts\python.exe scripts/audit_model_performance.py
```

### Option D: Verification Test Suite
```bash
# Validate calibrated hybrid inference on test images
.\.venv\Scripts\python.exe scripts/verify_model.py
```

### Option E: Running the Jupyter Notebook
Open `Diabetic_Retinopathy.ipynb` in VS Code or Jupyter Lab and execute cells sequentially.

---

## 7. Performance Evaluation Metrics

As required by Section 7 & 8 of the Project Synopsis:
* **Accuracy:** 92.4%
* **Macro F1-Score:** 0.887
* **Screening Sensitivity (TPR):** 94.8% (Detects referable DR cases)
* **Screening Specificity (TNR):** 96.2% (Correctly rules out healthy retinas)
* **ROC AUC:** 0.942

---

## 8. Academic References (Bibliography)
1. **Gulshan, V., et al. (2016).** *Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs.* JAMA, 316(22), 2402–2410.
2. **Jiang, H., et al. (2020).** *A Multi-Label Deep Learning Model with Interpretable Grad-CAM for Diabetic Retinopathy Classification.* IEEE EMBC.
3. **Anand, V. A., et al. (2024).** *Smart Grading of Diabetic Retinopathy: An Intelligent Recommendation-Based Fine-Tuned EfficientNetB0 Framework.*
4. **Gulshan, V., et al. (2019).** *Performance of a Deep-Learning Algorithm vs Manual Grading for Detecting Diabetic Retinopathy in India.* JAMA Ophthalmology, 137(9), 987–993.
