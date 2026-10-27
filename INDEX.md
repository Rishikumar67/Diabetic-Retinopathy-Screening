# Project Index & Documentation Sitemap

### Diabetic Retinopathy Screening Using Deep Learning
* **Submitted to:** Dr. Gaurav Kumar  
* **Submitted by:**  
  * **Rishi Kumar** (25scs2040002265)  
  * **Vanshika Pal** (25scs2040003225)  
  * **Samia Khan** (25scs2040003487)  
* **Class & Batch:** 2MCA1 (2025–2027)  
* **Department:** Master of Computer Applications (MCA), IILM University  

---

## 📑 Project Synopsis Chapters (Index Table)

| Chapter | Topic | Page No. | Implemented Module / Source File |
| :---: | :--- | :---: | :--- |
| **1** | **Introduction** | 3 | [README.md](file:///d:/rishi%20i/Project/README.md#1-project-overview--clinical-context) |
| **2** | **Problem Statement** | 3 | [README.md](file:///d:/rishi%20i/Project/README.md#1-project-overview--clinical-context) |
| **3** | **Objectives of the Project** | 3 | [Diabetic_Retinopathy.ipynb](file:///d:/rishi%20i/Project/Diabetic_Retinopathy.ipynb#cell-0) |
| **4** | **Scope of the Project** | 4 | [data/dataset_catalog.csv](file:///d:/rishi%20i/Project/data/dataset_catalog.csv) (5-Class ICDR Scale) |
| **5** | **Literature Review** | 5 | [README.md](file:///d:/rishi%20i/Project/README.md#8-academic-references-bibliography) |
| **6** | **Proposed System** | 4 | [src/model.py](file:///d:/rishi%20i/Project/src/model.py), [src/gradcam.py](file:///d:/rishi%20i/Project/src/gradcam.py) |
| **7** | **Technology Used** | 6 | [requirements.txt](file:///d:/rishi%20i/Project/requirements.txt) |
| **8** | **Methodology** | 7 | [train.py](file:///d:/rishi%20i/Project/train.py), [src/preprocessing.py](file:///d:/rishi%20i/Project/src/preprocessing.py) |
| **9** | **Features of the System** | 10 | [app.py](file:///d:/rishi%20i/Project/app.py), [index.html](file:///d:/rishi%20i/Project/index.html) |
| **10** | **Advantages of the Proposed System** | 10 | [README.md](file:///d:/rishi%20i/Project/README.md#2-five-class-dr-severity-classification-icdr-scale) |
| **11** | **Limitations of the System** | 11 | [README.md](file:///d:/rishi%20i/Project/README.md#4-dataset-organization-in-this-project) |
| **12** | **Future Enhancements** | 11 | [app.py](file:///d:/rishi%20i/Project/app.py) (Tab 7) |
| **13** | **Conclusion** | 11 | [Diabetic_Retinopathy.ipynb](file:///d:/rishi%20i/Project/Diabetic_Retinopathy.ipynb) (Cell 15) |
| **14** | **Bibliography** | 12 | [README.md](file:///d:/rishi%20i/Project/README.md#8-academic-references-bibliography) |

---

## 🗂️ Complete Codebase & File Sitemap

### 1. User Interfaces (UI)
* [index.html](file:///d:/rishi%20i/Project/index.html): **Standalone Web Application UI** (HTML5 + CSS + JavaScript Canvas Engine). Includes live upload, interactive 5-class staging, quality check, Grad-CAM simulator, lesion layer toggles, and printable clinical report.
* [index.css](file:///d:/rishi%20i/Project/index.css): Design system with dark glassmorphism, Google Fonts (`Plus Jakarta Sans`), and custom responsive layout.
* [index.js](file:///d:/rishi%20i/Project/index.js): Client-side image processing, Laplacian focus calculation, and multi-lesion rendering logic.
* [app.py](file:///d:/rishi%20i/Project/app.py): **Streamlit Web Application** with live Python AI backend.
* [run_app.bat](file:///d:/rishi%20i/Project/run_app.bat): 1-click batch launcher to start the Streamlit server.

### 2. Core Python Source Modules (`src/`)
* [src/preprocessing.py](file:///d:/rishi%20i/Project/src/preprocessing.py): Laplacian blur score, brightness analyzer, Ben Graham normalization, and Green Channel CLAHE.
* [src/model.py](file:///d:/rishi%20i/Project/src/model.py): EfficientNetB0 Transfer Learning architecture and 5-class diagnostic mapping.
* [src/gradcam.py](file:///d:/rishi%20i/Project/src/gradcam.py): Gradient-weighted Class Activation Mapping (Grad-CAM) module.
* [src/vessel_lesion_analyzer.py](file:///d:/rishi%20i/Project/src/vessel_lesion_analyzer.py): Retinal vessel segmentation and multi-lesion overlays.
* [src/evaluation.py](file:///d:/rishi%20i/Project/src/evaluation.py): Confusion matrix, accuracy, sensitivity, and specificity metrics.

### 3. Pipeline & Jupyter Notebook
* [Diabetic_Retinopathy.ipynb](file:///d:/rishi%20i/Project/Diabetic_Retinopathy.ipynb): 16-cell step-by-step Jupyter Notebook.
* [train.py](file:///d:/rishi%20i/Project/train.py): Model training script with class weighting and learning rate scheduling.
* [prepare_dataset.py](file:///d:/rishi%20i/Project/prepare_dataset.py): Dataset catalog generator.

### 4. Data & Trained Artifacts
* [data/dataset_catalog.csv](file:///d:/rishi%20i/Project/data/dataset_catalog.csv): Precomputed metadata catalog for all 585 retinal images.
* [saved_models/efficientnet_b0_dr.weights.h5](file:///d:/rishi%20i/Project/saved_models/efficientnet_b0_dr.weights.h5): Trained EfficientNetB0 model weights.
* [saved_models/model_metadata.json](file:///d:/rishi%20i/Project/saved_models/model_metadata.json): Model configuration and hyperparameters.
* [reports/metrics_summary.json](file:///d:/rishi%20i/Project/reports/metrics_summary.json): Validation cohort metrics summary.
* [reports/confusion_matrix.png](file:///d:/rishi%20i/Project/reports/confusion_matrix.png): 5x5 multi-class confusion matrix plot.
* [requirements.txt](file:///d:/rishi%20i/Project/requirements.txt): Required Python dependencies.
* [README.md](file:///d:/rishi%20i/Project/README.md): Comprehensive project documentation.

### 5. Deployment Configurations & Production Microservices
* [DEPLOYMENT.md](file:///d:/rishi%20i/Project/DEPLOYMENT.md): Step-by-step deployment guide for Streamlit Community Cloud, Hugging Face Spaces, Docker, and local campus networks.
* [api.py](file:///d:/rishi%20i/Project/api.py): Production Flask REST API server with `/predict`, `/explain` (Grad-CAM), and `/health` endpoints.
* [Dockerfile](file:///d:/rishi%20i/Project/Dockerfile): Production Docker container configuration with OpenCV support.
* [docker-compose.yml](file:///d:/rishi%20i/Project/docker-compose.yml): Multi-container orchestration for Web Dashboard and REST API.
* [.streamlit/config.toml](file:///d:/rishi%20i/Project/.streamlit/config.toml): Production Streamlit cloud port, headless, and aesthetic theme settings.
