# 🚀 Model & Application Deployment Guide

### Diabetic Retinopathy Screening System Using Deep Learning
* **Authors:** Rishi Kumar (25scs2040002265), Vanshika Pal (25scs2040003225), Samia Khan (25scs2040003487)
* **Institution:** Department of Computer Applications, IILM University
* **Supervisor:** Dr. Gaurav Kumar (2MCA1, Batch 2025–2027)

---

## 🌟 Quick Overview of Deployment Targets

| Method | Target Platform | Type | Cost | Setup Time | Best For |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Option 1** | **Local Network / Intranet** | Live Dashboard | Free | **Instant (0 min)** | Viva demo, campus lab presentation, Wi-Fi testing |
| **Option 2** | **Streamlit Community Cloud** | Web Application | Free | **2 mins** | Public worldwide URL (`.streamlit.app`) |
| **Option 3** | **Hugging Face Spaces** | ML Cloud Space | Free | **3 mins** | Academic portfolio, AI community showcase |
| **Option 4** | **Docker Containerization** | Microservice | Free/Varies | **5 mins** | AWS, GCP, Azure, DigitalOcean, Render |
| **Option 5** | **Production REST API (`api.py`)** | Headless Backend | Free | **1 min** | Mobile apps, EHR integration, hospital databases |

---

## 🖥️ Option 1: Instant Local & Campus Wi-Fi Deployment (Active Now!)

Your Streamlit application is **already live and running** on your local machine and campus/home network.

### 1. On Your Current Computer:
Open your browser and visit:
* **[http://localhost:8501](http://localhost:8501)**

### 2. On Any Phone, Tablet, or Colleague's Laptop (Same Wi-Fi):
Anyone connected to the same Wi-Fi network can access the live screening system using your local IP:
* **`http://192.168.1.3:8501`**

*(Make sure Windows Firewall allows inbound connections on port 8501 if prompted).*

---

## ☁️ Option 2: Streamlit Community Cloud (Free 1-Click Worldwide Hosting)

Deploy your application to a free, public URL (e.g., `https://rishi-dr-screening.streamlit.app`).

### Step-by-Step Instructions:

1. **Push your project to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Deploy Diabetic Retinopathy Screening System"
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/diabetic-retinopathy-screening.git
   git push -u origin main
   ```

2. **Go to Streamlit Community Cloud:**
   * Visit: [share.streamlit.io](https://share.streamlit.io)
   * Sign in with your **GitHub account**.

3. **Deploy New App:**
   * Click the **"New app"** button.
   * **Repository:** Select your repository `diabetic-retinopathy-screening`.
   * **Branch:** `main`
   * **Main file path:** `app.py`
   * Click **"Deploy!"**

4. **Result:**
   Streamlit will automatically read `.streamlit/config.toml` and `requirements.txt`, install dependencies, load `saved_models/efficientnet_b0_dr.weights.h5`, and give you a permanent public URL to share with professors, external examiners, or clinics!

---

## 🤗 Option 3: Hugging Face Spaces (Free Cloud Hosting)

Hugging Face provides free CPU/GPU spaces for machine learning projects:

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and create an account.
2. Click **"Create new Space"**.
3. Fill in:
   * **Space Name:** `diabetic-retinopathy-screening`
   * **License:** `apache-2.0` or `mit`
   * **Select SDK:** **Streamlit**
   * **Space Hardware:** Free (CPU basic)
4. Push your files to the Hugging Face Space repository using git:
   ```bash
   git remote add space https://huggingface.co/spaces/<YOUR_HF_USERNAME>/diabetic-retinopathy-screening
   git push space main
   ```
5. Your Space will build and be live at `https://huggingface.co/spaces/<YOUR_HF_USERNAME>/diabetic-retinopathy-screening`!

---

## 🐳 Option 4: Docker Container Deployment

The project includes a ready-to-run [Dockerfile](file:///d:/rishi%20i/Project/Dockerfile) and [docker-compose.yml](file:///d:/rishi%20i/Project/docker-compose.yml).

### 1. Build & Run with Docker Compose:
```bash
docker compose up --build
```
This starts both:
* **Streamlit Dashboard:** `http://localhost:8501`
* **Production REST API:** `http://localhost:5000`

### 2. Run Standalone Docker Container:
```bash
# Build Docker image
docker build -t dr-screening-system .

# Run container on port 8501
docker run -p 8501:8501 dr-screening-system
```

### 3. Deploy to Cloud (Render / Railway / GCP Cloud Run):
Connect your GitHub repository to [Render.com](https://render.com) or [Railway.app](https://railway.app), select **Docker**, and it will build and deploy automatically.

---

## 🔌 Option 5: Production Headless REST API (`api.py`)

For integrating the AI model into hospital EHR systems, mobile applications (iOS/Android), or custom websites, use the dedicated Flask REST API:

### 1. Start the API Server:
```bash
.\.venv\Scripts\python.exe api.py
```
*API runs on `http://localhost:5000` (or `PORT` environment variable).*

### 2. Health Check Endpoint:
```bash
curl http://localhost:5000/health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "backbone": "EfficientNetB0 (Transfer Learning)",
  "classes": [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
  ],
  "weights_present": true
}
```

### 3. Predict Endpoint (`POST /predict`):
Send a retinal fundus image as multipart form data:
```bash
curl -X POST -F "file=@images/IDRID_test_IDRiD_57.jpg" http://localhost:5000/predict
```
**JSON Response:**
```json
{
  "success": true,
  "diagnostic_result": {
    "predicted_grade": 2,
    "stage_name": "Moderate Diabetic Retinopathy",
    "short_name": "Moderate DR",
    "confidence_pct": 89.4,
    "any_dr_probability_pct": 98.2,
    "referable_dr_probability_pct": 93.0,
    "is_referable": true,
    "risk_level": "Moderate Risk",
    "clinical_description": "Hard lipid exudates and dot-blot hemorrhages present.",
    "referral_recommendation": "Ophthalmologist referral within 2-4 months for dilated fundus exam."
  },
  "quality_assessment": {
    "status": "PASS",
    "is_good_quality": true,
    "laplacian_focus_score": 178.6,
    "mean_illumination": 98.6,
    "contrast_score": 36.4
  }
}
```

### 4. Grad-CAM Explainability Endpoint (`POST /explain`):
```bash
curl -X POST -F "file=@images/IDRID_test_IDRiD_57.jpg" http://localhost:5000/explain
```
Returns a base64-encoded visual heatmap JPEG highlighting pathological retinal lesions!

---

## 📋 Summary of Deployment Files Created in This Project

1. [.streamlit/config.toml](file:///d:/rishi%20i/Project/.streamlit/config.toml): Cloud production port, headless, and UI theme settings.
2. [api.py](file:///d:/rishi%20i/Project/api.py): Production Flask REST API with `/predict`, `/explain`, and `/health` endpoints.
3. [Dockerfile](file:///d:/rishi%20i/Project/Dockerfile): Multi-stage containerization with OpenCV system dependencies.
4. [docker-compose.yml](file:///d:/rishi%20i/Project/docker-compose.yml): Multi-container orchestration (Dashboard + API).
5. [.dockerignore](file:///d:/rishi%20i/Project/.dockerignore): Fast, lightweight Docker build exclusions.
6. [run_app.bat](file:///d:/rishi%20i/Project/run_app.bat): 1-click Windows desktop deployment launcher.
7. [DEPLOYMENT.md](file:///d:/rishi%20i/Project/DEPLOYMENT.md): This comprehensive operational deployment guide.
