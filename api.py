"""
Diabetic Retinopathy Screening System - Production REST API
Authors: Rishi Kumar, Vanshika Pal, Samia Khan
Institution: IILM University

Provides production HTTP REST endpoints for automated fundus screening,
severity staging, quality assessment, and Grad-CAM explainability.
"""

import os
import io
import base64
import json
from typing import Dict, Any
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify

# Project AI modules
from src.model import load_or_create_model, predict_fundus_severity, SEVERITY_CLASSES
from src.preprocessing import (
    load_image, check_image_quality, preprocess_for_model,
    apply_clahe_green, apply_ben_graham_enhancement
)
from src.gradcam import compute_gradcam_heatmap, overlay_gradcam_on_image

app = Flask(__name__, static_folder=".", static_url_path="")
# Standard WSGI / Serverless handler aliases (Vercel, AWS Lambda, Gunicorn)
application = app
handler = app

# Global model instance with resilient lazy loading
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "efficientnet_b0_dr.weights.h5")
_cached_model = None

def get_model():
    """Lazily load or build the EfficientNet model when first needed."""
    global _cached_model
    if _cached_model is None:
        try:
            print(f"Initializing EfficientNetB0 AI model from {WEIGHTS_PATH}...")
            _cached_model = load_or_create_model(WEIGHTS_PATH if os.path.exists(WEIGHTS_PATH) else None)
            print("Model ready for inference.")
        except Exception as e:
            print(f"Notice: Deferred model initialization: {e}")
    return _cached_model

# Warm up model if possible
try:
    _cached_model = get_model()
except Exception:
    _cached_model = None

@app.after_request
def add_cors_headers(response):
    """Enable Cross-Origin Resource Sharing (CORS) for web frontends."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route("/", methods=["GET"])
def index():
    # If client prefers HTML or opens in browser, serve index.html if available
    accept = request.headers.get("Accept", "")
    if ("text/html" in accept or "*/*" in accept) and os.path.exists(os.path.join(os.path.dirname(__file__), "index.html")):
        from flask import send_file
        return send_file(os.path.join(os.path.dirname(__file__), "index.html"))
    return jsonify({
        "system": "Diabetic Retinopathy AI Screening REST API",
        "version": "1.0.0",
        "institution": "IILM University",
        "authors": ["Rishi Kumar", "Vanshika Pal", "Samia Khan"],
        "endpoints": {
            "GET /health": "System status and model health",
            "POST /predict": "Upload fundus image for DR severity & referral prediction",
            "POST /explain": "Upload fundus image for Grad-CAM explainability heatmap"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    active_model = get_model()
    return jsonify({
        "status": "healthy",
        "model_loaded": active_model is not None,
        "backbone": "EfficientNetB0 (Transfer Learning)",
        "classes": [c["name"] for c in SEVERITY_CLASSES.values()],
        "weights_present": os.path.exists(WEIGHTS_PATH)
    })

def _parse_uploaded_image(req) -> np.ndarray:
    """Helper to extract RGB image from multipart file or JSON base64."""
    if "file" in req.files:
        file_bytes = req.files["file"].read()
        return load_image(file_bytes)
    elif req.is_json and "image_base64" in req.json:
        raw_b64 = req.json["image_base64"]
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        decoded = base64.b64decode(raw_b64)
        return load_image(decoded)
    else:
        raise ValueError("No retinal fundus image provided. Send as multipart 'file' or JSON 'image_base64'.")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        img_rgb = _parse_uploaded_image(request)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    active_model = get_model()
    if active_model is None:
        return jsonify({"error": "AI screening model is currently initializing or unavailable."}), 503

    # 1. Quality Check
    quality = check_image_quality(img_rgb)

    # 2. Preprocess & Predict
    norm_input = preprocess_for_model(img_rgb)
    prediction = predict_fundus_severity(active_model, norm_input)

    response_payload = {
        "success": True,
        "diagnostic_result": {
            "predicted_grade": prediction["predicted_grade"],
            "stage_name": prediction["stage_name"],
            "short_name": prediction["short_name"],
            "confidence_pct": round(prediction["confidence"] * 100, 2),
            "any_dr_probability_pct": round(prediction["any_dr_probability"] * 100, 2),
            "referable_dr_probability_pct": round(prediction["referable_probability"] * 100, 2),
            "is_referable": prediction["is_referable"],
            "risk_level": prediction["risk_level"],
            "clinical_description": prediction["clinical_description"],
            "referral_recommendation": prediction["referral_recommendation"]
        },
        "probabilities": {
            k: round(v * 100, 2) for k, v in prediction["probabilities"].items()
        },
        "quality_assessment": {
            "status": quality["status"],
            "is_good_quality": quality["is_good_quality"],
            "laplacian_focus_score": round(quality["blur_score"], 2),
            "mean_illumination": round(quality["brightness"], 2),
            "contrast_score": round(quality["contrast"], 2),
            "issues": quality["issues"]
        }
    }
    return jsonify(response_payload)

@app.route("/explain", methods=["POST"])
def explain():
    try:
        img_rgb = _parse_uploaded_image(request)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    active_model = get_model()
    if active_model is None:
        return jsonify({"error": "AI screening model is currently initializing or unavailable."}), 503

    norm_input = preprocess_for_model(img_rgb)
    pred = predict_fundus_severity(active_model, norm_input)
    grade = pred["predicted_grade"]

    heatmap = compute_gradcam_heatmap(active_model, norm_input, class_idx=grade)
    superimposed, colored_heatmap = overlay_gradcam_on_image(img_rgb, heatmap, alpha=0.45)

    # Convert overlaid image to base64 JPEG
    success, buffer = cv2.imencode(".jpg", cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
    if not success:
        return jsonify({"error": "Failed to encode heatmap"}), 500

    b64_img = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "success": True,
        "predicted_grade": grade,
        "stage_name": pred["stage_name"],
        "heatmap_base64": f"data:image/jpeg;base64,{b64_img}"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Diabetic Retinopathy REST API on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
