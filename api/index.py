"""
Diabetic Retinopathy Screening System - Vercel Serverless Function Entrypoint
Institution: IILM University
Authors: Rishi Kumar, Vanshika Pal, Samia Khan | Supervisor: Dr. Gaurav Kumar
"""
from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response_data = {
            "status": "online",
            "service": "Diabetic Retinopathy AI Screening System",
            "version": "1.0.0",
            "institution": "IILM University",
            "authors": [
                "Rishi Kumar (25scs2040002265)",
                "Vanshika Pal (25scs2040003225)",
                "Samia Khan (25scs2040003487)"
            ],
            "supervisor": "Dr. Gaurav Kumar",
            "architecture": "EfficientNetB0 + Grad-CAM Explainable AI",
            "endpoints": {
                "dashboard": "/",
                "api_health": "/api"
            }
        }
        self.wfile.write(json.dumps(response_data, indent=2).encode('utf-8'))

    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response_data = {
            "status": "success",
            "message": "Automated fundus screening client is active in the web dashboard."
        }
        self.wfile.write(json.dumps(response_data).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
