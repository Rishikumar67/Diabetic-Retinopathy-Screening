"""
Diabetic Retinopathy Screening System - Vercel Serverless Function Entrypoint
Institution: IILM University
Authors: Rishi Kumar, Vanshika Pal, Samia Khan | Supervisor: Dr. Gaurav Kumar
"""
import json

def app(environ, start_response):
    """WSGI callable for /api endpoint."""
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
        "architecture": "EfficientNetB0 + Grad-CAM Explainable AI"
    }
    body = json.dumps(response_data, indent=2).encode('utf-8')
    start_response('200 OK', [
        ('Content-Type', 'application/json; charset=utf-8'),
        ('Content-Length', str(len(body))),
        ('Access-Control-Allow-Origin', '*')
    ])
    return [body]

# Aliases
application = app
handler = app
