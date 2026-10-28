"""
Diabetic Retinopathy Screening System - Vercel Python Application Entrypoint
Authors: Rishi Kumar, Vanshika Pal, Samia Khan
Supervisor: Dr. Gaurav Kumar | Institution: IILM University
"""
import os
import mimetypes
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def app(environ, start_response):
    """
    Standard WSGI callable for Vercel Python Runtime.
    Serves the screening web client (index.html, CSS, JS, images)
    and provides the /api endpoint.
    """
    method = environ.get('REQUEST_METHOD', 'GET')
    path = environ.get('PATH_INFO', '/').lstrip('/')

    # Handle CORS Preflight
    if method == 'OPTIONS':
        start_response('200 OK', [
            ('Access-Control-Allow-Origin', '*'),
            ('Access-Control-Allow-Methods', 'GET, POST, OPTIONS'),
            ('Access-Control-Allow-Headers', 'Content-Type'),
            ('Content-Length', '0')
        ])
        return []

    # API Endpoint
    if path == 'api' or path.startswith('api/'):
        response_data = {
            "status": "online",
            "service": "Diabetic Retinopathy AI Screening Platform",
            "version": "1.0.0",
            "institution": "IILM University",
            "authors": [
                "Rishi Kumar (25scs2040002265)",
                "Vanshika Pal (25scs2040003225)",
                "Samia Khan (25scs2040003487)"
            ],
            "supervisor": "Dr. Gaurav Kumar",
            "architecture": "EfficientNetB0 + Grad-CAM Explainable AI",
            "status_message": "Automated fundus screening client is active."
        }
        body = json.dumps(response_data, indent=2).encode('utf-8')
        start_response('200 OK', [
            ('Content-Type', 'application/json; charset=utf-8'),
            ('Content-Length', str(len(body))),
            ('Access-Control-Allow-Origin', '*')
        ])
        return [body]

    # Default to index.html for root path
    if not path or path == '/':
        path = 'index.html'

    file_path = os.path.join(BASE_DIR, path)

    # Prevent directory traversal attacks
    if not os.path.abspath(file_path).startswith(BASE_DIR):
        start_response('403 Forbidden', [('Content-Type', 'text/plain')])
        return [b'Forbidden']

    # Serve static asset
    if os.path.exists(file_path) and os.path.isfile(file_path):
        mime_type, _ = mimetypes.guess_type(file_path)
        if path.endswith('.js'):
            mime_type = 'application/javascript'
        elif path.endswith('.css'):
            mime_type = 'text/css'
        elif not mime_type:
            mime_type = 'application/octet-stream'

        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            start_response('200 OK', [
                ('Content-Type', mime_type),
                ('Content-Length', str(len(content))),
                ('Access-Control-Allow-Origin', '*'),
                ('Cache-Control', 'public, max-age=0, must-revalidate')
            ])
            return [content]
        except Exception as e:
            start_response('500 Internal Server Error', [('Content-Type', 'text/plain')])
            return [f"Server error: {e}".encode('utf-8')]

    # Fallback to index.html for SPA routing if available
    index_file = os.path.join(BASE_DIR, 'index.html')
    if os.path.exists(index_file):
        with open(index_file, 'rb') as f:
            content = f.read()
        start_response('200 OK', [
            ('Content-Type', 'text/html; charset=utf-8'),
            ('Content-Length', str(len(content))),
            ('Access-Control-Allow-Origin', '*')
        ])
        return [content]

    start_response('404 Not Found', [('Content-Type', 'text/plain')])
    return [b'Not Found']

# Standard WSGI aliases
application = app
handler = app
