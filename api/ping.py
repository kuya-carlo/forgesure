from flask import Flask, jsonify

from plate_service import PlateOptimizationService

app = Flask(__name__)


@app.route('/', methods=['GET'])
def handler():
    """Simple health check for Vercel serverless function."""
    return jsonify({"status": "ok", "message": "API is running"}), 200


# Expose `app` for the Vercel Python runtime
