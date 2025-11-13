from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/', methods=['GET'])
def handler():
    """Simple health check for Vercel serverless function."""
    return jsonify({"status": "ok", "message": "API is running"}), 200
