from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.

    Returns:
        JSON response with service status and HTTP 200 OK.
    """
    return jsonify({
        "status": "healthy",
        "message": "CropScan AI Backend service is running",
        "version": "0.1.0"
    }), 200
