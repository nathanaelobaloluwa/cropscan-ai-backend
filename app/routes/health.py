from flask import Blueprint, jsonify

from app.services.predictor import get_prediction_service

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """
    Process liveness probe endpoint.
    Verifies that the Flask web application process is responsive.
    Reports model readiness truthfully without forcing model loading.

    Returns:
        JSON response with service status and HTTP 200 OK.
    """
    predictor = get_prediction_service()
    return jsonify({
        "status": "healthy",
        "message": "CropScan AI Backend service is running",
        "version": "0.1.0",
        "model_ready": predictor.is_model_loaded,
    }), 200


@health_bp.route("/health/ready", methods=["GET"])
def readiness_check():
    """
    Readiness probe endpoint.
    Truthfully reports whether the AI vision model is loaded into memory
    and ready to serve inference requests without cold-start latency.
    Does NOT force synchronous loading (preserves lazy initialization).

    Returns:
        HTTP 200 if model is loaded into memory.
        HTTP 503 if model is not yet loaded into memory.
    """
    predictor = get_prediction_service()
    is_ready = predictor.is_model_loaded

    return jsonify({
        "status": "ready" if is_ready else "not_ready",
        "model_ready": is_ready,
        "model_name": predictor.model_name,
        "message": (
            "Vision model is loaded into memory and ready for inference."
            if is_ready
            else "Vision model is not yet loaded into memory (lazy initialization active)."
        ),
    }), 200 if is_ready else 503
