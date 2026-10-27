from flask import Blueprint, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from app.services.image_validator import ImageValidationError, validate_and_load_image
from app.services.predictor import (
    InferenceExecutionError,
    ModelLoadError,
    PredictionError,
    get_prediction_service,
)

predict_bp = Blueprint("predict", __name__, url_prefix="/api")

SUPPORTED_LANGUAGES = {"en", "yo"}


@predict_bp.route("/predict", methods=["POST"])
def predict():
    """
    POST /api/predict

    Receives a leaf image upload via multipart form-data, performs security
    and raster validation, runs genuine inference on MobileNetV2, and returns
    verified predictions along with localized agronomic guidance in English or Yoruba.

    Form Parameters:
        image: File (JPEG, PNG, or WebP up to 10 MB).
        lang: Optional string ('en' or 'yo', defaults to 'en').

    Response (Phase 3):
        Returns HTTP 200 with genuine prediction, model confidence, assessment
        status, and localized guidance.
    """
    # 1. Validate Language Parameter
    raw_lang = request.form.get("lang") or request.args.get("lang") or "en"
    lang = str(raw_lang).strip().lower()

    if lang not in SUPPORTED_LANGUAGES:
        return jsonify({
            "error": "invalid_language",
            "message": "Supported languages are 'en' (English) and 'yo' (Yoruba).",
        }), 400

    # 2. Validate Presence of Image File
    if "image" not in request.files:
        return jsonify({
            "error": "missing_image",
            "message": "No image file provided in multipart form data (field 'image').",
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "empty_image",
            "message": "The uploaded image file is empty or has no filename.",
        }), 400

    # 3. Validate and Safely Load Image in Memory
    try:
        metadata, rgb_image = validate_and_load_image(file)
    except ImageValidationError as e:
        return jsonify({
            "error": e.error_code,
            "message": e.message,
        }), e.status_code

    # 4. Execute Real Model Inference
    try:
        predictor = get_prediction_service()
        prediction = predictor.predict(rgb_image, lang=lang)
    except ModelLoadError as e:
        return jsonify({
            "error": e.error_code,
            "message": e.message,
        }), e.status_code
    except (InferenceExecutionError, PredictionError) as e:
        return jsonify({
            "error": e.error_code,
            "message": e.message,
        }), e.status_code
    except Exception as e:
        return jsonify({
            "error": "unexpected_inference_error",
            "message": f"An unexpected error occurred during prediction: {str(e)}",
        }), 500

    # 5. Return Genuine Prediction and Localized Guidance
    return jsonify({
        "status": "success",
        "language": lang,
        "image_metadata": metadata,
        "prediction": prediction,
    }), 200


@predict_bp.errorhandler(RequestEntityTooLarge)
def handle_oversized_payload(e):
    """Handle uploads exceeding MAX_CONTENT_LENGTH (10 MB)."""
    return jsonify({
        "error": "payload_too_large",
        "message": "Uploaded image exceeds the maximum permitted size of 10 MB.",
    }), 413

