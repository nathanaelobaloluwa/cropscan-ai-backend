import logging

from flask import Blueprint, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from app.services.guidance import GuidanceService
from app.services.image_validator import ImageValidationError, validate_and_load_image
from app.services.predictor import (
    InferenceExecutionError,
    ModelLoadError,
    PredictionError,
    get_prediction_service,
)

logger = logging.getLogger(__name__)

predict_bp = Blueprint("predict", __name__, url_prefix="/api")

SUPPORTED_LANGUAGES = {"en", "yo"}


@predict_bp.route("/predict", methods=["POST"])
def predict():
    """
    POST /api/predict

    Receives a leaf image upload and user crop selection via multipart form-data,
    performs security and raster validation, runs genuine inference on MobileNetV2,
    protects against crop mismatches, and returns verified predictions along with
    localized agronomic guidance in English or Yoruba.

    Form Parameters:
        image: File (JPEG, PNG, or WebP up to 10 MB).
        crop: Required string ('tomato', 'maize', 'bell_pepper', 'citrus', 'potato').
        lang: Optional string ('en' or 'yo', defaults to 'en').

    Response (Phase 5):
        Returns HTTP 200 with genuine prediction, selected crop, predicted crop,
        crop match status, model confidence, assessment status, and localized guidance.
    """
    # 1. Validate Language Parameter
    raw_lang = request.form.get("lang") or request.args.get("lang") or "en"
    lang = str(raw_lang).strip().lower()

    if lang not in SUPPORTED_LANGUAGES:
        return jsonify({
            "error": "invalid_language",
            "message": "Supported languages are 'en' (English) and 'yo' (Yoruba).",
        }), 400

    # 2. Validate Crop Selection Parameter
    raw_crop = request.form.get("crop") or request.args.get("crop")
    if not raw_crop or not str(raw_crop).strip():
        return jsonify({
            "error": "missing_crop",
            "message": (
                "Field 'crop' is required. Supported crops are: 'tomato', 'maize', 'bell_pepper', 'citrus', 'potato'."
                if lang == "en"
                else "A nílò orúkọ irúgbìn ní 'crop'. Àwọn irúgbìn tí a tìlẹ́yìn ni: 'tomato', 'maize', 'bell_pepper', 'citrus', 'potato'."
            ),
        }), 400

    selected_crop = GuidanceService.normalize_crop(raw_crop)
    if selected_crop is None:
        return jsonify({
            "error": "unsupported_crop",
            "message": (
                f"Unsupported crop '{raw_crop}'. Supported crops are: 'tomato', 'maize', 'bell_pepper', 'citrus', 'potato'."
                if lang == "en"
                else f"Irúgbìn '{raw_crop}' kò tíì wà ní àmúlò. Àwọn irúgbìn tí a tìlẹ́yìn ni: 'tomato', 'maize', 'bell_pepper', 'citrus', 'potato'."
            ),
        }), 400

    # 3. Validate Presence of Image File
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

    # 4. Validate and Safely Load Image in Memory
    try:
        metadata, rgb_image = validate_and_load_image(file)
    except ImageValidationError as e:
        return jsonify({
            "error": e.error_code,
            "message": e.message,
        }), e.status_code

    # 5. Execute Real Model Inference with Crop Mismatch Protection
    try:
        predictor = get_prediction_service()
        prediction = predictor.predict(rgb_image, selected_crop=selected_crop, lang=lang)
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
    except Exception:
        logger.exception("An unexpected error occurred during prediction.")
        return jsonify({
            "error": "unexpected_inference_error",
            "message": "An unexpected error occurred during prediction.",
        }), 500

    # 6. Return Genuine Prediction, Crop Mismatch Protection & Localized Guidance
    return jsonify({
        "status": "success",
        "selected_crop": selected_crop,
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

