from flask import Blueprint, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from app.services.image_validator import ImageValidationError, validate_image_upload

predict_bp = Blueprint("predict", __name__, url_prefix="/api")

SUPPORTED_LANGUAGES = {"en", "yo"}


@predict_bp.route("/predict", methods=["POST"])
def predict():
    """
    POST /api/predict

    Receives a leaf image upload via multipart form-data, performs comprehensive
    security and format validation, and returns the upload status.

    Form Parameters:
        image: File (JPEG, PNG, or WebP up to 10 MB).
        lang: Optional string ('en' or 'yo', defaults to 'en').

    Response (Phase 2):
        Returns status 'model_not_configured' with verified image metadata.
        No fabricated predictions or fake confidence scores are returned.
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

    # 3. Validate Image Content Safely in Memory
    try:
        metadata = validate_image_upload(file)
    except ImageValidationError as e:
        return jsonify({
            "error": e.error_code,
            "message": e.message,
        }), e.status_code

    # 4. Transparent Model-Pending Response
    # The real ML model is not integrated yet in Phase 2.
    # We explicitly return model_not_configured without inventing predictions.
    response_message = (
        "Image validated successfully. Plant disease detection model is not configured yet."
        if lang == "en"
        else "Àwòrán náà wà ní àlàáfíà. Àwòṣe àyẹ̀wò àrùn kò tíì wà ní àmúlò."
    )

    return jsonify({
        "status": "model_not_configured",
        "message": response_message,
        "language": lang,
        "image_metadata": metadata,
    }), 200


@predict_bp.errorhandler(RequestEntityTooLarge)
def handle_oversized_payload(e):
    """Handle uploads exceeding MAX_CONTENT_LENGTH (10 MB)."""
    return jsonify({
        "error": "payload_too_large",
        "message": "Uploaded image exceeds the maximum permitted size of 10 MB.",
    }), 413
