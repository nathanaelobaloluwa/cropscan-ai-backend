import io
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from app.services.predictor import InferenceExecutionError, ModelLoadError


def create_in_memory_image(
    format_name: str = "JPEG",
    size: tuple = (120, 120),
    color: tuple = (34, 139, 34)
) -> io.BytesIO:
    """Helper to generate in-memory image bytes without touching disk."""
    buf = io.BytesIO()
    # WebP and PNG support RGBA; JPEG requires RGB or L
    mode = "RGB" if format_name.upper() == "JPEG" else "RGBA"
    fill_color = color if mode == "RGB" else (*color, 255)
    img = Image.new(mode, size, fill_color)
    img.save(buf, format=format_name)
    buf.seek(0)
    return buf


class TestPredictEndpoint:
    """Automated tests for POST /api/predict image upload, crop selection, validation & inference."""

    # 1. Valid Formats & Genuine Model Prediction
    def test_valid_jpeg_upload_returns_genuine_prediction(self, client):
        """Verify valid JPEG upload with crop selection returns 200 with genuine prediction."""
        img_buf = create_in_memory_image("JPEG", size=(200, 150))
        data = {
            "image": (img_buf, "leaf_sample.jpg", "image/jpeg"),
            "crop": "tomato",
            "lang": "en",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 200
        assert response.is_json

        body = response.get_json()
        assert body["status"] == "success"
        assert body["selected_crop"] == "tomato"
        assert body["language"] == "en"

        metadata = body["image_metadata"]
        assert metadata["format"] == "JPEG"
        assert metadata["width"] == 200
        assert metadata["height"] == 150
        assert metadata["color_mode"] == "RGB"
        assert metadata["size_bytes"] > 0

        # Verify real prediction structure
        assert "prediction" in body
        pred = body["prediction"]
        assert isinstance(pred["class_index"], int)
        assert 0 <= pred["class_index"] <= 37
        assert isinstance(pred["class_label"], str)
        assert isinstance(pred["confidence"], float)
        assert 0.0 <= pred["confidence"] <= 1.0
        assert pred["confidence_threshold"] == 0.40
        assert pred["is_provisional_threshold"] is True
        assert pred["selected_crop"] == "tomato"
        assert "crop_match" in pred
        assert pred["assessment_status"] in (
            "confident",
            "inconclusive_low_confidence",
            "recognized_unsupported_crop",
            "crop_mismatch",
        )
        assert pred["guidance"] is not None

    def test_valid_png_upload(self, client):
        """Verify valid PNG upload with crop selection returns 200, correct PNG metadata, and prediction."""
        img_buf = create_in_memory_image("PNG", size=(100, 100))
        data = {
            "image": (img_buf, "leaf.png", "image/png"),
            "crop": "tomato",
            "lang": "en",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 200
        body = response.get_json()
        assert body["status"] == "success"
        assert body["selected_crop"] == "tomato"
        assert body["image_metadata"]["format"] == "PNG"
        assert "prediction" in body

    def test_valid_webp_upload(self, client):
        """Verify valid WebP upload with crop selection returns 200, correct WebP metadata, and prediction."""
        img_buf = create_in_memory_image("WEBP", size=(80, 80))
        data = {
            "image": (img_buf, "leaf.webp", "image/webp"),
            "crop": "maize",
            "lang": "en",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 200
        body = response.get_json()
        assert body["status"] == "success"
        assert body["selected_crop"] == "maize"
        assert body["image_metadata"]["format"] == "WEBP"
        assert "prediction" in body

    def test_yoruba_language_support(self, client):
        """Verify valid upload with lang='yo' returns Yoruba guidance."""
        img_buf = create_in_memory_image("JPEG", size=(64, 64))
        data = {
            "image": (img_buf, "leaf.jpg", "image/jpeg"),
            "crop": "tomato",
            "lang": "yo",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 200
        body = response.get_json()
        assert body["status"] == "success"
        assert body["language"] == "yo"
        assert body["prediction"]["guidance"]["language"] == "yo"

    def test_default_language_defaults_to_english(self, client):
        """Verify that omitting the lang parameter defaults to 'en'."""
        img_buf = create_in_memory_image("JPEG", size=(64, 64))
        data = {
            "image": (img_buf, "leaf.jpg", "image/jpeg"),
            "crop": "tomato",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 200
        body = response.get_json()
        assert body["language"] == "en"
        assert body["prediction"]["guidance"]["language"] == "en"

    # 2. Crop Selection Validation & Mismatch Protection
    def test_missing_crop_rejected_with_400(self, client):
        """Verify that omitting the crop field returns HTTP 400 missing_crop."""
        img_buf = create_in_memory_image("JPEG", size=(64, 64))
        data = {
            "image": (img_buf, "leaf.jpg", "image/jpeg"),
            "lang": "en",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 400
        body = response.get_json()
        assert body["error"] == "missing_crop"
        assert "crop" in body["message"]

    def test_empty_crop_rejected_with_400(self, client):
        """Verify that whitespace-only crop string returns HTTP 400 missing_crop."""
        img_buf = create_in_memory_image("JPEG", size=(64, 64))
        data = {
            "image": (img_buf, "leaf.jpg", "image/jpeg"),
            "crop": "   ",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 400
        body = response.get_json()
        assert body["error"] == "missing_crop"

    def test_unsupported_crop_cassava_rejected_with_400(self, client):
        """Verify that selecting unsupported cassava returns HTTP 400 unsupported_crop."""
        img_buf = create_in_memory_image("JPEG", size=(64, 64))
        data = {
            "image": (img_buf, "leaf.jpg", "image/jpeg"),
            "crop": "cassava",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 400
        body = response.get_json()
        assert body["error"] == "unsupported_crop"
        assert "cassava" in body["message"]
        assert "tomato" in body["message"]

    def test_unsupported_crop_yam_rejected_with_400(self, client):
        """Verify that selecting unsupported yam returns HTTP 400 unsupported_crop."""
        img_buf = create_in_memory_image("JPEG", size=(64, 64))
        data = {
            "image": (img_buf, "leaf.jpg", "image/jpeg"),
            "crop": "yam",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 400
        body = response.get_json()
        assert body["error"] == "unsupported_crop"

    def test_missing_crop_yoruba_message(self, client):
        """Verify that missing crop error returns Yoruba message when lang='yo'."""
        img_buf = create_in_memory_image("JPEG", size=(64, 64))
        data = {
            "image": (img_buf, "leaf.jpg", "image/jpeg"),
            "lang": "yo",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 400
        body = response.get_json()
        assert body["error"] == "missing_crop"
        assert "A nílò orúkọ irúgbìn" in body["message"]

    def test_crop_mismatch_returns_200_safe_mismatch(self, client):
        """
        Verify that when the model detects symptoms belonging to a different crop family,
        the endpoint returns 200 with assessment_status: 'crop_mismatch', crop_match: False,
        and withholds disease treatment advice.
        """
        img_buf = create_in_memory_image("JPEG", size=(64, 64))
        # User claims leaf is potato, but we mock model predicting Corn Common Rust (maize)
        data = {
            "image": (img_buf, "leaf.jpg", "image/jpeg"),
            "crop": "potato",
            "lang": "en",
        }

        mock_prediction = {
            "assessment_status": "crop_mismatch",
            "model_name": "mock_model",
            "selected_crop": "potato",
            "predicted_crop": "maize",
            "crop_match": False,
            "class_index": 8,
            "class_label": "Corn (Maize) with Common Rust",
            "confidence": 0.95,
            "confidence_threshold": 0.40,
            "is_provisional_threshold": True,
            "disclaimer": "Crop mismatch detected.",
            "guidance": {
                "status": "crop_mismatch",
                "selected_crop": "potato",
                "predicted_crop": "maize",
                "model_class_id": "Corn (Maize) with Common Rust",
                "language": "en",
                "is_healthy": None,
                "message": "Crop mismatch detected: guidance withheld.",
                "data": None,
            },
        }

        with patch("app.routes.predict.get_prediction_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_svc.predict.return_value = mock_prediction
            mock_get_svc.return_value = mock_svc

            response = client.post(
                "/api/predict",
                data=data,
                content_type="multipart/form-data"
            )

            assert response.status_code == 200
            body = response.get_json()
            assert body["status"] == "success"
            assert body["selected_crop"] == "potato"
            pred = body["prediction"]
            assert pred["assessment_status"] == "crop_mismatch"
            assert pred["crop_match"] is False
            assert pred["guidance"]["is_healthy"] is None
            assert pred["guidance"]["data"] is None

    # 3. Model Failure Error Responses
    def test_model_loading_failure_returns_503(self, client):
        """Verify that a model loading failure returns HTTP 503 service unavailable."""
        img_buf = create_in_memory_image("JPEG", size=(64, 64))
        data = {
            "image": (img_buf, "leaf.jpg", "image/jpeg"),
            "crop": "tomato",
        }

        with patch("app.routes.predict.get_prediction_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_svc.predict.side_effect = ModelLoadError("Model weights corrupted")
            mock_get_svc.return_value = mock_svc

            response = client.post(
                "/api/predict",
                data=data,
                content_type="multipart/form-data"
            )

            assert response.status_code == 503
            body = response.get_json()
            assert body["error"] == "model_load_error"
            assert "corrupted" in body["message"]

    def test_inference_execution_failure_returns_500(self, client):
        """Verify that an inference forward pass failure returns HTTP 500 error."""
        img_buf = create_in_memory_image("JPEG", size=(64, 64))
        data = {
            "image": (img_buf, "leaf.jpg", "image/jpeg"),
            "crop": "tomato",
        }

        with patch("app.routes.predict.get_prediction_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_svc.predict.side_effect = InferenceExecutionError("Forward pass failed")
            mock_get_svc.return_value = mock_svc

            response = client.post(
                "/api/predict",
                data=data,
                content_type="multipart/form-data"
            )

            assert response.status_code == 500
            body = response.get_json()
            assert body["error"] == "inference_failed"

    # 4. Rejection of Missing or Empty Files
    def test_missing_image_field_returns_400(self, client):
        """Verify request with missing image field is rejected with 400."""
        response = client.post(
            "/api/predict",
            data={"crop": "tomato", "lang": "en"},
            content_type="multipart/form-data"
        )

        assert response.status_code == 400
        body = response.get_json()
        assert body["error"] == "missing_image"

    def test_empty_image_returns_400(self, client):
        """Verify 0-byte upload is rejected with 400 empty_image."""
        empty_buf = io.BytesIO(b"")
        data = {
            "image": (empty_buf, "empty.jpg", "image/jpeg"),
            "crop": "tomato",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 400
        body = response.get_json()
        assert body["error"] == "empty_image"

    # 5. Format and Decoding Security Rejections
    def test_unsupported_format_gif_rejected(self, client):
        """Verify GIF format is rejected with 400 unsupported_format."""
        gif_buf = create_in_memory_image("GIF", size=(50, 50))
        data = {
            "image": (gif_buf, "animated.gif", "image/gif"),
            "crop": "tomato",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 400
        body = response.get_json()
        assert body["error"] == "unsupported_format"
        assert "GIF" in body["message"]

    def test_unsupported_format_bmp_rejected(self, client):
        """Verify BMP format is rejected with 400 unsupported_format."""
        bmp_buf = create_in_memory_image("BMP", size=(50, 50))
        data = {
            "image": (bmp_buf, "bitmap.bmp", "image/bmp"),
            "crop": "tomato",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 400
        body = response.get_json()
        assert body["error"] == "unsupported_format"

    def test_spoofed_mime_type_rejected(self, client):
        """
        Verify that a text file disguised with a .jpg filename and image/jpeg
        MIME header is rejected by Pillow raster validation.
        """
        fake_jpg = io.BytesIO(b"This is a text document, not a real image file.")
        data = {
            "image": (fake_jpg, "disguised.jpg", "image/jpeg"),
            "crop": "tomato",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 400
        body = response.get_json()
        assert body["error"] in ("invalid_image", "corrupt_image")

    def test_corrupt_image_rejected(self, client):
        """Verify truncated image bytes trigger corrupt_image error."""
        corrupt_buf = io.BytesIO(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 15)
        data = {
            "image": (corrupt_buf, "corrupt.jpg", "image/jpeg"),
            "crop": "tomato",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 400
        body = response.get_json()
        assert body["error"] in ("corrupt_image", "invalid_image")

    # 6. Payload Size Limit (10 MB Ceiling)
    def test_oversized_upload_rejected_with_413(self, client):
        """Verify upload exceeding 10 MB is rejected with HTTP 413."""
        # 10 MB + 1 KB
        oversized_data = io.BytesIO(b"A" * (10 * 1024 * 1024 + 1024))
        data = {
            "image": (oversized_data, "huge.jpg", "image/jpeg"),
            "crop": "tomato",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 413
        body = response.get_json()
        assert body["error"] == "payload_too_large"

    # 7. Language Validation
    def test_invalid_language_code_rejected(self, client):
        """Verify unsupported language code returns 400 invalid_language."""
        img_buf = create_in_memory_image("JPEG")
        data = {
            "image": (img_buf, "leaf.jpg", "image/jpeg"),
            "crop": "tomato",
            "lang": "fr",  # French not supported in prototype
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 400
        body = response.get_json()
        assert body["error"] == "invalid_language"
        assert "en" in body["message"]
        assert "yo" in body["message"]

    # 8. HTTP Method Constraints
    def test_get_method_not_allowed(self, client):
        """Verify GET /api/predict returns 405 Method Not Allowed."""
        response = client.get("/api/predict")
        assert response.status_code == 405

