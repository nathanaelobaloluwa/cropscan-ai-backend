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
    """Automated tests for POST /api/predict image upload, validation & inference."""

    # 1. Valid Formats & Genuine Model Prediction
    def test_valid_jpeg_upload_returns_genuine_prediction(self, client):
        """Verify valid JPEG upload returns 200 with genuine prediction and metadata."""
        img_buf = create_in_memory_image("JPEG", size=(200, 150))
        data = {
            "image": (img_buf, "leaf_sample.jpg", "image/jpeg"),
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
        assert pred["assessment_status"] in ("confident", "inconclusive_low_confidence", "recognized_unsupported_crop")
        assert pred["guidance"] is not None

    def test_valid_png_upload(self, client):
        """Verify valid PNG upload returns 200, correct PNG metadata, and prediction."""
        img_buf = create_in_memory_image("PNG", size=(100, 100))
        data = {
            "image": (img_buf, "leaf.png", "image/png"),
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
        assert body["image_metadata"]["format"] == "PNG"
        assert "prediction" in body

    def test_valid_webp_upload(self, client):
        """Verify valid WebP upload returns 200, correct WebP metadata, and prediction."""
        img_buf = create_in_memory_image("WEBP", size=(80, 80))
        data = {
            "image": (img_buf, "leaf.webp", "image/webp"),
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
        assert body["image_metadata"]["format"] == "WEBP"
        assert "prediction" in body

    def test_yoruba_language_support(self, client):
        """Verify valid upload with lang='yo' returns Yoruba guidance."""
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

    def test_model_loading_failure_returns_503(self, client):
        """Verify that a model loading failure returns HTTP 503 service unavailable."""
        img_buf = create_in_memory_image("JPEG", size=(64, 64))
        data = {"image": (img_buf, "leaf.jpg", "image/jpeg")}

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
        data = {"image": (img_buf, "leaf.jpg", "image/jpeg")}

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


    # 2. Rejection of Missing or Empty Files
    def test_missing_image_field_returns_400(self, client):
        """Verify request with missing image field is rejected with 400."""
        response = client.post(
            "/api/predict",
            data={"lang": "en"},
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
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 400
        body = response.get_json()
        assert body["error"] == "empty_image"

    # 3. Format and Decoding Security Rejections
    def test_unsupported_format_gif_rejected(self, client):
        """Verify GIF format is rejected with 400 unsupported_format."""
        gif_buf = create_in_memory_image("GIF", size=(50, 50))
        data = {
            "image": (gif_buf, "animated.gif", "image/gif"),
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
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 400
        body = response.get_json()
        assert body["error"] in ("corrupt_image", "invalid_image")

    # 4. Payload Size Limit (10 MB Ceiling)
    def test_oversized_upload_rejected_with_413(self, client):
        """Verify upload exceeding 10 MB is rejected with HTTP 413."""
        # 10 MB + 1 KB
        oversized_data = io.BytesIO(b"A" * (10 * 1024 * 1024 + 1024))
        data = {
            "image": (oversized_data, "huge.jpg", "image/jpeg"),
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 413
        body = response.get_json()
        assert body["error"] == "payload_too_large"

    # 5. Language Validation
    def test_invalid_language_code_rejected(self, client):
        """Verify unsupported language code returns 400 invalid_language."""
        img_buf = create_in_memory_image("JPEG")
        data = {
            "image": (img_buf, "leaf.jpg", "image/jpeg"),
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

    # 6. HTTP Method Constraints
    def test_get_method_not_allowed(self, client):
        """Verify GET /api/predict returns 405 Method Not Allowed."""
        response = client.get("/api/predict")
        assert response.status_code == 405
