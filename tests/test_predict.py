import io
import pytest
from PIL import Image


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
    """Automated tests for POST /api/predict image upload & validation."""

    # 1. Valid Formats & Model-Not-Configured Response
    def test_valid_jpeg_upload_returns_model_not_configured(self, client):
        """Verify valid JPEG upload returns 200 with model_not_configured and metadata."""
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
        assert body["status"] == "model_not_configured"
        assert body["language"] == "en"
        assert "not configured yet" in body["message"]

        metadata = body["image_metadata"]
        assert metadata["format"] == "JPEG"
        assert metadata["width"] == 200
        assert metadata["height"] == 150
        assert metadata["color_mode"] == "RGB"
        assert metadata["size_bytes"] > 0

        # Verify no fabricated predictions exist
        assert "prediction" not in body
        assert "class_index" not in body
        assert "confidence" not in body

    def test_valid_png_upload(self, client):
        """Verify valid PNG upload returns 200 and correct PNG metadata."""
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
        assert body["status"] == "model_not_configured"
        assert body["image_metadata"]["format"] == "PNG"

    def test_valid_webp_upload(self, client):
        """Verify valid WebP upload returns 200 and correct WebP metadata."""
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
        assert body["status"] == "model_not_configured"
        assert body["image_metadata"]["format"] == "WEBP"

    def test_yoruba_language_support(self, client):
        """Verify valid upload with lang='yo' returns Yoruba status message."""
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
        assert body["status"] == "model_not_configured"
        assert body["language"] == "yo"
        assert "Àwòrán náà" in body["message"]

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
