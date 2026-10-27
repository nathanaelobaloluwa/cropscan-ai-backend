"""
CropScan AI - Image Quality Screening Test Suite.

Automated unit and integration tests covering:
- Grayscale luminance and contrast measurement.
- Sharpness variance measurement with border artifact elimination.
- Green and vegetation-consistent color distribution (foliage, chlorotic, necrotic).
- Dimension robustness (small, medium, large within limits).
- Screening decisions at threshold boundaries.
- Pipeline integration: model inference prevention, guidance suppression, and safe error handling.
- API endpoint integration via POST /api/predict.
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import pytest
import torch

from app import create_app
from app.services.image_screener import (
    DEFAULT_MAX_LUMINANCE,
    DEFAULT_MIN_CONTRAST,
    DEFAULT_MIN_LUMINANCE,
    ScreeningResult,
    measure_color_distribution,
    measure_luminance_and_contrast,
    measure_sharpness,
    screen_image_quality,
)
from app.services.predictor import PredictionService, create_image_processor


# ---------------------------------------------------------------------------
# Synthetic Image Generators (Deterministic, Clean Provenance)
# ---------------------------------------------------------------------------

def _make_solid_image(size=(100, 100), color=(34, 139, 34)) -> Image.Image:
    """Creates a deterministic solid color image."""
    return Image.new("RGB", size, color)


def _make_gradient_image(size=(100, 100)) -> Image.Image:
    """Creates a horizontal linear grayscale gradient image."""
    w, h = size
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return Image.fromarray(arr, mode="L").convert("RGB")


def _make_high_contrast_pattern(size=(100, 100), grid_size=10) -> Image.Image:
    """Creates a sharp black-and-white checkerboard pattern."""
    w, h = size
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            if ((x // grid_size) + (y // grid_size)) % 2 == 0:
                draw.rectangle([x, y, x + grid_size - 1, y + grid_size - 1], fill=0)
    return img.convert("RGB")


def _make_in_memory_file(image: Image.Image, format_name="JPEG") -> io.BytesIO:
    """Encodes a PIL Image into an in-memory BytesIO buffer."""
    buf = io.BytesIO()
    image.save(buf, format=format_name)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Unit Tests: Luminance and Contrast
# ---------------------------------------------------------------------------

class TestLuminanceAndContrast:
    """Unit tests for grayscale luminance and standard-deviation contrast."""

    def test_pure_black_image(self):
        img = _make_solid_image(color=(0, 0, 0))
        mean_lum, contrast = measure_luminance_and_contrast(img)
        assert mean_lum == 0.0
        assert contrast == 0.0

    def test_pure_white_image(self):
        img = _make_solid_image(color=(255, 255, 255))
        mean_lum, contrast = measure_luminance_and_contrast(img)
        assert mean_lum == 255.0
        assert contrast == 0.0

    def test_neutral_gray_image(self):
        img = _make_solid_image(color=(128, 128, 128))
        mean_lum, contrast = measure_luminance_and_contrast(img)
        assert mean_lum == 128.0
        assert contrast == 0.0

    def test_gradient_contrast(self):
        img = _make_gradient_image(size=(100, 100))
        mean_lum, contrast = measure_luminance_and_contrast(img)
        assert 120.0 <= mean_lum <= 135.0
        assert contrast > 50.0  # High standard deviation across gradient


# ---------------------------------------------------------------------------
# Unit Tests: Sharpness & Blur
# ---------------------------------------------------------------------------

class TestSharpnessMeasurement:
    """Unit tests for interior Laplacian variance sharpness measurement."""

    def test_uniform_surface_has_zero_sharpness(self):
        """Uniform color must produce zero interior Laplacian variance (no edge padding artifacts)."""
        img = _make_solid_image(color=(100, 150, 80))
        sharpness = measure_sharpness(img)
        assert sharpness == 0.0

    def test_sharp_pattern_vs_blurred_pattern(self):
        """A high-contrast pattern must score higher than its blurred counterpart."""
        sharp_img = _make_high_contrast_pattern(size=(120, 120), grid_size=8)
        blurred_img = sharp_img.filter(ImageFilter.GaussianBlur(radius=4))

        sharp_score = measure_sharpness(sharp_img)
        blurred_score = measure_sharpness(blurred_img)

        assert sharp_score > 50.0
        assert blurred_score < sharp_score
        assert blurred_score < 20.0


# ---------------------------------------------------------------------------
# Unit Tests: Color Distribution
# ---------------------------------------------------------------------------

class TestColorDistribution:
    """Unit tests for green foliage, chlorotic yellow, and necrotic brown detection."""

    def test_green_foliage(self):
        """Forest green leaf color must be detected as both green and vegetation."""
        img = _make_solid_image(color=(34, 139, 34))
        green_ratio, veg_ratio = measure_color_distribution(img)
        assert green_ratio == 1.0
        assert veg_ratio == 1.0

    def test_chlorotic_yellow_leaf_tissue(self):
        """Chlorotic / yellow diseased tissue is not green but is recognized as vegetation."""
        img = _make_solid_image(color=(190, 190, 40))
        green_ratio, veg_ratio = measure_color_distribution(img)
        assert green_ratio == 0.0  # R == G, not G > R
        assert veg_ratio == 1.0    # Recognized as vegetation-consistent

    def test_necrotic_brown_blight_tissue(self):
        """Necrotic / brown leaf tissue is not green but is recognized as vegetation."""
        img = _make_solid_image(color=(130, 80, 30))
        green_ratio, veg_ratio = measure_color_distribution(img)
        assert green_ratio == 0.0  # R > G > B
        assert veg_ratio == 1.0    # Recognized as vegetation-consistent

    def test_blue_sky_non_vegetation(self):
        """Blue sky / non-leaf color produces zero vegetation ratio."""
        img = _make_solid_image(color=(60, 120, 230))
        green_ratio, veg_ratio = measure_color_distribution(img)
        assert green_ratio == 0.0
        assert veg_ratio == 0.0

    def test_red_object_non_vegetation(self):
        """Red object / non-leaf color produces zero vegetation ratio."""
        img = _make_solid_image(color=(220, 30, 30))
        green_ratio, veg_ratio = measure_color_distribution(img)
        assert green_ratio == 0.0
        assert veg_ratio == 0.0


# ---------------------------------------------------------------------------
# Unit Tests: Dimensions Robustness
# ---------------------------------------------------------------------------

class TestImageDimensionsRobustness:
    """Unit tests across small, medium, and large supported image dimensions."""

    def test_minimum_supported_dimensions(self):
        """10x10 minimum dimension supported by validator."""
        img = _make_solid_image(size=(10, 10), color=(46, 125, 50))
        res = screen_image_quality(img)
        assert res.is_acceptable is True
        assert res.metrics["mean_luminance"] > 0

    def test_medium_dimensions(self):
        """256x256 typical web/mobile resolution."""
        img = _make_solid_image(size=(256, 256), color=(46, 125, 50))
        res = screen_image_quality(img)
        assert res.is_acceptable is True

    def test_large_dimensions_capped_for_fast_analysis(self):
        """1200x1200 large uploaded photo."""
        img = _make_solid_image(size=(1200, 1200), color=(46, 125, 50))
        res = screen_image_quality(img)
        assert res.is_acceptable is True


# ---------------------------------------------------------------------------
# Unit Tests: Screening Threshold Boundaries & Decisions
# ---------------------------------------------------------------------------

class TestScreeningDecisionsAndBoundaries:
    """Unit tests for threshold boundaries and conservative rejection criteria."""

    def test_darkness_threshold_boundary_rejected(self):
        """Mean luminance 5.0 is below min_luminance 8.0 -> rejected as image_too_dark."""
        img = _make_solid_image(color=(5, 5, 5))
        res = screen_image_quality(img, min_luminance=DEFAULT_MIN_LUMINANCE)
        assert res.is_acceptable is False
        assert res.rejection_code == "image_too_dark"
        assert "extremely dark" in res.message.lower()

    def test_darkness_threshold_boundary_accepted(self):
        """Mean luminance 12.0 is above min_luminance 8.0 -> passes darkness check."""
        # Mix dark pixels with some illumination so mean > 8
        img = _make_solid_image(color=(12, 25, 12))
        res = screen_image_quality(img, min_luminance=DEFAULT_MIN_LUMINANCE)
        assert res.metrics["mean_luminance"] >= DEFAULT_MIN_LUMINANCE
        assert res.rejection_code != "image_too_dark"

    def test_overexposure_threshold_rejected(self):
        """Pure white image (255) with zero contrast -> rejected as image_too_bright."""
        img = _make_solid_image(color=(255, 255, 255))
        res = screen_image_quality(img, max_luminance=DEFAULT_MAX_LUMINANCE, min_contrast=DEFAULT_MIN_CONTRAST)
        assert res.is_acceptable is False
        assert res.rejection_code == "image_too_bright"
        assert "overexposed" in res.message.lower()

    def test_uniform_gray_canvas_rejected(self):
        """Flat uniform gray surface (zero contrast, non-plant) -> rejected as image_lacks_features."""
        img = _make_solid_image(color=(120, 120, 120))
        res = screen_image_quality(img, min_contrast=DEFAULT_MIN_CONTRAST)
        assert res.is_acceptable is False
        assert res.rejection_code == "image_lacks_features"
        assert "flat, uniform" in res.message.lower()

    def test_uniform_green_test_image_accepted(self):
        """
        A synthetic solid green leaf image has low contrast but high vegetation ratio.
        It must pass conservative screening so unit-test fixtures are not falsely rejected.
        """
        img = _make_solid_image(color=(34, 139, 34))
        res = screen_image_quality(img)
        assert res.is_acceptable is True
        assert res.rejection_code is None

    def test_sharpness_evaluative_threshold(self):
        """When min_sharpness is configured, blurred images are rejected as image_too_blurry."""
        sharp_img = _make_high_contrast_pattern(size=(120, 120), grid_size=8)
        blurred_img = sharp_img.filter(ImageFilter.GaussianBlur(radius=5))

        # With min_sharpness=None (default), blur is not rejected
        res_default = screen_image_quality(blurred_img, min_sharpness=None)
        assert res_default.is_acceptable is True

        # With min_sharpness=20.0, blurred image is rejected
        res_strict = screen_image_quality(blurred_img, min_sharpness=20.0)
        assert res_strict.is_acceptable is False
        assert res_strict.rejection_code == "image_too_blurry"
        assert "blurred" in res_strict.message.lower()

        # Sharp image passes the threshold
        res_sharp = screen_image_quality(sharp_img, min_sharpness=20.0)
        assert res_sharp.is_acceptable is True


# ---------------------------------------------------------------------------
# Integration Tests: Prediction Pipeline & Safe Degradation
# ---------------------------------------------------------------------------

class TestPredictorScreeningPipeline:
    """Integration tests verifying predictor behaves safely on screened images."""

    @pytest.fixture
    def mock_service(self):
        """PredictionService with mocked model and processor."""
        service = PredictionService(lazy_load=True, confidence_threshold=0.40)
        service._processor = create_image_processor()
        service._model = MagicMock()
        return service

    def test_dark_image_bypasses_model_inference(self, mock_service):
        """An extremely dark image must bypass model forward pass and return safe fallback."""
        black_img = _make_solid_image(color=(2, 2, 2))
        result = mock_service.predict(black_img, selected_crop="tomato", lang="en")

        # Verify model was NEVER invoked
        mock_service._model.assert_not_called()

        assert result["assessment_status"] == "unreliable_image_quality"
        assert result["crop_match"] is None
        assert result["class_index"] is None
        assert result["confidence"] is None
        assert result["screening_details"]["is_acceptable"] is False
        assert result["screening_details"]["rejection_code"] == "image_too_dark"

        # Verify treatment guidance is withheld and plant is not marked healthy
        guidance = result["guidance"]
        assert guidance["status"] == "unreliable_assessment"
        assert guidance["is_healthy"] is None
        assert guidance["data"] is None

    def test_overexposed_image_bypasses_model_inference(self, mock_service):
        """An overexposed image must bypass model forward pass."""
        white_img = _make_solid_image(color=(255, 255, 255))
        result = mock_service.predict(white_img, selected_crop="tomato", lang="en")

        mock_service._model.assert_not_called()
        assert result["assessment_status"] == "unreliable_image_quality"
        assert result["screening_details"]["rejection_code"] == "image_too_bright"
        assert result["guidance"]["data"] is None

    def test_featureless_canvas_bypasses_model_inference(self, mock_service):
        """A uniform gray canvas must bypass model forward pass."""
        gray_img = _make_solid_image(color=(128, 128, 128))
        result = mock_service.predict(gray_img, selected_crop="tomato", lang="en")

        mock_service._model.assert_not_called()
        assert result["assessment_status"] == "unreliable_image_quality"
        assert result["screening_details"]["rejection_code"] == "image_lacks_features"
        assert result["guidance"]["data"] is None

    def test_screening_internal_exception_fails_safely(self, mock_service):
        """If image screener encounters an unexpected error, fail safely without diagnosing."""
        normal_img = _make_solid_image(color=(34, 139, 34))

        with patch("app.services.predictor.screen_image_quality", side_effect=RuntimeError("Corrupt buffer")):
            result = mock_service.predict(normal_img, selected_crop="tomato", lang="en")

        mock_service._model.assert_not_called()
        assert result["assessment_status"] == "unreliable_image_quality"
        assert result["screening_details"]["rejection_code"] == "screening_error"
        assert result["guidance"]["status"] == "unreliable_assessment"
        assert result["guidance"]["data"] is None
        assert result["guidance"]["is_healthy"] is None

    def test_acceptable_image_proceeds_to_model_inference(self, mock_service):
        """An acceptable image proceeds to model inference and attaches screening details."""
        mock_output = MagicMock()
        mock_logits = torch.zeros((1, 38), dtype=torch.float32)
        mock_logits[0, 29] = 10.0  # Tomato early blight
        mock_output.logits = mock_logits
        mock_service._model.return_value = mock_output

        normal_img = _make_solid_image(color=(34, 139, 34))
        result = mock_service.predict(normal_img, selected_crop="tomato", lang="en")

        assert mock_service._model.called
        assert result["assessment_status"] == "confident"
        assert result["screening_details"]["is_acceptable"] is True
        assert result["screening_details"]["rejection_code"] is None
        assert "mean_luminance" in result["screening_details"]["metrics"]


# ---------------------------------------------------------------------------
# Integration Tests: HTTP API Endpoint (POST /api/predict)
# ---------------------------------------------------------------------------

class TestPredictApiScreeningEndpoint:
    """End-to-end HTTP tests verifying /api/predict handles screened uploads safely."""

    @pytest.fixture
    def client(self):
        app = create_app("testing")
        with app.test_client() as client:
            yield client

    def test_pitch_black_upload_returns_200_with_screening_rejection(self, client):
        """Pitch-black image returns 200 with assessment_status: 'unreliable_image_quality'."""
        black_img = _make_solid_image(size=(150, 150), color=(0, 0, 0))
        img_buf = _make_in_memory_file(black_img, "JPEG")

        data = {
            "image": (img_buf, "pitch_black.jpg", "image/jpeg"),
            "crop": "tomato",
            "lang": "en",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        assert response.is_json
        body = response.get_json()

        assert body["status"] == "success"
        prediction = body["prediction"]
        assert prediction["assessment_status"] == "unreliable_image_quality"
        assert prediction["crop_match"] is None
        assert prediction["confidence"] is None
        assert prediction["screening_details"]["is_acceptable"] is False
        assert prediction["screening_details"]["rejection_code"] == "image_too_dark"
        assert prediction["guidance"]["data"] is None
        assert prediction["guidance"]["is_healthy"] is None

    def test_overexposed_upload_returns_screening_rejection(self, client):
        """Pure white image returns 200 with image_too_bright."""
        white_img = _make_solid_image(size=(150, 150), color=(255, 255, 255))
        img_buf = _make_in_memory_file(white_img, "JPEG")

        data = {
            "image": (img_buf, "blown_out.jpg", "image/jpeg"),
            "crop": "tomato",
            "lang": "en",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        body = response.get_json()
        prediction = body["prediction"]
        assert prediction["assessment_status"] == "unreliable_image_quality"
        assert prediction["screening_details"]["rejection_code"] == "image_too_bright"
        assert prediction["guidance"]["data"] is None

    def test_flat_gray_canvas_returns_screening_rejection(self, client):
        """Uniform gray canvas returns 200 with image_lacks_features."""
        gray_img = _make_solid_image(size=(150, 150), color=(128, 128, 128))
        img_buf = _make_in_memory_file(gray_img, "JPEG")

        data = {
            "image": (img_buf, "gray_box.jpg", "image/jpeg"),
            "crop": "tomato",
            "lang": "en",
        }

        response = client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        body = response.get_json()
        prediction = body["prediction"]
        assert prediction["assessment_status"] == "unreliable_image_quality"
        assert prediction["screening_details"]["rejection_code"] == "image_lacks_features"
        assert prediction["guidance"]["data"] is None
