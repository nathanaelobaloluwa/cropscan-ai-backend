import json
from unittest.mock import MagicMock, patch
import pytest
import torch
from PIL import Image

from app.services.guidance import GuidanceService
from app.services.predictor import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_NAME,
    InferenceExecutionError,
    ModelLoadError,
    PredictionService,
    create_image_processor,
)


@pytest.fixture(scope="module")
def real_predictor():
    """Loads the genuine MobileNetV2 vision model once for testing."""
    service = PredictionService(lazy_load=False)
    return service


class TestPredictionServiceRealInference:
    """
    Automated tests verifying genuine inference using the real MobileNetV2 weights.
    These tests execute actual PyTorch forward passes without mocks.
    """

    def test_preprocessor_parameters(self):
        """Verify image processor reproduces exact documented preprocessing configuration."""
        processor = create_image_processor()
        assert processor.do_resize is True
        assert processor.size == {"shortest_edge": 256}
        assert processor.do_center_crop is True
        assert processor.crop_size == {"height": 224, "width": 224}
        assert processor.do_rescale is True
        assert pytest.approx(processor.rescale_factor, 1e-6) == 1 / 255.0
        assert processor.do_normalize is True
        assert list(processor.image_mean) == [0.5, 0.5, 0.5]
        assert list(processor.image_std) == [0.5, 0.5, 0.5]


    def test_real_model_loading_and_attributes(self, real_predictor):
        """Verify the model loads the real weights on CPU by default."""
        assert real_predictor.is_model_loaded is True
        assert real_predictor.device == "cpu"
        assert real_predictor.model_name == DEFAULT_MODEL_NAME
        assert real_predictor.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD

    def test_genuine_inference_on_sample_image(self, real_predictor):
        """
        Execute genuine inference with the real model on an in-memory sample image.
        Verifies genuine class index, score, and non-fabricated assessment output.
        """
        sample_img = Image.new("RGB", (256, 256), color=(46, 125, 50))
        result = real_predictor.predict(sample_img, lang="en")

        # 1. Genuine class index within 0-37
        assert isinstance(result["class_index"], int)
        assert 0 <= result["class_index"] <= 37

        # 2. Real class label from official mapping
        assert isinstance(result["class_label"], str)
        assert len(result["class_label"]) > 0

        # 3. Genuine model confidence score between 0.0 and 1.0
        assert isinstance(result["confidence"], float)
        assert 0.0 < result["confidence"] <= 1.0

        # 4. Assessment status and safety disclaimers
        assert result["assessment_status"] in (
            "confident",
            "inconclusive_low_confidence",
            "recognized_unsupported_crop",
        )
        assert "not a confirmed" in result["disclaimer"].lower() or "inconclusive" in result["disclaimer"].lower()
        assert result["guidance"] is not None

    def test_all_38_class_indices_match_model_config(self, real_predictor):
        """Confirm that local class_indices.json matches the real model's id2label configuration."""
        model_id2label = real_predictor._model.config.id2label
        guidance = real_predictor.guidance_service

        assert len(model_id2label) == 38
        assert guidance.total_model_classes == 38

        for idx_str, expected_label in model_id2label.items():
            idx = int(idx_str)
            resolved_label = guidance.get_class_label(idx)
            assert resolved_label == expected_label, f"Mismatch at class index {idx}"


class TestPredictionServiceUnitMocks:
    """
    Unit tests covering threshold boundary behavior, error paths, and fallbacks.
    Uses controlled mocked forward passes to test edge cases deterministically.
    Clearly distinguished from genuine inference tests.
    """

    @pytest.fixture
    def mock_predictor(self):
        """PredictionService with a mocked model and processor."""
        service = PredictionService(lazy_load=True, confidence_threshold=0.40)
        service._processor = create_image_processor()
        service._model = MagicMock()
        return service

    def _create_mock_logits(self, top_index: int, top_score: float, num_classes: int = 38):
        """Helper to create synthetic logits that produce exact softmax scores."""
        # Start with all zeros
        logits = torch.zeros((1, num_classes), dtype=torch.float32)
        # We calculate logits such that softmax produces top_score
        # Using a simple trick: log(top_score / remainder)
        remainder = (1.0 - top_score) / (num_classes - 1)
        logits[0, :] = torch.log(torch.tensor(remainder))
        logits[0, top_index] = torch.log(torch.tensor(top_score))
        return logits

    def test_model_loading_failure_raises_model_load_error(self):
        """Verify that a nonexistent or corrupt model path raises ModelLoadError."""
        service = PredictionService(model_name="nonexistent/fake-model-repo-cropscan-test")
        with pytest.raises(ModelLoadError) as exc_info:
            service.load_model()
        assert "Failed to load vision model" in str(exc_info.value)
        assert exc_info.value.status_code == 503

    def test_confidence_threshold_boundary_below(self, mock_predictor):
        """
        Boundary test: confidence = 0.39 < threshold = 0.40.
        Must return assessment_status: 'inconclusive_low_confidence' and unreliable fallback.
        """
        mock_output = MagicMock()
        mock_output.logits = self._create_mock_logits(top_index=29, top_score=0.39)
        mock_predictor._model.return_value = mock_output

        img = Image.new("RGB", (100, 100), (34, 139, 34))
        result = mock_predictor.predict(img, lang="en")

        assert result["assessment_status"] == "inconclusive_low_confidence"
        assert result["class_index"] == 29
        assert result["class_label"] == "Tomato with Early Blight"
        assert result["confidence"] == 0.39
        assert result["confidence_threshold"] == 0.40
        assert "below the provisional operational threshold" in result["disclaimer"]
        assert result["guidance"]["status"] == "unreliable_assessment"

    def test_confidence_threshold_boundary_at(self, mock_predictor):
        """
        Boundary test: confidence = 0.40 == threshold = 0.40.
        Must be accepted as confident.
        """
        mock_output = MagicMock()
        mock_output.logits = self._create_mock_logits(top_index=29, top_score=0.40)
        mock_predictor._model.return_value = mock_output

        img = Image.new("RGB", (100, 100), (34, 139, 34))
        result = mock_predictor.predict(img, lang="en")

        assert result["assessment_status"] == "confident"
        assert result["class_index"] == 29
        assert result["confidence"] == 0.40
        assert result["guidance"]["status"] == "success"

    def test_confidence_threshold_boundary_above(self, mock_predictor):
        """
        Boundary test: confidence = 0.88 > threshold = 0.40.
        Must return confident with localized guidance.
        """
        mock_output = MagicMock()
        mock_output.logits = self._create_mock_logits(top_index=29, top_score=0.88)
        mock_predictor._model.return_value = mock_output

        img = Image.new("RGB", (100, 100), (34, 139, 34))
        result = mock_predictor.predict(img, lang="en")

        assert result["assessment_status"] == "confident"
        assert result["confidence"] == 0.88
        assert result["guidance"]["status"] == "success"
        assert result["guidance"]["data"]["disease_id"] == "tomato_early_blight"

    def test_guidance_retrieval_yoruba(self, mock_predictor):
        """Verify retrieval of draft Yoruba guidance for confident predictions."""
        mock_output = MagicMock()
        mock_output.logits = self._create_mock_logits(top_index=29, top_score=0.75)
        mock_predictor._model.return_value = mock_output

        img = Image.new("RGB", (100, 100), (34, 139, 34))
        result = mock_predictor.predict(img, lang="yo")

        assert result["assessment_status"] == "confident"
        assert result["guidance"]["language"] == "yo"
        assert result["guidance"]["data"]["translation_review_status"] == "draft_translation_pending_native_expert_review"

    def test_recognized_unsupported_crop(self, mock_predictor):
        """
        Verify that a class recognized by the model but outside prototype scope
        (e.g., Apple Scab, index 0) returns recognized_unsupported_crop.
        """
        mock_output = MagicMock()
        mock_output.logits = self._create_mock_logits(top_index=0, top_score=0.82)
        mock_predictor._model.return_value = mock_output

        img = Image.new("RGB", (100, 100), (34, 139, 34))
        result = mock_predictor.predict(img, lang="en")

        assert result["assessment_status"] == "recognized_unsupported_crop"
        assert result["class_index"] == 0
        assert result["class_label"] == "Apple Scab"
        assert result["guidance"]["is_healthy"] is False

    def test_invalid_model_output_index(self, mock_predictor):
        """Verify handling if model produces an index outside 0-37."""
        mock_output = MagicMock()
        # Mock logits with 50 classes where index 45 is top
        mock_output.logits = self._create_mock_logits(top_index=45, top_score=0.90, num_classes=50)
        mock_predictor._model.return_value = mock_output

        img = Image.new("RGB", (100, 100), (34, 139, 34))
        result = mock_predictor.predict(img, lang="en")

        assert result["assessment_status"] == "invalid_model_output"
        assert result["class_index"] == 45
        assert result["class_label"] is None
        assert result["guidance"]["status"] == "unreliable_assessment"

    def test_crop_mismatch_withholds_disease_guidance(self, mock_predictor):
        """
        Verify that a crop mismatch (e.g. user selected tomato, but model predicted apple)
        returns crop_mismatch, withholds treatment guidance, and never claims healthy.
        """
        mock_output = MagicMock()
        # Class index 2 is Cedar Apple Rust (crop: apple)
        mock_output.logits = self._create_mock_logits(top_index=2, top_score=0.85)
        mock_predictor._model.return_value = mock_output

        img = Image.new("RGB", (100, 100), (34, 139, 34))
        result = mock_predictor.predict(img, selected_crop="tomato", lang="en")

        assert result["assessment_status"] == "crop_mismatch"
        assert result["crop_match"] is False
        assert result["selected_crop"] == "tomato"
        assert result["predicted_crop"] == "apple"
        assert result["class_index"] == 2
        assert result["class_label"] == "Cedar Apple Rust"
        assert result["confidence"] == 0.85

        guidance = result["guidance"]
        assert guidance["status"] == "crop_mismatch"
        assert guidance["is_healthy"] is None
        assert guidance["data"] is None
        assert "withheld" in guidance["message"].lower()
        assert "does not prove" in result["disclaimer"].lower()

    def test_crop_mismatch_yoruba(self, mock_predictor):
        """Verify that crop mismatch returns localized Yoruba explanation."""
        mock_output = MagicMock()
        # Class index 8 is Corn Common Rust (crop: maize)
        mock_output.logits = self._create_mock_logits(top_index=8, top_score=0.92)
        mock_predictor._model.return_value = mock_output

        img = Image.new("RGB", (100, 100), (34, 139, 34))
        result = mock_predictor.predict(img, selected_crop="tomato", lang="yo")

        assert result["assessment_status"] == "crop_mismatch"
        assert result["crop_match"] is False
        assert result["selected_crop"] == "tomato"
        assert result["predicted_crop"] == "maize"
        assert result["guidance"]["status"] == "crop_mismatch"
        assert result["guidance"]["is_healthy"] is None
        assert result["guidance"]["data"] is None
        assert "Àìbáramu irúgbìn wáyé" in result["guidance"]["message"]

    def test_crop_match_confident_returns_treatment_guidance(self, mock_predictor):
        """Verify matching crop selection with high confidence returns confident status and treatment."""
        mock_output = MagicMock()
        mock_output.logits = self._create_mock_logits(top_index=29, top_score=0.88)
        mock_predictor._model.return_value = mock_output

        img = Image.new("RGB", (100, 100), (34, 139, 34))
        result = mock_predictor.predict(img, selected_crop="tomato", lang="en")

        assert result["assessment_status"] == "confident"
        assert result["crop_match"] is True
        assert result["selected_crop"] == "tomato"
        assert result["predicted_crop"] == "tomato"
        assert result["guidance"]["status"] == "success"
        assert result["guidance"]["data"]["disease_id"] == "tomato_early_blight"

    def test_crop_match_low_confidence(self, mock_predictor):
        """Verify matching crop selection below confidence threshold returns inconclusive_low_confidence."""
        mock_output = MagicMock()
        mock_output.logits = self._create_mock_logits(top_index=29, top_score=0.35)
        mock_predictor._model.return_value = mock_output

        img = Image.new("RGB", (100, 100), (34, 139, 34))
        result = mock_predictor.predict(img, selected_crop="tomato", lang="en")

        assert result["assessment_status"] == "inconclusive_low_confidence"
        assert result["crop_match"] is True
        assert result["selected_crop"] == "tomato"
        assert result["confidence"] == 0.35
        assert result["guidance"]["status"] == "unreliable_assessment"

    def test_inference_execution_exception_handling(self, mock_predictor):
        """Verify that forward pass failures raise InferenceExecutionError."""
        mock_predictor._model.side_effect = RuntimeError("CUDA/CPU tensor memory error")

        img = Image.new("RGB", (100, 100), (34, 139, 34))
        with pytest.raises(InferenceExecutionError) as exc_info:
            mock_predictor.predict(img, selected_crop="tomato", lang="en")

        assert "Inference execution failed" in str(exc_info.value)
        assert exc_info.value.status_code == 500

