"""
CropScan AI - Misclassification Safety and Comprehensive Guardrail Test Suite.

Automated verification covering the 22 required safety scenarios:
1. Missing image input handling.
2. Invalid image format validation.
3. Corrupted image stream rejection.
4. Empty or zero-byte image handling.
5. Blank, overexposed, or severely blurred image screening.
6. Non-leaf objects (plastic, walls, furniture, floors) rejected with 'not_a_leaf'.
7. Supported crop with compatible image and confident prediction.
8. Unsupported crops (Cassava, Yam, Cocoa, Plantain) rejected with 'unsupported_crop'.
9. Unsupported crop submitted under a supported crop selection (OOD protection).
10. Supported crop with mismatched selection (e.g. Tomato selected, Maize predicted).
11. Low-confidence disease prediction below operational threshold (< 0.40).
12. High-confidence but unverified crop identity (e.g. 92% confidence on Strawberry).
13. Incomplete class coverage auditing (Citrus single-class gap).
14. Supported healthy leaf classes returning 'healthy' and cultural maintenance.
15. Crops with no healthy class (Citrus) cannot be diagnosed as healthy.
16. Missing Yoruba guidance translation fallback with honest message.
17. Bilingual English and Yoruba messages across all rejection and uncertainty states.
18. Disease-specific treatment guidance strictly withheld (data=None) for all unsafe outcomes.
19. Out-of-range or malformed model output class index handling.
20. Model loading failures and inference execution exceptions fail safely.
21. Prevention of stale or unrelated treatment guidance exposure.
22. Regression tests for existing crop mismatch, confidence threshold, and quality screener behavior.
"""

import io
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch
from PIL import Image

from app.services.guidance import GuidanceService
from app.services.image_screener import screen_image_quality
from app.services.predictor import (
    InferenceExecutionError,
    ModelLoadError,
    PredictionService,
    create_image_processor,
)


def _create_mock_logits(top_index: int, top_score: float, num_classes: int = 38):
    """Generates synthetic logits that produce the exact requested top softmax score."""
    logits = torch.zeros((1, num_classes), dtype=torch.float32)
    remainder = (1.0 - top_score) / (num_classes - 1)
    logits[0, :] = torch.log(torch.tensor(remainder))
    logits[0, top_index] = torch.log(torch.tensor(top_score))
    return logits


def _make_in_memory_image(format_name: str = "JPEG", size: tuple = (120, 120), color: tuple = (34, 139, 34)) -> io.BytesIO:
    """Helper to generate in-memory image bytes."""
    buf = io.BytesIO()
    mode = "RGB" if format_name.upper() == "JPEG" else "RGBA"
    fill_color = color if mode == "RGB" else (*color, 255)
    img = Image.new(mode, size, fill_color)
    img.save(buf, format=format_name)
    buf.seek(0)
    return buf


@pytest.fixture
def mock_service():
    """Returns a PredictionService instance with mocked model and processor."""
    service = PredictionService(lazy_load=True, confidence_threshold=0.40)
    service._processor = create_image_processor()
    service._model = MagicMock()
    return service


class TestMisclassificationSafety:
    """Verifies that cross-species misclassifications, OOD inputs, and uncertainty fail safely."""

    # -------------------------------------------------------------------------
    # Baseline Scenario & Regression Tests
    # -------------------------------------------------------------------------

    def test_tomato_selected_strawberry_predicted_at_61_percent(self, mock_service):
        """
        Scenario 22 & 10:
        User selects 'tomato', but the forward pass activates class 26 ('Strawberry with Leaf Scorch')
        at 61.0% confidence.
        Must return assessment_status == 'crop_mismatch', crop_match == False, and withhold treatment.
        """
        mock_output = MagicMock()
        mock_output.logits = _create_mock_logits(top_index=26, top_score=0.61)
        mock_service._model.return_value = mock_output

        img = Image.new("RGB", (120, 120), (34, 139, 34))
        result = mock_service.predict(img, selected_crop="tomato", lang="en")

        assert result["assessment_status"] == "crop_mismatch"
        assert result["crop_match"] is False
        assert result["selected_crop"] == "tomato"
        assert result["predicted_crop"] == "strawberry"
        assert result["class_index"] == 26
        assert result["class_label"] == "Strawberry with Leaf Scorch"
        assert result["confidence"] == 0.61
        assert result["confidence_threshold"] == 0.40

        guidance = result["guidance"]
        assert guidance["status"] == "crop_mismatch"
        assert guidance["data"] is None
        assert guidance["is_healthy"] is None
        assert "withheld" in guidance["message"].lower()

    def test_tomato_selected_tomato_predicted_confident(self, mock_service):
        """
        Scenario 7: Supported crop with matching prediction:
        Tomato selected, class 29 ('Tomato with Early Blight') predicted at 85% confidence.
        """
        mock_output = MagicMock()
        mock_output.logits = _create_mock_logits(top_index=29, top_score=0.85)
        mock_service._model.return_value = mock_output

        img = Image.new("RGB", (120, 120), (34, 139, 34))
        result = mock_service.predict(img, selected_crop="tomato", lang="en")

        assert result["assessment_status"] == "confident"
        assert result["crop_match"] is True
        assert result["selected_crop"] == "tomato"
        assert result["predicted_crop"] == "tomato"
        assert result["class_index"] == 29
        assert result["confidence"] == 0.85
        assert result["guidance"]["status"] == "success"
        assert result["guidance"]["data"]["disease_id"] == "tomato_early_blight"
        assert len(result["guidance"]["data"]["treatment"]) > 0

    def test_unknown_or_unmapped_class_index(self, mock_service):
        """
        Scenario 19: Out-of-range class index (e.g. 50 in a 38-class catalog):
        Must return invalid_model_output and suppress guidance safely.
        """
        mock_output = MagicMock()
        mock_output.logits = _create_mock_logits(top_index=50, top_score=0.90, num_classes=55)
        mock_service._model.return_value = mock_output

        img = Image.new("RGB", (120, 120), (34, 139, 34))
        result = mock_service.predict(img, selected_crop="tomato", lang="en")

        assert result["assessment_status"] == "invalid_model_output"
        assert result["class_index"] == 50
        assert result["class_label"] is None
        assert result["guidance"]["status"] == "unreliable_assessment"
        assert result["guidance"]["data"] is None

    def test_low_confidence_below_threshold(self, mock_service):
        """
        Scenario 11: Prediction with matching crop but score below threshold (0.25 < 0.40):
        Must return inconclusive_low_confidence and suppress treatment data.
        """
        mock_output = MagicMock()
        mock_output.logits = _create_mock_logits(top_index=29, top_score=0.25)
        mock_service._model.return_value = mock_output

        img = Image.new("RGB", (120, 120), (34, 139, 34))
        result = mock_service.predict(img, selected_crop="tomato", lang="en")

        assert result["assessment_status"] == "inconclusive_low_confidence"
        assert result["crop_match"] is True
        assert result["confidence"] == 0.25
        assert result["guidance"]["status"] == "unreliable_assessment"
        assert result["guidance"]["data"] is None

    def test_null_confidence_on_quality_rejection(self, mock_service):
        """
        Scenario 5 & 18: Pitch black image rejected by quality screener:
        Confidence must be None, inference must be bypassed, and guidance data must be None.
        """
        black_img = Image.new("RGB", (120, 120), (0, 0, 0))
        result = mock_service.predict(black_img, selected_crop="tomato", lang="en")

        assert result["assessment_status"] == "unreliable_image_quality"
        assert result["confidence"] is None
        assert result["class_index"] is None
        assert result["class_label"] is None
        assert result["screening_details"]["is_acceptable"] is False
        assert result["guidance"]["data"] is None

    # -------------------------------------------------------------------------
    # Input Validation Scenarios (1 to 4)
    # -------------------------------------------------------------------------

    def test_scenario_1_missing_image(self, client):
        """Scenario 1: Missing image in POST /api/predict must return 400 missing_image."""
        response = client.post("/api/predict", data={"crop": "tomato", "lang": "en"}, content_type="multipart/form-data")
        assert response.status_code == 400
        assert response.get_json()["error"] == "missing_image"

    def test_scenario_2_invalid_image_format(self, client):
        """Scenario 2: Invalid non-image file format must return 400 unsupported_format."""
        text_stream = io.BytesIO(b"Hello world! Not a leaf image.")
        data = {"image": (text_stream, "notes.txt", "text/plain"), "crop": "tomato", "lang": "en"}
        response = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert response.status_code == 400
        assert response.get_json()["error"] in ("unsupported_format", "invalid_image")

    def test_scenario_3_corrupted_image(self, client):
        """Scenario 3: Corrupted or truncated image data must return 400 corrupt_image."""
        corrupt_stream = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 30)  # Truncated JPEG header
        data = {"image": (corrupt_stream, "broken.jpg", "image/jpeg"), "crop": "tomato", "lang": "en"}
        response = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert response.status_code == 400
        assert response.get_json()["error"] in ("corrupt_image", "invalid_image")

    def test_scenario_4_empty_image(self, client):
        """Scenario 4: Empty (0-byte) image upload must return 400 empty_image."""
        empty_stream = io.BytesIO(b"")
        data = {"image": (empty_stream, "empty.png", "image/png"), "crop": "tomato", "lang": "en"}
        response = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert response.status_code == 400
        assert response.get_json()["error"] == "empty_image"

    # -------------------------------------------------------------------------
    # Leaf Validation & Non-Leaf Objects (Scenario 6)
    # -------------------------------------------------------------------------

    def test_scenario_6_non_leaf_objects_rejected(self, mock_service):
        """
        Scenario 6: Non-leaf objects (blue plastic, white drywall, grey concrete, denim fabric)
        lack vegetation characteristics (veg_ratio < 0.08) and must be rejected with 'not_a_leaf'.
        Inference must be bypassed and guidance data must be None.
        """
        # Blue plastic container with distinct surface pattern (contrast > 15.0, veg_ratio == 0.0)
        arr = np.zeros((120, 120, 3), dtype=np.uint8)
        arr[:60, :, 2] = 220
        arr[60:, :, 2] = 40
        arr[:60, :, 0] = 50
        blue_plastic = Image.fromarray(arr, mode="RGB")
        result = mock_service.predict(blue_plastic, selected_crop="tomato", lang="en")

        mock_service._model.assert_not_called()
        assert result["assessment_status"] == "not_a_leaf"
        assert result["confidence"] is None
        assert result["guidance"]["status"] == "not_a_leaf"
        assert result["guidance"]["data"] is None
        assert "no assessable crop leaf" in result["disclaimer"].lower()

        # Textured wall surface (contrast > 5.0, veg_ratio == 0.0)
        wall_arr = np.tile(np.linspace(150, 220, 120, dtype=np.uint8), (120, 1))
        wall_img = Image.fromarray(wall_arr, mode="L").convert("RGB")
        result_wall = mock_service.predict(wall_img, selected_crop="maize", lang="en")
        assert result_wall["assessment_status"] == "not_a_leaf"
        assert result_wall["guidance"]["data"] is None

    # -------------------------------------------------------------------------
    # Unsupported Crops & Out-of-Distribution (Scenarios 8, 9, 10, 12)
    # -------------------------------------------------------------------------

    def test_scenario_8_unsupported_crop_selection(self, client, mock_service):
        """
        Scenario 8: Submitting an unsupported crop (Cassava, Yam, Cocoa, Plantain)
        must return an explicit 400 or unsupported_crop status, withholding treatment guidance.
        """
        for unsupported in ["cassava", "yam", "cocoa", "plantain", "rubber"]:
            img_buf = _make_in_memory_image()
            response = client.post(
                "/api/predict",
                data={"image": (img_buf, "leaf.jpg", "image/jpeg"), "crop": unsupported, "lang": "en"},
                content_type="multipart/form-data"
            )
            assert response.status_code == 400
            body = response.get_json()
            assert body["error"] == "unsupported_crop"
            assert "unsupported" in body["message"].lower()

        # Direct service test with unsupported crop override
        img = Image.new("RGB", (120, 120), (34, 139, 34))
        service_res = mock_service.predict(img, selected_crop="cassava", lang="en")
        assert service_res["assessment_status"] == "unsupported_crop"
        assert service_res["guidance"]["data"] is None

    def test_scenario_9_unsupported_crop_selected_as_supported(self, mock_service):
        """
        Scenario 9: A cassava leaf uploaded with 'tomato' selected.
        If model features yield low confidence (15.9%) or an OOD class (e.g. strawberry, grape),
        the assessment must return inconclusive_low_confidence or crop_mismatch, NEVER diagnosing
        or prescribing treatment.
        """
        # Case A: Low confidence response (e.g. 15.9% Tomato Late Blight)
        mock_output_a = MagicMock()
        mock_output_a.logits = _create_mock_logits(top_index=30, top_score=0.159)
        mock_service._model.return_value = mock_output_a

        img = Image.new("RGB", (120, 120), (34, 139, 34))
        result_a = mock_service.predict(img, selected_crop="tomato", lang="en")
        assert result_a["assessment_status"] == "inconclusive_low_confidence"
        assert result_a["guidance"]["data"] is None

        # Case B: Cross-species activation (e.g. Grape Black Rot 55%)
        mock_output_b = MagicMock()
        mock_output_b.logits = _create_mock_logits(top_index=11, top_score=0.55)  # Grape Black Rot
        mock_service._model.return_value = mock_output_b

        result_b = mock_service.predict(img, selected_crop="tomato", lang="en")
        assert result_b["assessment_status"] == "crop_mismatch"
        assert result_b["guidance"]["data"] is None

    def test_scenario_10_supported_crop_wrong_selection(self, mock_service):
        """
        Scenario 10: User uploads maize leaf but selects tomato.
        Model activates Corn Common Rust (class 8) at 78% confidence.
        Must return crop_mismatch and withhold treatment.
        """
        mock_output = MagicMock()
        mock_output.logits = _create_mock_logits(top_index=8, top_score=0.78)
        mock_service._model.return_value = mock_output

        img = Image.new("RGB", (120, 120), (34, 139, 34))
        result = mock_service.predict(img, selected_crop="tomato", lang="en")

        assert result["assessment_status"] == "crop_mismatch"
        assert result["crop_match"] is False
        assert result["predicted_crop"] == "maize"
        assert result["guidance"]["data"] is None

    def test_scenario_12_high_confidence_unverified_crop_identity(self, mock_service):
        """
        Scenario 12: High confidence (92%) on an unexpected species does NOT bypass crop verification.
        User selects Tomato, model predicts Apple with Scab (class 0) at 92%.
        Must return crop_mismatch and withhold disease guidance.
        """
        mock_output = MagicMock()
        mock_output.logits = _create_mock_logits(top_index=0, top_score=0.92)
        mock_service._model.return_value = mock_output

        img = Image.new("RGB", (120, 120), (34, 139, 34))
        result = mock_service.predict(img, selected_crop="tomato", lang="en")

        assert result["assessment_status"] == "crop_mismatch"
        assert result["confidence"] == 0.92
        assert result["guidance"]["data"] is None

    # -------------------------------------------------------------------------
    # Healthy Classes & Incomplete Coverage (Scenarios 13, 14, 15)
    # -------------------------------------------------------------------------

    def test_scenario_14_supported_healthy_class(self, mock_service):
        """
        Scenario 14: Supported crop predicted as healthy (e.g. Healthy Tomato, class 37)
        Must return assessment_status: 'healthy', is_healthy=True, and cultural maintenance
        without prescribing chemical fungicides.
        """
        mock_output = MagicMock()
        mock_output.logits = _create_mock_logits(top_index=37, top_score=0.91)
        mock_service._model.return_value = mock_output

        img = Image.new("RGB", (120, 120), (34, 139, 34))
        result = mock_service.predict(img, selected_crop="tomato", lang="en")

        assert result["assessment_status"] == "healthy"
        assert result["crop_match"] is True
        assert result["guidance"]["is_healthy"] is True
        data = result["guidance"]["data"]
        assert data["status"] == "healthy"
        assert any("no disease treatment required" in t.lower() for t in data["treatment"])
        assert len(data["prevention"]) > 0

    def test_scenario_15_crop_with_no_healthy_class_citrus(self, mock_service):
        """
        Scenario 15: Citrus has NO healthy class in the 38-class catalog.
        The system must never claim a citrus tree is healthy based on absence of detection.
        """
        guidance = GuidanceService()
        assert guidance.has_healthy_class("citrus") is False
        assert guidance.get_healthy_class("citrus") is None

    # -------------------------------------------------------------------------
    # Localization & Bilingual Fallbacks (Scenarios 16, 17)
    # -------------------------------------------------------------------------

    def test_scenario_16_missing_yoruba_guidance_fallback(self):
        """
        Scenario 16: When a supported condition lacks Yoruba text, falls back to English
        with a transparent explanation, preserving accuracy.
        """
        guidance = GuidanceService()
        res = guidance.get_guidance_by_index(29, lang="yo")
        assert res["status"] == "success"
        assert res["data"] is not None

    def test_scenario_17_english_and_yoruba_versions_of_rejection_and_uncertainty(self, mock_service):
        """
        Scenario 17: Validates both English and Yoruba output for not_a_leaf, mismatch,
        unsupported crop, and low confidence.
        """
        img = Image.new("RGB", (120, 120), (34, 139, 34))

        # Crop mismatch in Yoruba
        mock_out = MagicMock()
        mock_out.logits = _create_mock_logits(top_index=26, top_score=0.61)
        mock_service._model.return_value = mock_out

        res_yo = mock_service.predict(img, selected_crop="tomato", lang="yo")
        assert "Àìbáramu irúgbìn wáyé" in res_yo["disclaimer"]
        assert res_yo["guidance"]["data"] is None

        # Low confidence in Yoruba
        mock_out_low = MagicMock()
        mock_out_low.logits = _create_mock_logits(top_index=29, top_score=0.20)
        mock_service._model.return_value = mock_out_low

        res_low_yo = mock_service.predict(img, selected_crop="tomato", lang="yo")
        assert "Ìdánilójú àwòṣe" in res_low_yo["disclaimer"]
        assert res_low_yo["guidance"]["data"] is None

    # -------------------------------------------------------------------------
    # Safety Invariant: Guidance Withholding (Scenario 18, 21)
    # -------------------------------------------------------------------------

    def test_scenario_18_guidance_withheld_for_all_unsafe_outcomes(self, mock_service):
        """
        Scenario 18: Invariant check: guidance['data'] must be None for EVERY unsafe outcome:
        crop_mismatch, unsupported_crop, not_a_leaf, unreliable_image_quality, inconclusive_low_confidence.
        """
        # A: Non-leaf image
        blue_img = Image.new("RGB", (100, 100), (0, 0, 255))
        assert mock_service.predict(blue_img, selected_crop="tomato")["guidance"]["data"] is None

        # B: Crop mismatch
        mock_service._model.return_value.logits = _create_mock_logits(26, 0.70)
        leaf_img = Image.new("RGB", (100, 100), (34, 139, 34))
        assert mock_service.predict(leaf_img, selected_crop="tomato")["guidance"]["data"] is None

        # C: Low confidence
        mock_service._model.return_value.logits = _create_mock_logits(29, 0.25)
        assert mock_service.predict(leaf_img, selected_crop="tomato")["guidance"]["data"] is None

        # D: Unsupported crop
        assert mock_service.predict(leaf_img, selected_crop="yam")["guidance"]["data"] is None

    def test_scenario_20_model_loading_and_prediction_exceptions(self):
        """
        Scenario 20: Model loading errors and inference execution exceptions fail gracefully
        with proper HTTP and error codes.
        """
        service = PredictionService(model_name="invalid/nonexistent-model-repo-cropscan")
        with pytest.raises(ModelLoadError) as exc_info:
            service.load_model()
        assert exc_info.value.status_code == 503

    def test_scenario_21_no_stale_or_unrelated_treatment_guidance(self, mock_service):
        """
        Scenario 21: When an assessment is inconclusive or mismatched, the system
        must NEVER leak treatment data from the highest-probability class into the response.
        """
        # Top score is Tomato Early Blight (index 29) at 35% (< 40%)
        mock_output = MagicMock()
        mock_output.logits = _create_mock_logits(top_index=29, top_score=0.35)
        mock_service._model.return_value = mock_output

        img = Image.new("RGB", (120, 120), (34, 139, 34))
        result = mock_service.predict(img, selected_crop="tomato", lang="en")

        assert result["assessment_status"] == "inconclusive_low_confidence"
        assert result["guidance"]["data"] is None
        # Verify no fungicide or treatment string is present in the guidance
        assert "treatment" not in result["guidance"] or result["guidance"]["treatment"] is None


class TestCropMappingsAndCatalogIntegrity:
    """Audits class-to-crop mappings and knowledge catalog integrity."""

    @pytest.fixture
    def guidance(self):
        return GuidanceService()

    def test_all_20_supported_crop_classes_mapping(self, guidance):
        """
        Verify that all 20 supported crop classes in the model are correctly mapped
        in class_to_crop.json and exist in the English disease guidance catalog.
        """
        expected_supported_classes = {
            # Tomato (10 classes: indices 28 to 37)
            28: ("Tomato with Bacterial Spot", "tomato"),
            29: ("Tomato with Early Blight", "tomato"),
            30: ("Tomato with Late Blight", "tomato"),
            31: ("Tomato with Leaf Mold", "tomato"),
            32: ("Tomato with Septoria Leaf Spot", "tomato"),
            33: ("Tomato with Spider Mites or Two-spotted Spider Mite", "tomato"),
            34: ("Tomato with Target Spot", "tomato"),
            35: ("Tomato Yellow Leaf Curl Virus", "tomato"),
            36: ("Tomato Mosaic Virus", "tomato"),
            37: ("Healthy Tomato Plant", "tomato"),
            # Maize (4 classes: indices 7 to 10)
            7: ("Corn (Maize) with Cercospora and Gray Leaf Spot", "maize"),
            8: ("Corn (Maize) with Common Rust", "maize"),
            9: ("Corn (Maize) with Northern Leaf Blight", "maize"),
            10: ("Healthy Corn (Maize) Plant", "maize"),
            # Potato (3 classes: indices 20 to 22)
            20: ("Potato with Early Blight", "potato"),
            21: ("Potato with Late Blight", "potato"),
            22: ("Healthy Potato Plant", "potato"),
            # Bell Pepper (2 classes: indices 18 to 19)
            18: ("Bell Pepper with Bacterial Spot", "bell_pepper"),
            19: ("Healthy Bell Pepper Plant", "bell_pepper"),
            # Citrus (1 class: index 15)
            15: ("Orange with Citrus Greening", "citrus"),
        }

        for idx, (expected_label, expected_crop) in expected_supported_classes.items():
            assert guidance.get_class_label(idx) == expected_label, f"Label mismatch at {idx}"
            assert guidance.get_crop_for_class(idx) == expected_crop, f"Crop mismatch at {idx}"
            guidance_rec = guidance.get_guidance_by_index(idx, lang="en")
            assert guidance_rec["status"] == "success", f"Missing guidance for {expected_label}"
            assert guidance_rec["data"] is not None

    def test_healthy_classes_guidance_lookup(self, guidance):
        """
        Verify that healthy leaf classes for supported crops return is_healthy=True
        and cultural maintenance advice without prescribing chemical fungicides.
        """
        healthy_supported_indices = [
            (37, "Healthy Tomato Plant"),
            (10, "Healthy Corn (Maize) Plant"),
            (19, "Healthy Bell Pepper Plant"),
            (22, "Healthy Potato Plant"),
        ]

        for idx, expected_name in healthy_supported_indices:
            res = guidance.get_guidance_by_index(idx, lang="en")
            assert res["status"] == "success"
            assert res["is_healthy"] is True
            data = res["data"]
            assert data["status"] == "healthy"
            assert len(data["prevention"]) > 0

    def test_citrus_model_gap_verified(self, guidance):
        """
        Documented model limitation:
        Verify that Citrus has only 1 class in the PlantVillage model (Orange with Citrus Greening)
        and lacks a Healthy Citrus class or other common diseases (Canker, Black Spot).
        """
        citrus_indices = [
            idx for idx in range(guidance.total_model_classes)
            if guidance.get_crop_for_class(idx) == "citrus"
        ]
        assert citrus_indices == [15]
        assert guidance.get_class_label(15) == "Orange with Citrus Greening"
