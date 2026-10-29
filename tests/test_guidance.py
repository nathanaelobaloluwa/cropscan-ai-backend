import pytest
from app.services.guidance import GuidanceService


@pytest.fixture
def guidance_service():
    """Provides a fresh GuidanceService instance."""
    return GuidanceService()


class TestGuidanceService:
    """Automated tests for GuidanceService and the Disease Knowledge Base."""

    # 1. Class-Index Resolution
    def test_class_indices_count(self, guidance_service):
        """Verify the service loaded all 38 verified model classes."""
        assert guidance_service.total_model_classes == 38

    def test_class_index_resolution(self, guidance_service):
        """Verify that known class indices resolve correctly."""
        # Index 7 -> Corn (Maize) with Cercospora and Gray Leaf Spot
        assert guidance_service.get_class_label(7) == "Corn (Maize) with Cercospora and Gray Leaf Spot"
        assert guidance_service.get_class_label("7") == "Corn (Maize) with Cercospora and Gray Leaf Spot"

        # Index 15 -> Orange with Citrus Greening
        assert guidance_service.get_class_label(15) == "Orange with Citrus Greening"

        # Index 29 -> Tomato with Early Blight
        assert guidance_service.get_class_label(29) == "Tomato with Early Blight"

        # Index 37 -> Healthy Tomato Plant
        assert guidance_service.get_class_label(37) == "Healthy Tomato Plant"

    # 2. English & Yoruba Retrieval
    def test_english_guidance_retrieval(self, guidance_service):
        """Verify retrieval of valid English guidance records."""
        result = guidance_service.get_guidance_by_index(29, lang="en")

        assert result["status"] == "success"
        assert result["language"] == "en"
        assert result["model_class_id"] == "Tomato with Early Blight"
        assert result["is_healthy"] is False

        data = result["data"]
        assert data is not None
        assert data["crop"] == "Tomato"
        assert data["display_name"] == "Early Blight"
        assert data["scientific_name"] == "Alternaria solani"
        assert data["disease_id"] == "tomato_early_blight"
        assert len(data["symptoms"]) > 0
        assert len(data["treatment"]) > 0
        assert len(data["prevention"]) > 0
        assert data["expert_review_required"] is True
        assert len(data["sources"]) > 0

    def test_yoruba_guidance_retrieval(self, guidance_service):
        """Verify retrieval of valid Yoruba guidance records."""
        result = guidance_service.get_guidance_by_index(29, lang="yo")

        assert result["status"] == "success"
        assert result["language"] == "yo"
        assert result["model_class_id"] == "Tomato with Early Blight"
        assert result["is_healthy"] is False

        data = result["data"]
        assert data is not None
        assert "Tòmátò" in data["crop"]
        assert "Early Blight" in data["display_name"]
        assert data["disease_id"] == "tomato_early_blight"
        assert data["expert_review_required"] is True
        assert data["translation_review_status"] == "draft_translation_pending_native_expert_review"
        assert len(data["sources"]) > 0

    def test_yoruba_fallback_when_translation_missing(self, guidance_service):
        """Verify fallback to English when Yoruba guidance is missing for a valid class."""
        # Temporarily mock a class in EN only
        guidance_service._disease_info_en["Test Only In English"] = {
            "disease_id": "test_only_in_en",
            "model_class_id": "Test Only In English",
            "crop": "Tomato",
            "display_name": "Test Disease",
            "scientific_name": "Testus disease",
            "status": "diseased",
            "severity": "low",
            "description": "English only description.",
            "symptoms": ["Test spot"],
            "treatment": ["Test treatment"],
            "prevention": ["Test prevention"],
        }
        result = guidance_service.get_guidance_by_label("Test Only In English", lang="yo")
        assert result["status"] == "success"
        assert result["language"] == "en"
        assert "Yorùbá" in result["message"]
        assert result["data"]["display_name"] == "Test Disease"

    # 3. Missing Records & Invalid Indices
    def test_invalid_class_index_out_of_bounds(self, guidance_service):
        """Verify handling of class indices outside the 0-37 range."""
        result = guidance_service.get_guidance_by_index(99, lang="en")

        assert result["status"] == "invalid_class_index"
        assert result["data"] is None
        assert result["is_healthy"] is None
        assert "invalid" in result["message"].lower()

    def test_invalid_class_index_negative(self, guidance_service):
        """Verify handling of negative class indices."""
        result = guidance_service.get_guidance_by_index(-1, lang="yo")

        assert result["status"] == "invalid_class_index"
        assert result["data"] is None
        assert result["is_healthy"] is None
        assert result["language"] == "yo"

    # 4. Required Data Fields & Language Consistency
    def test_language_dataset_consistency(self, guidance_service):
        """Verify identical keys and shared schema between EN and YO datasets."""
        en_keys = set(guidance_service._disease_info_en.keys())
        yo_keys = set(guidance_service._disease_info_yo.keys())

        # English and Yoruba must cover the exact same set of classes
        assert en_keys == yo_keys
        assert len(en_keys) == 20

        required_fields = {
            "disease_id",
            "model_class_id",
            "crop",
            "display_name",
            "scientific_name",
            "status",
            "severity",
            "description",
            "symptoms",
            "treatment",
            "prevention",
            "expert_review_required",
            "translation_review_status",
            "sources",
        }

        for model_class in en_keys:
            en_record = guidance_service._disease_info_en[model_class]
            yo_record = guidance_service._disease_info_yo[model_class]

            assert required_fields.issubset(en_record.keys()), f"Missing fields in EN for {model_class}"
            assert required_fields.issubset(yo_record.keys()), f"Missing fields in YO for {model_class}"

            # disease_id and model_class_id must match across languages
            assert en_record["disease_id"] == yo_record["disease_id"]
            assert en_record["model_class_id"] == yo_record["model_class_id"]
            assert en_record["status"] == yo_record["status"]

            # Must require expert review
            assert en_record["expert_review_required"] is True
            assert yo_record["expert_review_required"] is True

            # Yoruba records must explicitly flag draft translation status
            assert yo_record["translation_review_status"] == "draft_translation_pending_native_expert_review"

            # Both must cite traceable agricultural sources
            assert len(en_record["sources"]) > 0
            assert len(yo_record["sources"]) > 0

    # 5. Supported-Crop Scope Verification
    def test_supported_crop_scope(self, guidance_service):
        """Verify that Maize, Tomato, Pepper, Citrus, and Potato are present."""
        supported_classes = guidance_service.supported_guidance_classes

        # Maize (4 classes)
        assert "Corn (Maize) with Cercospora and Gray Leaf Spot" in supported_classes
        assert "Corn (Maize) with Common Rust" in supported_classes
        assert "Corn (Maize) with Northern Leaf Blight" in supported_classes
        assert "Healthy Corn (Maize) Plant" in supported_classes

        # Citrus (1 class)
        assert "Orange with Citrus Greening" in supported_classes

        # Bell Pepper (2 classes)
        assert "Bell Pepper with Bacterial Spot" in supported_classes
        assert "Healthy Bell Pepper Plant" in supported_classes

        # Potato (3 classes)
        assert "Potato with Early Blight" in supported_classes
        assert "Potato with Late Blight" in supported_classes
        assert "Healthy Potato Plant" in supported_classes

        # Tomato (10 classes)
        assert "Tomato with Bacterial Spot" in supported_classes
        assert "Tomato with Early Blight" in supported_classes
        assert "Tomato with Late Blight" in supported_classes
        assert "Tomato with Leaf Mold" in supported_classes
        assert "Tomato with Septoria Leaf Spot" in supported_classes
        assert "Tomato with Spider Mites or Two-spotted Spider Mite" in supported_classes
        assert "Tomato with Target Spot" in supported_classes
        assert "Tomato Yellow Leaf Curl Virus" in supported_classes
        assert "Tomato Mosaic Virus" in supported_classes
        assert "Healthy Tomato Plant" in supported_classes

    def test_cassava_and_yam_are_not_present(self, guidance_service):
        """Confirm that cassava and yam are NOT present in the model or guidance."""
        for label in guidance_service._class_indices.values():
            assert "cassava" not in label.lower()
            assert "yam" not in label.lower()

        for label in guidance_service.supported_guidance_classes:
            assert "cassava" not in label.lower()
            assert "yam" not in label.lower()

    # 6. Safe Fallback Behavior
    def test_recognized_class_without_guidance_fallback(self, guidance_service):
        """
        Verify that a class recognized by the model that has no guidance record yet
        (e.g., Apple Scab) returns a safe fallback and NEVER claims the image is healthy.
        """
        # Index 0 is "Apple Scab", recognized by model but outside Nigerian staple scope
        result = guidance_service.get_guidance_by_index(0, lang="en")

        assert result["status"] == "recognized_unsupported_crop"
        assert result["model_class_id"] == "Apple Scab"
        assert result["data"] is None
        # MUST NEVER claim healthy
        assert result["is_healthy"] is False
        assert "Nigerian staple crops" in result["message"] or "consult" in result["message"].lower()

    def test_unreliable_assessment_fallback(self, guidance_service):
        """Verify that an unreliable assessment never claims healthy."""
        result_en = guidance_service.get_unreliable_assessment_fallback(lang="en")
        assert result_en["status"] == "unreliable_assessment"
        assert result_en["is_healthy"] is None
        assert result_en["data"] is None
        assert "reliably" in result_en["message"].lower()

        result_yo = guidance_service.get_unreliable_assessment_fallback(lang="yo")
        assert result_yo["status"] == "unreliable_assessment"
        assert result_yo["is_healthy"] is None
        assert result_yo["language"] == "yo"

    # 7. Crop Mapping & Mismatch Fallback
    def test_class_to_crop_mapping_coverage(self, guidance_service):
        """Verify that all 38 model classes (indices 0 to 37) map to an explicit crop identifier."""
        assert len(guidance_service._class_to_crop) == 38
        for i in range(38):
            crop = guidance_service.get_crop_for_class(i)
            assert crop is not None, f"Missing crop for class index {i}"
            assert isinstance(crop, str) and len(crop) > 0

    def test_crop_normalization_supported_and_aliases(self, guidance_service):
        """Verify normalization of supported crops and common user aliases."""
        assert guidance_service.normalize_crop("tomato") == "tomato"
        assert guidance_service.normalize_crop("Tomatoes") == "tomato"
        assert guidance_service.normalize_crop("maize") == "maize"
        assert guidance_service.normalize_crop("corn") == "maize"
        assert guidance_service.normalize_crop("bell_pepper") == "bell_pepper"
        assert guidance_service.normalize_crop("bell pepper") == "bell_pepper"
        assert guidance_service.normalize_crop("pepper") == "bell_pepper"
        assert guidance_service.normalize_crop("citrus") == "citrus"
        assert guidance_service.normalize_crop("orange") == "citrus"
        assert guidance_service.normalize_crop("potato") == "potato"
        assert guidance_service.normalize_crop("Irish Potato") == "potato"

    def test_crop_normalization_unsupported(self, guidance_service):
        """Verify that unsupported crops normalize to None."""
        assert guidance_service.normalize_crop("cassava") is None
        assert guidance_service.normalize_crop("yam") is None
        assert guidance_service.normalize_crop("wheat") is None
        assert guidance_service.normalize_crop("") is None
        assert guidance_service.normalize_crop(None) is None

    def test_is_crop_supported(self, guidance_service):
        """Verify boolean crop support checks."""
        assert guidance_service.is_crop_supported("tomato") is True
        assert guidance_service.is_crop_supported("corn") is True
        assert guidance_service.is_crop_supported("bell pepper") is True
        assert guidance_service.is_crop_supported("citrus") is True
        assert guidance_service.is_crop_supported("potato") is True
        assert guidance_service.is_crop_supported("cassava") is False
        assert guidance_service.is_crop_supported("apple") is False

    def test_get_crop_for_class_indices(self, guidance_service):
        """Verify exact crop lookup for representative model class indices."""
        assert guidance_service.get_crop_for_class(0) == "apple"
        assert guidance_service.get_crop_for_class(7) == "maize"
        assert guidance_service.get_crop_for_class(15) == "citrus"
        assert guidance_service.get_crop_for_class(18) == "bell_pepper"
        assert guidance_service.get_crop_for_class(21) == "potato"
        assert guidance_service.get_crop_for_class(29) == "tomato"

    def test_get_crop_mismatch_fallback_english_and_yoruba(self, guidance_service):
        """Verify that crop mismatch fallback withholds guidance and never claims healthy."""
        res_en = guidance_service.get_crop_mismatch_fallback(
            selected_crop="tomato",
            predicted_crop="apple",
            model_class_id="Cedar Apple Rust",
            lang="en"
        )
        assert res_en["status"] == "crop_mismatch"
        assert res_en["selected_crop"] == "tomato"
        assert res_en["predicted_crop"] == "apple"
        assert res_en["model_class_id"] == "Cedar Apple Rust"
        assert res_en["is_healthy"] is None
        assert res_en["data"] is None
        assert "Crop mismatch detected" in res_en["message"]
        assert "withheld" in res_en["message"]

        res_yo = guidance_service.get_crop_mismatch_fallback(
            selected_crop="tomato",
            predicted_crop="apple",
            model_class_id="Cedar Apple Rust",
            lang="yo"
        )
        assert res_yo["status"] == "crop_mismatch"
        assert res_yo["language"] == "yo"
        assert res_yo["is_healthy"] is None
        assert res_yo["data"] is None
        assert "Àìbáramu irúgbìn wáyé" in res_yo["message"]

