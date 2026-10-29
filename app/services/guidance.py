import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class GuidanceService:
    """
    GuidanceService resolves verified model predictions (class index or label)
    into structured, conservative agronomic guidance in English or Yoruba.

    Decoupling Principles:
    - Model Class IDs: The exact prediction string from the vision model (e.g. 'Tomato with Early Blight').
    - Disease IDs: Standardized machine slugs (e.g. 'tomato_early_blight').
    - Localized Display Names: Human-readable names formatted for farmers (e.g. 'Early Blight' or 'Àrùn Èérún Tòmátò').
    - Treatment & Prevention Records: Conservative agronomic management practices derived from verifiable sources.
    """

    SUPPORTED_LANGUAGES = ("en", "yo")
    DEFAULT_LANGUAGE = "en"
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"

    # Supported crop identifiers for the Nigerian prototype scope
    SUPPORTED_CROPS = ("tomato", "maize", "bell_pepper", "citrus", "potato")

    # Authoritative mapping of supported crops to their verified healthy class indices.
    # CRITICAL: 'citrus' is omitted because the 38-class PlantVillage model contains ONLY
    # 'Orange with Citrus Greening' (index 15) and has NO healthy citrus class.
    HEALTHY_CLASSES: Dict[str, int] = {
        "tomato": 37,        # "Healthy Tomato Plant"
        "maize": 10,         # "Healthy Corn (Maize) Plant"
        "potato": 22,        # "Healthy Potato Plant"
        "bell_pepper": 19,   # "Healthy Bell Pepper Plant"
    }

    # Common aliases mapping to canonical crop identifiers
    CROP_ALIASES = {
        "tomato": "tomato",
        "tomatoes": "tomato",
        "maize": "maize",
        "corn": "maize",
        "bell_pepper": "bell_pepper",
        "bell pepper": "bell_pepper",
        "bellpepper": "bell_pepper",
        "pepper": "bell_pepper",
        "peppers": "bell_pepper",
        "citrus": "citrus",
        "orange": "citrus",
        "citrus (orange)": "citrus",
        "potato": "potato",
        "potatoes": "potato",
        "irish potato": "potato",
    }

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else self.DATA_DIR
        self._class_indices: Dict[str, str] = {}
        self._class_to_crop: Dict[str, str] = {}
        self._disease_info_en: Dict[str, Dict[str, Any]] = {}
        self._disease_info_yo: Dict[str, Dict[str, Any]] = {}
        self._load_datasets()

    def _load_datasets(self) -> None:
        """Load datasets from disk into memory."""
        class_indices_file = self.data_dir / "class_indices.json"
        class_to_crop_file = self.data_dir / "class_to_crop.json"
        en_file = self.data_dir / "disease_info_en.json"
        yo_file = self.data_dir / "disease_info_yo.json"

        if not class_indices_file.exists():
            raise FileNotFoundError(f"Missing class indices mapping: {class_indices_file}")
        if not class_to_crop_file.exists():
            raise FileNotFoundError(f"Missing class-to-crop mapping: {class_to_crop_file}")
        if not en_file.exists():
            raise FileNotFoundError(f"Missing English disease info dataset: {en_file}")
        if not yo_file.exists():
            raise FileNotFoundError(f"Missing Yoruba disease info dataset: {yo_file}")

        with open(class_indices_file, "r", encoding="utf-8") as f:
            self._class_indices = json.load(f)

        with open(class_to_crop_file, "r", encoding="utf-8") as f:
            self._class_to_crop = json.load(f)

        with open(en_file, "r", encoding="utf-8") as f:
            self._disease_info_en = json.load(f)

        with open(yo_file, "r", encoding="utf-8") as f:
            self._disease_info_yo = json.load(f)

    def _normalize_lang(self, lang: Optional[str]) -> str:
        """Normalize and validate the language code."""
        if not lang or lang.lower() not in self.SUPPORTED_LANGUAGES:
            return self.DEFAULT_LANGUAGE
        return lang.lower()

    @classmethod
    def normalize_crop(cls, crop_name: Optional[str]) -> Optional[str]:
        """
        Normalize user-provided crop string to a canonical crop identifier.
        Returns None if the input is empty or unrecognized.
        """
        if not crop_name:
            return None
        cleaned = str(crop_name).strip().lower().replace("-", "_")
        return cls.CROP_ALIASES.get(cleaned)

    @classmethod
    def is_crop_supported(cls, crop_name: Optional[str]) -> bool:
        """Check if a crop identifier is within the supported prototype scope."""
        canonical = cls.normalize_crop(crop_name)
        return canonical in cls.SUPPORTED_CROPS

    def get_crop_for_class(self, class_index: Union[int, str]) -> Optional[str]:
        """
        Retrieve the biological crop identifier associated with a model class index.
        Returns None if class index is not found in the 38-class catalog.
        """
        str_index = str(class_index).strip()
        return self._class_to_crop.get(str_index)

    def get_crop_classes(self, crop_name: Optional[str]) -> List[int]:
        """Return list of model class indices belonging to the specified crop."""
        canonical = self.normalize_crop(crop_name)
        if not canonical:
            return []
        return [
            int(idx)
            for idx, c in self._class_to_crop.items()
            if c == canonical
        ]

    def get_healthy_class(self, crop_name: Optional[str]) -> Optional[int]:
        """
        Return the model class index for a supported crop's healthy class.
        Returns None if the crop has no healthy class in the model catalog (e.g. Citrus).
        """
        canonical = self.normalize_crop(crop_name)
        if not canonical:
            return None
        return self.HEALTHY_CLASSES.get(canonical)

    def has_healthy_class(self, crop_name: Optional[str]) -> bool:
        """Returns True if the crop has a verified healthy class in the model catalog."""
        return self.get_healthy_class(crop_name) is not None

    @property
    def total_model_classes(self) -> int:
        """Total number of classes recognized by the underlying model."""
        return len(self._class_indices)

    @property
    def supported_guidance_classes(self) -> List[str]:
        """List of model classes that have verified guidance records in the knowledge base."""
        return list(self._disease_info_en.keys())

    def get_class_label(self, class_index: Union[int, str]) -> Optional[str]:
        """
        Resolve an integer or string class index to the official model class label.
        Returns None if the index is not recognized by the model.
        """
        str_index = str(class_index).strip()
        return self._class_indices.get(str_index)

    def get_guidance_by_index(
        self,
        class_index: Union[int, str],
        lang: str = "en"
    ) -> Dict[str, Any]:
        """
        Retrieve guidance for a prediction given its model output class index.

        Distinguishes four distinct outcomes:
        1. 'success': Valid class with complete localized guidance.
        2. 'recognized_unsupported_crop': Recognized by model, but outside prototype scope.
        3. 'invalid_class_index': Index is not within model range.
        4. Safe structured fallbacks that NEVER assume health.
        """
        lang = self._normalize_lang(lang)
        model_class_id = self.get_class_label(class_index)

        # Case 1: Invalid class index
        if model_class_id is None:
            return {
                "status": "invalid_class_index",
                "class_index": class_index,
                "model_class_id": None,
                "language": lang,
                "is_healthy": None,
                "message": (
                    f"Class index '{class_index}' is invalid. The model recognizes 38 classes (indices 0 to 37)."
                    if lang == "en"
                    else f"Nọ́ḿbà àrùn '{class_index}' kò tọ́. Àwọn àrùn 38 péré (0 sí 37) ni a mọ̀."
                ),
                "data": None,
            }

        return self.get_guidance_by_label(model_class_id, lang=lang, class_index=class_index)

    def get_guidance_by_label(
        self,
        model_class_id: str,
        lang: str = "en",
        class_index: Optional[Union[int, str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve guidance for a prediction given its official model class string.
        """
        lang = self._normalize_lang(lang)
        catalog = self._disease_info_yo if lang == "yo" else self._disease_info_en

        # If class_index was not provided, look it up in reverse
        if class_index is None:
            for idx_str, label in self._class_indices.items():
                if label == model_class_id:
                    class_index = int(idx_str)
                    break

        record = catalog.get(model_class_id)

        # Fallback when Yoruba guidance is missing for a supported crop
        if record is None and lang == "yo" and model_class_id in self._disease_info_en:
            en_record = self._disease_info_en[model_class_id]
            return {
                "status": "success",
                "class_index": class_index,
                "model_class_id": model_class_id,
                "language": "en",
                "is_healthy": (en_record.get("status") == "healthy"),
                "message": (
                    "Ìtọ́sọ́nà ìtọ́jú ní èdè Yorùbá kò tíì wà lárọ̀ọ́wọ́tó fún àrùn yìí. "
                    "A pèsè ìtọ́sọ́nà lédè Gẹ̀ẹ́sì fún ìrànlọ́wọ́ rẹ."
                ),
                "data": en_record,
            }

        # Case 2: Recognized by model, but no localized guidance record yet
        if record is None:
            return {
                "status": "recognized_unsupported_crop",
                "class_index": class_index,
                "model_class_id": model_class_id,
                "language": lang,
                "is_healthy": False,
                "message": (
                    f"CropScan AI recognizes this condition as '{model_class_id}'. "
                    "However, our current prototype focuses on Nigerian staple crops (Maize, Tomato, Bell Pepper, Citrus, Potato). "
                    "Detailed local treatment and prevention guidance is not yet documented for this crop. "
                    "Please consult a local agricultural extension officer for assistance."
                    if lang == "en"
                    else f"CropScan AI dá àrùn yìí mọ̀ gẹ́gẹ́ bí '{model_class_id}'. "
                    "Àmọ́, ètò wa fojú sùn àwọn ohun ọ̀gbìn ilẹ̀ Nàìjíríà (Àgbàdo, Tòmátò, Ata, Ọsàn, Pọtétò) ní báyìí. "
                    "Ìtọ́sọ́nà ìtọ́jú ní èdè Yorùbá kò tíì wà fún irúgbìn yìí. "
                    "Jọ̀wọ́ kan sí òṣìṣẹ́ àgbẹ̀ agbègbè rẹ fún ìrànlọ́wọ́."
                ),
                "data": None,
            }

        # Case 3: Fully supported guidance record
        return {
            "status": "success",
            "class_index": class_index,
            "model_class_id": model_class_id,
            "language": lang,
            "is_healthy": (record.get("status") == "healthy"),
            "message": "Guidance record retrieved successfully.",
            "data": record,
        }

    def get_unreliable_assessment_fallback(self, lang: str = "en") -> Dict[str, Any]:
        """
        Structured fallback when an image cannot be reliably assessed
        (e.g., blurry image, poor lighting, or confidence threshold not met).
        Never claims the plant is healthy.
        """
        lang = self._normalize_lang(lang)
        return {
            "status": "unreliable_assessment",
            "class_index": None,
            "model_class_id": None,
            "language": lang,
            "is_healthy": None,
            "message": (
                "The image could not be reliably assessed. "
                "Please capture a clear, well-lit photo of a single leaf against a plain background, "
                "or consult your local agricultural extension service for on-field diagnosis."
                if lang == "en"
                else "A kò lè ṣe àyẹ̀wò àwòrán yìí dáradára. "
                "Jọ̀wọ́ ya àwòrán tó mọ́lẹ̀ kedere ti ewé kan ṣoṣo níbi tí ìmọ́lẹ̀ ti dára, "
                "tàbí kí o kan sí òṣìṣẹ́ àgbẹ̀ agbègbè rẹ fún àyẹ̀wò tààrà lórí oko."
            ),
            "data": None,
        }

    def get_crop_mismatch_fallback(
        self,
        selected_crop: str,
        predicted_crop: Optional[str],
        model_class_id: Optional[str],
        lang: str = "en"
    ) -> Dict[str, Any]:
        """
        Structured fallback when the user's selected crop does not match the model's predicted crop.
        Withholds disease-specific treatment guidance to prevent dangerous chemical misapplication.
        Never claims the plant is healthy.
        """
        lang = self._normalize_lang(lang)
        pred_display = predicted_crop.replace("_", " ").title() if predicted_crop else "Unknown"
        selected_display = selected_crop.replace("_", " ").title()

        return {
            "status": "crop_mismatch",
            "selected_crop": selected_crop,
            "predicted_crop": predicted_crop,
            "model_class_id": model_class_id,
            "language": lang,
            "is_healthy": None,
            "message": (
                f"Crop mismatch detected: You submitted this leaf as '{selected_display}', "
                f"but the vision model detected visual features resembling '{model_class_id or 'an unlisted disease'}' "
                f"({pred_display}). Disease-specific treatment guidance is withheld to prevent chemical misapplication. "
                "User crop selection does not prove the botanical identity of the plant. "
                "Please verify your crop selection or consult your local agricultural extension service for on-field diagnosis."
                if lang == "en"
                else f"Àìbáramu irúgbìn wáyé: O fi ewé yìí ránṣẹ́ gẹ́gẹ́ bí '{selected_display}', "
                f"ṣùgbọ́n àwòṣe náà dá àwọn àmì àrùn mọ̀ tó jọ '{model_class_id or 'àrùn kan'}' "
                f"({pred_display}). A dáwọ́ ìtọ́sọ́nà ìtọ́jú dúró láti dènà lílo oògùn tí kò tọ́. "
                "Yíyan irúgbìn kò fi ìdí rẹ̀ múlẹ̀ pé irúgbìn náà jẹ́ bẹ́ẹ̀ gan-an. "
                "Jọ̀wọ́ ṣàyẹ̀wò irúgbìn tí o yàn tàbí kí o kan sí òṣìṣẹ́ àgbẹ̀ agbègbè rẹ fún àyẹ̀wò lórí oko."
            ),
            "data": None,
        }

    def get_unsupported_crop_fallback(
        self,
        crop: Optional[str],
        lang: str = "en"
    ) -> Dict[str, Any]:
        """
        Structured fallback when an unsupported crop is submitted or detected.
        Withholds disease-specific treatment to prevent chemical misapplication.
        Never claims the plant is healthy.
        """
        lang = self._normalize_lang(lang)
        crop_display = str(crop).replace("_", " ").title() if crop else "Unknown"

        return {
            "status": "unsupported_crop",
            "selected_crop": crop,
            "predicted_crop": None,
            "model_class_id": None,
            "language": lang,
            "is_healthy": None,
            "message": (
                f"Crop '{crop_display}' is not currently supported in CropScan AI. "
                "Our validated catalog covers Maize, Tomato, Bell Pepper, Citrus, and Potato. "
                "Disease-specific treatment guidance is withheld to prevent misdiagnosis. "
                "Please consult a local agricultural extension officer."
                if lang == "en"
                else f"Irúgbìn '{crop_display}' kò tíì wà nínú àkójọ CropScan AI ní báyìí. "
                "Àwọn ohun ọ̀gbìn tí a fọwọ́ sí ni Àgbàdo, Tòmátò, Ata, Ọsàn, àti Pọtétò. "
                "A dáwọ́ ìtọ́sọ́nà ìtọ́jú dúró láti dènà lílo oògùn tí kò tọ́. "
                "Jọ̀wọ́ kan sí òṣìṣẹ́ àgbẹ̀ agbègbè rẹ fún ìrànlọ́wọ́."
            ),
            "data": None,
        }

    def get_not_a_leaf_fallback(self, lang: str = "en") -> Dict[str, Any]:
        """
        Structured fallback when an image does not contain an assessable crop leaf.
        """
        lang = self._normalize_lang(lang)
        return {
            "status": "not_a_leaf",
            "class_index": None,
            "model_class_id": None,
            "language": lang,
            "is_healthy": None,
            "message": (
                "No assessable crop leaf was detected in this photograph. "
                "Automated screening detected non-plant objects or insufficient leaf tissue. "
                "Please upload a clear, well-lit photo of a single crop leaf."
                if lang == "en"
                else "A kò rí ewé irúgbìn tó ṣeé wò nínú àwòrán yìí. "
                "Ìmọ̀ ẹ̀rọ dá àwọn ohun tí kì í ṣe ewéko tàbí àìtó àmì ewé mọ̀. "
                "Jọ̀wọ́ ya àwòrán ewé kan ṣoṣo ní tòsí kedere."
            ),
            "data": None,
        }

