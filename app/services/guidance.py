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

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else self.DATA_DIR
        self._class_indices: Dict[str, str] = {}
        self._disease_info_en: Dict[str, Dict[str, Any]] = {}
        self._disease_info_yo: Dict[str, Dict[str, Any]] = {}
        self._load_datasets()

    def _load_datasets(self) -> None:
        """Load datasets from disk into memory."""
        class_indices_file = self.data_dir / "class_indices.json"
        en_file = self.data_dir / "disease_info_en.json"
        yo_file = self.data_dir / "disease_info_yo.json"

        if not class_indices_file.exists():
            raise FileNotFoundError(f"Missing class indices mapping: {class_indices_file}")
        if not en_file.exists():
            raise FileNotFoundError(f"Missing English disease info dataset: {en_file}")
        if not yo_file.exists():
            raise FileNotFoundError(f"Missing Yoruba disease info dataset: {yo_file}")

        with open(class_indices_file, "r", encoding="utf-8") as f:
            self._class_indices = json.load(f)

        with open(en_file, "r", encoding="utf-8") as f:
            self._disease_info_en = json.load(f)

        with open(yo_file, "r", encoding="utf-8") as f:
            self._disease_info_yo = json.load(f)

    def _normalize_lang(self, lang: Optional[str]) -> str:
        """Normalize and validate the language code."""
        if not lang or lang.lower() not in self.SUPPORTED_LANGUAGES:
            return self.DEFAULT_LANGUAGE
        return lang.lower()

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
