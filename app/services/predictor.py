import logging
import os
from typing import Any, Dict, Optional, Tuple, Union
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, MobileNetV2ImageProcessorPil

from app.services.guidance import GuidanceService

logger = logging.getLogger(__name__)

# Default provisional configuration
DEFAULT_MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
DEFAULT_CONFIDENCE_THRESHOLD = 0.40
DEFAULT_DEVICE = "cpu"


class PredictionError(Exception):
    """Base exception for prediction and inference errors."""

    def __init__(self, message: str, error_code: str = "inference_error", status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code


class ModelLoadError(PredictionError):
    """Raised when the ML model weights or processor fail to load."""

    def __init__(self, message: str):
        super().__init__(message, error_code="model_load_error", status_code=503)


class InferenceExecutionError(PredictionError):
    """Raised when an error occurs during tensor transformation or forward pass."""

    def __init__(self, message: str):
        super().__init__(message, error_code="inference_failed", status_code=500)


def create_image_processor() -> MobileNetV2ImageProcessorPil:
    """
    Constructs the MobileNetV2 image processor with the exact preprocessing
    verified from the official model configuration:
    - Shortest-edge resize to 256 pixels
    - Bilinear resampling (PIL.Image.BILINEAR)
    - Center crop to 224 x 224 pixels
    - Pixel rescaling by 1/255.0 (rescale_factor: 0.00392156862745098)
    - Normalization with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5] per RGB channel
    """
    return MobileNetV2ImageProcessorPil(
        do_resize=True,
        size={"shortest_edge": 256},
        resample=Image.BILINEAR,
        do_center_crop=True,
        crop_size={"height": 224, "width": 224},
        do_rescale=True,
        rescale_factor=1 / 255.0,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )


class PredictionService:
    """
    Dedicated inference service for CropScan AI MobileNetV2 plant disease model.

    Features:
    - Singleton / cached model loading: loads weights once, reuses across requests.
    - CPU inference by default, device configurable via environment or argument.
    - Exact verified preprocessing pipeline reproducing model configuration.
    - Softmax score extraction over 38 classes mapped to class_indices.json.
    - Configurable provisional confidence threshold (default 0.40).
    - Conservative agronomic guidance integration via GuidanceService.
    - Clear safety disclaimers: raw scores are uncalibrated and not clinical diagnoses.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        device: Optional[str] = None,
        guidance_service: Optional[GuidanceService] = None,
        lazy_load: bool = True,
    ):
        self.model_name = model_name or os.getenv("CROPSCAN_MODEL_NAME", DEFAULT_MODEL_NAME)

        env_threshold = os.getenv("PREDICTION_CONFIDENCE_THRESHOLD")
        if confidence_threshold is not None:
            self.confidence_threshold = float(confidence_threshold)
        elif env_threshold is not None:
            try:
                self.confidence_threshold = float(env_threshold)
            except ValueError:
                logger.warning(
                    f"Invalid PREDICTION_CONFIDENCE_THRESHOLD '{env_threshold}'. Falling back to {DEFAULT_CONFIDENCE_THRESHOLD}."
                )
                self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        else:
            self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD

        self.device = device or os.getenv("PREDICTOR_DEVICE", DEFAULT_DEVICE)
        self.guidance_service = guidance_service or GuidanceService()
        self._model: Optional[AutoModelForImageClassification] = None
        self._processor: Optional[MobileNetV2ImageProcessorPil] = None

        if not lazy_load:
            self.load_model()

    @property
    def is_model_loaded(self) -> bool:
        """Check whether the model weights and processor are currently in memory."""
        return self._model is not None and self._processor is not None

    def load_model(self) -> None:
        """
        Loads the pretrained model and image processor once into memory.
        Safe to call multiple times (cached singleton pattern).
        """
        if self.is_model_loaded:
            return

        try:
            logger.info(
                f"Initializing CropScan AI vision model '{self.model_name}' on device '{self.device}'."
            )
            # Attempt loading from local HF cache first for offline resilience
            try:
                model = AutoModelForImageClassification.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                )
            except Exception:
                model = AutoModelForImageClassification.from_pretrained(
                    self.model_name,
                )

            model.eval()
            model.to(self.device)
            self._model = model
            self._processor = create_image_processor()
            logger.info("CropScan AI vision model loaded successfully.")

        except Exception as e:
            self._model = None
            self._processor = None
            logger.error(f"Failed to load vision model '{self.model_name}': {e}", exc_info=True)
            raise ModelLoadError(
                f"Failed to load vision model '{self.model_name}': {str(e)}"
            ) from e

    def predict(
        self,
        image: Image.Image,
        selected_crop: Optional[str] = None,
        lang: str = "en"
    ) -> Dict[str, Any]:
        """
        Executes genuine inference on a validated PIL Image with crop mismatch protection.

        Args:
            image: A valid PIL Image instance.
            selected_crop: Optional user-selected crop identifier (e.g. 'tomato', 'maize').
            lang: Localized guidance language ('en' or 'yo').

        Returns:
            dict containing class index, label, selected_crop, predicted_crop, crop_match,
            raw model score, assessment status, safety disclaimers, and localized agronomic guidance.

        Raises:
            ModelLoadError: If model cannot be loaded.
            InferenceExecutionError: If forward pass or tensor transformation fails.
        """
        if not self.is_model_loaded:
            self.load_model()

        # Ensure image is in RGB mode (3 channels)
        if image.mode != "RGB":
            image = image.convert("RGB")

        try:
            # 1. Preprocess image
            inputs = self._processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            # 2. Genuine inference forward pass
            with torch.no_grad():
                outputs = self._model(pixel_values=pixel_values)
                logits = outputs.logits[0]
                probabilities = torch.nn.functional.softmax(logits, dim=-1)

                top_prob, top_idx = torch.topk(probabilities, 1)
                predicted_class_index = int(top_idx.item())
                confidence_score = float(top_prob.item())

        except Exception as e:
            logger.error(f"Inference execution failed: {e}", exc_info=True)
            raise InferenceExecutionError(
                f"Inference execution failed during forward pass: {str(e)}"
            ) from e

        # 3. Resolve class label and biological crop from official mappings
        class_label = self.guidance_service.get_class_label(predicted_class_index)
        predicted_crop = self.guidance_service.get_crop_for_class(predicted_class_index)

        normalized_selected_crop = (
            self.guidance_service.normalize_crop(selected_crop)
            if selected_crop
            else None
        )

        # Handle invalid model output index (outside 0-37)
        if class_label is None:
            fallback_guidance = self.guidance_service.get_unreliable_assessment_fallback(lang=lang)
            return {
                "assessment_status": "invalid_model_output",
                "model_name": self.model_name,
                "selected_crop": normalized_selected_crop or selected_crop,
                "predicted_crop": None,
                "crop_match": None,
                "class_index": predicted_class_index,
                "class_label": None,
                "confidence": round(confidence_score, 4),
                "confidence_threshold": self.confidence_threshold,
                "is_provisional_threshold": True,
                "disclaimer": (
                    "Model output index is outside the recognized 38-class catalog. "
                    "This output is treated as invalid and inconclusive."
                ),
                "guidance": fallback_guidance,
            }

        # 4. Check for Crop Mismatch
        is_crop_match = (
            (normalized_selected_crop == predicted_crop)
            if normalized_selected_crop is not None
            else True
        )

        if not is_crop_match:
            # Crop mismatch: user-selected crop differs from predicted class's crop family
            mismatch_guidance = self.guidance_service.get_crop_mismatch_fallback(
                selected_crop=normalized_selected_crop or selected_crop,
                predicted_crop=predicted_crop,
                model_class_id=class_label,
                lang=lang,
            )
            selected_display = (normalized_selected_crop or selected_crop).replace("_", " ").title()
            pred_display = predicted_crop.replace("_", " ").title() if predicted_crop else "Unknown"

            return {
                "assessment_status": "crop_mismatch",
                "model_name": self.model_name,
                "selected_crop": normalized_selected_crop or selected_crop,
                "predicted_crop": predicted_crop,
                "crop_match": False,
                "class_index": predicted_class_index,
                "class_label": class_label,
                "confidence": round(confidence_score, 4),
                "confidence_threshold": self.confidence_threshold,
                "is_provisional_threshold": True,
                "disclaimer": (
                    f"Crop mismatch detected: The image was submitted as '{selected_display}', but model visual "
                    f"features resemble '{class_label}' ({pred_display}). User crop selection does not prove "
                    "ground-truth botanical identity. Disease-specific treatment guidance is withheld to prevent "
                    "chemical misapplication. Consult a local agricultural extension officer."
                ),
                "guidance": mismatch_guidance,
            }

        # 5. Evaluate against provisional confidence threshold
        is_confident = confidence_score >= self.confidence_threshold

        if not is_confident:
            # Below provisional threshold: treat as inconclusive without diagnosing
            fallback_guidance = self.guidance_service.get_unreliable_assessment_fallback(lang=lang)
            return {
                "assessment_status": "inconclusive_low_confidence",
                "model_name": self.model_name,
                "selected_crop": normalized_selected_crop or selected_crop,
                "predicted_crop": predicted_crop,
                "crop_match": True,
                "class_index": predicted_class_index,
                "class_label": class_label,
                "confidence": round(confidence_score, 4),
                "confidence_threshold": self.confidence_threshold,
                "is_provisional_threshold": True,
                "disclaimer": (
                    f"Model confidence ({confidence_score:.2%}) is below the provisional operational threshold "
                    f"({self.confidence_threshold:.2%}). Raw model scores are not calibrated probabilities of "
                    "correctness. Confidence thresholding cannot reliably detect unsupported crops, non-leaf "
                    "images, or out-of-distribution inputs. User crop selection does not prove botanical identity. "
                    "This result is inconclusive and must not be treated as a confirmed diagnosis."
                ),
                "guidance": fallback_guidance,
            }

        # 6. Retrieve localized guidance for confident, matching predictions
        guidance_result = self.guidance_service.get_guidance_by_index(
            predicted_class_index, lang=lang
        )

        assessment_status = (
            "confident"
            if guidance_result["status"] == "success"
            else guidance_result["status"]
        )

        return {
            "assessment_status": assessment_status,
            "model_name": self.model_name,
            "selected_crop": normalized_selected_crop or selected_crop,
            "predicted_crop": predicted_crop,
            "crop_match": True,
            "class_index": predicted_class_index,
            "class_label": class_label,
            "confidence": round(confidence_score, 4),
            "confidence_threshold": self.confidence_threshold,
            "is_provisional_threshold": True,
            "disclaimer": (
                "Automated screening result only; not a confirmed agricultural diagnosis. "
                "Raw model scores are uncalibrated and do not guarantee accuracy. "
                "User crop selection does not prove botanical identity. "
                "Always consult local agricultural extension officers before applying treatments."
            ),
            "guidance": guidance_result,
        }



# Singleton instance container
_predictor_singleton: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """
    Returns the singleton PredictionService instance.
    Loads the model once and reuses it across subsequent requests.
    """
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = PredictionService()
    return _predictor_singleton


def set_prediction_service(service: Optional[PredictionService]) -> None:
    """
    Sets or resets the singleton PredictionService instance.
    Primarily used for testing with dependency injection.
    """
    global _predictor_singleton
    _predictor_singleton = service
