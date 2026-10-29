import gc
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
from flask import current_app, has_app_context
from PIL import Image
from transformers import AutoModelForImageClassification, MobileNetV2ImageProcessorPil

from app.config import DEFAULT_CONFIDENCE_THRESHOLD, validate_confidence_threshold
from app.services.guidance import GuidanceService
from app.services.image_screener import screen_image_quality

logger = logging.getLogger(__name__)

# Default provisional configuration
DEFAULT_MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
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

        # Resolve validated confidence threshold:
        # Priority 1: Explicit parameter override (validated)
        # Priority 2: Current Flask app config (if in active app context)
        # Priority 3: Environment variable fallback (validated)
        if confidence_threshold is not None:
            self.confidence_threshold = validate_confidence_threshold(
                confidence_threshold,
                default=DEFAULT_CONFIDENCE_THRESHOLD,
                source_name="explicit confidence_threshold",
            )
        elif has_app_context() and "PREDICTION_CONFIDENCE_THRESHOLD" in current_app.config:
            self.confidence_threshold = validate_confidence_threshold(
                current_app.config["PREDICTION_CONFIDENCE_THRESHOLD"],
                default=DEFAULT_CONFIDENCE_THRESHOLD,
                source_name="app.config['PREDICTION_CONFIDENCE_THRESHOLD']",
            )
        else:
            self.confidence_threshold = validate_confidence_threshold(
                os.getenv("PREDICTION_CONFIDENCE_THRESHOLD"),
                default=DEFAULT_CONFIDENCE_THRESHOLD,
                source_name="PREDICTION_CONFIDENCE_THRESHOLD",
            )

        self.device = device or os.getenv("PREDICTOR_DEVICE", DEFAULT_DEVICE)
        self.guidance_service = guidance_service or GuidanceService()
        self._model: Optional[AutoModelForImageClassification] = None
        self._processor: Optional[MobileNetV2ImageProcessorPil] = None

        # Constrain PyTorch thread pool to 1 thread to avoid memory inflation from OpenMP/MKL
        # on constrained cloud instances (e.g. Render 512 MB memory limit)
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

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

            # Freeze model parameters to disable autograd tracking overhead
            for param in model.parameters():
                param.requires_grad = False

            model.eval()
            model.to(self.device)
            self._model = model
            self._processor = create_image_processor()

            # Clean up temporary deserialization dictionaries and garbage
            gc.collect()
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
        lang: str = "en",
        screening_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes genuine inference on a validated PIL Image with conservative
        pre-inference quality screening and crop mismatch protection.

        Args:
            image: A valid PIL Image instance.
            selected_crop: Optional user-selected crop identifier (e.g. 'tomato', 'maize').
            lang: Localized guidance language ('en' or 'yo').
            screening_options: Optional parameter overrides for image quality screener.

        Returns:
            dict containing class index, label, selected_crop, predicted_crop, crop_match,
            raw model score, assessment status, safety disclaimers, screening details,
            and localized agronomic guidance.

        Raises:
            ModelLoadError: If model cannot be loaded.
            InferenceExecutionError: If forward pass or tensor transformation fails.
        """
        normalized_selected_crop = (
            self.guidance_service.normalize_crop(selected_crop)
            if selected_crop
            else None
        )

        # 1. Conservative pre-inference image quality and leaf presence screening
        screening_opts = dict(screening_options or {})
        if "check_leaf_presence" not in screening_opts and "min_vegetation_ratio" not in screening_opts:
            screening_opts["check_leaf_presence"] = True

        try:
            screening_res = screen_image_quality(image, **screening_opts)
        except Exception as e:
            logger.error(
                f"Image quality screening encountered an unexpected error: {e}",
                exc_info=True,
            )
            fallback_guidance = self.guidance_service.get_unreliable_assessment_fallback(lang=lang)
            return {
                "assessment_status": "unreliable_image_quality",
                "model_name": self.model_name,
                "selected_crop": normalized_selected_crop or selected_crop,
                "predicted_crop": None,
                "crop_match": None,
                "class_index": None,
                "class_label": None,
                "confidence": None,
                "confidence_threshold": self.confidence_threshold,
                "is_provisional_threshold": True,
                "screening_details": {
                    "is_acceptable": False,
                    "rejection_code": "screening_error",
                    "message": "Image quality screening encountered an internal error.",
                    "metrics": {},
                },
                "disclaimer": (
                    "Image quality screening could not reliably assess this photograph. "
                    "Inference was bypassed for safety. "
                    "Please capture a clear, well-lit photo of a single leaf."
                ),
                "guidance": fallback_guidance,
            }

        # If rejected by screening, prevent model inference and suppress treatment guidance
        if not screening_res.is_acceptable:
            rejection_code = screening_res.rejection_code
            if rejection_code in ("not_a_leaf", "no_leaf_detected"):
                assessment_status = "not_a_leaf"
                fallback_guidance = self.guidance_service.get_not_a_leaf_fallback(lang=lang)
                disclaimer = (
                    "A kò rí ewé irúgbìn tó ṣeé wò nínú àwòrán yìí. Ìmọ̀ ẹ̀rọ dá àwọn ohun tí kì í ṣe ewéko tàbí àìtó àmì ewé mọ̀. Jọ̀wọ́ ya àwòrán ewé kan ṣoṣo ní tòsí kedere."
                    if lang == "yo"
                    else "No assessable crop leaf was detected in this photograph. Automated screening detected non-plant objects or insufficient leaf tissue. Inference was bypassed for safety. Please upload a clear photo of a single crop leaf."
                )
            else:
                assessment_status = "unreliable_image_quality"
                fallback_guidance = self.guidance_service.get_unreliable_assessment_fallback(lang=lang)
                disclaimer = (
                    (
                        f"Àyẹ̀wò dídára àwòrán kọ àwòrán yìí ({screening_res.rejection_code}): "
                        f"{screening_res.message} A kò ṣe àyẹ̀wò fún ààbò rẹ. "
                        "Jọ̀wọ́ ya àwòrán ewé kan ṣoṣo tó mọ́lẹ̀ kedere."
                    )
                    if lang == "yo"
                    else (
                        f"Image quality screening rejected this photograph ({screening_res.rejection_code}): "
                        f"{screening_res.message} Inference was bypassed for safety. "
                        "Please capture a clear, well-lit photo of a single leaf against a plain background."
                    )
                )

            return {
                "assessment_status": assessment_status,
                "model_name": self.model_name,
                "selected_crop": normalized_selected_crop or selected_crop,
                "predicted_crop": None,
                "crop_match": None,
                "class_index": None,
                "class_label": None,
                "confidence": None,
                "confidence_threshold": self.confidence_threshold,
                "is_provisional_threshold": True,
                "screening_details": screening_res.to_dict(),
                "disclaimer": disclaimer,
                "guidance": fallback_guidance,
            }

        # Check if selected crop was provided but is unsupported
        if selected_crop is not None and not self.guidance_service.is_crop_supported(selected_crop):
            unsupported_guidance = self.guidance_service.get_unsupported_crop_fallback(
                crop=selected_crop, lang=lang
            )
            return {
                "assessment_status": "unsupported_crop",
                "model_name": self.model_name,
                "selected_crop": selected_crop,
                "predicted_crop": None,
                "crop_match": False,
                "class_index": None,
                "class_label": None,
                "confidence": None,
                "confidence_threshold": self.confidence_threshold,
                "is_provisional_threshold": True,
                "screening_details": screening_res.to_dict(),
                "disclaimer": (
                    f"Irúgbìn '{selected_crop}' kò tíì wà nínú àkójọ CropScan AI. A dáwọ́ ìtọ́sọ́nà dúró láti dènà ewu lílo oògùn tí kò tọ́."
                    if lang == "yo"
                    else f"Crop '{selected_crop}' is not currently supported in CropScan AI. Guidance is withheld to prevent chemical misapplication."
                ),
                "guidance": unsupported_guidance,
            }

        if not self.is_model_loaded:
            self.load_model()

        # Ensure image is in RGB mode (3 channels)
        if image.mode != "RGB":
            image = image.convert("RGB")

        try:
            # 2. Preprocess image
            inputs = self._processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            # 3. Genuine inference forward pass
            inference_context = (
                torch.inference_mode()
                if hasattr(torch, "inference_mode")
                else torch.no_grad()
            )
            with inference_context:
                outputs = self._model(pixel_values=pixel_values)
                logits = outputs.logits[0] if hasattr(outputs, "logits") else outputs[0]
                probabilities = torch.nn.functional.softmax(logits, dim=-1)

                top_prob, top_idx = torch.topk(probabilities, 1)
                predicted_class_index = int(top_idx.item())
                confidence_score = float(top_prob.item())

            # 4. Uncertainty & Crop Probability Distribution Analysis
            probs_np = probabilities.detach().cpu().numpy()
            probs_clipped = np.clip(probs_np, 1e-12, 1.0)
            entropy = float(-np.sum(probs_clipped * np.log(probs_clipped)))
            num_cls = len(probs_np)
            max_entropy = float(np.log(num_cls)) if num_cls > 1 else 1.0
            normalized_entropy = round(entropy / max_entropy, 4)

            # Crop-level probability aggregation across all model classes
            crop_probabilities: Dict[str, float] = {}
            for idx_val, p_val in enumerate(probs_np):
                c_name = self.guidance_service.get_crop_for_class(idx_val)
                if c_name:
                    crop_probabilities[c_name] = crop_probabilities.get(c_name, 0.0) + float(p_val)
            crop_probabilities = {k: round(v, 4) for k, v in crop_probabilities.items()}

            # Immediately release intermediate PyTorch tensors and collect garbage
            del inputs, pixel_values, outputs, logits, probabilities
            gc.collect()

        except Exception as e:
            logger.error(f"Inference execution failed: {e}", exc_info=True)
            raise InferenceExecutionError(
                f"Inference execution failed during forward pass: {str(e)}"
            ) from e

        # 5. Resolve class label and biological crop from official mappings
        class_label = self.guidance_service.get_class_label(predicted_class_index)
        predicted_crop = self.guidance_service.get_crop_for_class(predicted_class_index)

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
                "screening_details": screening_res.to_dict(),
                "entropy": normalized_entropy,
                "crop_probabilities": crop_probabilities,
                "disclaimer": (
                    "Model output index is outside the recognized 38-class catalog. "
                    "This output is treated as invalid and inconclusive."
                ),
                "guidance": fallback_guidance,
            }

        # 6. Check for Crop Mismatch
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
                "screening_details": screening_res.to_dict(),
                "entropy": normalized_entropy,
                "crop_probabilities": crop_probabilities,
                "disclaimer": (
                    (
                        f"Àìbáramu irúgbìn wáyé: O fi ewé yìí ránṣẹ́ gẹ́gẹ́ bí '{selected_display}', ṣùgbọ́n àwòṣe náà dá àwọn àmì "
                        f"àrùn mọ̀ tó jọ '{class_label}' ({pred_display}). Yíyan irúgbìn kò fi ìdí rẹ̀ múlẹ̀ pé irúgbìn "
                        "náà jẹ́ bẹ́ẹ̀ gan-an. A dáwọ́ ìtọ́sọ́nà ìtọ́jú dúró láti dènà lílo oògùn tí kò tọ́. "
                        "Jọ̀wọ́ kan sí òṣìṣẹ́ àgbẹ̀ agbègbè rẹ."
                    )
                    if lang == "yo"
                    else (
                        f"Crop mismatch detected: The image was submitted as '{selected_display}', but model visual "
                        f"features resemble '{class_label}' ({pred_display}). User crop selection does not prove "
                        "ground-truth botanical identity. Disease-specific treatment guidance is withheld to prevent "
                        "chemical misapplication. Consult a local agricultural extension officer."
                    )
                ),
                "guidance": mismatch_guidance,
            }

        # 7. Evaluate against provisional confidence threshold
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
                "screening_details": screening_res.to_dict(),
                "entropy": normalized_entropy,
                "crop_probabilities": crop_probabilities,
                "disclaimer": (
                    (
                        f"Ìdánilójú àwòṣe ({confidence_score:.2%}) kéré sí ìwọ̀n tí a là kalẹ̀ "
                        f"({self.confidence_threshold:.2%}). Àbájáde yìí kò tíì ní àrídájú, "
                        "a kò sì gbọdọ̀ lò ó bí àyẹ̀wò àìsàn tó péye."
                    )
                    if lang == "yo"
                    else (
                        f"Model confidence ({confidence_score:.2%}) is below the provisional operational threshold "
                        f"({self.confidence_threshold:.2%}). Raw model scores are not calibrated probabilities of "
                        "correctness. Confidence thresholding cannot reliably detect unsupported crops, non-leaf "
                        "images, or out-of-distribution inputs. User crop selection does not prove botanical identity. "
                        "This result is inconclusive and must not be treated as a confirmed diagnosis."
                    )
                ),
                "guidance": fallback_guidance,
            }

        # 8. Retrieve localized guidance for confident, matching predictions
        guidance_result = self.guidance_service.get_guidance_by_index(
            predicted_class_index, lang=lang
        )

        is_healthy = bool(guidance_result.get("is_healthy", False))
        if is_healthy:
            effective_crop = normalized_selected_crop or predicted_crop
            if self.guidance_service.has_healthy_class(effective_crop):
                assessment_status = "healthy"
            else:
                # Crop lacks a verified healthy class in model (e.g. Citrus) - must never claim healthy
                assessment_status = "inconclusive_low_confidence"
                guidance_result = self.guidance_service.get_unreliable_assessment_fallback(lang=lang)
        else:
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
            "screening_details": screening_res.to_dict(),
            "entropy": normalized_entropy,
            "crop_probabilities": crop_probabilities,
            "disclaimer": (
                (
                    "Àbájáde àyẹ̀wò ìmọ̀ ẹ̀rọ nìkan ni èyí; kì í ṣe àyẹ̀wò àrídájú lórí oko. "
                    "Àwọn nọ́ḿbà àwòṣe kò tíì fìdí múlẹ̀ pípé. "
                    "Yíyan irúgbìn kò fi ìdí rẹ̀ múlẹ̀ pé irúgbìn náà jẹ́ bẹ́ẹ̀ gan-an. "
                    "Máa kan sí àwọn agbóṣẹ́ àgbẹ̀ agbègbè rẹ kí o tó lo oògùn ètò ìtọ́jú."
                )
                if lang == "yo"
                else (
                    "Automated screening result only; not a confirmed agricultural diagnosis. "
                    "Raw model scores are uncalibrated and do not guarantee accuracy. "
                    "User crop selection does not prove botanical identity. "
                    "Always consult local agricultural extension officers before applying treatments."
                )
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
