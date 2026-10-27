"""
CropScan AI Services Package.

This package houses backend business logic and service modules:
- GuidanceService: Resolving model predictions to verified agronomic guidance in English and Yoruba.
- ImageValidator: In-memory image validation, format verification, and security checks.
"""

from app.services.guidance import GuidanceService
from app.services.image_validator import (
    ImageValidationError,
    validate_and_load_image,
    validate_image_upload,
)
from app.services.predictor import (
    InferenceExecutionError,
    ModelLoadError,
    PredictionError,
    PredictionService,
    get_prediction_service,
    set_prediction_service,
)

__all__ = [
    "GuidanceService",
    "ImageValidationError",
    "InferenceExecutionError",
    "ModelLoadError",
    "PredictionError",
    "PredictionService",
    "get_prediction_service",
    "set_prediction_service",
    "validate_and_load_image",
    "validate_image_upload",
]

