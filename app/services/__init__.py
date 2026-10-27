"""
CropScan AI Services Package.

This package houses backend business logic and service modules:
- GuidanceService: Resolving model predictions to verified agronomic guidance in English and Yoruba.
- ImageValidator: In-memory image validation, format verification, and security checks.
"""

from app.services.guidance import GuidanceService
from app.services.image_validator import ImageValidationError, validate_image_upload

__all__ = [
    "GuidanceService",
    "ImageValidationError",
    "validate_image_upload",
]
