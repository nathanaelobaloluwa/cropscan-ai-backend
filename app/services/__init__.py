"""
CropScan AI Services Package.

This package houses backend business logic and service modules:
- GuidanceService: Resolving model predictions to verified agronomic guidance in English and Yoruba.
- PredictionService: Preprocessing leaf images and running model inference (Phase 3).
"""

from app.services.guidance import GuidanceService

__all__ = ["GuidanceService"]
