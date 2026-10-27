"""
CropScan AI Services Package.

This package houses backend business logic and service modules:
- PredictionService: Preprocessing leaf images and running model inference.
- GuidanceService: Retrieving disease descriptions, treatment steps, and prevention
  recommendations in both English and Yoruba.

Keeping business logic inside dedicated services decouples domain logic
from HTTP routing and request parsing, making code easy to test and maintain.
"""
