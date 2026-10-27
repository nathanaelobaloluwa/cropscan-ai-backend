import logging
import math
import os
from typing import Any, Optional
from dotenv import load_dotenv

# Load variables from .env if present
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.40


def validate_confidence_threshold(
    value: Any,
    default: float = DEFAULT_CONFIDENCE_THRESHOLD,
    source_name: str = "PREDICTION_CONFIDENCE_THRESHOLD",
) -> float:
    """
    Validates and converts a confidence threshold to a finite float in [0.0, 1.0].

    Rules:
    - If value is None, returns default (0.40) without warning.
    - If value is an empty or whitespace-only string, logs warning and returns default.
    - If value is non-numeric, logs warning and returns default.
    - If value is NaN, Infinity, or -Infinity (not finite), logs warning and returns default.
    - If value is < 0.0 or > 1.0, logs warning and returns default.
    - Otherwise, returns float(value).
    """
    if value is None:
        return default

    # If it's a string, strip whitespace and check if empty
    if isinstance(value, str):
        val_str = value.strip()
        if not val_str:
            logger.warning(
                f"Empty or whitespace-only {source_name}. Falling back to default {default:.2f}."
            )
            return default
    else:
        val_str = str(value)

    try:
        numeric_val = float(value)
    except (ValueError, TypeError):
        logger.warning(
            f"Invalid non-numeric {source_name} '{value}'. Falling back to default {default:.2f}."
        )
        return default

    if not math.isfinite(numeric_val):
        logger.warning(
            f"Non-finite {source_name} '{value}' (NaN or Infinity). Falling back to default {default:.2f}."
        )
        return default

    if not (0.0 <= numeric_val <= 1.0):
        logger.warning(
            f"{source_name} '{value}' out of valid range [0.0, 1.0]. Falling back to default {default:.2f}."
        )
        return default

    return numeric_val


class Config:
    """Base configuration settings shared across environments."""
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-cropscan")
    TESTING = False
    DEBUG = False
    JSON_SORT_KEYS = False

    # Upload constraints
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB upload ceiling

    # ML Model & Predictor Configuration
    CROPSCAN_MODEL_NAME = os.getenv(
        "CROPSCAN_MODEL_NAME",
        "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    )
    # Provisional confidence threshold (default 0.40) validated to finite float in [0.0, 1.0]
    PREDICTION_CONFIDENCE_THRESHOLD = validate_confidence_threshold(
        os.getenv("PREDICTION_CONFIDENCE_THRESHOLD")
    )
    PREDICTOR_DEVICE = os.getenv("PREDICTOR_DEVICE", "cpu")


class DevelopmentConfig(Config):
    """Configuration for local development."""
    DEBUG = True


class TestingConfig(Config):
    """Configuration for automated unit and integration tests."""
    TESTING = True
    DEBUG = True


class ProductionConfig(Config):
    """Configuration for production deployment."""
    DEBUG = False


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
