import logging
import math
import os
from typing import Any, Optional
from dotenv import load_dotenv

# Load variables from .env if present
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.40
DEFAULT_PORT: int = 5000
DEFAULT_DEV_SECRET_KEY: str = "dev-secret-key-cropscan"
INSECURE_DEV_SECRET_KEYS: set[str] = {
    "dev-secret-key-cropscan",
    "dev-secret-key-cropscan-change-in-production",
}


def validate_port(value: Any, default: int = DEFAULT_PORT) -> int:
    """
    Validates and converts a port setting to an integer in the inclusive range 1..65535.

    Rules:
    - If value is None: returns default (5000) without warning.
    - If value is a boolean or float: logs warning and returns default.
    - If value is an empty or whitespace-only string: logs warning and returns default.
    - If value is a non-integer string (e.g. 'abc', '5000.5'): logs warning and returns default.
    - If value is an integer outside [1, 65535]: logs warning and returns default.
    - Otherwise returns the validated integer port.
    """
    if value is None:
        return default

    # Booleans are subclasses of int in Python; reject them explicitly
    if isinstance(value, bool):
        logger.warning(
            f"Invalid non-integer PORT '{value}'. Falling back to default {default}."
        )
        return default

    # Floats (e.g. 5000.5 or 5000.0) are rejected as invalid port types
    if isinstance(value, float):
        logger.warning(
            f"Invalid non-integer PORT '{value}'. Falling back to default {default}."
        )
        return default

    if isinstance(value, str):
        val_str = value.strip()
        if not val_str:
            logger.warning(
                f"Empty or whitespace-only PORT. Falling back to default {default}."
            )
            return default
        try:
            port_val = int(val_str)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid non-integer PORT '{value}'. Falling back to default {default}."
            )
            return default
    elif isinstance(value, int):
        port_val = value
    else:
        try:
            port_val = int(value)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid non-integer PORT '{value}'. Falling back to default {default}."
            )
            return default

    if not (1 <= port_val <= 65535):
        logger.warning(
            f"PORT '{value}' out of valid range [1, 65535]. Falling back to default {default}."
        )
        return default

    return port_val


def validate_production_secret(secret_key: Optional[str]) -> str:
    """
    Validates that a production secret key is provided, non-empty, non-whitespace,
    and not equal to any known insecure development default.

    Raises:
        ValueError: If secret_key is missing, empty, whitespace-only,
            or equals an insecure development default. Never logs or echoes the secret value.
    """
    if secret_key is None:
        raise ValueError(
            "Production requires a valid, non-default SECRET_KEY environment variable. "
            "No SECRET_KEY was provided."
        )
    if not isinstance(secret_key, str) or not secret_key.strip():
        raise ValueError(
            "Production requires a valid, non-default SECRET_KEY environment variable. "
            "SECRET_KEY cannot be empty or whitespace-only."
        )
    stripped = secret_key.strip()
    if stripped in INSECURE_DEV_SECRET_KEYS or stripped.startswith("dev-secret-key-cropscan"):
        raise ValueError(
            "Production requires a valid, non-default SECRET_KEY environment variable. "
            "The default development key is not permitted in production."
        )
    return stripped


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
    SECRET_KEY = os.getenv("SECRET_KEY", DEFAULT_DEV_SECRET_KEY)
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
    SECRET_KEY = os.getenv("SECRET_KEY")


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
