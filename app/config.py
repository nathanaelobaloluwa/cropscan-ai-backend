import os
from dotenv import load_dotenv

# Load variables from .env if present
load_dotenv()


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
    # Provisional confidence threshold (0.40) - documented as provisional for prototype
    PREDICTION_CONFIDENCE_THRESHOLD = float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.40"))
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
