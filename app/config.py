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
