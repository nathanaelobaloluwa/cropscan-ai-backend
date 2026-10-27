import os
from flask import Flask
from app.config import config_by_name
from app.routes.health import health_bp


def create_app(config_name=None):
    """
    Application factory for CropScan AI Backend.

    Args:
        config_name (str, optional): Name of configuration to load
            ('development', 'testing', 'production'). Defaults to None,
            which resolves to the FLASK_ENV environment variable or 'development'.

    Returns:
        Flask: The configured Flask application instance.
    """
    if config_name is None:
        config_name = os.getenv("FLASK_ENV", "development")

    app = Flask(__name__)

    # Load configuration
    config_class = config_by_name.get(config_name, config_by_name["default"])
    app.config.from_object(config_class)

    # Register blueprints
    app.register_blueprint(health_bp)

    return app
