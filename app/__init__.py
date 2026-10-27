import os
from flask import Flask, jsonify
from werkzeug.exceptions import RequestEntityTooLarge

from app.config import config_by_name
from app.routes.health import health_bp
from app.routes.predict import predict_bp


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
    app.register_blueprint(predict_bp)

    # Global error handlers
    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_entity_too_large(e):
        return jsonify({
            "error": "payload_too_large",
            "message": "Uploaded file exceeds the maximum permitted size of 10 MB.",
        }), 413

    return app
