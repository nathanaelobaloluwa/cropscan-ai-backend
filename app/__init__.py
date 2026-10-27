import os
from flask import Flask, jsonify
from werkzeug.exceptions import (
    HTTPException,
    InternalServerError,
    MethodNotAllowed,
    NotFound,
    RequestEntityTooLarge,
)

from app.config import (
    ProductionConfig,
    config_by_name,
    validate_production_secret,
)
from app.routes.health import health_bp
from app.routes.predict import predict_bp


def create_app(config_name=None):
    """
    Application factory for CropScan AI Backend.

    Args:
        config_name (str or type, optional): Name of configuration to load
            ('development', 'testing', 'production') or a Config class.
            Defaults to None, which resolves to the FLASK_ENV environment
            variable or 'development'.

    Returns:
        Flask: The configured Flask application instance.
    """
    if config_name is None:
        config_name = os.getenv("FLASK_ENV", "development")

    app = Flask(__name__)

    # Load configuration
    if isinstance(config_name, type):
        config_class = config_name
    else:
        config_class = config_by_name.get(config_name, config_by_name["default"])
    app.config.from_object(config_class)

    # Validate production configuration safeguards
    is_production = (
        config_name == "production"
        or (isinstance(config_class, type) and issubclass(config_class, ProductionConfig))
    )
    if is_production:
        env_secret = os.getenv("SECRET_KEY")
        secret_to_validate = env_secret if env_secret is not None else app.config.get("SECRET_KEY")
        validated_secret = validate_production_secret(secret_to_validate)
        app.config["SECRET_KEY"] = validated_secret

    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(predict_bp)

    # Global error handlers
    @app.errorhandler(NotFound)
    @app.errorhandler(404)
    def handle_not_found(e):
        return jsonify({
            "error": "not_found",
            "message": "The requested resource was not found.",
        }), 404

    @app.errorhandler(MethodNotAllowed)
    @app.errorhandler(405)
    def handle_method_not_allowed(e):
        return jsonify({
            "error": "method_not_allowed",
            "message": "The HTTP method is not allowed for the requested URL.",
        }), 405

    @app.errorhandler(RequestEntityTooLarge)
    @app.errorhandler(413)
    def handle_request_entity_too_large(e):
        return jsonify({
            "error": "payload_too_large",
            "message": "Uploaded file exceeds the maximum permitted size of 10 MB.",
        }), 413

    @app.errorhandler(InternalServerError)
    @app.errorhandler(500)
    def handle_internal_server_error(e):
        return jsonify({
            "error": "internal_server_error",
            "message": "An internal server error occurred.",
        }), 500

    @app.errorhandler(Exception)
    def handle_unhandled_exception(e):
        if isinstance(e, HTTPException):
            return e
        app.logger.error(f"Unhandled server error: {e}", exc_info=True)
        return jsonify({
            "error": "internal_server_error",
            "message": "An internal server error occurred.",
        }), 500

    return app
