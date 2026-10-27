def test_app_starts(app):
    """
    Test that the application factory successfully instantiates the app
    with the testing configuration.
    """
    assert app is not None
    assert app.config["TESTING"] is True


def test_health_endpoint_success(client):
    """
    Test that GET /health returns HTTP 200 OK and expected JSON schema.
    """
    response = client.get("/health")

    # Verify status code
    assert response.status_code == 200

    # Verify headers and response format
    assert response.is_json

    # Verify payload content
    payload = response.get_json()
    assert payload is not None
    assert payload.get("status") == "healthy"
    assert "message" in payload
    assert payload.get("version") == "0.1.0"


def test_health_endpoint_post_not_allowed(client):
    """
    Test that unsupported HTTP methods (e.g. POST /health) return 405 Method Not Allowed with JSON.
    """
    response = client.post("/health")
    assert response.status_code == 405
    assert response.is_json
    body = response.get_json()
    assert body["error"] == "method_not_allowed"
    assert "method is not allowed" in body["message"].lower()


def test_nonexistent_route_returns_404_json(client):
    """
    Test that requesting an unknown route returns HTTP 404 Not Found with standardized JSON.
    """
    response = client.get("/api/nonexistent_endpoint_for_test")
    assert response.status_code == 404
    assert response.is_json
    body = response.get_json()
    assert body["error"] == "not_found"
    assert body["message"] == "The requested resource was not found."


def test_unhandled_server_exception_returns_500_json(app, client):
    """
    Test that an unhandled server exception:
    1. Triggers the app-level 500 error handler returning HTTP 500.
    2. Returns standardized JSON error response.
    3. Does NOT expose internal error details (e.g. PRIVATE_HTTP_ERROR_DETAIL_456).
    """
    from unittest.mock import MagicMock, patch

    secret_detail = "PRIVATE_HTTP_ERROR_DETAIL_456: /internal/db/credentials.vault"
    mock_fn = MagicMock(side_effect=RuntimeError(secret_detail))

    with patch.dict(app.view_functions, {"health.health_check": mock_fn}):
        response = client.get("/health")

    assert response.status_code == 500
    assert response.is_json
    body = response.get_json()
    assert body["error"] == "internal_server_error"
    assert body["message"] == "An internal server error occurred."
    assert "PRIVATE_HTTP_ERROR_DETAIL_456" not in body["message"]
    assert "/internal/db" not in body["message"]


def test_readiness_endpoint_when_model_not_loaded_returns_503(client):
    """
    Test that GET /health/ready returns HTTP 503 and not_ready status
    when the vision model has not yet been loaded into memory.
    """
    from app.services.predictor import get_prediction_service, set_prediction_service
    # Ensure predictor singleton has is_model_loaded=False
    predictor = get_prediction_service()
    original_model = predictor._model
    predictor._model = None
    try:
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.is_json
        body = response.get_json()
        assert body["status"] == "not_ready"
        assert body["model_ready"] is False
        assert "not yet loaded" in body["message"].lower()
    finally:
        predictor._model = original_model


def test_readiness_endpoint_when_model_loaded_returns_200(client):
    """
    Test that GET /health/ready returns HTTP 200 and ready status
    when the vision model is successfully loaded in memory.
    """
    from unittest.mock import MagicMock
    from app.services.predictor import get_prediction_service
    predictor = get_prediction_service()
    original_model = predictor._model
    original_proc = predictor._processor
    predictor._model = MagicMock()
    predictor._processor = MagicMock()
    try:
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.is_json
        body = response.get_json()
        assert body["status"] == "ready"
        assert body["model_ready"] is True
        assert "ready for inference" in body["message"].lower()
    finally:
        predictor._model = original_model
        predictor._processor = original_proc


def test_cors_get_health_allowed_origin(client):
    """Verify that an allowed origin receives Access-Control-Allow-Origin on GET /health."""
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert response.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"


def test_cors_get_health_disallowed_origin(client):
    """Verify that a disallowed origin does NOT receive Access-Control-Allow-Origin."""
    response = client.get("/health", headers={"Origin": "https://malicious-attacker.com"})
    assert response.status_code == 200
    assert response.headers.get("Access-Control-Allow-Origin") is None


def test_cors_preflight_options_allowed_origin(client):
    """Verify that OPTIONS preflight request from allowed origin returns 200 with CORS headers."""
    response = client.options(
        "/api/predict",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
    methods = response.headers.get("Access-Control-Allow-Methods", "")
    assert "POST" in methods


def test_cors_preflight_options_disallowed_origin(client):
    """Verify that OPTIONS preflight request from disallowed origin does NOT receive allow-origin."""
    response = client.options(
        "/api/predict",
        headers={
            "Origin": "https://malicious-attacker.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.headers.get("Access-Control-Allow-Origin") is None

