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

