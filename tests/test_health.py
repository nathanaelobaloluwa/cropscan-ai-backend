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
    Test that unsupported HTTP methods (e.g. POST) return 405 Method Not Allowed.
    """
    response = client.post("/health")
    assert response.status_code == 405
