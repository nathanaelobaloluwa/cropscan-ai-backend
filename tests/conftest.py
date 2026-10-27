import pytest
from app import create_app


@pytest.fixture
def app():
    """
    Create and configure a clean Flask application instance for each test.
    Uses 'testing' configuration where TESTING is set to True.
    """
    app = create_app("testing")
    yield app


@pytest.fixture
def client(app):
    """
    Create a Flask test client.
    Allows simulating HTTP requests without running an actual HTTP server.
    """
    return app.test_client()
