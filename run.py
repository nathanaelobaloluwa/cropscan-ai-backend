import os
from app import create_app
from app.config import validate_port

# Instantiate the application using the current environment setting
env = os.getenv("FLASK_ENV", "development")
app = create_app(env)

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = validate_port(os.getenv("PORT"))
    debug = app.config.get("DEBUG", True)

    print(f"-> Starting CropScan AI Backend ({env} environment)")
    print(f"-> Serving on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
