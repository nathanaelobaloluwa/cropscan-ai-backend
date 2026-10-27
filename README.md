# CropScan AI - Backend

CropScan AI is a plant disease detection backend API designed to empower farmers and agricultural stakeholders. Farmers can submit crop leaf images, obtain fast predictions from an open-weight vision model, and access actionable disease treatment and prevention recommendations available in both English and Yoruba.

---

## Architecture & Project Structure

The codebase is organized using Flask's **Application Factory Pattern** and **Blueprints** to ensure modularity, ease of testing, and maintainability across hackathon development phases:

```text
cropscan-ai-backend/
├── app/
│   ├── __init__.py          # Application factory (create_app)
│   ├── config.py            # Environment configurations (dev, test, prod)
│   ├── routes/
│   │   ├── __init__.py      # Blueprint aggregation
│   │   └── health.py        # Health check endpoint (GET /health)
│   ├── services/
│   │   └── __init__.py      # Business logic (model inference, guidance)
│   └── data/
│       ├── __init__.py      # Static dataset package
│       └── README.md        # Documentation for class mappings and translations
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Pytest fixtures (app, client)
│   └── test_health.py       # Health check & startup unit tests
├── .env.example             # Template for local environment variables
├── .gitignore               # Ignored files (venv, env, caches)
├── requirements.txt         # Core dependencies
├── run.py                   # Server entry point
└── README.md                # Project documentation
```

### Why this structure?
- **Application Factory (`create_app`)**: Allows creating isolated app instances with distinct configurations (e.g., testing vs. development).
- **Blueprints (`app/routes`)**: Keeps HTTP endpoints decoupled by feature domain.
- **Dedicated Services (`app/services`)**: Separates domain logic (image processing, model inference, translation lookups) from HTTP request handling.
- **Data Isolation (`app/data`)**: Keeps model class index mappings and bilingual guidance dictionaries organized and versioned without hardcoding into logic.

---

## Prerequisites

- **Python 3.11+** (verified on Python 3.11.9)
- **Git**
- **Windows PowerShell** (or command prompt / bash)

---

## Getting Started

### 1. Clone the Repository
```powershell
git clone https://github.com/nathanaelobaloluwa/cropscan-ai-backend.git
cd cropscan-ai-backend
```

### 2. Activate the Virtual Environment

On Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

> **Note**: If PowerShell displays an execution policy error, enable local script execution for the current user once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. Install Dependencies

Install the core dependencies:
```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a local `.env` file from the provided template:
```powershell
Copy-Item .env.example .env
```

---

## Running the Server

Start the Flask development server:
```powershell
python run.py
```

The server will start locally:
```text
-> Starting CropScan AI Backend (development environment)
-> Serving on http://127.0.0.1:5000
```

---

## Running the Automated Tests

Run the test suite using `pytest`:
```powershell
pytest -v
```

All tests will run against an isolated test client without needing a live network connection or model weights.

---

## Available Endpoints

### 1. Health Check

Checks the status and availability of the CropScan AI backend service.

- **URL**: `/health`
- **Method**: `GET`
- **Response Format**: `application/json`
- **Status Code**: `200 OK`

#### Example Request (PowerShell)
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/health" -Method Get
```

#### Example Request (curl)
```bash
curl -X GET http://127.0.0.1:5000/health
```

#### Example Response Body
```json
{
  "status": "healthy",
  "message": "CropScan AI Backend service is running",
  "version": "0.1.0"
}
```
