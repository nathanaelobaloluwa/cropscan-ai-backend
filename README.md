# CropScan AI - Backend

CropScan AI is a plant disease detection and localized agronomic advisory backend API designed to empower smallholder farmers and agricultural stakeholders in West Africa. Farmers can submit leaf photographs, select their target crop, obtain fast inference from an open-weight vision model, and access actionable disease treatment and prevention recommendations available in both English and Yoruba.

---

## 1. Architecture & Project Structure

The codebase is organized using Flask's **Application Factory Pattern** and **Blueprints** to ensure modularity, maintainability, clean separation of concerns, and isolated automated testing:

```text
cropscan-ai-backend/
├── app/
│   ├── __init__.py              # Application factory (create_app), error handlers, and production safeguards
│   ├── config.py                # Environment configurations, port validation, secret hardening & threshold bounds
│   ├── routes/
│   │   ├── __init__.py          # Blueprint aggregation
│   │   ├── health.py            # Health check endpoint (GET /health)
│   │   └── predict.py           # Image upload, crop validation & prediction endpoint (POST /api/predict)
│   ├── services/
│   │   ├── __init__.py          # Service layer exports
│   │   ├── guidance.py          # GuidanceService (English & Yoruba crop/disease resolution)
│   │   ├── image_validator.py   # In-memory image decoding, format verification & security guards
│   │   └── predictor.py         # PredictionService (MobileNetV2 inference, threshold & guidance integration)
│   └── data/
│       ├── __init__.py          # Static dataset package
│       ├── class_indices.json   # Verified 38-class model index mapping
│       ├── class_to_crop.json   # Explicit 38-class biological crop family mapping
│       ├── disease_info_en.json # Agronomic guidance in English
│       ├── disease_info_yo.json # Agronomic guidance in Yoruba (draft review)
│       └── README.md            # Knowledge base architecture & caveats
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures (clean app and test client)
│   ├── test_config.py           # Configuration validation, port parsing & production secret unit/integration tests
│   ├── test_guidance.py         # Knowledge base retrieval, crop mapping & GuidanceService unit tests
│   ├── test_health.py           # Health endpoint, routing, 404, 405, and 500 error handler tests
│   ├── test_predict.py          # Endpoint integration, crop selection, upload security & error handling tests
│   └── test_predictor.py        # ML service unit tests, boundary tests & genuine inference tests
├── .env.example                 # Documented template for local environment variables
├── .gitignore                   # Version control ignore rules (venv, .env, caches)
├── requirements.txt             # Pinned core dependencies (Flask, Pillow, PyTorch, Transformers, pytest)
├── run.py                       # Local development server entry point with validated port handling
└── README.md                    # Project documentation
```

### Decoupled Data Principles
- **Model Class IDs**: Exact label produced by the vision model (e.g., `"Tomato with Early Blight"`).
- **Disease IDs**: Standardized, language-agnostic machine slugs (e.g., `"tomato_early_blight"`).
- **Display Names**: Farmer-friendly localized titles (e.g., `"Early Blight"` or `"Àrùn Èérún Tòmátò"`).
- **Guidance Records**: Conservative cultural and preventive measures referencing established agricultural bodies (FAO, IITA, CIMMYT, CIP, WorldVeg).

---

## 2. Machine Learning Vision Model & Inference

### Model Specifications
- **Model Identifier**: `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`
- **Architecture**: MobileNetV2 for Image Classification (`MobileNetV2ForImageClassification`)
- **Base Architecture**: `google/mobilenet_v2_1.0_224` (Apache 2.0 license)
- **Training Dataset**: Kaggle New Plant Diseases Dataset (PlantVillage, CC0 Public Domain)
- **Model Size**: ~14.2 MB (`model.safetensors`, 314 weight tensors)
- **Default Device**: CPU inference (`PREDICTOR_DEVICE=cpu`)
- **Singleton Lifecycle**: Model weights and image processor are loaded once into memory on service initialization or first request, preventing per-request reloading overhead.

### Verified Preprocessing Pipeline
CropScan AI reproduces the exact preprocessing pipeline from the official model configuration without duplicate normalization:
1. **Shortest-edge resize**: 256 pixels
2. **Resampling method**: Bilinear interpolation (`PIL.Image.BILINEAR`)
3. **Center crop**: 224 × 224 pixels
4. **Pixel rescaling**: Multiplied by `1 / 255.0` (`0.00392156862745098`)
5. **Channel normalization**: Mean `[0.5, 0.5, 0.5]` and standard deviation `[0.5, 0.5, 0.5]` per RGB channel
6. **Backend Implementation**: Powered by `MobileNetV2ImageProcessorPil` (Pillow-native backend, avoiding the heavy `torchvision` dependency).

### Confidence Thresholding & Safeguards
- **Provisional Threshold**: Defaults to `0.40` (configurable via `PREDICTION_CONFIDENCE_THRESHOLD`).
- **Hardened Validation**: Threshold configuration is strictly validated to a finite float in `[0.0, 1.0]`. Missing, empty, non-numeric (`"abc"`), `NaN`, `Infinity`, `-Infinity`, or out-of-range values log a descriptive warning and safely fall back to `0.40`.
- **Provisional Baseline**: This threshold is strictly provisional for screening in this prototype and has not been validated through agricultural field trials.
- **Uncalibrated Scores**: Raw model scores are computed via Softmax over 38 logits. They represent relative model activation among trained classes, **not calibrated probabilities** that the diagnosis is correct.
- **Not a Confirmed Diagnosis**: CropScan AI provides automated screening aids, not a botanical or phytopathological diagnosis. Farmers must consult local agricultural extension officers before applying chemical treatments.
- **Out-of-Distribution Vulnerability**: Confidence thresholding alone **cannot reliably detect** unsupported crops (e.g., cassava, yam), non-leaf images, or arbitrary out-of-distribution inputs. An image of a shoe or weeds may still yield an arbitrary prediction if forced through closed-set softmax classification.

---

## 3. Initial Prototype Scope (Nigerian-Relevant Crops)

The initial knowledge base focuses on high-priority Nigerian staple crops supported by the 38-class model:
- **Maize / Corn (Àgbàdo)**: Cercospora / Gray Leaf Spot, Common Rust, Northern Leaf Blight, and Healthy.
- **Tomato (Tòmátò)**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Two-spotted Spider Mite, Target Spot, Yellow Leaf Curl Virus (TYLCV), Mosaic Virus (ToMV), and Healthy.
- **Bell Pepper (Ata Tàtàṣé)**: Bacterial Spot and Healthy.
- **Citrus / Orange (Ọsàn)**: Citrus Greening (Huanglongbing).
- **Potato (Ànàmọ́ Gẹ̀ẹ́sì / Pọtétò)**: Early Blight, Late Blight, and Healthy.

> [!NOTE]
> **Excluded Crops**: Staple African crops such as **Cassava** and **Yam** are **NOT** present in the 38-class PlantVillage dataset and are strictly excluded from model mappings. Any future cassava support will require a dedicated secondary model (such as Google CropNet Cassava).

---

## 4. Prerequisites

- **Python 3.11+** (verified on Python 3.11.9)
- **Git**
- **Windows PowerShell** (or command prompt / bash)

---

## 5. Local Setup & Installation

### 1. Clone the Repository
```powershell
git clone https://github.com/nathanaelobaloluwa/cropscan-ai-backend.git
cd cropscan-ai-backend
```

### 2. Create and Activate Virtual Environment
On Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> **Note**: If PowerShell displays an execution policy error, enable script execution for the current user once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. Install Dependencies
Install the pinned dependencies from `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the provided `.env.example` file to `.env`:
```powershell
Copy-Item .env.example .env
```

---

## 6. Running the Local Development Server

Start the local server using `run.py`:
```powershell
python run.py
```

Or using Flask's CLI:
```powershell
flask --app app run --debug
```

Expected startup output:
```text
-> Starting CropScan AI Backend (development environment)
-> Serving on http://127.0.0.1:5000
```

---

## 7. Running the Automated Test Suite

Run the complete regression test suite using `pytest`:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

All **91 automated tests** execute in `~6s` with 100% pass rate:
- `tests/test_config.py` (30 tests): Port validation, confidence threshold validation, production secret validation, and app factory config loading.
- `tests/test_guidance.py` (17 tests): Knowledge base retrieval, 38-class biological mapping, crop normalization and alias resolution, Yoruba/English consistency, and safe fallbacks.
- `tests/test_health.py` (5 tests): Health endpoint, startup checks, 404 Not Found, 405 Method Not Allowed, and 500 server error JSON response standardization.
- `tests/test_predict.py` (22 tests): End-to-end `POST /api/predict` tests, required crop selection, unsupported crop rejection, format validation, 10 MB payload ceiling, language localization, and error handling.
- `tests/test_predictor.py` (17 tests): Exact preprocessor verification, real MobileNetV2 weight loading, genuine inference on real sample images, threshold boundaries, crop mismatch protection, and exception handling.

To run focused test suites:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_config.py -v
.\.venv\Scripts\python.exe -m pytest tests/test_predict.py -v
```

---

## 8. Environment Variable Reference

The application uses `python-dotenv` to load environment variables from `.env` if present. Supported configuration variables:

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `FLASK_APP` | string | `run.py` | Entry point module for Flask CLI. |
| `FLASK_ENV` | string | `development` | Environment mode (`development`, `testing`, `production`). In `production`, strict `SECRET_KEY` validation is enforced. |
| `HOST` | string | `127.0.0.1` | Network interface to bind. Use `0.0.0.0` in Docker or PaaS environments. |
| `PORT` | integer | `5000` | TCP port (`1` to `65535`). Validated at startup; invalid or out-of-range values log a warning and safely fall back to `5000`. |
| `SECRET_KEY` | string | `dev-secret-key-cropscan` | Cryptographic secret. In `production`, this **must** be set to a custom, non-default secret key. If missing or default, startup aborts. |
| `CROPSCAN_MODEL_NAME` | string | `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` | Hugging Face model repository or local directory path. |
| `PREDICTOR_DEVICE` | string | `cpu` | PyTorch inference device (`cpu` or `cuda`). |
| `PREDICTION_CONFIDENCE_THRESHOLD` | float | `0.40` | Screening confidence threshold in `[0.0, 1.0]`. Validated at startup; invalid values fall back to `0.40` with a warning. |

---

## 9. API Reference

### 1. Health Check
Checks the status and process liveness of the CropScan AI backend service.

> [!NOTE]
> **What `/health` Verifies**: This endpoint verifies that the Flask application process is active and responding to HTTP requests. It **does NOT** execute model inference, benchmark GPU availability, or connect to external databases.

- **URL**: `/health`
- **Method**: `GET`
- **Response Format**: `application/json`
- **Status Code**: `200 OK`

#### Example Request
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/health" -Method Get
```

#### Example Response Body
```json
{
  "status": "healthy",
  "message": "CropScan AI Backend service is running",
  "version": "0.1.0"
}
```

---

### 2. Leaf Image Upload & Prediction
Receives a crop leaf photograph and target crop selection via multipart form-data, performs raster validation, executes MobileNetV2 vision inference, protects against crop mismatches, and returns localized agronomic guidance in English or Yoruba.

- **URL**: `/api/predict`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Form Parameters**:
  - `image` *(required, file)*: Image file (JPEG, PNG, or WebP). Maximum file size is **10 MB**.
  - `crop` *(required, string)*: Target crop selection. Valid canonical identifiers and aliases:
    - `tomato` (Tomato)
    - `maize` (Maize / Corn; aliases: `corn`, `agbado`)
    - `bell_pepper` (Bell Pepper; aliases: `bell pepper`, `pepper`, `tatase`)
    - `citrus` (Citrus / Orange; aliases: `orange`, `osan`)
    - `potato` (Irish Potato; aliases: `irish potato`, `poteto`)
  - `lang` *(optional, string)*: Language for response guidance (`en` for English, `yo` for Yoruba). Defaults to `en`.

#### Validation & Safety Safeguards
- **Required Crop Selection**: Enforces explicit crop context from the farmer to prevent cross-family misdiagnosis.
- **Explicit 38-Class Biological Mapping**: Maps all 38 model class indices directly to their biological crop family (`app/data/class_to_crop.json`).
- **Crop Mismatch Protection**: If the model's top predicted class belongs to a different crop family than the user's selection (e.g., user selects `tomato` but model predicts `Corn Common Rust` or `Apple Scab`), CropScan AI returns `assessment_status: "crop_mismatch"` and `crop_match: false`.
- **Withheld Treatment Guidance**: Whenever a crop mismatch occurs, disease treatment recommendations are strictly withheld (`is_healthy: null`, `data: null`) to prevent dangerous chemical misapplication.
- **Safe Confidence Thresholding**: Predictions matching the selected crop but scoring below the validated threshold (`0.40`) return `inconclusive_low_confidence` and advisory guidance.
- **In-Memory Decoding & Security**: Images are verified against true raster byte headers and decoded strictly in memory with decompression bomb guards (10,000 px max dimension).

#### Example Request (PowerShell)
```powershell
$form = @{
    image = Get-Item "C:\path\to\leaf_sample.jpg"
    crop  = "tomato"
    lang  = "en"
}
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict" -Method Post -Form $form
```

#### Example Response (Genuine Prediction - Confident with Matching Crop)
```json
{
  "status": "success",
  "selected_crop": "tomato",
  "language": "en",
  "image_metadata": {
    "format": "JPEG",
    "width": 800,
    "height": 600,
    "color_mode": "RGB",
    "size_bytes": 104520
  },
  "prediction": {
    "assessment_status": "confident",
    "crop_match": true,
    "selected_crop": "tomato",
    "predicted_crop": "tomato",
    "model_name": "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
    "class_index": 29,
    "class_label": "Tomato with Early Blight",
    "confidence": 0.8845,
    "confidence_threshold": 0.40,
    "is_provisional_threshold": true,
    "disclaimer": "Automated screening result only; not a confirmed agricultural diagnosis. User crop selection does not prove or guarantee botanical crop identity. Raw model scores are uncalibrated. Always consult local agricultural extension officers before applying treatments.",
    "guidance": {
      "status": "success",
      "class_index": 29,
      "model_class_id": "Tomato with Early Blight",
      "language": "en",
      "is_healthy": false,
      "message": "Guidance record retrieved successfully.",
      "data": {
        "disease_id": "tomato_early_blight",
        "crop": "Tomato",
        "display_name": "Early Blight",
        "scientific_name": "Alternaria solani",
        "severity": "moderate",
        "symptoms": [
          "Dark brown to black spots with concentric rings (target-like pattern) on older leaves.",
          "Yellow halo surrounding leaf lesions.",
          "Premature defoliation starting from the lower canopy upwards."
        ],
        "treatment": [
          "Prune and destroy infected lower leaves immediately to prevent fungal spore dispersal.",
          "Apply approved copper-based or bio-fungicide sprays during high humidity according to manufacturer instructions."
        ],
        "prevention": [
          "Practice 3-year crop rotation avoiding Solanaceous crops (potato, pepper, eggplant).",
          "Water at the base of plants via drip or furrow; avoid overhead watering that wets foliage.",
          "Apply clean straw or plastic mulch to prevent soil-borne fungal spores from splashing onto lower leaves."
        ],
        "expert_review_required": true,
        "sources": [
          "IITA Plant Health Guide: Tomato Pest & Disease Management in West Africa",
          "FAO Integrated Pest Management: Solanaceous Crops"
        ]
      }
    }
  }
}
```

#### Example Response (Crop Mismatch - Treatment Guidance Withheld)
```json
{
  "status": "success",
  "selected_crop": "tomato",
  "language": "en",
  "image_metadata": {
    "format": "JPEG",
    "width": 800,
    "height": 600,
    "color_mode": "RGB",
    "size_bytes": 98200
  },
  "prediction": {
    "assessment_status": "crop_mismatch",
    "crop_match": false,
    "selected_crop": "tomato",
    "predicted_crop": "apple",
    "model_name": "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
    "class_index": 0,
    "class_label": "Apple with Apple Scab",
    "confidence": 0.9120,
    "confidence_threshold": 0.40,
    "is_provisional_threshold": true,
    "disclaimer": "Automated screening result only; not a confirmed agricultural diagnosis. User crop selection does not prove or guarantee botanical crop identity. Raw model scores are uncalibrated. Always consult local agricultural extension officers before applying treatments.",
    "guidance": {
      "status": "crop_mismatch",
      "class_index": null,
      "model_class_id": "Apple with Apple Scab",
      "language": "en",
      "is_healthy": null,
      "message": "Crop mismatch detected. You selected 'tomato', but the model predicted characteristics resembling 'apple' ('Apple with Apple Scab'). Because of this mismatch, disease-specific treatment recommendations are withheld for safety. Please verify that the selected crop matches the photo, ensure only a single clear leaf is photographed, or consult a local agricultural extension specialist.",
      "data": null
    }
  }
}
```

#### Example Response (Inconclusive Low Confidence)
```json
{
  "status": "success",
  "selected_crop": "tomato",
  "language": "en",
  "image_metadata": {
    "format": "JPEG",
    "width": 256,
    "height": 256,
    "color_mode": "RGB",
    "size_bytes": 12450
  },
  "prediction": {
    "assessment_status": "inconclusive_low_confidence",
    "crop_match": true,
    "selected_crop": "tomato",
    "predicted_crop": "tomato",
    "model_name": "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
    "class_index": 30,
    "class_label": "Tomato with Late Blight",
    "confidence": 0.2168,
    "confidence_threshold": 0.40,
    "is_provisional_threshold": true,
    "disclaimer": "Model confidence (21.68%) is below the provisional operational threshold (40.00%). User crop selection does not prove or guarantee botanical crop identity. Raw model scores are not calibrated probabilities of correctness. Confidence thresholding cannot reliably detect unsupported crops, non-leaf images, or out-of-distribution inputs. This result is inconclusive and must not be treated as a confirmed diagnosis.",
    "guidance": {
      "status": "unreliable_assessment",
      "class_index": null,
      "model_class_id": null,
      "language": "en",
      "is_healthy": null,
      "message": "The image could not be reliably assessed. Please capture a clear, well-lit photo of a single leaf against a plain background, or consult your local agricultural extension service for on-field diagnosis.",
      "data": null
    }
  }
}
```

### Standard Error Responses
All API error responses use a consistent JSON format: `{"error": "<error_code>", "message": "<descriptive_message>"}`.

| Status Code | Error Code | Description / Example |
| :---: | :--- | :--- |
| **400** | `missing_crop` | `{"error": "missing_crop", "message": "Field 'crop' is required. Supported crops are: 'tomato', 'maize', 'bell_pepper', 'citrus', 'potato'."}` |
| **400** | `unsupported_crop` | `{"error": "unsupported_crop", "message": "Unsupported crop 'cassava'. Supported crops are: 'tomato', 'maize', 'bell_pepper', 'citrus', 'potato'."}` |
| **400** | `missing_image` | `{"error": "missing_image", "message": "No image file provided in multipart form data (field 'image')."}` |
| **400** | `empty_image` | `{"error": "empty_image", "message": "The uploaded image file is empty or has no filename."}` |
| **400** | `unsupported_format` | `{"error": "unsupported_format", "message": "Unsupported image format 'GIF'. CropScan AI accepts JPEG, PNG, and WebP images only."}` |
| **400** | `invalid_language` | `{"error": "invalid_language", "message": "Supported languages are 'en' (English) and 'yo' (Yoruba)."}` |
| **400** | `corrupt_image` | `{"error": "corrupt_image", "message": "Uploaded image file is truncated or corrupted."}` |
| **400** | `spoofed_mime_type` | `{"error": "spoofed_mime_type", "message": "Image content does not match reported MIME type."}` |
| **404** | `not_found` | `{"error": "not_found", "message": "The requested resource was not found."}` |
| **405** | `method_not_allowed` | `{"error": "method_not_allowed", "message": "The HTTP method is not allowed for the requested URL."}` |
| **413** | `payload_too_large` | `{"error": "payload_too_large", "message": "Uploaded image exceeds the maximum permitted size of 10 MB."}` |
| **500** | `unexpected_inference_error` | `{"error": "unexpected_inference_error", "message": "An unexpected error occurred during prediction."}` |
| **500** | `internal_server_error` | `{"error": "internal_server_error", "message": "An internal server error occurred."}` |
| **503** | `model_not_loaded` | `{"error": "model_not_loaded", "message": "Vision model is not loaded or unavailable."}` |

---

## 10. Production Deployment Considerations

### Development vs. Production WSGI Servers
`python run.py` executes Flask's built-in development server. This server is intended strictly for local development and debugging; it is single-process, lacks request concurrency controls, and is not hardened against connection exhaustion attacks.

In production environments, a dedicated WSGI server should be used to serve the application factory:

#### Proposed Production Deployment on Linux / Containers (Gunicorn)
```bash
# Proposed deployment step (requires pip install gunicorn):
export FLASK_ENV=production
export SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export PORT=5000

gunicorn "app:create_app('production')" \
    --bind 0.0.0.0:${PORT:-5000} \
    --workers 2 \
    --threads 2 \
    --timeout 120
```

#### Proposed Production Deployment on Windows (Waitress)
```powershell
# Proposed deployment step (requires pip install waitress):
$env:FLASK_ENV = "production"
$env:SECRET_KEY = [System.Guid]::NewGuid().ToString("N")
$env:PORT = "5000"

waitress-serve --port=5000 "app:create_app('production')"
```

> [!NOTE]
> **Dependency Status**: `gunicorn` and `waitress` are deployment-time servers and are not bundled in `requirements.txt` to keep the local prototype environment lightweight. The commands above are proposed deployment workflows.

### Production Security & Configuration Requirements
1. **Mandatory Production `SECRET_KEY`**:
   - In production mode (`FLASK_ENV=production` or `create_app('production')`), the application strictly validates that `SECRET_KEY` is provided, non-empty, and does not match the default development key (`dev-secret-key-cropscan`).
   - If `SECRET_KEY` is missing or insecure, application startup raises `ValueError` immediately.
   - Never commit `.env` files or hardcode production keys in the codebase. Supply `SECRET_KEY` via environment variables in your hosting provider's dashboard (Render, Railway, Fly.io, AWS Secrets Manager, etc.).
2. **Port Binding**:
   - `PORT` is validated by `validate_port()`. Invalid values safely fall back to `5000`.
   - In containerized environments, ensure the server binds to `HOST=0.0.0.0` rather than `127.0.0.1` so traffic can reach the container.
3. **Health Check Probes**:
   - The `GET /health` endpoint can be configured as an HTTP liveness probe on cloud platforms and load balancers.
4. **Verification Disclaimer**:
   - External cloud deployment has not been executed or verified as part of this local repository.

---

## 11. Model & Safety Limitations

1. **Closed-Set Vision Model Constraints**:
   - MobileNetV2 uses a closed-set softmax classification head over 38 classes.
   - Any image submitted to the model is forced to distribute probability across those 38 classes.
   - If an unsupported crop (e.g., cassava, yam) or non-leaf image is uploaded with a supported crop selection, a false confident prediction can still occur.
2. **Uncalibrated Confidence Scores**:
   - Softmax probabilities reflect relative activation, not calibrated statistical probability of diagnosis correctness.
3. **Crop Selection is Not Botanical Proof**:
   - User-selected crop indicates farmer context; it does not scientifically prove botanical identity.
4. **Informational Guidance Only**:
   - All disease treatment recommendations are educational and conservative. Farmers must consult qualified agricultural extension officers before purchasing or applying commercial fungicides/pesticides.
5. **Yoruba Translation Review Status**:
   - All Yoruba guidance records are tagged with `"translation_review_status": "draft_translation_pending_native_expert_review"` and `"expert_review_required": true`.
6. **Model Licensing**:
   - The fine-tuned weights (`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`) carry a license tag of **`other`** in Hugging Face metadata.
   - Commercial rights have not been verified; formal author clearance or re-training from CC0 weights is required prior to commercial release.

---

## 12. Future Work (Planned)

The following items are planned improvements and not yet implemented:
- **Upstream Crop Verification / Anomaly Filter**: Lightweight binary classifier or open-set out-of-distribution detector to reject non-leaf images and unsupported species before disease inference.
- **African Staple Model Integration**: Dedicated cassava and yam disease classification models (e.g. Google CropNet Cassava).
- **Official Docker Packaging**: Multi-stage `Dockerfile` with Waitress/Gunicorn for single-command container deployment.
- **Field Validation Trials**: Real-world validation on photographs taken under direct African field conditions (direct sun, varied soil, complex backgrounds).
- **Native Speaker Validation**: Comprehensive review and refinement of Yoruba agricultural terminology with regional agronomists.
