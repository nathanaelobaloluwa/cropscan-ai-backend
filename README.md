# CropScan AI - Backend

CropScan AI is a plant disease detection backend API designed to empower farmers and agricultural stakeholders. Farmers can submit crop leaf images, obtain fast predictions from an open-weight vision model, and access actionable disease treatment and prevention recommendations available in both English and Yoruba.

---

## Architecture & Project Structure

The codebase is organized using Flask's **Application Factory Pattern** and **Blueprints** to ensure modularity, ease of testing, and maintainability across hackathon development phases:

```text
cropscan-ai-backend/
├── app/
│   ├── __init__.py              # Application factory (create_app)
│   ├── config.py                # Environment configurations (dev, test, prod; 10 MB limit)
│   ├── routes/
│   │   ├── __init__.py          # Blueprint aggregation
│   │   ├── health.py            # Health check endpoint (GET /health)
│   │   └── predict.py           # Image upload & prediction endpoint (POST /api/predict)
│   ├── services/
│   │   ├── __init__.py          # Service layer exports
│   │   ├── guidance.py          # GuidanceService (English & Yoruba resolution)
│   │   └── image_validator.py   # In-memory image decoding, format & security validation
│   └── data/
│       ├── __init__.py          # Static dataset package
│       ├── class_indices.json   # Verified 38-class model index mapping
│       ├── disease_info_en.json # Agronomic guidance in English
│       ├── disease_info_yo.json # Agronomic guidance in Yoruba (draft review)
│       └── README.md            # Knowledge base architecture & caveats
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures (app, client)
│   ├── test_health.py           # Health check & startup unit tests
│   ├── test_guidance.py         # Knowledge base & GuidanceService unit tests
│   └── test_predict.py          # Image upload validation & endpoint tests
├── .env.example                 # Template for local environment variables
├── .gitignore                   # Ignored files (venv, env, caches)
├── requirements.txt             # Core dependencies (Flask, Pillow, pytest)
├── run.py                       # Server entry point
└── README.md                    # Project documentation
```

### Decoupled Data Principles
- **Model Class IDs**: Exact prediction label from the model (e.g., `"Tomato with Early Blight"`).
- **Disease IDs**: Standardized machine slugs (e.g., `"tomato_early_blight"`).
- **Display Names**: Farmer-friendly localized titles (e.g., `"Early Blight"` or `"Àrùn Èérún Tòmátò"`).
- **Guidance Records**: Conservative cultural and preventive measures referencing verified agricultural bodies (FAO, IITA, CIMMYT, CIP, WorldVeg).

---

## Initial Prototype Scope (Nigerian-Relevant Crops)

The initial knowledge base focuses on high-priority Nigerian staple crops supported by the 38-class model:
- **Maize / Corn (Àgbàdo)**: Cercospora / Gray Leaf Spot, Common Rust, Northern Leaf Blight, and Healthy.
- **Tomato (Tòmátò)**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Two-spotted Spider Mite, Target Spot, Yellow Leaf Curl Virus (TYLCV), Mosaic Virus (ToMV), and Healthy.
- **Bell Pepper (Ata Tàtàṣé)**: Bacterial Spot and Healthy.
- **Citrus / Orange (Ọsàn)**: Citrus Greening (Huanglongbing).
- **Potato (Ànàmọ́ Gẹ̀ẹ́sì / Pọtétò)**: Early Blight, Late Blight, and Healthy.

> [!NOTE]
> **Excluded Crops**: Staple African crops such as **Cassava** and **Yam** are **NOT** present in the 38-class PlantVillage dataset and are strictly excluded from model mappings. Any future cassava support will require a dedicated secondary model (such as Google CropNet Cassava).

---

## Model Licensing Uncertainty & Known Limitations

1. **Licensing Status**:
   - The fine-tuned weights (`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`) carry a license metadata tag of **`other`**.
   - While the base architecture (`google/mobilenet_v2_1.0_224`) is Apache 2.0 and the PlantVillage dataset was released under Public Domain (CC0 1.0), the fine-tuned weights lack a standard open-source commercial grant.
   - **Prototype constraint**: The model weights are suitable for hackathon demonstration and academic evaluation, but commercial production requires formal author clarification or custom training from CC0 weights.

2. **Evaluation Accuracy vs. Real-World Field Performance**:
   - The reported evaluation accuracy of **95.41%** reflects validation on laboratory-captured leaves on solid backgrounds.
   - **Real-world field performance is unverified**: In-field photographs containing complex backgrounds (soil, fingers, weeds, varied sunlight) typically exhibit lower accuracy.
   - CropScan AI treats model predictions as diagnostic screening aids, not definitive agronomic diagnoses.

3. **Translation Review Status**:
   - All Yoruba guidance records carry `"translation_review_status": "draft_translation_pending_native_expert_review"` and `"expert_review_required": true`.
   - Content should be formally reviewed by bilingual agronomists before public release.

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

All 28 tests execute in `~2s` against isolated test fixtures with programmatically generated in-memory images, requiring no live network connection or heavy model weights.

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

Receives a crop leaf photograph, performs validation, and returns assessment status.

- **URL**: `/api/predict`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `image` *(required, file)*: Image file (JPEG, PNG, or WebP). Maximum file size is **10 MB**.
  - `lang` *(optional, string)*: Language for response guidance (`en` for English, `yo` for Yoruba). Defaults to `en`.

#### Validation & Security Safeguards
- **In-Memory Decoding**: Uploaded files are decoded strictly in memory using Pillow; files are never written to disk.
- **Format Verification**: The true raster header and byte streams are verified (protecting against spoofed `.jpg` extensions containing text or scripts).
- **Decompression Bomb Protection**: Guard rails enforce maximum dimensions (10,000 px) to prevent memory-exhaustion attacks.
- **Transparent Model Status**: Because model weights are not integrated in Phase 2, valid uploads return `status: "model_not_configured"` with validated image metadata. No fabricated predictions or fake confidence scores are returned.

#### Example Request (PowerShell)
```powershell
$form = @{
    image = Get-Item "C:\path\to\leaf_sample.jpg"
    lang  = "en"
}
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict" -Method Post -Form $form
```

#### Example Response (Valid Upload - Model Pending Integration)
```json
{
  "status": "model_not_configured",
  "message": "Image validated successfully. Plant disease detection model is not configured yet.",
  "language": "en",
  "image_metadata": {
    "format": "JPEG",
    "width": 800,
    "height": 600,
    "color_mode": "RGB",
    "size_bytes": 104520
  }
}
```

#### Example Error Response (Unsupported Format - 400 Bad Request)
```json
{
  "error": "unsupported_format",
  "message": "Unsupported image format 'GIF'. CropScan AI accepts JPEG, PNG, and WebP images only."
}
```

#### Example Error Response (Oversized File - 413 Payload Too Large)
```json
{
  "error": "payload_too_large",
  "message": "Uploaded image exceeds the maximum permitted size of 10 MB."
}
```
