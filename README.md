# CropScan AI - Backend

CropScan AI is a plant disease detection backend API designed to empower farmers and agricultural stakeholders. Farmers can submit crop leaf images, obtain fast predictions from an open-weight vision model, and access actionable disease treatment and prevention recommendations available in both English and Yoruba.

---

## Architecture & Project Structure

The codebase is organized using Flask's **Application Factory Pattern** and **Blueprints** to ensure modularity, ease of testing, and maintainability across hackathon development phases:

```text
cropscan-ai-backend/
├── app/
│   ├── __init__.py              # Application factory (create_app)
│   ├── config.py                # Environment configurations (10 MB limit, model & threshold settings)
│   ├── routes/
│   │   ├── __init__.py          # Blueprint aggregation
│   │   ├── health.py            # Health check endpoint (GET /health)
│   │   └── predict.py           # Image upload & prediction endpoint (POST /api/predict)
│   ├── services/
│   │   ├── __init__.py          # Service layer exports
│   │   ├── guidance.py          # GuidanceService (English & Yoruba resolution)
│   │   ├── image_validator.py   # In-memory image decoding, format & security validation
│   │   └── predictor.py         # PredictionService (MobileNetV2 inference, threshold & guidance)
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
│   ├── test_predict.py          # Endpoint integration & upload security tests
│   └── test_predictor.py        # ML service unit & genuine inference tests
├── .env.example                 # Template for local environment variables
├── .gitignore                   # Ignored files (venv, env, caches)
├── requirements.txt             # Core dependencies (Flask, Pillow, PyTorch, Transformers, pytest)
├── run.py                       # Server entry point
└── README.md                    # Project documentation
```

### Decoupled Data Principles
- **Model Class IDs**: Exact prediction label from the model (e.g., `"Tomato with Early Blight"`).
- **Disease IDs**: Standardized machine slugs (e.g., `"tomato_early_blight"`).
- **Display Names**: Farmer-friendly localized titles (e.g., `"Early Blight"` or `"Àrùn Èérún Tòmátò"`).
- **Guidance Records**: Conservative cultural and preventive measures referencing verified agricultural bodies (FAO, IITA, CIMMYT, CIP, WorldVeg).

---

## Machine Learning Vision Model & Inference

### Model Specifications
- **Model Name**: `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`
- **Model Architecture**: MobileNetV2 for Image Classification (`MobileNetV2ForImageClassification`)
- **Base Model**: `google/mobilenet_v2_1.0_224` (Apache 2.0 license)
- **Training Dataset**: Kaggle New Plant Diseases Dataset (PlantVillage, CC0 Public Domain)
- **Model Size**: ~14.2 MB (`model.safetensors`, 314 weight tensors)
- **Default Device**: CPU inference (`PREDICTOR_DEVICE=cpu`)
- **Singleton Lifecycle**: Model weights and preprocessor are loaded once into memory upon service initialization or first request, preventing per-request reloading overhead.

### Documented & Verified Preprocessing Pipeline
CropScan AI reproduces the exact preprocessing pipeline from the model configuration without duplicate normalization:
1. **Shortest-edge resize**: 256 pixels
2. **Resampling method**: Bilinear interpolation (`PIL.Image.BILINEAR`)
3. **Center crop**: 224 × 224 pixels
4. **Pixel rescaling**: Multiplied by `1 / 255.0` (`0.00392156862745098`)
5. **Channel normalization**: Mean `[0.5, 0.5, 0.5]` and standard deviation `[0.5, 0.5, 0.5]` per RGB channel
6. **Backend Implementation**: Powered by `MobileNetV2ImageProcessorPil` (Pillow-native backend, avoiding the heavy `torchvision` dependency).

### Confidence Thresholding & Uncertainty Limitations
- **Provisional Threshold**: Defaulting to `0.40` (configurable via `PREDICTION_CONFIDENCE_THRESHOLD`).
- **Provisional Nature**: This threshold is strictly provisional for screening in this hackathon prototype and has not been validated through agricultural field trials.
- **Uncalibrated Scores**: Raw model scores are computed via Softmax over 38 logits. They represent relative model activation among trained classes, **not calibrated probabilities** that the diagnosis is correct.
- **Not a Confirmed Diagnosis**: CropScan AI provides automated screening aids, not a botanical or phytopathological diagnosis. Farmers must consult local agricultural extension officers before applying chemical treatments.
- **Out-of-Distribution Vulnerability**: Confidence thresholding alone **cannot reliably detect** unsupported crops (e.g., cassava, yam), non-leaf images, or arbitrary out-of-distribution inputs. An image of a shoe or weeds may still yield an arbitrary prediction if forced through closed-set softmax classification.

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
   - The fine-tuned weights (`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`) carry a license metadata tag of **`other`** in the Hugging Face repository.
   - While the base architecture (`google/mobilenet_v2_1.0_224`) is Apache 2.0 and the PlantVillage dataset was released under Public Domain (CC0 1.0), the fine-tuned repository lacks an explicit license file defining commercial redistribution rights.
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

Install the pinned dependencies:
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

Run the complete test suite using `pytest`:
```powershell
pytest -v
```

All **42 automated tests** execute in `~3s`:
- `tests/test_health.py`: Health endpoint and startup checks.
- `tests/test_guidance.py`: Knowledge base retrieval, 38-class coverage, Yoruba/English consistency, and safe fallbacks.
- `tests/test_predictor.py`: Preprocessing verification, real MobileNetV2 model loading, genuine inference on sample images, confidence threshold boundary behavior, and exception handling.
- `tests/test_predict.py`: End-to-end endpoint tests for `POST /api/predict`, format verification, payload limit (10 MB), language localization, and error statuses (400, 413, 500, 503).

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

Receives a crop leaf photograph, performs security and format validation, runs genuine inference on MobileNetV2, and returns localized agronomic guidance in English or Yoruba.

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
- **Genuine Model Inference**: Uses the pretrained MobileNetV2 model weights. Never fabricates predictions or confidence scores.
- **Safe Fallback**: Predictions below the provisional confidence threshold (0.40) return `inconclusive_low_confidence` and safe fallback instructions without diagnosing.

#### Example Request (PowerShell)
```powershell
$form = @{
    image = Get-Item "C:\path\to\leaf_sample.jpg"
    lang  = "en"
}
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict" -Method Post -Form $form
```

#### Example Response (Genuine Prediction - Confident with Guidance)
```json
{
  "status": "success",
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
    "model_name": "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
    "class_index": 29,
    "class_label": "Tomato with Early Blight",
    "confidence": 0.8845,
    "confidence_threshold": 0.40,
    "is_provisional_threshold": true,
    "disclaimer": "Automated screening result only; not a confirmed agricultural diagnosis. Raw model scores are uncalibrated and do not guarantee accuracy. Always consult local agricultural extension officers before applying treatments.",
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

#### Example Response (Inconclusive Low Confidence)
```json
{
  "status": "success",
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
    "model_name": "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
    "class_index": 30,
    "class_label": "Tomato with Late Blight",
    "confidence": 0.2168,
    "confidence_threshold": 0.40,
    "is_provisional_threshold": true,
    "disclaimer": "Model confidence (21.68%) is below the provisional operational threshold (40.00%). Raw model scores are not calibrated probabilities of correctness. Confidence thresholding cannot reliably detect unsupported crops, non-leaf images, or out-of-distribution inputs. This result is inconclusive and must not be treated as a confirmed diagnosis.",
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

