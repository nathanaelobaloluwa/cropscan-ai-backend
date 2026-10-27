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
│       ├── class_to_crop.json   # Explicit 38-class biological crop family mapping
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

All **58 automated tests** execute in `~27s`:
- `tests/test_health.py`: Health endpoint and startup checks (3 tests).
- `tests/test_guidance.py`: Knowledge base retrieval, 38-class coverage, explicit `class_to_crop` mapping, crop normalization and alias resolution, Yoruba/English consistency, and safe fallbacks (17 tests).
- `tests/test_predictor.py`: Preprocessing verification, real MobileNetV2 model loading, genuine inference on sample images, confidence threshold boundary behavior, crop mismatch safety protection, and exception handling (16 tests).
- `tests/test_predict.py`: End-to-end endpoint tests for `POST /api/predict`, required crop selection, unsupported crop rejection (cassava, yam), format verification, payload limit (10 MB), language localization, and error statuses (400, 413, 500, 503) (22 tests).

---

## Real Leaf Image Empirical Validation (Phase 4)

To validate live model inference and end-to-end endpoint integration prior to adding new features, 7 authentic real leaf photographs were evaluated through the actual `POST /api/predict` endpoint using the loaded MobileNetV2 model weights.

### Image Sources & Licensing
1. **PlantVillage Dataset** (`spMohanty/PlantVillage-Dataset`): Licensed under **CC0 1.0 Universal (Public Domain Dedication)**. Laboratory-photographed leaves on uniform backgrounds across standardized disease and healthy classes.
2. **Wikimedia Commons** (`File:Cassava plant leaf.jpg`): Licensed under **CC BY-SA 4.0**. High-resolution in-field photograph of a cassava (*Manihot esculenta*) leaf, an African staple crop completely absent from the 38-class PlantVillage dataset.

### Empirical Test Matrix & Observed Outputs

| Test Image | Crop & True Condition | Expected Label | Predicted Class | Raw Score | Assessment Status | Localized Guidance | Validation Outcome |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| `tomato_early_blight.jpg` | Tomato (Diseased) | Tomato with Early Blight | Tomato with Early Blight (idx 29) | `0.8394` | `confident` | `success` (`tomato_early_blight`) | **Accurate detection & guidance** |
| `tomato_healthy.jpg` | Tomato (Healthy) | Healthy Tomato Plant | Healthy Tomato Plant (idx 37) | `1.0000` | `confident` | `success` (`healthy_tomato`) | **Accurate detection (no false alarm)** |
| `corn_common_rust.jpg` | Maize (Diseased) | Corn (Maize) with Common Rust | Corn with Common Rust (idx 8) | `0.9989` | `confident` | `success` (`corn_common_rust`) | **Accurate detection & guidance** |
| `bell_pepper_healthy.jpg` | Bell Pepper (Healthy) | Healthy Bell Pepper Plant | Healthy Bell Pepper (idx 19) | `0.9999` | `confident` | `success` (`healthy_pepper`) | **Accurate detection (no false alarm)** |
| `potato_late_blight.jpg` | Potato (Diseased) | Potato with Late Blight | Potato with Late Blight (idx 21) | `0.6684` | `confident` | `success` (`potato_late_blight`) | **Accurate detection & guidance** |
| `apple_scab.jpg` | Apple (Diseased, Out-of-Scope) | Apple Scab (idx 0) | Apple Scab (idx 0) | `0.9979` | `recognized_unsupported_crop` | `recognized_unsupported_crop` | **Safe scope separation (never claims healthy)** |
| `cassava_leaf_unsupported.jpg` | Cassava (Unsupported Crop) | Out-of-Distribution (None) | Cedar Apple Rust (idx 2) | `0.3192` | `inconclusive_low_confidence` | `unreliable_assessment` | **Safely rejected via threshold (0.3192 < 0.40)** |

### Critical Empirical Insights & Limitations

1. **Closed-Set Vulnerability on Out-of-Distribution Inputs**:
   - Because MobileNetV2 uses a closed-set softmax classification head over 38 classes, any image passed into the model is mathematically forced to distribute probability across those 38 classes.
   - For the **Cassava leaf**, the model assigned `0.3192` to `"Cedar Apple Rust"` (index 2).
   - Because `0.3192` fell below the provisional threshold (`0.40`), CropScan AI safely prevented a false diagnosis and returned an inconclusive advisory response (`is_healthy: None`).
   - **Crucial Caution**: Confidence thresholding alone is **not** an out-of-distribution detector. A different unsupported crop or complex background image could easily produce an activation $\ge 0.40$, resulting in a confident misdiagnosis.

2. **Provisional Nature of the 0.40 Threshold**:
   - The threshold of `0.40` was selected as a provisional operational baseline. It is **not clinically or statistically validated**.
   - Setting a threshold cannot be validated on 7 images; genuine operational calibration requires hundreds of field-collected images across target ecologies.

3. **Architectural Recommendations for Future Phases**:
   - **Frontend Crop Selector**: Require users to specify their target crop (e.g., Tomato, Maize) before diagnosis. If the model's top prediction does not match the user's selected crop family, the system should reject the image as mismatched.
   - **Two-Stage Classifier Pipeline**: Introduce an upstream lightweight crop-species classifier (or open-set anomaly detector) to reject non-leaf or unsupported crops (e.g., cassava, yam, weeds) before disease inference.

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

Receives a crop leaf photograph and user crop selection, performs security and format validation, runs genuine inference on MobileNetV2, protects against crop prediction mismatches, and returns localized agronomic guidance in English or Yoruba.

- **URL**: `/api/predict`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `image` *(required, file)*: Image file (JPEG, PNG, or WebP). Maximum file size is **10 MB**.
  - `crop` *(required, string)*: Target crop selection. Valid canonical identifiers are:
    - `tomato` (Tomato)
    - `maize` (Maize / Corn; aliases: `corn`, `agbado`)
    - `bell_pepper` (Bell Pepper; aliases: `bell pepper`, `pepper`, `tatase`)
    - `citrus` (Citrus / Orange; aliases: `orange`, `osan`)
    - `potato` (Irish Potato; aliases: `irish potato`, `poteto`)
  - `lang` *(optional, string)*: Language for response guidance (`en` for English, `yo` for Yoruba). Defaults to `en`.

#### Validation & Safety Safeguards (Phase 5)
- **Required Crop Selection**: Enforces explicit crop context from the farmer to prevent closed-set misdiagnosis across botanical families.
- **Explicit 38-Class Biological Mapping**: Maps all 38 model class indices directly to their biological crop family (`app/data/class_to_crop.json`).
- **Crop Mismatch Protection**: If the model's top predicted class belongs to a different crop family than the user's selection (e.g., user selects `tomato` but model predicts `Corn Common Rust` or `Apple Scab`), CropScan AI returns `assessment_status: "crop_mismatch"` and `crop_match: false`.
- **Withheld Treatment Guidance**: Whenever a crop mismatch is detected, disease-specific treatment recommendations are strictly withheld (`is_healthy: null`, `data: null`) to prevent dangerous chemical misapplication.
- **Safe Confidence Thresholding**: Predictions matching the selected crop but scoring below the provisional confidence threshold (`0.40`) return `inconclusive_low_confidence` and advisory guidance.
- **In-Memory Decoding & Security**: Images are verified against true raster byte headers and decoded strictly in memory with decompression bomb guards (10,000 px max dimension).

#### Crucial Machine Learning Safety Limitations
> [!IMPORTANT]
> - **Crop Selection Does NOT Prove Botanical Identity**: Crop selection is a user-asserted screening constraint, **NOT** automated botanical verification. The system cannot independently prove that an uploaded leaf truly belongs to the selected crop.
> - **In-Crop Out-of-Distribution Vulnerability**: If a user submits weeds, non-leaf objects, or an unsupported disease while selecting a supported crop (e.g., submitting cassava as `tomato`), the closed-set model will still force an activation among the tomato classes. If that activation exceeds `0.40`, a false positive can occur.
> - **Provisional Baseline**: Thresholds and mismatch rules are operational screening safeguards for prototyping and must never replace on-field diagnostic verification by agricultural extension officers.

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

#### Example Error Response (Missing Crop - 400 Bad Request)
```json
{
  "error": "missing_crop",
  "message": "Field 'crop' is required. Supported crops are: 'tomato', 'maize', 'bell_pepper', 'citrus', 'potato'."
}
```

#### Example Error Response (Unsupported Crop - 400 Bad Request)
```json
{
  "error": "unsupported_crop",
  "message": "Unsupported crop 'cassava'. Supported crops are: 'tomato', 'maize', 'bell_pepper', 'citrus', 'potato'."
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

