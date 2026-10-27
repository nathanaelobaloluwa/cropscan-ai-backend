# CropScan AI Data Directory

This directory contains the core domain data assets for the CropScan AI application:

- `class_indices.json`: Verified 38-class mapping from model output index to official model class label string.
- `disease_info_en.json`: Structured agronomic disease and plant health guidance in English.
- `disease_info_yo.json`: Structured agronomic disease and plant health guidance in Yoruba.

---

## Decoupled Architecture

To ensure maintainability and scientific rigor, prediction identifiers are decoupled from display names and localized treatment records:

1. **Model Class ID**: The exact string produced by the computer vision model (e.g., `"Tomato with Early Blight"`).
2. **Disease ID**: A standardized, language-agnostic machine identifier (e.g., `"tomato_early_blight"`).
3. **Display Name**: Human-readable title tailored for farmers in English or Yoruba.
4. **Treatment & Prevention Records**: Conservative, non-prescriptive agricultural advice referencing verified agricultural bodies (FAO, IITA, CIMMYT, CIP, WorldVeg).

---

## Initial Prototype Scope (Nigerian-Relevant Crops)

The initial knowledge base focuses on high-priority Nigerian staple crops supported by the 38-class model:
- **Maize (Àgbàdo)**: Classes `7`, `8`, `9`, `10`
- **Tomato (Tòmátò)**: Classes `28`, `29`, `30`, `31`, `32`, `33`, `34`, `35`, `36`, `37`
- **Bell Pepper (Ata Tàtàṣé)**: Classes `18`, `19`
- **Citrus (Ọsàn)**: Class `15`
- **Potato (Ànàmọ́ Gẹ̀ẹ́sì / Pọtétò)**: Classes `20`, `21`, `22`
- **Healthy Classifications**: Explicit positive health records for each supported crop.

> [!NOTE]
> **Excluded Crops**: Key African staple crops such as **Cassava** and **Yam** are **NOT** present in the 38-class PlantVillage dataset and are excluded from model mappings. Any future cassava support will require a separate, verified model (such as Google CropNet Cassava).

---

## Model Licensing Uncertainty & Known Limitations

1. **Licensing Status**:
   - The fine-tuned model weights (`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`) carry a license metadata tag of **`other`**.
   - While the base architecture (`google/mobilenet_v2_1.0_224`) is licensed under Apache 2.0 and the underlying PlantVillage dataset was dedicated to the Public Domain (CC0 1.0), the fine-tuned weights themselves lack a standardized open-source license grant.
   - **Prototype constraint**: The model weights are suitable for hackathon demonstration and academic evaluation, but commercial deployment requires written clarification from the author or re-training from CC0 weights.

2. **Evaluation Accuracy vs. Real-World Field Performance**:
   - The reported evaluation accuracy of **95.41%** on Hugging Face was measured on the New Plant Diseases validation split, which consists of laboratory-captured leaves photographed against uniform, solid backgrounds.
   - **Real-world field accuracy is unverified**: In-field photographs with complex backgrounds (soil, fingers, weeds, varied natural sunlight) typically experience significant accuracy degradation.
   - CropScan AI treats model predictions as screening aids, not definitive agronomic diagnoses.

3. **Translation Review Status**:
   - All Yoruba records in `disease_info_yo.json` carry `"translation_review_status": "draft_translation_pending_native_expert_review"` and `"expert_review_required": true`.
   - Translations should undergo formal validation by bilingual agronomists before public release.
