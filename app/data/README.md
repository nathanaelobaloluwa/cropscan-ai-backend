# Data Directory

This directory is reserved for application data assets, including:
- `class_indices.json`: Mapping of model output index to disease labels.
- `disease_info_en.json`: Disease details, treatment, and prevention instructions in English.
- `disease_info_yo.json`: Disease details, treatment, and prevention instructions in Yoruba.

Keeping structured domain data decoupled from service logic allows easy updates, audits, and localization without code changes.
