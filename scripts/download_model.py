"""
CropScan AI - Build-Time Model Pre-caching Script.

Executes during the Render build phase (e.g. `pip install -r requirements.txt && python scripts/download_model.py`)
to download and cache the MobileNetV2 plant disease vision model on disk before the web service starts.

Benefits:
- Eliminates on-demand HuggingFace HTTP download during user prediction requests.
- Prevents memory spikes and request timeouts during first inference.
- Ensures the live container can run with local_files_only=True.
"""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"


def pre_cache_model() -> None:
    model_name = os.getenv("CROPSCAN_MODEL_NAME", DEFAULT_MODEL_NAME)
    logger.info(f"Pre-caching vision model '{model_name}' during build phase...")
    try:
        from transformers import AutoModelForImageClassification
        AutoModelForImageClassification.from_pretrained(model_name)
        logger.info(f"Vision model '{model_name}' successfully pre-cached on disk.")
    except Exception as e:
        logger.error(f"Failed to pre-cache model '{model_name}': {e}", exc_info=True)
        # Do not fail build hard if network has transient blips; runtime fallback still exists
        sys.exit(0)


if __name__ == "__main__":
    pre_cache_model()

