"""
CropScan AI - Image Quality Screening Module.

Provides conservative pre-inference image quality screening to detect and filter
out obviously unusable inputs (e.g. pitch-black frames, completely overexposed/blown-out
images, or uniform non-plant canvases) before invoking deep learning vision models.

Design Principles & Limitations:
- Conservative Rejection: Only rejects inputs that are unambiguously unusable.
- No False Out-of-Distribution Claims: Color and blur heuristics alone cannot prove
  an image is a plant or leaf. This module screens image quality, not botanical identity.
- Preserves Imperfect Images: Diseased, yellowed, senescent, or chlorotic leaves
  are never rejected on the basis of lacking green pixels.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageStat

logger = logging.getLogger(__name__)

# Conservative default screening thresholds
DEFAULT_MIN_LUMINANCE: float = 8.0     # Mean luminance below this is pitch black
DEFAULT_MAX_LUMINANCE: float = 250.0   # Mean luminance above this is blown out
DEFAULT_MIN_CONTRAST: float = 5.0      # Standard deviation below this indicates flat/uniform field
DEFAULT_MIN_SHARPNESS: Optional[float] = None  # Sharpness threshold (optional/evaluative)

# Maximum dimension to analyze during screening (ensures constant-time execution)
SCREENING_ANALYSIS_SIZE: Tuple[int, int] = (512, 512)

# Laplacian 3x3 kernel with offset=128 to keep intermediate values positive
_LAPLACIAN_KERNEL = ImageFilter.Kernel(
    (3, 3),
    [0, 1, 0, 1, -4, 1, 0, 1, 0],
    scale=1,
    offset=128,
)


@dataclass(frozen=True)
class ScreeningResult:
    """
    Structured outcome of pre-inference image quality screening.

    Attributes:
        is_acceptable: True if the image passes all conservative screening checks.
        rejection_code: Machine-readable code if rejected (e.g. 'image_too_dark').
        message: Human-readable explanation of rejection or warning.
        metrics: Dictionary of computed quantitative image quality measurements.
    """
    is_acceptable: bool
    rejection_code: Optional[str]
    message: Optional[str]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the screening result to a plain dictionary."""
        return asdict(self)


def _prepare_thumbnail(image: Image.Image) -> Image.Image:
    """Creates a capped-resolution copy of the image for fast, constant-time screening."""
    w, h = image.size
    if w <= SCREENING_ANALYSIS_SIZE[0] and h <= SCREENING_ANALYSIS_SIZE[1]:
        return image.copy()
    thumb = image.copy()
    thumb.thumbnail(SCREENING_ANALYSIS_SIZE, Image.Resampling.BILINEAR)
    return thumb


def measure_luminance_and_contrast(image: Image.Image) -> Tuple[float, float]:
    """
    Measures the mean luminance and contrast (standard deviation) of an image in grayscale.

    Args:
        image: A valid PIL Image instance.

    Returns:
        tuple: (mean_luminance, contrast) in range [0.0, 255.0] and [0.0, ~127.5].
    """
    thumb = _prepare_thumbnail(image)
    gray = thumb.convert("L")
    stat = ImageStat.Stat(gray)
    mean_lum = round(float(stat.mean[0]), 2)
    contrast = round(float(stat.stddev[0]), 2)
    return mean_lum, contrast


def measure_sharpness(image: Image.Image) -> float:
    """
    Measures high-frequency texture and edge sharpness using interior Laplacian variance.

    The 1-pixel image border is cropped after filtering to prevent boundary padding
    artifacts from inflating the variance of flat or blurred images.

    Args:
        image: A valid PIL Image instance.

    Returns:
        float: Interior Laplacian variance (higher indicates sharper edges/texture).
    """
    thumb = _prepare_thumbnail(image)
    gray = thumb.convert("L")
    lap = gray.filter(_LAPLACIAN_KERNEL)

    # Crop 1-pixel border to eliminate filter boundary padding
    if lap.width > 2 and lap.height > 2:
        lap = lap.crop((1, 1, lap.width - 1, lap.height - 1))

    stat = ImageStat.Stat(lap)
    return round(float(stat.var[0]), 2)


def measure_color_distribution(image: Image.Image) -> Tuple[float, float]:
    """
    Computes green-dominant pixel ratio and broad vegetation-consistent pixel ratio.

    Vegetation-consistent pixels include:
    - Green foliage: G > R and G > B with minimum brightness (G > 30).
    - Chlorotic / yellow leaf tissue: R > 80, G > 80, B < 120, |R - G| < 50.
    - Necrotic / brown blight tissue: R > 60, G > 40, B < 60, R >= G > B.

    Important:
    The vegetation ratio is a weak diagnostic signal. A low ratio must never be used
    alone to reject valid non-green diseased leaves.

    Args:
        image: A valid PIL Image instance.

    Returns:
        tuple: (green_pixel_ratio, vegetation_pixel_ratio) both in range [0.0, 1.0].
    """
    thumb = _prepare_thumbnail(image).convert("RGB")
    arr = np.asarray(thumb, dtype=np.float32)

    total_pixels = arr.shape[0] * arr.shape[1]
    if total_pixels == 0:
        return 0.0, 0.0

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # Green foliage
    green_mask = (g > r) & (g > b) & (g > 30.0)

    # Yellow / chlorotic leaf tissue
    yellow_mask = (
        (r > 80.0)
        & (g > 80.0)
        & (b < 120.0)
        & (np.abs(r - g) < 50.0)
    )

    # Brown / necrotic leaf blight tissue
    brown_mask = (
        (r > 60.0)
        & (g > 40.0)
        & (b < 60.0)
        & (r >= g)
        & (g > b)
    )

    veg_mask = green_mask | yellow_mask | brown_mask

    green_ratio = round(float(np.sum(green_mask) / total_pixels), 4)
    veg_ratio = round(float(np.sum(veg_mask) / total_pixels), 4)

    return green_ratio, veg_ratio


def screen_image_quality(
    image: Image.Image,
    min_luminance: float = DEFAULT_MIN_LUMINANCE,
    max_luminance: float = DEFAULT_MAX_LUMINANCE,
    min_contrast: float = DEFAULT_MIN_CONTRAST,
    min_sharpness: Optional[float] = DEFAULT_MIN_SHARPNESS,
) -> ScreeningResult:
    """
    Executes conservative pre-inference image quality screening on an uploaded PIL Image.

    Evaluation checks:
    1. Extreme darkness: mean_luminance < min_luminance (e.g. < 8.0).
    2. Extreme overexposure: mean_luminance > max_luminance and contrast < min_contrast.
    3. Blank / featureless canvas: contrast < min_contrast and vegetation_ratio < 0.01.
    4. Severe blur: sharpness < min_sharpness (only evaluated if min_sharpness is provided).

    Args:
        image: PIL Image to evaluate.
        min_luminance: Minimum allowable mean luminance (default 8.0).
        max_luminance: Maximum allowable mean luminance before checking overexposure (default 250.0).
        min_contrast: Minimum allowable contrast standard deviation (default 5.0).
        min_sharpness: Optional minimum sharpness variance threshold.

    Returns:
        ScreeningResult containing acceptance flag, optional rejection code,
        human-readable message, and computed metrics.
    """
    mean_lum, contrast = measure_luminance_and_contrast(image)
    sharpness = measure_sharpness(image)
    green_ratio, veg_ratio = measure_color_distribution(image)

    metrics = {
        "mean_luminance": mean_lum,
        "contrast": contrast,
        "sharpness": sharpness,
        "green_pixel_ratio": green_ratio,
        "vegetation_pixel_ratio": veg_ratio,
    }

    # 1. Extreme Darkness Check
    if mean_lum < min_luminance:
        logger.warning(
            f"Image quality screening rejected image: mean luminance {mean_lum:.2f} < {min_luminance:.2f} (too dark)."
        )
        return ScreeningResult(
            is_acceptable=False,
            rejection_code="image_too_dark",
            message=(
                f"The uploaded image is extremely dark (mean luminance {mean_lum:.2f} is below {min_luminance:.2f}) "
                "and lacks discernible visual features."
            ),
            metrics=metrics,
        )

    # 2. Extreme Overexposure Check
    if mean_lum > max_luminance and contrast < min_contrast:
        logger.warning(
            f"Image quality screening rejected image: mean luminance {mean_lum:.2f} > {max_luminance:.2f} "
            f"with contrast {contrast:.2f} < {min_contrast:.2f} (overexposed)."
        )
        return ScreeningResult(
            is_acceptable=False,
            rejection_code="image_too_bright",
            message=(
                f"The uploaded image is completely overexposed (mean luminance {mean_lum:.2f} is above {max_luminance:.2f} "
                "with negligible contrast) and lacks discernible visual features."
            ),
            metrics=metrics,
        )

    # 3. Blank / Featureless Uniform Canvas Check (non-plant solid color)
    if contrast < min_contrast and veg_ratio < 0.01:
        logger.warning(
            f"Image quality screening rejected image: contrast {contrast:.2f} < {min_contrast:.2f} "
            f"and vegetation ratio {veg_ratio:.4f} < 0.01 (lacks visual features)."
        )
        return ScreeningResult(
            is_acceptable=False,
            rejection_code="image_lacks_features",
            message=(
                f"The uploaded image is a flat, uniform surface (contrast {contrast:.2f} is below {min_contrast:.2f} "
                "with no discernible vegetation characteristics)."
            ),
            metrics=metrics,
        )

    # 4. Optional Sharpness / Blur Check (only if caller/evaluator explicitly configures threshold)
    if min_sharpness is not None and sharpness < min_sharpness:
        logger.warning(
            f"Image quality screening rejected image: sharpness {sharpness:.2f} < {min_sharpness:.2f} (too blurry)."
        )
        return ScreeningResult(
            is_acceptable=False,
            rejection_code="image_too_blurry",
            message=(
                f"The uploaded image lacks sufficient sharpness (sharpness score {sharpness:.2f} is below "
                f"threshold {min_sharpness:.2f}) and appears severely blurred or out of focus."
            ),
            metrics=metrics,
        )

    # All conservative checks passed
    return ScreeningResult(
        is_acceptable=True,
        rejection_code=None,
        message=None,
        metrics=metrics,
    )
