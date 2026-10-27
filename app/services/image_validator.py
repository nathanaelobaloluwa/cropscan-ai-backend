import io
from typing import Any, Dict, Optional, Tuple
from PIL import Image, UnidentifiedImageError

# Maximum dimension limit (pixels) to protect against decompression bombs
MAX_IMAGE_DIMENSION = 10000
# Maximum file size: 10 MB in bytes
MAX_IMAGE_FILE_SIZE = 10 * 1024 * 1024
# Supported formats
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageValidationError(Exception):
    """Custom exception raised during image validation."""

    def __init__(self, error_code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code


def validate_image_upload(file_storage: Any) -> Dict[str, Any]:
    """
    Safely validates an uploaded image file in memory without saving to disk.

    Performs:
    1. Presence and non-empty checks.
    2. File size ceiling checks (10 MB).
    3. Safe header and raster decode checks using Pillow.
    4. Image format verification (JPEG, PNG, WebP only).
    5. Dimension sanity checks and decompression bomb protection.
    6. Complete raster load check to catch corrupt/truncated data.

    Args:
        file_storage: Werkzeug FileStorage object from request.files['image'].

    Returns:
        dict: Verified image metadata if valid.

    Raises:
        ImageValidationError: If validation fails for any reason.
    """
    if file_storage is None:
        raise ImageValidationError(
            error_code="missing_image",
            message="No image file provided in multipart form data (field 'image').",
            status_code=400,
        )

    # Read data into memory
    data = file_storage.read()

    # Empty check
    if not data or len(data) == 0:
        raise ImageValidationError(
            error_code="empty_image",
            message="The uploaded image file is empty (0 bytes).",
            status_code=400,
        )

    # File size ceiling check (10 MB)
    if len(data) > MAX_IMAGE_FILE_SIZE:
        raise ImageValidationError(
            error_code="payload_too_large",
            message=f"Uploaded image exceeds the maximum permitted size of 10 MB (received {len(data)} bytes).",
            status_code=413,
        )

    stream = io.BytesIO(data)

    try:
        # Step 1: Open stream and verify headers
        with Image.open(stream) as img:
            img_format = img.format
            if img_format not in ALLOWED_FORMATS:
                raise ImageValidationError(
                    error_code="unsupported_format",
                    message=(
                        f"Unsupported image format '{img_format or 'UNKNOWN'}'. "
                        "CropScan AI accepts JPEG, PNG, and WebP images only."
                    ),
                    status_code=400,
                )

            # verify() verifies stream integrity without fully decoding raster pixels
            img.verify()

        # Step 2: Re-open a fresh stream to inspect dimensions, mode, and load pixels
        # (img.verify() exhausts the image stream, so reopening is required by Pillow)
        stream_reopen = io.BytesIO(data)
        with Image.open(stream_reopen) as img_loaded:
            width, height = img_loaded.size

            # Check dimensions sanity
            if width <= 0 or height <= 0:
                raise ImageValidationError(
                    error_code="invalid_dimensions",
                    message=f"Invalid image dimensions: {width}x{height}.",
                    status_code=400,
                )

            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                raise ImageValidationError(
                    error_code="decompression_bomb",
                    message=f"Image dimensions ({width}x{height}) exceed maximum allowed limit of {MAX_IMAGE_DIMENSION}px.",
                    status_code=400,
                )

            # Step 3: Fully decode raster pixels to detect corrupt or truncated streams
            img_loaded.load()

            metadata = {
                "format": img_format,
                "width": width,
                "height": height,
                "color_mode": img_loaded.mode,
                "size_bytes": len(data),
            }

            return metadata

    except ImageValidationError:
        # Re-raise known validation errors
        raise
    except Image.DecompressionBombError as e:
        raise ImageValidationError(
            error_code="decompression_bomb",
            message="Image triggered decompression bomb protection: dimensions are excessively large.",
            status_code=400,
        ) from e
    except (UnidentifiedImageError, SyntaxError, ValueError) as e:
        raise ImageValidationError(
            error_code="invalid_image",
            message="The uploaded file is not a valid or supported image file.",
            status_code=400,
        ) from e
    except OSError as e:
        # Typically raised on truncated or corrupted image streams
        raise ImageValidationError(
            error_code="corrupt_image",
            message="The uploaded image file is corrupt or truncated.",
            status_code=400,
        ) from e
