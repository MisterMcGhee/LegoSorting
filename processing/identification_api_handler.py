# processing/identification_api_handler.py
"""
identification_api_handler.py - API communication for piece identification

This module handles communication with the Brickognize API to identify
Lego pieces from images.

RESPONSIBILITIES:
- Send images to Brickognize API
- Parse API responses
- Return IdentificationResult dataclass
- Handle errors and validation

DOES NOT:
- Lookup categories (that's category_lookup_module)
- Assign bins (that's bin_assignment_module)
- Save images (that's capture_controller)
- Make business logic decisions
"""

import os
import requests
import logging
from typing import Optional

from processing.processing_data_models import IdentificationResult
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig

# Set up module logger
logger = logging.getLogger(__name__)


class IdentificationAPIHandler:
    """
    Handles communication with Brickognize API for Lego piece identification.

    This is a focused module that only handles API communication.
    All configuration comes from enhanced_config_manager.
    """

    # API endpoint is constant - Brickognize's public API URL
    API_URL = "https://api.brickognize.com/predict/"

    # Supported image formats
    VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif']

    def __init__(self, config_manager: EnhancedConfigManager):
        """
        Initialize API handler with configuration.

        Args:
            config_manager: Configuration manager instance
        """
        # Get API configuration from config manager
        module_config = config_manager.get_module_config(ModuleConfig.API.value)

        self.timeout = module_config["timeout"]
        self.retry_count = module_config["retry_count"]

        logger.info(
            f"API handler initialized "
            f"(timeout={self.timeout}s, retries={self.retry_count})"
        )

    def identify_piece(self, image_path: str) -> IdentificationResult:
        """
        Identify a Lego piece from an image file.

        This sends the image to Brickognize API and returns the
        top identification result.

        Args:
            image_path: Path to the image file to identify

        Returns:
            IdentificationResult with element_id, name, and confidence

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If file format invalid or API returns no results
            requests.RequestException: If network/API error occurs
        """
        # Validate file exists
        if not os.path.exists(image_path):
            error_msg = f"Image file not found: {image_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Validate file extension and determine MIME type
        ext = os.path.splitext(image_path)[1].lower()

        # Map extensions to MIME types
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif'
        }

        if ext not in mime_types:
            error_msg = f"Invalid image format: {ext} (supported: {list(mime_types.keys())})"
            logger.error(error_msg)
            raise ValueError(error_msg)

        mime_type = mime_types[ext]

        logger.debug(f"Sending image to API: {image_path} (MIME: {mime_type})")

        # Send image to API
        try:
            with open(image_path, 'rb') as image_file:
                # Include filename and MIME type explicitly (API requires this)
                files = {
                    'query_image': (
                        os.path.basename(image_path),  # Filename
                        image_file,  # File data
                        mime_type  # MIME type (REQUIRED)
                    )
                }

                # Make API request
                response = requests.post(
                    self.API_URL,
                    files=files,
                    timeout=self.timeout
                )

                # Check for HTTP errors
                response.raise_for_status()

                # Parse JSON response
                data = response.json()

            # Extract identification results
            if "items" in data and data["items"]:
                # Get top match (highest confidence)
                item = data["items"][0]

                # Create result dataclass
                result = IdentificationResult(
                    element_id=item.get("id", "unknown"),
                    name=item.get("name", "Unknown Piece"),
                    confidence=item.get("score", 0.0)
                )

                logger.info(
                    f"Piece identified: {result.element_id} ({result.name}) "
                    f"with confidence {result.confidence:.2f}"
                )

                return result

            else:
                # API returned empty results
                error_msg = "No items found in API response - piece not recognized"
                logger.warning(error_msg)
                raise ValueError(error_msg)

        # Handle specific error cases
        except requests.exceptions.Timeout:
            error_msg = f"API request timed out after {self.timeout} seconds"
            logger.error(error_msg)
            raise requests.RequestException(error_msg)

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            logger.error(error_msg)
            raise requests.RequestException(error_msg)

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error: {str(e)}"
            logger.error(error_msg)
            raise requests.RequestException(error_msg)

        except ValueError as e:
            # JSON parsing error or no results
            logger.error(f"Response parsing error: {e}")
            raise

        except Exception as e:
            # Catch any unexpected errors
            error_msg = f"Unexpected error in API call: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise requests.RequestException(error_msg)

# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_identification_api_handler(
        config_manager: EnhancedConfigManager
) -> IdentificationAPIHandler:
    """
    Factory function to create an identification API handler.

    This provides a consistent interface for creating API handlers
    across the application.

    Args:
        config_manager: Configuration manager instance

    Returns:
        Initialized IdentificationAPIHandler

    Example:
        config_manager = create_config_manager()
        api_handler = create_identification_api_handler(config_manager)
        result = api_handler.identify_piece("LegoPictures/piece_001.jpg")
    """
    return IdentificationAPIHandler(config_manager)


# ============================================================================
# MODULE TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test the identification API handler with images from LegoPictures directory.

    This will identify all .jpg images in the LegoPictures directory.
    """
    import glob
    from enhanced_config_manager import create_config_manager

    # Set up logging for test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Testing Identification API Handler...\n")

    # Create config manager and API handler
    config_manager = create_config_manager()
    api_handler = create_identification_api_handler(config_manager)

    # Find all images in LegoPictures directory
    image_dir = "LegoPictures"
    if not os.path.exists(image_dir):
        logger.error(f"Directory not found: {image_dir}")
        logger.info("Please create the directory and add some test images")
        exit(1)

    # Get all jpg images
    image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.gif"]
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(image_dir, pattern)))

    if not image_files:
        logger.error(f"No images found in {image_dir}")
        logger.info("Please add some .jpg, .png, or .gif images to test with")
        exit(1)

    logger.info(f"Found {len(image_files)} images to test\n")

    # Test each image
    successful = 0
    failed = 0

    for image_path in image_files:
        logger.info(f"Testing: {os.path.basename(image_path)}")

        try:
            result = api_handler.identify_piece(image_path)

            logger.info(f"  ✓ SUCCESS")
            logger.info(f"    Element ID: {result.element_id}")
            logger.info(f"    Name: {result.name}")
            logger.info(f"    Confidence: {result.confidence:.2%}")
            logger.info("")

            successful += 1

        except FileNotFoundError as e:
            logger.error(f"  ✗ FAILED: File not found - {e}")
            failed += 1

        except ValueError as e:
            logger.error(f"  ✗ FAILED: Validation error - {e}")
            failed += 1

        except requests.RequestException as e:
            logger.error(f"  ✗ FAILED: API error - {e}")
            failed += 1

        except Exception as e:
            logger.error(f"  ✗ FAILED: Unexpected error - {e}")
            failed += 1

    # Print summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info(f"  Total images: {len(image_files)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info("=" * 60)

    if successful == len(image_files):
        logger.info("✓ ALL TESTS PASSED!")
    else:
        logger.warning(f"⚠ {failed} test(s) failed")