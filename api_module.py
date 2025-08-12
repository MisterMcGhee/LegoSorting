"""
api_module.py - API communication for the Lego sorting application

This module handles all communication with external APIs for piece identification.
Currently supports only the Brickognize API, but structured to allow future expansion.

The module uses configurable network parameters (timeout, retry count) while keeping
the API endpoint and core functionality simple and focused.
"""

import os
import requests
import logging
from typing import Dict, Any, Optional

from error_module import APIError, log_and_return
from enhanced_config_manager import ModuleConfig

# Get module logger
logger = logging.getLogger(__name__)


class BrickognizeAPI:
    """
    Handles communication with the Brickognize API for Lego piece identification.

    This class manages the HTTP communication with Brickognize's image recognition API.
    Network parameters (timeout, retry count) are configurable, while the API endpoint
    itself is hardcoded since it won't change.
    """

    # API endpoint is constant - Brickognize's public API URL
    API_URL = "https://api.brickognize.com/predict/"

    # Supported image formats for the API
    VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif']

    def __init__(self, timeout: int = 30, retry_count: int = 3):
        """
        Initialize the Brickognize API client with network parameters.

        Args:
            timeout: Maximum seconds to wait for API response (default: 30)
            retry_count: Number of times to retry on failure (default: 3)

        Note: These parameters affect network behavior only. The API endpoint
        and authentication (if ever needed) would be handled separately.
        """
        # Store network configuration
        self.timeout = timeout
        self.retry_count = retry_count

        # Log initialization with actual values being used
        logger.info(f"Initialized BrickognizeAPI with timeout={timeout}s, retry_count={retry_count}")

    def send_to_api(self, image_path: str) -> Dict[str, Any]:
        """
        Send an image to the Brickognize API for Lego piece identification.

        This method handles the complete API interaction including:
        1. File validation
        2. HTTP request with configured timeout
        3. Response parsing
        4. Error handling

        Args:
            image_path: Path to the image file to identify

        Returns:
            Dictionary containing either:
            - Success: {"id": piece_id, "name": piece_name, "score": confidence_score}
            - Failure: {"error": error_message}

        Note: The retry logic is handled by the error_module's retry decorator
        if configured. This method focuses on a single API call attempt.
        """
        # Step 1: Validate the input file exists and is readable
        if not os.path.isfile(image_path):
            error_message = f"File not found: {image_path}"
            logger.error(error_message)
            return {"error": error_message}

        # Step 2: Check file is not empty (common error with failed captures)
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            error_message = f"File is empty: {image_path}"
            logger.error(error_message)
            return {"error": error_message}

        # Step 3: Validate file extension is supported by API
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension not in self.VALID_EXTENSIONS:
            error_message = (
                f"Invalid file type '{file_extension}'. "
                f"Supported formats: {', '.join(self.VALID_EXTENSIONS)}"
            )
            logger.error(error_message)
            return {"error": error_message}

        # Step 4: Prepare and send the API request
        try:
            # Log the API call for debugging
            logger.info(f"Sending {os.path.basename(image_path)} to Brickognize API")
            logger.debug(f"API parameters: timeout={self.timeout}s, URL={self.API_URL}")

            # Open and prepare the image file for upload
            with open(image_path, "rb") as image_file:
                # Format required by Brickognize API
                files = {
                    "query_image": (
                        os.path.basename(image_path),  # filename
                        image_file,  # file object
                        "image/jpeg"  # MIME type
                    )
                }

                # Make the HTTP POST request with configured timeout
                response = requests.post(
                    self.API_URL,
                    files=files,
                    timeout=self.timeout  # Use configured timeout
                )

            # Step 5: Check HTTP response status
            if response.status_code != 200:
                error_message = f"API request failed with status code: {response.status_code}"
                logger.error(error_message)

                # Include response body for debugging if available
                try:
                    logger.debug(f"Response body: {response.text[:500]}")  # First 500 chars
                except:
                    pass  # Don't fail if we can't read response

                return {"error": error_message}

            # Step 6: Parse JSON response
            try:
                data = response.json()
            except ValueError as json_error:
                error_message = f"Invalid JSON in API response: {str(json_error)}"
                logger.error(error_message)
                return {"error": error_message}

            # Step 7: Extract piece information from response
            # Brickognize returns results in an "items" array
            if "items" in data and data["items"]:
                # Get the top match (highest confidence)
                item = data["items"][0]

                # Extract the relevant fields
                result = {
                    "id": item.get("id", "unknown"),
                    "name": item.get("name", "Unknown Piece"),
                    "score": item.get("score", 0.0)
                }

                # Log successful identification
                logger.info(
                    f"Piece identified: {result['id']} "
                    f"({result['name']}) "
                    f"with confidence {result['score']:.2f}"
                )

                return result
            else:
                # API returned empty results - piece not recognized
                error_message = "No items found in API response - piece not recognized"
                logger.warning(error_message)
                return {"error": error_message}

        # Step 8: Handle network and other errors
        except requests.exceptions.Timeout:
            error_message = f"API request timed out after {self.timeout} seconds"
            logger.error(error_message)
            return {"error": error_message}

        except requests.exceptions.ConnectionError as conn_error:
            error_message = f"Connection error: {str(conn_error)}"
            logger.error(error_message)
            return {"error": error_message}

        except requests.exceptions.RequestException as req_error:
            error_message = f"Request error: {str(req_error)}"
            logger.error(error_message)
            return {"error": error_message}

        except Exception as unexpected_error:
            # Catch any other unexpected errors
            error_message = f"Unexpected error in API call: {str(unexpected_error)}"
            logger.error(error_message, exc_info=True)  # Include stack trace
            return {"error": error_message}


def create_api_client(api_type: str = "brickognize",
                      config_manager: Optional[Any] = None) -> BrickognizeAPI:
    """
    Factory function to create an API client with optional configuration.

    This function provides a consistent interface for creating API clients across
    the application. It supports configuration through the config manager while
    maintaining sensible defaults if no configuration is provided.

    Args:
        api_type: Type of API client to create. Currently only "brickognize" is supported.
                 This parameter exists for future extensibility if other APIs are added.

        config_manager: Optional configuration manager instance. If provided, network
                       settings (timeout, retry_count) will be read from configuration.
                       If None, default values will be used.

    Returns:
        BrickognizeAPI: Configured API client instance

    Raises:
        APIError: If an unsupported API type is requested

    Example:
        # With configuration manager (typical usage in main application)
        api_client = create_api_client("brickognize", config_manager)

        # Without configuration (useful for testing or standalone scripts)
        api_client = create_api_client("brickognize")
    """
    # Log which API type is being created
    logger.info(f"Creating API client of type: {api_type}")

    if api_type == "brickognize":
        # Determine configuration source and create client accordingly

        if config_manager is not None:
            # Configuration manager is available - use configured values
            logger.debug("Using configuration manager for API settings")

            try:
                # Get the API module configuration
                api_config = config_manager.get_module_config(ModuleConfig.API.value)

                # Extract only the network-related settings
                # Other settings like api_key or base_url are ignored for Brickognize
                timeout = api_config.get("timeout", 30)
                retry_count = api_config.get("retry_count", 3)

                logger.info(
                    f"Creating BrickognizeAPI with configured settings: "
                    f"timeout={timeout}s, retry_count={retry_count}"
                )

                return BrickognizeAPI(
                    timeout=timeout,
                    retry_count=retry_count
                )

            except Exception as config_error:
                # If configuration reading fails, fall back to defaults
                logger.warning(
                    f"Error reading API configuration, using defaults: {config_error}"
                )
                return BrickognizeAPI()  # Uses default parameters

        else:
            # No configuration manager provided - use defaults
            logger.debug("No configuration manager provided, using default API settings")
            return BrickognizeAPI()  # Uses default parameters

    else:
        # Unsupported API type requested
        error_message = (
            f"Unsupported API type: '{api_type}'. "
            f"Currently only 'brickognize' is supported."
        )
        logger.error(error_message)
        raise APIError(error_message)


# Module-level convenience function for backward compatibility
def identify_piece(image_path: str, config_manager: Optional[Any] = None) -> Dict[str, Any]:
    """
    Convenience function to identify a Lego piece from an image.

    This is a simplified interface that creates a client and makes a single API call.
    Useful for scripts or testing where you don't need to maintain a client instance.

    Args:
        image_path: Path to the image file
        config_manager: Optional configuration manager

    Returns:
        Dictionary with identification results or error

    Example:
        result = identify_piece("LegoPictures/piece_001.jpg")
        if "error" not in result:
            print(f"Found piece: {result['name']}")
    """
    client = create_api_client("brickognize", config_manager)
    return client.send_to_api(image_path)


# Example usage and testing
# Replace the existing __main__ section at the bottom of api_module.py with this:

# Example usage and testing
if __name__ == "__main__":
    """
    Test code for the API module.
    Can be run directly from IDE or command line.

    Usage:
        From command line: python api_module.py [image_path]
        From IDE: Just run the file (uses default test image)
    """
    import sys
    import os

    # Set up logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Determine test image path
    if len(sys.argv) > 1:
        # Command line argument provided
        test_image_path = sys.argv[1]
        print(f"Using image from command line argument: {test_image_path}")
    else:
        # No argument - running from IDE, use default test images
        print("No command line argument provided - checking for test images...")

        # List of possible test image locations (in order of preference)
        possible_test_images = [
            "LegoPictures/test_piece.jpg",
            "LegoPictures/test_piece.png",
            "LegoPictures/test.jpg",
            "LegoPictures/test.png",
            # Check if ANY image exists in LegoPictures for testing
            [f"LegoPictures/{f}" for f in os.listdir("LegoPictures")
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            if os.path.exists("LegoPictures") else []
        ]

        # Find first existing test image
        test_image_path = None
        for possible_path in possible_test_images:
            if os.path.exists(possible_path):
                test_image_path = possible_path
                print(f"Found test image: {test_image_path}")
                break

        if test_image_path is None:
            print("\n" + "=" * 60)
            print("ERROR: No test image found!")
            print("=" * 60)
            print("\nTo run tests from IDE, please do ONE of the following:")
            print("1. Place a test image at: LegoPictures/test_piece.jpg")
            print("2. Place any .jpg or .png image in the LegoPictures folder")
            print("3. Run with command line: python api_module.py <image_path>")
            print("\nCurrent working directory:", os.getcwd())

            # Check if LegoPictures exists
            if os.path.exists("LegoPictures"):
                files = os.listdir("LegoPictures")
                if files:
                    print(f"\nFiles found in LegoPictures: {files[:5]}")  # Show first 5
                else:
                    print("\nLegoPictures folder exists but is empty!")
            else:
                print("\nLegoPictures folder does not exist!")
                print("Creating LegoPictures folder...")
                os.makedirs("LegoPictures", exist_ok=True)
                print("Folder created. Please add a test image.")

            sys.exit(1)

    # Verify the test image actually exists before proceeding
    if not os.path.exists(test_image_path):
        print(f"\nERROR: Image file does not exist: {test_image_path}")
        print(f"Current working directory: {os.getcwd()}")
        sys.exit(1)

    # Display test information
    print("\n" + "=" * 60)
    print("API MODULE TEST SUITE")
    print("=" * 60)
    print(f"Test image: {test_image_path}")
    print(f"File size: {os.path.getsize(test_image_path):,} bytes")
    print("=" * 60 + "\n")

    # Test 1: Create client without configuration
    print("=== Test 1: Client without configuration ===")
    try:
        client_no_config = create_api_client("brickognize")
        print(f"✓ Created client with defaults: timeout={client_no_config.timeout}s")
        print(f"  Retry count: {client_no_config.retry_count}")
    except Exception as e:
        print(f"✗ Failed to create client: {e}")
        sys.exit(1)

    # Test 2: Test API call
    print("\n=== Test 2: API call test ===")
    print(f"Sending {os.path.basename(test_image_path)} to Brickognize API...")

    result = client_no_config.send_to_api(test_image_path)

    if "error" in result:
        print(f"✗ API Error: {result['error']}")
        print("\nTroubleshooting tips:")
        print("- Check your internet connection")
        print("- Verify the image is a clear photo of a Lego piece")
        print("- Ensure the image file is not corrupted")
    else:
        print(f"✓ Success! Identified piece:")
        print(f"  ID:         {result['id']}")
        print(f"  Name:       {result['name']}")
        print(f"  Confidence: {result['score']:.2%}")

        # Additional info for high/low confidence
        if result['score'] > 0.9:
            print(f"  → High confidence match!")
        elif result['score'] < 0.5:
            print(f"  → Low confidence - piece might be unclear or unusual")

    # Test 3: Test with simulated config manager
    print("\n=== Test 3: Client with simulated configuration ===")


    # Create a mock config manager for testing
    class MockConfigManager:
        def get_module_config(self, module_name):
            return {
                "timeout": 60,  # Longer timeout
                "retry_count": 5,  # More retries
                "api_key": "not_used",  # Ignored for Brickognize
                "base_url": "not_used"  # Ignored for Brickognize
            }


    try:
        mock_config = MockConfigManager()
        client_with_config = create_api_client("brickognize", mock_config)
        print(f"✓ Created client with custom config:")
        print(f"  Timeout: {client_with_config.timeout}s (vs default 30s)")
        print(f"  Retry count: {client_with_config.retry_count} (vs default 3)")
    except Exception as e:
        print(f"✗ Failed to create configured client: {e}")

    # Test 4: Test error handling with invalid API type
    print("\n=== Test 4: Error handling test ===")
    try:
        invalid_client = create_api_client("invalid_api_type")
        print("✗ Should have raised an error for invalid API type!")
    except APIError as e:
        print(f"✓ Correctly raised APIError: {e}")
    except Exception as e:
        print(f"✗ Unexpected error type: {e}")

    # Test 5: Test convenience function
    print("\n=== Test 5: Convenience function test ===")
    try:
        quick_result = identify_piece(test_image_path)
        if "error" not in quick_result:
            print(f"✓ identify_piece() worked: Found {quick_result['name']}")
        else:
            print(f"✗ identify_piece() returned error: {quick_result['error']}")
    except Exception as e:
        print(f"✗ identify_piece() failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)
    print("\nNote: This test uses the REAL Brickognize API.")
    print("Results depend on your internet connection and image quality.")
    print("For best results, use a clear image of a single Lego piece")
    print("on a plain background with good lighting.")