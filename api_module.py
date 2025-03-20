"""
api_module.py - API communication for the Lego sorting application

This module handles all communication with external APIs for piece identification.
"""

import os
import requests
import logging
from typing import Dict, Any

from error_module import APIError, log_and_return

# Get module logger
logger = logging.getLogger(__name__)


class BrickognizeAPI:
    """Handles communication with the Brickognize API for Lego piece identification."""

    @staticmethod
    def send_to_api(image_path: str) -> Dict[str, Any]:
        """Send image to Brickognize API for identification.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with API response or error information
        """
        url = "https://api.brickognize.com/predict/"
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']

        # Input validation with exceptions
        if not os.path.isfile(image_path):
            return log_and_return(
                APIError(f"File not found: {image_path}"),
                logger
            )

        if os.path.getsize(image_path) == 0:
            return log_and_return(
                APIError("File is empty"),
                logger
            )

        if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
            return log_and_return(
                APIError(f"Invalid file type. Allowed: {', '.join(valid_extensions)}"),
                logger
            )

        try:
            logger.info(f"Sending {os.path.basename(image_path)} to Brickognize API")
            with open(image_path, "rb") as image_file:
                files = {"query_image": (image_path, image_file, "image/jpeg")}
                response = requests.post(url, files=files)

            if response.status_code != 200:
                return log_and_return(
                    APIError(f"API request failed: {response.status_code}"),
                    logger
                )

            data = response.json()
            if "items" in data and data["items"]:
                item = data["items"][0]
                result = {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "score": item.get("score")
                }
                logger.info(f"Piece identified: {result['id']} ({result['name']}) with score {result['score']}")
                return result

            return log_and_return(
                APIError("No items found in response"),
                logger
            )

        except Exception as e:
            return log_and_return(
                APIError(f"Request error: {str(e)}"),
                logger
            )


# Factory function
def create_api_client(api_type: str = "brickognize", config_manager=None) -> BrickognizeAPI:
    """Create an API client of the specified type.

    Args:
        api_type: Type of API client ("brickognize" or future implementations)
        config_manager: Configuration manager for API settings

    Returns:
        API client instance

    Raises:
        APIError: If an unsupported API type is specified
    """
    logger.info(f"Creating API client of type: {api_type}")

    if api_type == "brickognize":
        return BrickognizeAPI()
    else:
        logger.error(f"Unsupported API type: {api_type}")
        raise APIError(f"Unsupported API type: {api_type}. Currently only 'brickognize' is supported.")