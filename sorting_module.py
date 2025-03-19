import csv
from typing import Dict, Any


class SortingStrategy:
    """Base class for Lego piece sorting with dynamic bin assignment."""

    def __init__(self, config: Dict[str, Any], categories_data: Dict[str, Dict[str, str]]):
        """Initialize sorting strategy.

        Args:
            config: Configuration dictionary
            categories_data: Dictionary mapping element_ids to category information
        """
        self.config = config
        self.categories_data = categories_data
        self.overflow_bin = config.get("overflow_bin", 9)

        # For dynamic bin assignment
        self.category_to_bin = {}  # Maps category keys to bin numbers
        self.next_available_bin = 0
        self.max_bins = config.get("max_bins", 9)  # Maximum number of bins (excluding overflow)

    def get_bin(self, element_id: str, confidence: float) -> int:
        """Determine bin for a piece (to be implemented by specific strategies).

        Args:
            element_id: Element ID of the Lego piece
            confidence: Confidence score of the identification

        Returns:
            Bin number
        """
        raise NotImplementedError("Subclasses must implement get_bin method")


class PrimaryCategorySorter(SortingStrategy):
    """Sort Lego pieces by their primary category."""

    def get_bin(self, element_id: str, confidence: float) -> int:
        """Assign bin based on primary category.

        Args:
            element_id: Element ID of the Lego piece
            confidence: Confidence score of the identification

        Returns:
            Bin number
        """
        if element_id not in self.categories_data:
            return self.overflow_bin

        piece_info = self.categories_data[element_id]
        category_key = piece_info.get("primary_category")

        # If we've already assigned a bin to this category, use it
        if category_key in self.category_to_bin:
            return self.category_to_bin[category_key]

        # If we have room for a new bin, assign the next available one
        if self.next_available_bin < self.max_bins:
            bin_number = self.next_available_bin
            self.category_to_bin[category_key] = bin_number
            self.next_available_bin += 1
            return bin_number

        # Otherwise, use the overflow bin
        return self.overflow_bin

    def get_description(self) -> str:
        """Get description of this sorting strategy.

        Returns:
            Human-readable description
        """
        return "Sorting by primary category"


class SecondaryCategorySorter(SortingStrategy):
    """Sort Lego pieces by their secondary category within a specific primary category."""

    def __init__(self, config: Dict[str, Any], categories_data: Dict[str, Dict[str, str]]):
        super().__init__(config, categories_data)
        self.target_primary = config.get("target_primary_category")

        if not self.target_primary:
            raise ValueError("Secondary category sorting requires a target primary category")

    def get_bin(self, element_id: str, confidence: float) -> int:
        """Assign bin based on secondary category within target primary category.

        Args:
            element_id: Element ID of the Lego piece
            confidence: Confidence score of the identification

        Returns:
            Bin number
        """
        if element_id not in self.categories_data:
            return self.overflow_bin

        piece_info = self.categories_data[element_id]

        # Check if this piece belongs to the target primary category
        if piece_info.get("primary_category") != self.target_primary:
            return self.overflow_bin

        # Use secondary category as the key
        category_key = piece_info.get("secondary_category")

        # If we've already assigned a bin to this category, use it
        if category_key in self.category_to_bin:
            return self.category_to_bin[category_key]

        # If we have room for a new bin, assign the next available one
        if self.next_available_bin < self.max_bins:
            bin_number = self.next_available_bin
            self.category_to_bin[category_key] = bin_number
            self.next_available_bin += 1
            return bin_number

        # Otherwise, use the overflow bin
        return self.overflow_bin

    def get_description(self) -> str:
        """Get description of this sorting strategy.

        Returns:
            Human-readable description
        """
        return f"Sorting by secondary category within primary category '{self.target_primary}'"


class TertiaryCategorySorter(SortingStrategy):
    """Sort Lego pieces by their tertiary category within specific primary and secondary categories."""

    def __init__(self, config: Dict[str, Any], categories_data: Dict[str, Dict[str, str]]):
        super().__init__(config, categories_data)
        self.target_primary = config.get("target_primary_category")
        self.target_secondary = config.get("target_secondary_category")

        if not self.target_primary or not self.target_secondary:
            raise ValueError("Tertiary category sorting requires target primary and secondary categories")

    def get_bin(self, element_id: str, confidence: float) -> int:
        """Assign bin based on tertiary category within target primary and secondary categories.

        Args:
            element_id: Element ID of the Lego piece
            confidence: Confidence score of the identification

        Returns:
            Bin number
        """
        if element_id not in self.categories_data:
            return self.overflow_bin

        piece_info = self.categories_data[element_id]

        # Check if this piece belongs to the target primary and secondary categories
        if (piece_info.get("primary_category") != self.target_primary or
                piece_info.get("secondary_category") != self.target_secondary):
            return self.overflow_bin

        # Use tertiary category as the key
        category_key = piece_info.get("tertiary_category")

        # If we've already assigned a bin to this category, use it
        if category_key in self.category_to_bin:
            return self.category_to_bin[category_key]

        # If we have room for a new bin, assign the next available one
        if self.next_available_bin < self.max_bins:
            bin_number = self.next_available_bin
            self.category_to_bin[category_key] = bin_number
            self.next_available_bin += 1
            return bin_number

        # Otherwise, use the overflow bin
        return self.overflow_bin

    def get_description(self) -> str:
        """Get description of this sorting strategy.

        Returns:
            Human-readable description
        """
        return f"Sorting by tertiary category within '{self.target_primary}/{self.target_secondary}'"


class SortingManager:
    """Manages piece sorting based on configured strategies."""

    def __init__(self, config_manager):
        """Initialize sorting manager.

        Args:
            config_manager: Configuration manager object
        """
        self.config_manager = config_manager
        self.sorting_config = config_manager.get_section("sorting")
        self.categories_data = {}

        # Load categories data
        csv_path = config_manager.get("piece_identifier", "csv_path", "Lego_Categories.csv")
        self._load_categories(csv_path)

        # Set up sorting strategy based on config
        self.strategy = self._configure_strategy()

    def _load_categories(self, filepath: str) -> None:
        """Load piece categories from CSV file.

        Args:
            filepath: Path to CSV file
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    self.categories_data[row['element_id']] = {
                        'name': row['name'],
                        'primary_category': row['primary_category'],
                        'secondary_category': row['secondary_category'],
                        'tertiary_category': row.get('tertiary_category', '')
                    }
            print(f"Loaded {len(self.categories_data)} pieces from {filepath}")
        except FileNotFoundError:
            print(f"Error: Could not find file at {filepath}")
        except Exception as e:
            print(f"Error reading CSV file: {e}")

    def _configure_strategy(self) -> SortingStrategy:
        """Configure sorting strategy based on config settings.

        Returns:
            Configured SortingStrategy
        """
        # Get the strategy type or default to primary
        strategy_type = self.sorting_config.get("strategy", "primary")

        try:
            if strategy_type == "primary":
                return PrimaryCategorySorter(self.sorting_config, self.categories_data)

            elif strategy_type == "secondary":
                return SecondaryCategorySorter(self.sorting_config, self.categories_data)

            elif strategy_type == "tertiary":
                return TertiaryCategorySorter(self.sorting_config, self.categories_data)

            else:
                print(f"Warning: Unknown sorting strategy '{strategy_type}'.")
                print("Defaulting to primary category sorting.")
                return PrimaryCategorySorter(self.sorting_config, self.categories_data)

        except ValueError as e:
            print(f"Error configuring sorting strategy: {e}")
            print("Defaulting to primary category sorting.")
            return PrimaryCategorySorter(self.sorting_config, self.categories_data)

    def identify_piece(self, api_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process API result and determine sorting bin.

        Args:
            api_result: Result from Brickognize API

        Returns:
            Dictionary with identification results and bin number
        """
        # Extract information from API result
        element_id = api_result.get("id")
        confidence = api_result.get("score", 0)

        # Create result dictionary
        result = {
            "id": element_id,
            "confidence": confidence,
            "bin_number": self.sorting_config.get("overflow_bin", 9)  # Default to overflow bin
        }

        # Check confidence threshold
        confidence_threshold = self.config_manager.get(
            "piece_identifier", "confidence_threshold", 0.7
        )
        if confidence < confidence_threshold:
            result["error"] = "Low confidence score"
            return result

        # Check if piece exists in our database
        if not element_id or element_id not in self.categories_data:
            result["error"] = "Piece not found in dictionary"
            return result

        # Add piece information to result
        piece_info = self.categories_data[element_id]
        result.update({
            "element_id": element_id,
            "name": piece_info['name'],
            "primary_category": piece_info['primary_category'],
            "secondary_category": piece_info['secondary_category'],
            "tertiary_category": piece_info.get('tertiary_category', '')
        })

        # Get bin number from strategy
        result["bin_number"] = self.strategy.get_bin(element_id, confidence)

        return result

    def get_strategy_description(self) -> str:
        """Get description of the current sorting strategy.

        Returns:
            Human-readable description of the sorting strategy
        """
        return self.strategy.get_description()

    def get_bin_mapping(self) -> Dict[str, int]:
        """Get current mapping of categories to bins.

        Returns:
            Dictionary mapping category names to bin numbers
        """
        return self.strategy.category_to_bin.copy()


def create_sorting_manager(config_manager):
    """Factory function to create a sorting manager.

    Args:
        config_manager: Configuration manager object

    Returns:
        SortingManager instance
    """
    return SortingManager(config_manager)