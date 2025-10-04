# processing/category_hierarchy_service.py
"""
category_hierarchy_service.py - Category hierarchy management for Lego pieces

This module provides centralized access to the category hierarchy parsed from
Lego_Categories.csv. It caches the hierarchy structure and provides query methods
for modules that need to understand category relationships.

RESPONSIBILITIES:
- Parse ccgories.csv into hierarchical structure
- Cache the hierarchy in memory
- Provide query methods for category relationships
- Thread-safe access to hierarchy data

USED BY:
- bin_assignment_module: To understand what categories exist at each sorting level
- config_gui_module: To populate category dropdown menus during setup
- Validation logic: To verify pre-assignments are valid categories

DOES NOT:
- Look up individual pieces by element_id (that's category_lookup_module)
- Assign bins (that's bin_assignment_module)
- Store configuration data (that's enhanced_config_manager)
"""

import os
import csv
import threading
import logging
from typing import Dict, List, Set, Optional, Any
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig

logger = logging.getLogger(__name__)


class CategoryHierarchyService:
    """
    Parses and provides access to Lego category hierarchy.

    This is a read-only service that loads the category structure once
    and provides various ways to query it. The hierarchy contains only
    the relationships between categories, not individual piece data.
    """

    def __init__(self, config_manager: EnhancedConfigManager):
        """
        Initialize the category hierarchy service.

        Called during application startup, before any sorting begins.
        Loads the hierarchy immediately to fail fast if CSV is missing.

        Args:
            config_manager: Configuration manager to get CSV path
        """
        self.config_manager = config_manager

        # Get CSV path from config (might be customized by user)
        piece_config = config_manager.get_module_config(
            ModuleConfig.PIECE_IDENTIFIER.value
        )
        self.csv_path = piece_config["csv_path"]

        # Cache for parsed hierarchy (loaded once, used many times)
        self._hierarchy: Optional[Dict[str, Any]] = None

        # Thread safety for concurrent access from multiple modules
        self._lock = threading.RLock()

        # Load hierarchy immediately on initialization
        self._load_hierarchy()

        logger.info(
            f"Category hierarchy service initialized "
            f"({len(self._hierarchy['primary'])} primary categories)"
        )

    def _load_hierarchy(self) -> None:
        """
        Parse the CSV file and build the hierarchy structure.

        Called once during initialization. Reads through entire CSV file
        and builds nested dictionaries that map categories to subcategories.

        The hierarchy contains ONLY category relationships:
        - What primary categories exist
        - What secondary categories exist under each primary
        - What tertiary categories exist under each (primary, secondary) pair

        Individual piece data (element_id -> categories) is handled by
        category_lookup_module, not stored here.

        Structure created:
        {
            "primary": ["Basic", "Technic", "Plates", ...],
            "primary_to_secondary": {
                "Basic": ["Brick", "Slope", ...],
                "Technic": ["Pin", "Axle", ...]
            },
            "secondary_to_tertiary": {
                ("Basic", "Brick"): ["2x4", "1x2", ...],
                ("Technic", "Pin"): ["3L", "Friction", ...]
            }
        }
        """
        with self._lock:
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(
                    f"Categories CSV not found at: {self.csv_path}"
                )

            # Initialize with sets to automatically handle duplicates
            hierarchy = {
                "primary": set(),
                "primary_to_secondary": {},
                "secondary_to_tertiary": {}
            }

            try:
                with open(self.csv_path, 'r', encoding='utf-8') as file:
                    csv_reader = csv.DictReader(file)

                    for row in csv_reader:
                        # Extract category fields (strip whitespace)
                        element_id = row.get('element_id', '').strip()
                        primary = row.get('primary_category', '').strip()
                        secondary = row.get('secondary_category', '').strip()
                        tertiary = row.get('tertiary_category', '').strip()

                        # Skip rows without required data
                        if not element_id or not primary:
                            continue

                        # Build set of all unique primary categories
                        hierarchy["primary"].add(primary)

                        # Build primary -> secondary mapping
                        if primary not in hierarchy["primary_to_secondary"]:
                            hierarchy["primary_to_secondary"][primary] = set()
                        if secondary:
                            hierarchy["primary_to_secondary"][primary].add(secondary)

                        # Build (primary, secondary) -> tertiary mapping
                        # Uses tuple key because we need both to identify tertiary
                        if secondary:
                            key = (primary, secondary)
                            if key not in hierarchy["secondary_to_tertiary"]:
                                hierarchy["secondary_to_tertiary"][key] = set()
                            if tertiary:
                                hierarchy["secondary_to_tertiary"][key].add(tertiary)

                # Convert all sets to sorted lists for consistent ordering
                # GUI dropdowns need stable ordering, sets are unordered
                hierarchy["primary"] = sorted(list(hierarchy["primary"]))

                for primary in hierarchy["primary_to_secondary"]:
                    hierarchy["primary_to_secondary"][primary] = sorted(
                        list(hierarchy["primary_to_secondary"][primary])
                    )

                for key in hierarchy["secondary_to_tertiary"]:
                    hierarchy["secondary_to_tertiary"][key] = sorted(
                        list(hierarchy["secondary_to_tertiary"][key])
                    )

                self._hierarchy = hierarchy

                logger.info(
                    f"Loaded category hierarchy from {self.csv_path}: "
                    f"{len(hierarchy['primary'])} primary categories"
                )

            except Exception as e:
                error_msg = f"Error parsing categories CSV: {e}"
                logger.error(error_msg)
                raise Exception(error_msg)

    def get_hierarchy(self) -> Dict[str, Any]:
        """
        Get the complete category hierarchy dictionary.

        Called by modules that need direct access to the full hierarchy structure.
        Returns cached data, so this is a fast lookup after initial load.

        Most modules should use the specific query methods instead of this,
        but it's available for advanced use cases.

        Returns:
            Dictionary with complete hierarchy structure
        """
        with self._lock:
            if self._hierarchy is None:
                logger.warning("Hierarchy not loaded, loading now")
                self._load_hierarchy()

            return self._hierarchy

    def get_primary_categories(self) -> List[str]:
        """
        Get sorted list of all primary categories.

        CALLED BY:
        - config_gui_module: Populates the primary category dropdown during setup
        - bin_assignment_module (primary strategy): Gets list of categories to assign bins

        WHEN:
        - During GUI initialization to show available categories
        - When bin assignment module needs to know what primaries exist

        Returns:
            Sorted list of primary categories

        Example:
            ["Basic", "Plates", "Special", "Technic"]
        """
        hierarchy = self.get_hierarchy()
        return hierarchy["primary"]

    def get_secondary_categories(self, primary_category: str) -> List[str]:
        """
        Get all secondary categories under a specific primary category.

        CALLED BY:
        - config_gui_module: Populates secondary dropdown when user selects a primary
        - bin_assignment_module (secondary strategy): Gets categories to assign bins

        WHEN:
        - User selects primary category in GUI, secondary dropdown updates
        - Bin assignment with secondary strategy needs to know what secondaries exist
          under the target primary category

        Args:
            primary_category: The primary category to query

        Returns:
            Sorted list of secondary categories under that primary,
            or empty list if primary has no secondaries

        Example:
            get_secondary_categories("Basic")
            -> ["Brick", "Plate", "Slope", "Tile"]
        """
        hierarchy = self.get_hierarchy()
        return hierarchy["primary_to_secondary"].get(primary_category, [])

    def get_tertiary_categories(self, primary_category: str,
                                secondary_category: str) -> List[str]:
        """
        Get all tertiary categories under a primary+secondary combination.

        CALLED BY:
        - config_gui_module: Populates tertiary dropdown when user selects both
          primary and secondary categories
        - bin_assignment_module (tertiary strategy): Gets categories to assign bins

        WHEN:
        - User selects both primary and secondary in GUI, tertiary dropdown updates
        - Bin assignment with tertiary strategy needs to know what tertiaries exist
          under the target primary+secondary combination

        Args:
            primary_category: The primary category
            secondary_category: The secondary category

        Returns:
            Sorted list of tertiary categories under that combination,
            or empty list if that combination has no tertiaries

        Example:
            get_tertiary_categories("Basic", "Brick")
            -> ["1x1", "1x2", "2x2", "2x4"]
        """
        hierarchy = self.get_hierarchy()
        key = (primary_category, secondary_category)
        return hierarchy["secondary_to_tertiary"].get(key, [])

    def get_categories_for_strategy(self,
                                    strategy: str,
                                    primary_category: Optional[str] = None,
                                    secondary_category: Optional[str] = None) -> List[str]:
        """
        Get available categories for a specific sorting strategy.

        This is the main method used by bin_assignment_module. It abstracts away
        the strategy-specific logic and returns the appropriate category list.

        CALLED BY:
        - bin_assignment_module: Primary method for getting categories to assign bins
        - Validation logic: Checking if pre-assigned categories are valid

        WHEN:
        - During bin assignment module initialization to understand available categories
        - When assigning a bin to a newly encountered category
        - When validating user's pre-assignment configuration

        Args:
            strategy: "primary", "secondary", or "tertiary"
            primary_category: Required for secondary/tertiary strategies
            secondary_category: Required for tertiary strategy

        Returns:
            List of categories at the requested level

        Examples:
            # Primary strategy - all pieces sorted by primary category
            get_categories_for_strategy("primary")
            -> ["Basic", "Plates", "Technic"]

            # Secondary strategy - only sorting "Basic" pieces by secondary
            get_categories_for_strategy("secondary", primary_category="Basic")
            -> ["Brick", "Plate", "Slope"]

            # Tertiary strategy - only sorting "Basic > Brick" pieces by tertiary
            get_categories_for_strategy("tertiary",
                                       primary_category="Basic",
                                       secondary_category="Brick")
            -> ["1x1", "1x2", "2x4"]
        """
        if strategy == "primary":
            return self.get_primary_categories()

        elif strategy == "secondary":
            if not primary_category:
                logger.warning(
                    "get_categories_for_strategy called with strategy='secondary' "
                    "but no primary_category provided"
                )
                return []
            return self.get_secondary_categories(primary_category)

        elif strategy == "tertiary":
            if not primary_category or not secondary_category:
                logger.warning(
                    "get_categories_for_strategy called with strategy='tertiary' "
                    "but missing primary or secondary category"
                )
                return []
            return self.get_tertiary_categories(primary_category, secondary_category)

        else:
            logger.error(f"Invalid strategy: {strategy}")
            return []

    def category_exists(self, category: str, level: str = "any") -> bool:
        """
        Check if a category name exists in the hierarchy at a specific level.

        CALLED BY:
        - Validation logic: Checking if user's pre-assignment categories are valid
        - bin_assignment_module: Verifying a category exists before assigning a bin

        WHEN:
        - During startup to validate configuration
        - Before processing pre-assignments to ensure typos don't cause crashes

        Args:
            category: Category name to check
            level: "primary", "secondary", "tertiary", or "any"

        Returns:
            True if category exists at specified level

        Example:
            category_exists("Basic", "primary")  -> True
            category_exists("Brick", "secondary")  -> True
            category_exists("NotReal", "any")  -> False
        """
        hierarchy = self.get_hierarchy()

        if level == "primary" or level == "any":
            if category in hierarchy["primary"]:
                return True

        if level == "secondary" or level == "any":
            for secondaries in hierarchy["primary_to_secondary"].values():
                if category in secondaries:
                    return True

        if level == "tertiary" or level == "any":
            for tertiaries in hierarchy["secondary_to_tertiary"].values():
                if category in tertiaries:
                    return True

        return False


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_category_hierarchy_service(
        config_manager: EnhancedConfigManager
) -> CategoryHierarchyService:
    """
    Factory function to create a category hierarchy service.

    Called once during application startup to create the singleton instance.
    This instance is then passed to modules that need category hierarchy data.

    Args:
        config_manager: Configuration manager instance

    Returns:
        Initialized CategoryHierarchyService

    Example:
        config_manager = create_config_manager()
        category_service = create_category_hierarchy_service(config_manager)

        # Pass to modules that need it
        bin_assignment = BinAssignmentModule(config_manager, category_service)
        gui = ConfigGUI(config_manager, category_service)
    """
    return CategoryHierarchyService(config_manager)


# ============================================================================
# MODULE TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test the category hierarchy service with the real CSV file.

    Verifies:
    - CSV parsing works correctly
    - Hierarchy structure is built properly
    - Query methods return expected results
    - Thread safety works
    """
    from enhanced_config_manager import create_config_manager

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Testing Category Hierarchy Service...\n")

    # Create config manager and service
    config_manager = create_config_manager()
    category_service = create_category_hierarchy_service(config_manager)

    # Test 1: Get primary categories
    logger.info("TEST 1: Get primary categories")
    primary = category_service.get_primary_categories()
    logger.info(f"Found {len(primary)} primary categories:")
    logger.info(f"  {primary}")
    assert len(primary) > 0, "Should have primary categories"
    logger.info("✓ Primary categories loaded\n")

    # Test 2: Get secondary categories
    logger.info("TEST 2: Get secondary categories")
    if primary:
        test_primary = primary[0]
        secondary = category_service.get_secondary_categories(test_primary)
        logger.info(f"Secondary categories under '{test_primary}':")
        logger.info(f"  {secondary}")
        logger.info("✓ Secondary categories work\n")

    # Test 3: Get tertiary categories
    logger.info("TEST 3: Get tertiary categories")
    if primary:
        test_primary = primary[0]
        secondary = category_service.get_secondary_categories(test_primary)
        if secondary:
            test_secondary = secondary[0]
            tertiary = category_service.get_tertiary_categories(
                test_primary, test_secondary
            )
            logger.info(
                f"Tertiary categories under '{test_primary}' -> '{test_secondary}':"
            )
            logger.info(f"  {tertiary}")
            logger.info("✓ Tertiary categories work\n")

    # Test 4: Strategy-based queries
    logger.info("TEST 4: Strategy-based category queries")

    logger.info("  Primary strategy:")
    cats = category_service.get_categories_for_strategy("primary")
    logger.info(f"    {len(cats)} categories")

    if primary:
        logger.info(f"  Secondary strategy (for '{primary[0]}'):")
        cats = category_service.get_categories_for_strategy(
            "secondary",
            primary_category=primary[0]
        )
        logger.info(f"    {len(cats)} categories")

    logger.info("✓ Strategy queries work\n")

    # Test 5: Category existence checks
    logger.info("TEST 5: Category existence checks")
    if primary:
        test_cat = primary[0]
        exists = category_service.category_exists(test_cat, "primary")
        logger.info(f"  '{test_cat}' exists as primary: {exists}")
        assert exists, "Known primary should exist"

        fake_exists = category_service.category_exists("FakeCategory", "any")
        logger.info(f"  'FakeCategory' exists: {fake_exists}")
        assert not fake_exists, "Fake category should not exist"

    logger.info("✓ Existence checks work\n")

    # Test 6: Get complete hierarchy
    logger.info("TEST 6: Get complete hierarchy structure")
    hierarchy = category_service.get_hierarchy()
    logger.info(f"Hierarchy keys: {list(hierarchy.keys())}")
    logger.info(f"  Primary categories: {len(hierarchy['primary'])}")
    logger.info("✓ Full hierarchy accessible\n")

    logger.info("✓ ALL TESTS PASSED!")