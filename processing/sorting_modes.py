"""
sorting_modes.py - Sorting mode strategies for bin assignment

A sorting "mode" is the top-level choice of how identified pieces are routed
to physical bins. The identification pipeline (Brickognize API → element_id
lookup → category lookup) is the same for every mode; only the final
"which bin?" decision changes.

AVAILABLE MODES:
  - design_id: route by shape category. Takes a "tier" parameter — primary,
               secondary, or tertiary — each tier being a successive
               subdivision of the bulk feed.
  - bag:       route by the piece's original bag number in a specific LEGO
               set inventory, keyed on element_id.

RESPONSIBILITIES:
- Compute the sort key for an IdentifiedPiece
- List the sort keys available for pre-assignment validation
- Validate the mode-specific section of the sorting configuration

DOES NOT:
- Track bin assignments (that's bin_assignment_module)
- Load or parse category databases (that's category_lookup_module /
  category_hierarchy_service)
- Run API calls or element_id lookups (handled by the processing coordinator
  before sort_key_for is called)

USAGE:
    mode = create_sorting_mode(config_manager, category_service)
    key = mode.sort_key_for(identified_piece)   # "Basic", "bag_3", or "Unknown"
    bin_num = bin_assignment._assign_or_create_bin(key)
"""

import csv
import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Type

from enhanced_config_manager import EnhancedConfigManager, ModuleConfig

if TYPE_CHECKING:
    # Heavy imports (numpy, CSV parsing, etc.) are kept behind TYPE_CHECKING so
    # the validator can load MODE_REGISTRY without requiring the full runtime
    # stack just to know which modes exist.
    from processing.category_hierarchy_service import CategoryHierarchyService
    from processing.processing_data_models import IdentifiedPiece

logger = logging.getLogger(__name__)


# ============================================================================
# ABSTRACT BASE
# ============================================================================


class SortingMode(ABC):
    """
    Strategy for turning an identified piece into a sort key.

    A sort key is the string bin_assignment_module maps to a bin number.
    Returning "Unknown" routes the piece to the overflow bin.
    """

    @abstractmethod
    def sort_key_for(self, piece: "IdentifiedPiece") -> str:
        """Return the sort key for this piece, or "Unknown" to overflow."""

    def available_keys(self) -> List[str]:
        """
        Return the universe of sort keys that can be pre-assigned.

        Modes that don't have a finite enumerable set of keys (e.g. bag mode,
        where legitimate keys depend on the inventory file) should return
        whatever they can enumerate; callers use this list only for
        pre-assignment validation.
        """
        return []

    def is_valid_key(self, key: str) -> bool:
        """Check whether a key is valid for pre-assignment under this mode."""
        available = self.available_keys()
        return not available or key in available

    @classmethod
    @abstractmethod
    def validate_config(cls, sorting_config: Dict) -> List[str]:
        """Return a list of error messages for this mode's config section."""


# ============================================================================
# DESIGN-ID MODE (primary / secondary / tertiary tiers)
# ============================================================================


class DesignIdMode(SortingMode):
    """
    Route pieces by shape category, at a configurable tier.

    The three tiers form a progressive sort:
      - primary:   sort every piece by primary category.
      - secondary: take the bulk of a chosen primary and sort by secondary.
      - tertiary:  take the bulk of a chosen primary+secondary and sort by
                   tertiary.

    Pieces that don't match the tier filter route to overflow.
    """

    TIERS = ("primary", "secondary", "tertiary")

    def __init__(self, config_manager: EnhancedConfigManager,
                 category_service: "CategoryHierarchyService"):
        sorting_config = config_manager.get_module_config(ModuleConfig.SORTING.value)
        tier_config = sorting_config.get("design_id", {})

        self.category_service = category_service
        self.tier = tier_config.get("tier", "primary")
        self.target_primary = tier_config.get("target_primary_category", "")
        self.target_secondary = tier_config.get("target_secondary_category", "")

        logger.info(
            f"DesignIdMode initialized: tier={self.tier}, "
            f"target_primary={self.target_primary!r}, "
            f"target_secondary={self.target_secondary!r}"
        )

    def sort_key_for(self, piece: "IdentifiedPiece") -> str:
        if self.tier == "primary":
            return piece.primary_category or "Unknown"

        if self.tier == "secondary":
            if piece.primary_category != self.target_primary:
                return "Unknown"
            return piece.secondary_category or "Unknown"

        if self.tier == "tertiary":
            if (piece.primary_category != self.target_primary
                    or piece.secondary_category != self.target_secondary):
                return "Unknown"
            return piece.tertiary_category or "Unknown"

        logger.error(f"DesignIdMode reached unreachable tier: {self.tier!r}")
        return "Unknown"

    def available_keys(self) -> List[str]:
        return self.category_service.get_categories_for_strategy(
            self.tier,
            primary_category=self.target_primary if self.tier != "primary" else None,
            secondary_category=self.target_secondary if self.tier == "tertiary" else None,
        )

    @classmethod
    def validate_config(cls, sorting_config: Dict) -> List[str]:
        errors: List[str] = []
        tier_config = sorting_config.get("design_id", {})
        tier = tier_config.get("tier")

        if tier not in cls.TIERS:
            errors.append(
                f"Invalid design_id.tier: {tier!r} (must be one of {list(cls.TIERS)})"
            )
            return errors

        if tier in ("secondary", "tertiary"):
            if not tier_config.get("target_primary_category"):
                errors.append(
                    f"design_id.tier={tier!r} requires target_primary_category"
                )
        if tier == "tertiary":
            if not tier_config.get("target_secondary_category"):
                errors.append(
                    "design_id.tier='tertiary' requires target_secondary_category"
                )

        return errors


# ============================================================================
# BAG MODE (element_id → bag_number inventory)
# ============================================================================


class BagMode(SortingMode):
    """
    Route pieces by their original bag number in a LEGO set inventory.

    The inventory file maps element_id → bag_number. At sort time, the mode
    looks up the piece's element_id (populated upstream by
    element_id_lookup_module) and returns "bag_{n}". Pieces whose element_id
    isn't in the inventory route to overflow.
    """

    def __init__(self, config_manager: EnhancedConfigManager):
        sorting_config = config_manager.get_module_config(ModuleConfig.SORTING.value)
        bag_config = sorting_config.get("bag", {})

        self.inventory_path = bag_config.get("inventory_path", "")
        self.inventory_format = bag_config.get("inventory_format", "csv")
        self.set_id = bag_config.get("set_id", "")

        self._element_to_bag: Dict[str, int] = self._load_inventory(
            self.inventory_path, self.inventory_format
        )

        logger.info(
            f"BagMode initialized: set_id={self.set_id!r}, "
            f"inventory_path={self.inventory_path!r}, "
            f"{len(self._element_to_bag)} element mappings loaded"
        )

    @staticmethod
    def _load_inventory(path: str, fmt: str) -> Dict[str, int]:
        """
        Parse inventory file into {element_id: bag_number}.

        CSV format: two columns named `element_id` and `bag_number`.
        """
        if not path:
            return {}
        if not os.path.exists(path):
            logger.error(f"Bag inventory file not found: {path}")
            return {}

        mapping: Dict[str, int] = {}

        if fmt == "csv":
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    element_id = (row.get("element_id") or "").strip()
                    bag_raw = (row.get("bag_number") or "").strip()
                    if not element_id or not bag_raw:
                        continue
                    try:
                        mapping[element_id] = int(bag_raw)
                    except ValueError:
                        logger.warning(
                            f"Bag inventory: non-integer bag_number {bag_raw!r} "
                            f"for element_id {element_id}"
                        )
        else:
            logger.error(f"Unsupported bag inventory format: {fmt!r}")

        return mapping

    def sort_key_for(self, piece: "IdentifiedPiece") -> str:
        if piece.element_id is None:
            return "Unknown"
        bag = self._element_to_bag.get(piece.element_id)
        if bag is None:
            return "Unknown"
        return f"bag_{bag}"

    def available_keys(self) -> List[str]:
        unique_bags = sorted(set(self._element_to_bag.values()))
        return [f"bag_{n}" for n in unique_bags]

    @classmethod
    def validate_config(cls, sorting_config: Dict) -> List[str]:
        errors: List[str] = []
        bag_config = sorting_config.get("bag", {})
        path = bag_config.get("inventory_path", "")

        if not path:
            errors.append("bag.inventory_path is required for mode='bag'")
        elif not os.path.exists(path):
            errors.append(f"bag.inventory_path does not exist: {path}")

        fmt = bag_config.get("inventory_format", "csv")
        if fmt not in ("csv", "json"):
            errors.append(f"Invalid bag.inventory_format: {fmt!r}")

        return errors


# ============================================================================
# REGISTRY AND FACTORY
# ============================================================================


MODE_REGISTRY: Dict[str, Type[SortingMode]] = {
    "design_id": DesignIdMode,
    "bag": BagMode,
}


def create_sorting_mode(config_manager: EnhancedConfigManager,
                        category_service: "CategoryHierarchyService") -> SortingMode:
    """
    Build the sorting mode selected by the active configuration.

    DesignIdMode takes the category service; BagMode does not. The factory
    handles dispatch so callers pass both dependencies uniformly.
    """
    sorting_config = config_manager.get_module_config(ModuleConfig.SORTING.value)
    mode_name = sorting_config.get("mode", "design_id")

    if mode_name not in MODE_REGISTRY:
        raise ValueError(
            f"Unknown sorting mode: {mode_name!r} "
            f"(must be one of {sorted(MODE_REGISTRY.keys())})"
        )

    if mode_name == "design_id":
        return DesignIdMode(config_manager, category_service)
    if mode_name == "bag":
        return BagMode(config_manager)

    raise ValueError(f"Mode {mode_name!r} is in MODE_REGISTRY but has no factory branch")
