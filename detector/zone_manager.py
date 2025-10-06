"""
zone_manager.py - Manages conveyor belt zones and piece zone status

This module provides zone-specific business logic for the Lego sorting system.
It handles the division of the ROI into functional zones and manages the zone
status of tracked pieces as they move through the conveyor belt.

Zone Architecture:
- Entry Zone: Where pieces first appear (wait for full visibility)
- Valid Zone: Where pieces can be safely captured for identification
- Exit Zone: Where pieces trigger sorting mechanism

Key Responsibilities:
- Calculate zone boundaries within the ROI
- Update TrackedPiece zone status flags (in_exit_zone, exit_zone_entry_time)
- Provide filtering methods for other modules to get pieces by zone
- Maintain zone transition timing and detection

This module does NOT:
- Handle coordinate system conversions (that's detector_coordinator's job)
- Make capture decisions (that's capture_controller's job)
- Make sorting decisions (that's servo_controller's job)
- Manage ROI configuration (that's enhanced_config_manager's job)

The zone manager works with TrackedPiece objects in ROI coordinates,
focusing purely on zone business logic and piece status management.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from detector.detector_data_models import TrackedPiece, RegionOfInterest
from enhanced_config_manager import ModuleConfig

# Set up module logger
logger = logging.getLogger(__name__)


# ============================================================================
# ZONE TYPE DEFINITIONS
# ============================================================================
# Zone types and business logic definitions

class ZoneType(Enum):
    """
    Defines the different zones on the conveyor belt.
    Each zone has different rules for when pieces can be captured or sorted.
    """
    ENTRY = "entry"  # Where pieces first appear (wait for full visibility)
    VALID = "valid"  # Where we can safely capture images
    EXIT = "exit"  # Where we trigger the sorting mechanism


@dataclass
class ConveyorZone:
    """
    Defines a zone along the conveyor belt with specific rules.

    Zones are defined by x-coordinate ranges and determine when
    different operations (capture, sorting) can be performed.
    """
    zone_type: ZoneType  # What type of zone this is
    start_x: int  # Left boundary (inclusive)
    end_x: int  # Right boundary (inclusive)
    name: str  # Human-readable name

    def contains_piece(self, piece: TrackedPiece, roi_offset_x: int = 0) -> bool:
        """
        Check if a tracked piece is currently in this zone.

        Args:
            piece: TrackedPiece to check
            roi_offset_x: X offset of ROI within full frame

        Returns:
            True if piece center is within zone boundaries
        """
        # Convert piece center to frame coordinates
        piece_x = piece.center[0] + roi_offset_x
        return self.start_x <= piece_x <= self.end_x

    def width(self) -> int:
        """Get the width of this zone in pixels."""
        return self.end_x - self.start_x


# ============================================================================
# ZONE CONFIGURATION DATACLASS
# ============================================================================

@dataclass
class ZoneConfiguration:
    """
    Defines the zone layout within the ROI.

    This class encapsulates all zone boundary calculations and provides
    a clean interface for zone-related operations.
    """
    entry_zone: ConveyorZone  # Where pieces first appear
    valid_zone: ConveyorZone  # Where pieces can be captured
    exit_zone: ConveyorZone  # Where pieces trigger sorting
    roi: RegionOfInterest  # The full detection area

    def get_zone_by_type(self, zone_type: ZoneType) -> ConveyorZone:
        """
        Get a specific zone by its type.

        Args:
            zone_type: Type of zone to retrieve

        Returns:
            ConveyorZone object for the requested type

        Raises:
            ValueError: If zone_type is not recognized
        """
        zone_map = {
            ZoneType.ENTRY: self.entry_zone,
            ZoneType.VALID: self.valid_zone,
            ZoneType.EXIT: self.exit_zone
        }

        if zone_type not in zone_map:
            raise ValueError(f"Unknown zone type: {zone_type}")

        return zone_map[zone_type]

    def get_all_zones(self) -> List[ConveyorZone]:
        """Get all zones as a list for iteration."""
        return [self.entry_zone, self.valid_zone, self.exit_zone]


# ============================================================================
# MAIN ZONE MANAGER CLASS
# ============================================================================

class ZoneManager:
    """
    Manages conveyor belt zones and updates piece zone status.

    This class handles the business logic of determining which zone each
    piece is in and maintaining zone-related status flags on TrackedPiece
    objects. It provides filtering and querying capabilities for other modules.

    The zone manager works exclusively in ROI coordinates, maintaining
    consistency with the vision processor and piece tracker modules.
    """

    def __init__(self, config_manager):
        """
        Initialize the zone manager with unified configuration.

        Args:
            config_manager: Enhanced config manager instance (required)

        Raises:
            ValueError: If config_manager is None or invalid
        """
        if not config_manager:
            raise ValueError("ZoneManager requires a valid config_manager")

        logger.info("Initializing ZoneManager")

        self.config_manager = config_manager
        self._zone_config: Optional[ZoneConfiguration] = None

        # Load configuration from unified system
        self._load_configuration()

        logger.info("ZoneManager initialized successfully")

    def _load_configuration(self):
        """
        Load zone configuration parameters from enhanced_config_manager.

        Zone parameters come from both detector and detector_roi configurations
        to get zone percentages and ROI boundaries.
        """
        # Get detector configuration for zone percentages
        detector_config = self.config_manager.get_module_config(ModuleConfig.DETECTOR.value)

        # Zone percentage configuration
        self.entry_zone_percent = detector_config.get("entry_zone_percent", 0.15)
        self.exit_zone_percent = detector_config.get("exit_zone_percent", 0.15)

        # Zone transition timing
        self.zone_transition_debounce = detector_config.get("zone_transition_debounce", 0.1)

        logger.info("Zone configuration loaded from enhanced_config_manager")
        logger.info(f"Entry zone: {self.entry_zone_percent:.1%}, "
                    f"Exit zone: {self.exit_zone_percent:.1%}")

    # ========================================================================
    # ROI AND ZONE SETUP
    # ========================================================================

    def set_roi(self, roi: RegionOfInterest):
        """
        Set the ROI and calculate zone boundaries.

        This method computes the zone layout based on the ROI dimensions
        and configured zone percentages. All zones are defined in ROI
        coordinates (0,0 = top-left of ROI).

        Args:
            roi: Region of interest configuration
        """
        logger.info(f"Setting ROI and calculating zone boundaries: {roi.to_tuple()}")

        # Calculate zone boundaries based on ROI width and percentages
        entry_width = int(roi.width * self.entry_zone_percent)
        exit_width = int(roi.width * self.exit_zone_percent)

        # Define zones in ROI coordinates (x=0 is left edge of ROI)
        entry_zone = ConveyorZone(
            zone_type=ZoneType.ENTRY,
            start_x=0,  # Left edge of ROI
            end_x=entry_width,
            name="Entry Zone"
        )

        exit_zone = ConveyorZone(
            zone_type=ZoneType.EXIT,
            start_x=roi.width - exit_width,  # Right side of ROI
            end_x=roi.width,  # Right edge of ROI
            name="Exit Zone"
        )

        valid_zone = ConveyorZone(
            zone_type=ZoneType.VALID,
            start_x=entry_zone.end_x,  # After entry zone
            end_x=exit_zone.start_x,  # Before exit zone
            name="Valid Zone"
        )

        # Store zone configuration
        self._zone_config = ZoneConfiguration(
            entry_zone=entry_zone,
            valid_zone=valid_zone,
            exit_zone=exit_zone,
            roi=roi
        )

        logger.info(f"Zone boundaries calculated:")
        logger.info(f"  Entry: x=0 to x={entry_width}")
        logger.info(f"  Valid: x={entry_zone.end_x} to x={exit_zone.start_x}")
        logger.info(f"  Exit: x={exit_zone.start_x} to x={roi.width}")

    def get_zone_configuration(self) -> Optional[ZoneConfiguration]:
        """
        Get the current zone configuration.

        Returns:
            ZoneConfiguration object or None if ROI not set
        """
        return self._zone_config

    # ========================================================================
    # PIECE ZONE STATUS MANAGEMENT
    # ========================================================================

    def update_piece_zones(self, tracked_pieces: List[TrackedPiece]) -> List[TrackedPiece]:
        """
        Update zone status for all tracked pieces.

        This is the main entry point for zone management. It examines each
        piece's position and updates the zone-related flags on the TrackedPiece
        objects (in_exit_zone, exit_zone_entry_time).

        Args:
            tracked_pieces: List of TrackedPiece objects in ROI coordinates

        Returns:
            The same list of TrackedPiece objects with updated zone status

        Raises:
            ValueError: If zone configuration is not set
        """
        if not self._zone_config:
            raise ValueError("Zone configuration not set. Call set_roi() first.")

        current_time = time.time()

        for piece in tracked_pieces:
            # Determine which zone the piece is currently in
            current_zone = self._determine_piece_zone(piece)

            # Update zone-specific status flags
            self._update_piece_zone_status(piece, current_zone, current_time)

        logger.debug(f"Updated zone status for {len(tracked_pieces)} pieces")

        # NEW: DETERMINE RIGHTMOST PIECE IN EXIT ZONE
        # ========================================================================
        # Clear the flag for all pieces first
        for piece in tracked_pieces:
            piece.is_rightmost_in_exit_zone = False

        # Find all pieces currently in exit zone
        exit_zone_pieces = [p for p in tracked_pieces if p.in_exit_zone]

        if exit_zone_pieces:
            # Find the piece with the highest right_edge (furthest right)
            rightmost_piece = max(exit_zone_pieces, key=lambda p: p.right_edge)

            # Set the flag for this piece only
            rightmost_piece.is_rightmost_in_exit_zone = True

            logger.debug(
                f"Rightmost piece in exit zone: ID {rightmost_piece.id} "
                f"at x={rightmost_piece.right_edge:.1f}"
            )

        return tracked_pieces

    def _determine_piece_zone(self, piece: TrackedPiece) -> Optional[ZoneType]:
        """
        Determine which zone a piece is currently in.

        Uses the piece's center position (already in ROI coordinates) to
        determine zone membership. Pieces are assigned to zones based on
        their center point position.

        Args:
            piece: TrackedPiece to evaluate (in ROI coordinates)

        Returns:
            ZoneType enum or None if piece is outside all zones
        """
        piece_center_x = piece.center[0]  # Already in ROI coordinates

        # Check each zone to see if piece center falls within boundaries
        for zone in self._zone_config.get_all_zones():
            if zone.start_x <= piece_center_x <= zone.end_x:
                return zone.zone_type

        # Piece is outside all defined zones (shouldn't happen normally)
        logger.warning(f"Piece {piece.id} at x={piece_center_x} is outside all zones")
        return None

    def _update_piece_zone_status(self, piece: TrackedPiece, current_zone: Optional[ZoneType],
                                  current_time: float):
        """
        Update all zone-related status flags on a TrackedPiece.

        This method manages zone status for all zones, updating the flags that
        other modules use for decision making without needing coordinate calculations.

        Args:
            piece: TrackedPiece to update
            current_zone: Zone the piece is currently in
            current_time: Current timestamp
        """
        # Reset all zone flags first
        piece.in_entry_zone = False
        piece.in_valid_zone = False
        piece.in_exit_zone = False
        piece.exit_zone_entry_time = None

        # Set the appropriate zone flag based on current zone
        if current_zone == ZoneType.ENTRY:
            piece.in_entry_zone = True
            logger.debug(f"Piece {piece.id} is in entry zone")

        elif current_zone == ZoneType.VALID:
            piece.in_valid_zone = True
            logger.debug(f"Piece {piece.id} is in valid zone")

        elif current_zone == ZoneType.EXIT:
            piece.in_exit_zone = True
            # Set entry time if this is the first time in exit zone
            if piece.exit_zone_entry_time is None:
                piece.exit_zone_entry_time = current_time
                logger.debug(f"Piece {piece.id} entered exit zone")

        # If current_zone is None, all flags remain False (piece outside all zones)

    # ========================================================================
    # ZONE FILTERING AND QUERYING
    # ========================================================================

    def get_pieces_in_zone(self, tracked_pieces: List[TrackedPiece],
                           zone_type: ZoneType) -> List[TrackedPiece]:
        """
        Filter pieces by zone membership.

        This method provides an efficient way for other modules to get
        pieces in specific zones without having to implement zone logic.

        Args:
            tracked_pieces: List of TrackedPiece objects
            zone_type: Zone to filter by

        Returns:
            List of pieces currently in the specified zone

        Raises:
            ValueError: If zone configuration is not set
        """
        if not self._zone_config:
            raise ValueError("Zone configuration not set. Call set_roi() first.")

        target_zone = self._zone_config.get_zone_by_type(zone_type)
        pieces_in_zone = []

        for piece in tracked_pieces:
            piece_center_x = piece.center[0]  # ROI coordinates

            if target_zone.start_x <= piece_center_x <= target_zone.end_x:
                pieces_in_zone.append(piece)

        logger.debug(f"Found {len(pieces_in_zone)} pieces in {zone_type.value} zone")
        return pieces_in_zone

    def get_pieces_in_valid_zone(self, tracked_pieces: List[TrackedPiece]) -> List[TrackedPiece]:
        """
        Get pieces in the valid capture zone.

        This is a convenience method for the capture controller, which
        specifically needs pieces that are in the valid zone for processing.

        Args:
            tracked_pieces: List of TrackedPiece objects

        Returns:
            List of pieces in valid zone
        """
        return self.get_pieces_in_zone(tracked_pieces, ZoneType.VALID)

    def get_pieces_in_exit_zone(self, tracked_pieces: List[TrackedPiece]) -> List[TrackedPiece]:
        """
        Get pieces in the exit zone.

        This is a convenience method for the servo controller, which
        specifically needs pieces that are in the exit zone for sorting.

        Args:
            tracked_pieces: List of TrackedPiece objects

        Returns:
            List of pieces in exit zone
        """
        return self.get_pieces_in_zone(tracked_pieces, ZoneType.EXIT)

    def get_capture_ready_pieces(self, tracked_pieces: List[TrackedPiece]) -> List[TrackedPiece]:
        """
        Get pieces ready for capture (in valid zone AND fully visible).

        This combines zone filtering with visibility status to provide
        the capture controller with pieces that meet all requirements
        for image capture and processing.

        Args:
            tracked_pieces: List of TrackedPiece objects

        Returns:
            List of pieces ready for capture
        """
        valid_zone_pieces = self.get_pieces_in_valid_zone(tracked_pieces)

        # Filter for pieces that are also fully visible and stable
        capture_ready = [
            piece for piece in valid_zone_pieces
            if piece.fully_in_frame and not piece.captured and not piece.being_processed
        ]

        logger.debug(f"Found {len(capture_ready)} capture-ready pieces "
                     f"out of {len(valid_zone_pieces)} in valid zone")
        return capture_ready

    def get_exit_zone_pieces_by_timing(self, tracked_pieces: List[TrackedPiece],
                                       max_time_in_zone: float = None) -> List[TrackedPiece]:
        """
        Get pieces in exit zone filtered by how long they've been there.

        This helps the servo controller prioritize pieces that need immediate
        sorting versus pieces that just entered the exit zone.

        Args:
            tracked_pieces: List of TrackedPiece objects
            max_time_in_zone: Maximum time in seconds (None for all pieces)

        Returns:
            List of pieces in exit zone meeting timing criteria
        """
        exit_pieces = self.get_pieces_in_exit_zone(tracked_pieces)

        if max_time_in_zone is None:
            return exit_pieces

        current_time = time.time()
        urgent_pieces = []

        for piece in exit_pieces:
            if piece.exit_zone_entry_time is not None:
                time_in_zone = current_time - piece.exit_zone_entry_time
                if time_in_zone <= max_time_in_zone:
                    urgent_pieces.append(piece)

        logger.debug(f"Found {len(urgent_pieces)} urgent exit zone pieces "
                     f"(within {max_time_in_zone}s)")
        return urgent_pieces

    # ========================================================================
    # ZONE STATISTICS AND MONITORING
    # ========================================================================

    def get_zone_statistics(self, tracked_pieces: List[TrackedPiece]) -> Dict[str, Any]:
        """
        Get statistics about piece distribution across zones.

        This provides monitoring data for system performance analysis
        and debugging zone-related issues.

        Args:
            tracked_pieces: List of TrackedPiece objects

        Returns:
            Dictionary with zone distribution statistics
        """
        if not self._zone_config:
            return {"error": "Zone configuration not set"}

        # Count pieces in each zone
        zone_counts = {
            "entry": len(self.get_pieces_in_zone(tracked_pieces, ZoneType.ENTRY)),
            "valid": len(self.get_pieces_in_zone(tracked_pieces, ZoneType.VALID)),
            "exit": len(self.get_pieces_in_zone(tracked_pieces, ZoneType.EXIT))
        }

        # Calculate additional metrics
        total_pieces = len(tracked_pieces)
        capture_ready = len(self.get_capture_ready_pieces(tracked_pieces))

        # Calculate average time in exit zone for current exit pieces
        exit_pieces = self.get_pieces_in_exit_zone(tracked_pieces)
        avg_exit_time = 0.0
        if exit_pieces:
            current_time = time.time()
            exit_times = [
                current_time - piece.exit_zone_entry_time
                for piece in exit_pieces
                if piece.exit_zone_entry_time is not None
            ]
            if exit_times:
                avg_exit_time = sum(exit_times) / len(exit_times)

        return {
            "total_pieces": total_pieces,
            "zone_distribution": zone_counts,
            "capture_ready_pieces": capture_ready,
            "pieces_with_exit_timing": len([p for p in exit_pieces if p.exit_zone_entry_time]),
            "average_exit_zone_time": avg_exit_time,
            "zone_configuration": {
                "entry_percent": self.entry_zone_percent,
                "exit_percent": self.exit_zone_percent,
                "entry_width": self._zone_config.entry_zone.width(),
                "valid_width": self._zone_config.valid_zone.width(),
                "exit_width": self._zone_config.exit_zone.width()
            }
        }

    def get_zone_boundaries_info(self) -> Dict[str, Any]:
        """
        Get information about zone boundaries for visualization or debugging.

        Returns zone boundaries in ROI coordinates for use by other modules
        that need to understand the zone layout.

        Returns:
            Dictionary with zone boundary information
        """
        if not self._zone_config:
            return {"configured": False}

        return {
            "configured": True,
            "roi": self._zone_config.roi.to_tuple(),
            "zones": {
                "entry": {
                    "start_x": self._zone_config.entry_zone.start_x,
                    "end_x": self._zone_config.entry_zone.end_x,
                    "width": self._zone_config.entry_zone.width()
                },
                "valid": {
                    "start_x": self._zone_config.valid_zone.start_x,
                    "end_x": self._zone_config.valid_zone.end_x,
                    "width": self._zone_config.valid_zone.width()
                },
                "exit": {
                    "start_x": self._zone_config.exit_zone.start_x,
                    "end_x": self._zone_config.exit_zone.end_x,
                    "width": self._zone_config.exit_zone.width()
                }
            }
        }

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def reset_zone_status(self, tracked_pieces: List[TrackedPiece]):
        """
        Reset zone status for all pieces.

        This clears all zone-related flags and timing information.
        Useful when restarting the system or clearing the conveyor.

        Args:
            tracked_pieces: List of TrackedPiece objects to reset
        """
        logger.info("Resetting zone status for all tracked pieces")

        for piece in tracked_pieces:
            piece.in_entry_zone = False
            piece.in_valid_zone = False
            piece.in_exit_zone = False
            piece.exit_zone_entry_time = None

        logger.info(f"Reset zone status for {len(tracked_pieces)} pieces")

    def update_configuration(self):
        """
        Reload zone configuration from the config manager.

        This allows dynamic updates to zone percentages without
        restarting the entire system. ROI must be set again after
        calling this method.
        """
        logger.info("Reloading zone configuration")
        self._load_configuration()
        self._zone_config = None  # Force ROI recalculation


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_zone_manager(config_manager) -> ZoneManager:
    """
    Create a ZoneManager instance with unified configuration.

    This is the standard way to create a zone manager using the
    enhanced_config_manager system for consistent configuration management.

    Args:
        config_manager: Enhanced config manager instance (required)

    Returns:
        Configured ZoneManager instance

    Raises:
        ValueError: If config_manager is None or invalid
    """
    if config_manager is None:
        raise ValueError("create_zone_manager requires a valid config_manager")

    logger.info("Creating ZoneManager with enhanced_config_manager")
    return ZoneManager(config_manager)


# ============================================================================
# TESTING AND UTILITIES
# ============================================================================

def create_test_piece_in_zone(piece_id: int, zone_x: float, roi_width: int = 600) -> TrackedPiece:
    """
    Create a test TrackedPiece positioned in a specific zone.

    Args:
        piece_id: Unique ID for the test piece
        zone_x: X position within ROI (0 = left edge, roi_width = right edge)
        roi_width: Width of ROI for boundary calculations

    Returns:
        TrackedPiece positioned at the specified location
    """
    import numpy as np
    from detector_data_models import create_tracked_piece_from_detection, Detection

    # Create a simple detection at the specified position
    bbox = (int(zone_x - 20), 150, 40, 30)  # Center at zone_x
    contour = np.array([[[int(zone_x - 20), 150]], [[int(zone_x + 20), 150]],
                        [[int(zone_x + 20), 180]], [[int(zone_x - 20), 180]]])

    detection = Detection(
        contour=contour,
        bbox=bbox,
        area=1200,
        confidence=0.9
    )

    return create_tracked_piece_from_detection(detection, piece_id, time.time())


if __name__ == "__main__":
    """
    Test the zone manager with synthetic tracked pieces.

    This demonstrates zone boundary calculation, piece status updates,
    and filtering functionality.
    """
    import sys
    from detector_data_models import RegionOfInterest

    logging.basicConfig(level=logging.INFO)
    logger.info("Testing ZoneManager with synthetic data")

    # Create a mock config manager for testing
    class MockConfigManager:
        """Mock config manager with test configuration"""

        def get_module_config(self, module_name):
            if module_name == "detector":
                return {
                    "entry_zone_percent": 0.15,
                    "exit_zone_percent": 0.15,
                    "zone_transition_debounce": 0.1
                }
            return {}


    try:
        # Create zone manager with mock config
        mock_config = MockConfigManager()
        zone_manager = create_zone_manager(mock_config)

        # Set up ROI and calculate zones
        roi = RegionOfInterest(x=100, y=100, width=600, height=200)
        zone_manager.set_roi(roi)

        print(f"\nZoneManager Test Results:")
        print(f"=" * 50)

        # Show zone configuration
        zone_info = zone_manager.get_zone_boundaries_info()
        print(f"Zone Boundaries (ROI coordinates):")
        for zone_name, zone_data in zone_info["zones"].items():
            print(f"  {zone_name.title()}: x={zone_data['start_x']} to "
                  f"x={zone_data['end_x']} (width={zone_data['width']})")

        # Create test pieces in different zones
        test_pieces = [
            create_test_piece_in_zone(1, 50, 600),  # Entry zone
            create_test_piece_in_zone(2, 300, 600),  # Valid zone
            create_test_piece_in_zone(3, 550, 600),  # Exit zone
        ]

        # Update zone status
        updated_pieces = zone_manager.update_piece_zones(test_pieces)

        print(f"\nPiece Zone Status:")
        for piece in updated_pieces:
            zone = "entry" if piece.center[0] < 90 else ("exit" if piece.center[0] > 510 else "valid")
            print(f"  Piece {piece.id}: x={piece.center[0]:.0f}, zone={zone}, "
                  f"in_exit_zone={piece.in_exit_zone}")

        # Test filtering methods
        print(f"\nZone Filtering Results:")
        valid_pieces = zone_manager.get_pieces_in_valid_zone(updated_pieces)
        exit_pieces = zone_manager.get_pieces_in_exit_zone(updated_pieces)
        capture_ready = zone_manager.get_capture_ready_pieces(updated_pieces)

        print(f"  Valid zone pieces: {len(valid_pieces)}")
        print(f"  Exit zone pieces: {len(exit_pieces)}")
        print(f"  Capture ready pieces: {len(capture_ready)}")

        # Show statistics
        stats = zone_manager.get_zone_statistics(updated_pieces)
        print(f"\nZone Statistics:")
        for key, value in stats["zone_distribution"].items():
            print(f"  {key.title()} zone: {value} pieces")

        logger.info("ZoneManager test completed successfully")

    except Exception as e:
        logger.error(f"ZoneManager test failed: {e}")
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
