# processing/processing_coordinator.py
"""
processing_coordinator.py - Orchestrates the complete piece processing pipeline

This module coordinates all processing steps to transform a CapturePackage into
a complete IdentifiedPiece with element ID, categories, and bin assignment.

RESPONSIBILITIES:
- Initialize all processing sub-modules
- Orchestrate the processing pipeline in correct sequence
- Build IdentifiedPiece incrementally through pipeline stages
- Handle errors gracefully without crashing
- Log processing steps and failures

DOES NOT:
- Pull from queue (that's processing_worker)
- Control servos or hardware
- Track statistics (that's queue_manager's job)

PIPELINE SEQUENCE:
1. Create IdentifiedPiece with basic info (piece_id, image_path, timestamp)
2. API Handler → IdentificationResult → update IdentifiedPiece
3. Category Lookup → CategoryInfo → update IdentifiedPiece
4. Bin Assignment → BinAssignment → update IdentifiedPiece
5. Finalize and return complete IdentifiedPiece
"""

import logging
from typing import Optional
from detector.detector_data_models import CapturePackage
from processing.processing_data_models import (
    IdentifiedPiece,
    IdentificationResult,
    CategoryInfo,
    BinAssignment
)
from processing.identification_api_handler import IdentificationAPIHandler
from processing.category_lookup_module import CategoryLookup
from processing.bin_assignment_module import BinAssignmentModule
from enhanced_config_manager import EnhancedConfigManager

logger = logging.getLogger(__name__)


class ProcessingCoordinator:
    """
    Orchestrates the complete piece processing pipeline.

    This is the main interface for processing pieces. It coordinates all
    sub-modules and builds the complete IdentifiedPiece result.
    """

    def __init__(self,
                 api_handler: IdentificationAPIHandler,
                 category_lookup: CategoryLookup,
                 bin_assignment: BinAssignmentModule,
                 config_manager: EnhancedConfigManager):
        """
        Initialize processing coordinator with all sub-modules.

        Called during application startup. All sub-modules should already
        be initialized and passed in.

        Args:
            api_handler: Handles Brickognize API calls
            category_lookup: Looks up categories from CSV database
            bin_assignment: Assigns bins based on categories
            config_manager: Configuration manager (for future use)
        """
        self.api_handler = api_handler
        self.category_lookup = category_lookup
        self.bin_assignment = bin_assignment
        self.config_manager = config_manager

        # Statistics (optional - could be removed if not needed)
        self.pieces_processed = 0
        self.pieces_failed = 0

        logger.info("Processing coordinator initialized")

    def process_piece(self, capture_package: CapturePackage) -> IdentifiedPiece:
        """
        Process a captured piece through the complete pipeline.

        This is the main entry point called by processing workers. It runs
        the piece through all processing stages and returns a complete result.

        Called by: processing_worker.py for each piece pulled from queue

        Args:
            capture_package: Captured piece from detection pipeline

        Returns:
            IdentifiedPiece with all fields populated (or error state)

        Example:
            capture_package = queue_manager.get_next_piece()
            identified_piece = coordinator.process_piece(capture_package)
            # identified_piece.bin_number = 3
            # identified_piece.complete = True
        """
        # Create IdentifiedPiece with basic capture info
        identified_piece = IdentifiedPiece(
            piece_id=capture_package.piece_id,
            image_path=capture_package.image_path,
            capture_timestamp=capture_package.capture_timestamp
        )

        logger.info(f"Processing piece {capture_package.piece_id}")

        try:
            # Step 1: Identify piece using API
            self._run_api_identification(identified_piece, capture_package)

            # Step 2: Look up categories (only if API succeeded)
            if identified_piece.element_id:
                self._run_category_lookup(identified_piece)

            # Step 3: Assign bin (only if we have categories)
            if identified_piece.primary_category:
                self._run_bin_assignment(identified_piece)

            # Finalize the piece
            identified_piece.finalize()

            self.pieces_processed += 1
            logger.info(
                f"Piece {capture_package.piece_id} complete: "
                f"{identified_piece.name} → Bin {identified_piece.bin_number}"
            )

        except Exception as e:
            # Catch any unexpected errors to prevent worker crashes
            logger.error(
                f"Unexpected error processing piece {capture_package.piece_id}: {e}",
                exc_info=True
            )
            self.pieces_failed += 1
            # Piece will be incomplete but still returned

        return identified_piece

    def _run_api_identification(self,
                                identified_piece: IdentifiedPiece,
                                capture_package: CapturePackage) -> None:
        """
        Run API identification step of pipeline.

        Calls Brickognize API with the captured image and updates the
        IdentifiedPiece with element_id, name, and confidence.

        Called by: process_piece() as step 1

        Args:
            identified_piece: Piece being processed (updated in place)
            capture_package: Capture data with image path

        Side effects:
            Updates identified_piece.element_id, .name, .identification_confidence
        """
        try:
            logger.debug(f"Step 1: API identification for piece {identified_piece.piece_id}")

            # Call API handler
            api_result = self.api_handler.identify_piece(capture_package.image_path)

            # Update identified piece with API results
            identified_piece.update_from_identification(api_result)

            logger.debug(
                f"API result: element_id={api_result.element_id}, "
                f"confidence={api_result.confidence:.2f}"
            )

        except FileNotFoundError as e:
            # Image file missing - critical error
            logger.error(f"Image file not found: {e}")
            identified_piece.element_id = None

        except ValueError as e:
            # API returned no results or invalid data
            logger.warning(f"API identification failed: {e}")
            identified_piece.element_id = None

        except Exception as e:
            # Network error, timeout, or other API issue
            logger.error(f"API error: {e}")
            identified_piece.element_id = None

    def _run_category_lookup(self, identified_piece: IdentifiedPiece) -> None:
        """
        Run category lookup step of pipeline.

        Looks up the element_id in the category database and updates the
        IdentifiedPiece with category hierarchy information.

        Called by: process_piece() as step 2 (only if API succeeded)

        Args:
            identified_piece: Piece being processed (updated in place)

        Side effects:
            Updates identified_piece.primary_category, .secondary_category,
            .tertiary_category, .found_in_database
        """
        try:
            logger.debug(
                f"Step 2: Category lookup for element_id={identified_piece.element_id}"
            )

            # Look up categories in database
            category_info = self.category_lookup.get_categories(
                identified_piece.element_id,
                piece_name=identified_piece.name
            )

            # Update identified piece with category results
            identified_piece.update_from_categories(category_info)

            logger.debug(
                f"Categories: {category_info.primary_category}/"
                f"{category_info.secondary_category}/"
                f"{category_info.tertiary_category}"
            )

        except Exception as e:
            # Category lookup should not fail, but catch just in case
            logger.error(f"Category lookup error: {e}")
            identified_piece.primary_category = "Unknown"

    def _run_bin_assignment(self, identified_piece: IdentifiedPiece) -> None:
        """
        Run bin assignment step of pipeline.

        Assigns a bin based on the piece's categories and configured
        sorting strategy.

        Called by: process_piece() as step 3 (only if categories found)

        Args:
            identified_piece: Piece being processed (updated in place)

        Side effects:
            Updates identified_piece.bin_number
        """
        try:
            logger.debug(f"Step 3: Bin assignment for piece {identified_piece.piece_id}")

            # Create CategoryInfo from identified piece data
            category_info = CategoryInfo(
                element_id=identified_piece.element_id,
                primary_category=identified_piece.primary_category,
                secondary_category=identified_piece.secondary_category,
                tertiary_category=identified_piece.tertiary_category,
                found_in_database=identified_piece.found_in_database
            )

            # Assign bin
            bin_assignment = self.bin_assignment.assign_bin(category_info)

            # Update identified piece with bin result
            identified_piece.update_from_bin_assignment(bin_assignment)

            logger.debug(f"Assigned to bin {bin_assignment.bin_number}")

        except Exception as e:
            # Bin assignment should not fail, but catch just in case
            logger.error(f"Bin assignment error: {e}")
            identified_piece.bin_number = 0  # Overflow bin

    def get_statistics(self) -> dict:
        """
        Get processing statistics.

        Returns basic statistics about pieces processed. Can be expanded
        to include more metrics if needed.

        Returns:
            Dictionary with statistics
        """
        return {
            "pieces_processed": self.pieces_processed,
            "pieces_failed": self.pieces_failed,
            "success_rate": (
                self.pieces_processed / (self.pieces_processed + self.pieces_failed)
                if (self.pieces_processed + self.pieces_failed) > 0
                else 0.0
            )
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_processing_coordinator(
        api_handler: IdentificationAPIHandler,
        category_lookup: CategoryLookup,
        bin_assignment: BinAssignmentModule,
        config_manager: EnhancedConfigManager
) -> ProcessingCoordinator:
    """
    Factory function to create a processing coordinator.

    Called during application startup after all sub-modules are created.

    Args:
        api_handler: Initialized API handler
        category_lookup: Initialized category lookup
        bin_assignment: Initialized bin assignment module
        config_manager: Configuration manager

    Returns:
        Initialized ProcessingCoordinator

    Example:
        # During application startup
        config_manager = create_config_manager()
        category_service = create_category_hierarchy_service(config_manager)

        api_handler = create_identification_api_handler(config_manager)
        category_lookup = create_category_lookup(config_manager)
        bin_assignment = create_bin_assignment_module(config_manager, category_service)

        coordinator = create_processing_coordinator(
            api_handler,
            category_lookup,
            bin_assignment,
            config_manager
        )

        # Use in worker
        identified_piece = coordinator.process_piece(capture_package)
    """
    return ProcessingCoordinator(
        api_handler,
        category_lookup,
        bin_assignment,
        config_manager
    )


# ============================================================================
# MODULE TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test the processing coordinator with an actual image from LegoPictures.

    This verifies:
    - All sub-modules are initialized correctly
    - Pipeline runs in correct order through all steps
    - IdentifiedPiece is built correctly with real data
    - Errors are handled gracefully
    """
    import logging
    import os
    import numpy as np
    import time
    from enhanced_config_manager import create_config_manager
    from processing.category_hierarchy_service import create_category_hierarchy_service
    from processing.identification_api_handler import create_identification_api_handler
    from processing.category_lookup_module import create_category_lookup
    from processing.bin_assignment_module import create_bin_assignment_module

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("PROCESSING COORDINATOR TEST")
    print("=" * 70)

    # Check for test image
    image_dir = "LegoPictures"
    test_image_name = "test_piece.jpeg"
    test_image_path = os.path.join(image_dir, test_image_name)

    # Try .jpeg first, fall back to .jpg
    if not os.path.exists(test_image_path):
        test_image_name = "test_piece.jpg"
        test_image_path = os.path.join(image_dir, test_image_name)

    if not os.path.exists(test_image_path):
        print(f"\nERROR: Test image not found!")
        print(f"Please place a Lego piece image at: {image_dir}/test_piece.jpeg")
        print(f"(or {image_dir}/test_piece.jpg)")
        exit(1)

    print(f"\nFound test image: {test_image_path}")

    # Initialize all sub-modules
    print("\nInitializing sub-modules...")
    print("-" * 70)

    config_manager = create_config_manager()
    print("  Created config_manager")

    category_service = create_category_hierarchy_service(config_manager)
    print("  Created category_hierarchy_service")

    api_handler = create_identification_api_handler(config_manager)
    print("  Created identification_api_handler")

    category_lookup = create_category_lookup(config_manager)
    print("  Created category_lookup_module")

    bin_assignment = create_bin_assignment_module(config_manager, category_service)
    print("  Created bin_assignment_module")

    coordinator = create_processing_coordinator(
        api_handler,
        category_lookup,
        bin_assignment,
        config_manager
    )
    print("  Created processing_coordinator")

    # Create CapturePackage with actual image
    print("\nCreating CapturePackage...")
    print("-" * 70)

    mock_capture = CapturePackage(
        piece_id=1,
        processed_image=np.zeros((100, 100, 3), dtype=np.uint8),  # Placeholder
        capture_timestamp=time.time(),
        capture_position=(100, 200),
        original_bbox=(100, 200, 50, 50)
    )
    mock_capture.image_path = test_image_path

    print(f"  Piece ID: {mock_capture.piece_id}")
    print(f"  Image: {mock_capture.image_path}")
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mock_capture.capture_timestamp))}")

    # Process the piece through complete pipeline
    print("\nProcessing piece through pipeline...")
    print("-" * 70)

    identified_piece = coordinator.process_piece(mock_capture)

    # Display detailed results
    print("\nPIPELINE RESULTS")
    print("=" * 70)

    print("\n1. API IDENTIFICATION:")
    print(f"   Element ID:  {identified_piece.element_id or 'NOT FOUND'}")
    print(f"   Name:        {identified_piece.name or 'Unknown'}")
    print(
        f"   Confidence:  {identified_piece.confidence:.2%}" if identified_piece.confidence else "   Confidence:  N/A")

    print("\n2. CATEGORY LOOKUP:")
    print(f"   Primary:     {identified_piece.primary_category or 'Unknown'}")
    print(f"   Secondary:   {identified_piece.secondary_category or 'N/A'}")
    print(f"   Tertiary:    {identified_piece.tertiary_category or 'N/A'}")
    print(f"   Full Path:   {identified_piece.get_full_category_path()}")
    print(f"   In Database: {identified_piece.found_in_database}")

    print("\n3. BIN ASSIGNMENT:")
    print(f"   Bin Number:  {identified_piece.bin_number}")
    print(f"   Strategy:    {bin_assignment.strategy}")

    print("\n4. COMPLETION STATUS:")
    print(f"   Complete:    {identified_piece.complete}")
    print(f"   All Fields:  {identified_piece.is_complete}")

    # Show current bin assignments
    print("\nCURRENT BIN ASSIGNMENTS:")
    print("-" * 70)
    assignments = bin_assignment.get_bin_assignments()
    if assignments:
        for category, bin_num in sorted(assignments.items(), key=lambda x: x[1]):
            print(f"   Bin {bin_num}: {category}")
    else:
        print("   No dynamic assignments yet")

    # Show statistics
    print("\nCOORDINATOR STATISTICS:")
    print("-" * 70)
    stats = coordinator.get_statistics()
    print(f"   Pieces Processed: {stats['pieces_processed']}")
    print(f"   Pieces Failed:    {stats['pieces_failed']}")
    print(f"   Success Rate:     {stats['success_rate']:.1%}")

    # Show bin assignment statistics
    print("\nBIN ASSIGNMENT STATISTICS:")
    print("-" * 70)
    bin_stats = bin_assignment.get_statistics()
    print(f"   Strategy:            {bin_stats['strategy']}")
    print(f"   Max Bins:            {bin_stats['max_bins']}")
    print(f"   Assigned Categories: {bin_stats['assigned_categories']}")
    print(f"   Used Bins:           {bin_stats['used_bins']}")
    print(f"   Available Bins:      {bin_stats['available_bins']}")

    # Final summary
    print("\n" + "=" * 70)
    if identified_piece.complete:
        print("TEST PASSED - Piece fully processed through pipeline")
        print(f"Result: {identified_piece.name} → Bin {identified_piece.bin_number}")
    else:
        print("TEST COMPLETED - Piece processed with incomplete data")
        if not identified_piece.element_id:
            print("  Issue: API identification failed")
        elif not identified_piece.primary_category:
            print("  Issue: Category lookup failed")
        elif identified_piece.bin_number is None:
            print("  Issue: Bin assignment failed")
    print("=" * 70)