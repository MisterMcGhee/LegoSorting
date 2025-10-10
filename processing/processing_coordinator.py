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
- Notify registered callbacks when piece identification completes (for GUI updates)

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
6. Notify callbacks that piece is complete (for GUI updates)

CALLBACK SYSTEM:
This module implements the Observer pattern to notify external systems
(primarily the GUI) when piece identification completes. External systems
register callback functions during initialization, and these callbacks are
automatically invoked when processing finishes.
"""

import logging
from typing import Optional, Callable, Dict, List
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

    Callback Support:
    External systems can register callbacks to be notified when piece
    identification completes. This is primarily used by the GUI to update
    the "Recently Processed" display in real-time.
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
        # Store references to all sub-modules
        # These are used throughout the processing pipeline
        self.api_handler = api_handler
        self.category_lookup = category_lookup
        self.bin_assignment = bin_assignment
        self.config_manager = config_manager

        # ============================================================
        # STATISTICS TRACKING
        # ============================================================
        # Keep track of how many pieces we've processed successfully
        # and how many failed. This can be used for monitoring.
        self.pieces_processed = 0
        self.pieces_failed = 0

        # ============================================================
        # CALLBACK SYSTEM FOR IDENTIFICATION COMPLETION
        # ============================================================
        # This list stores callback functions that want to be notified
        # when piece identification completes. The GUI registers its
        # callback here so it can update the display in real-time.
        # Each callback should be a function that accepts an IdentifiedPiece
        self.identification_callbacks: List[Callable[[IdentifiedPiece], None]] = []

        logger.info("Processing coordinator initialized")

    # ========================================================================
    # CALLBACK REGISTRATION AND NOTIFICATION
    # ========================================================================
    # These methods implement the Observer pattern, allowing external systems
    # (like the GUI) to "subscribe" to piece identification events.

    def register_identification_callback(self, callback_function: Callable[[IdentifiedPiece], None]):
        """
        Register a callback function to be notified when identification completes.

        PURPOSE:
        This allows external systems (especially the GUI) to be notified
        immediately when a piece has been fully identified and assigned to
        a bin. The callback receives the complete IdentifiedPiece object
        with all fields populated.

        WHEN TO USE:
        Call this during initialization to register any listeners that
        need to know about identified pieces. The sorting_gui calls this
        during its startup to update the "Recently Processed" panel.

        CALLBACK SIGNATURE:
        The registered function must accept one argument:
            callback(identified_piece: IdentifiedPiece)

        WHERE CALLBACKS ARE TRIGGERED:
        Callbacks are triggered at the end of process_piece() after all
        processing stages complete (API identification, category lookup,
        bin assignment). Even pieces identified as "Unknown" trigger
        callbacks - the callback receiver can check identified_piece fields
        to determine success/failure.

        IDENTIFIED PIECE STATES:
        The IdentifiedPiece passed to callbacks may have different states:
        - Complete success: All fields populated, bin assigned
        - Unknown piece: element_id may be None, bin_number = 0 (overflow)
        - API failure: element_id None, categories None, bin = 0

        The callback receiver should check identified_piece.complete and
        other fields to determine the actual state.

        THREAD SAFETY:
        Callbacks are called synchronously on the processing thread
        (usually a worker thread). If your callback updates GUI, make
        sure to use Qt signals or thread-safe mechanisms.

        Args:
            callback_function: Function to call when identification completes.
                              Must accept (identified_piece: IdentifiedPiece)

        Example:
            # In sorting_gui.py __init__:
            processing.register_identification_callback(self.on_piece_identified)

            # Later, when identification completes:
            # GUI's on_piece_identified(identified_piece) is called automatically
            # GUI can then display the image, name, element_id, bin, etc.
        """
        # Add the callback function to our list
        self.identification_callbacks.append(callback_function)

        # Log that we registered this callback (helpful for debugging)
        logger.debug(
            f"Registered identification callback: {callback_function.__name__}"
        )

    def _notify_identification_callbacks(self, identified_piece: IdentifiedPiece):
        """
        Notify all registered callbacks that identification has completed.

        PURPOSE:
        This is an internal method that calls all registered callbacks
        when piece processing completes. It handles errors gracefully so
        one failing callback doesn't break the pipeline.

        WHEN CALLED:
        This is called at the end of process_piece(), after all processing
        stages complete and identified_piece.finalize() has been called.

        IMPORTANT TIMING:
        This is called AFTER the IdentifiedPiece is complete, which means:
        - API identification has run (success or failure)
        - Category lookup has run (if API succeeded)
        - Bin assignment has run (if categories found)
        - identified_piece.complete = True (or False if something failed)
        - identified_piece.finalize() has been called

        ERROR HANDLING:
        If a callback raises an exception, it is caught and logged,
        but other callbacks still execute. This prevents one bad
        callback from breaking the pipeline.

        PROCESSING WORKER THREAD:
        These callbacks execute on the processing worker thread, not
        the main thread. GUI callbacks should use signals or thread-safe
        methods to update UI elements.

        Args:
            identified_piece: Complete IdentifiedPiece object with all
                            processing results

        Example Flow:
            # At end of process_piece():
            identified_piece.finalize()  # Mark piece as complete
            self._notify_identification_callbacks(identified_piece)  # Tell everyone

            # This calls each registered callback:
            # 1. sorting_gui.on_piece_identified(identified_piece)
            # 2. any_other_registered_callback(identified_piece)
            # 3. etc.
        """
        # Loop through all registered callback functions
        for callback in self.identification_callbacks:
            try:
                # Call the callback with the complete identified piece
                # This is where the GUI gets notified!
                callback(identified_piece)

            except Exception as e:
                # If a callback crashes, log the error but keep going
                # This ensures one broken callback doesn't stop other callbacks
                # or break the entire processing pipeline
                logger.error(
                    f"Error in identification callback {callback.__name__}: {e}",
                    exc_info=True
                )

    def get_identified_pieces(self) -> Dict[int, IdentifiedPiece]:
        """
        Get dictionary of currently identified pieces.

        NOTE: This method is not implemented in ProcessingCoordinator.
        The main orchestrator (LegoSorting008) maintains the authoritative
        dictionary of identified pieces.

        PURPOSE:
        This placeholder method exists for interface compatibility but
        returns an empty dict. The GUI and other systems should get the
        identified pieces dictionary from the orchestrator instead:

            identified_dict = orchestrator.get_identified_pieces()

        WHY ORCHESTRATOR HANDLES THIS:
        - ProcessingCoordinator processes one piece at a time
        - It doesn't maintain state across multiple pieces
        - LegoSorting008 coordinates the entire system and tracks all pieces
        - This avoids duplication and keeps state management centralized

        Returns:
            Empty dictionary (actual data maintained by orchestrator)

        See Also:
            LegoSorting008.get_identified_pieces() - The actual implementation
        """
        # TODO: Implement based on architecture decision
        # For now, return empty dict to allow GUI to function
        # without breaking
        logger.warning(
            "get_identified_pieces() not fully implemented - "
            "returning empty dict"
        )
        return {}

    # ========================================================================
    # MAIN PROCESSING PIPELINE
    # ========================================================================
    # This is the core functionality - processing a captured piece through
    # all stages to get a complete identification and bin assignment.

    def process_piece(self, capture_package: CapturePackage) -> IdentifiedPiece:
        """
        Process a captured piece through the complete pipeline.

        This is the main entry point called by processing workers. It runs
        the piece through all processing stages and returns a complete result.

        FLOW:
        1. Create empty IdentifiedPiece with basic info
        2. Call API to identify the piece (get element_id and name)
        3. Look up categories in database (get primary/secondary/tertiary)
        4. Assign a bin based on categories and strategy
        5. Finalize the piece (mark as complete)
        6. Notify all registered callbacks (update GUI, etc.)
        7. Return the complete IdentifiedPiece

        Called by: processing_worker.py for each piece pulled from queue

        Args:
            capture_package: Captured piece from detection pipeline
                           Contains: piece_id, image_path, timestamp, bbox, etc.

        Returns:
            IdentifiedPiece with all fields populated (or error state if something failed)

        Example:
            # In processing_worker:
            capture_package = queue_manager.get_next_piece()
            identified_piece = coordinator.process_piece(capture_package)
            # identified_piece.bin_number = 3
            # identified_piece.complete = True
        """
        # ============================================================
        # STEP 0: CREATE IDENTIFIED PIECE CONTAINER
        # ============================================================
        # Create a new IdentifiedPiece object that will hold all the
        # information we gather during processing. Start with just the
        # basic info from the capture package.
        identified_piece = IdentifiedPiece(
            piece_id=capture_package.piece_id,
            image_path=capture_package.image_path,
            capture_timestamp=capture_package.capture_timestamp
        )

        logger.info(f"Processing piece {capture_package.piece_id}")

        try:
            # ============================================================
            # STEP 1: API IDENTIFICATION
            # ============================================================
            # Send the image to the Brickognize API to identify what piece
            # this is. This gives us the element_id, name, and confidence.
            self._run_api_identification(identified_piece, capture_package)

            # ============================================================
            # STEP 2: CATEGORY LOOKUP
            # ============================================================
            # Only run if API succeeded (we got an element_id)
            # Look up the element_id in our category database to get
            # the category hierarchy (primary/secondary/tertiary)
            if identified_piece.element_id:
                self._run_category_lookup(identified_piece)

            # ============================================================
            # STEP 3: BIN ASSIGNMENT
            # ============================================================
            # Only run if we have categories (category lookup succeeded)
            # Use the categories and sorting strategy to determine which
            # physical bin this piece should go into
            if identified_piece.primary_category:
                self._run_bin_assignment(identified_piece)

            # ============================================================
            # STEP 4: FINALIZE
            # ============================================================
            # Mark the piece as complete. This sets the 'complete' flag
            # to True, indicating we've finished all processing stages
            identified_piece.finalize()

            # Update our success counter
            self.pieces_processed += 1

            logger.info(
                f"Piece {capture_package.piece_id} complete: "
                f"{identified_piece.name} → Bin {identified_piece.bin_number}"
            )

        except Exception as e:
            # ============================================================
            # ERROR HANDLING
            # ============================================================
            # Catch any unexpected errors to prevent worker crashes
            # If something goes wrong, log it but don't crash the entire
            # processing pipeline. The piece will be returned with whatever
            # data we managed to gather (might be incomplete)
            logger.error(
                f"Unexpected error processing piece {capture_package.piece_id}: {e}",
                exc_info=True
            )
            self.pieces_failed += 1
            # Piece will be incomplete but still returned

        # ============================================================
        # STEP 5: NOTIFY CALLBACKS
        # ============================================================
        # After processing is complete (success or failure), notify all
        # registered callbacks. This is where the GUI gets updated!
        # Callbacks happen whether the piece was successfully identified
        # or not - the callback can check the fields to determine success
        self._notify_identification_callbacks(identified_piece)

        # Return the complete (or incomplete) identified piece
        return identified_piece

    # ========================================================================
    # PROCESSING STAGE IMPLEMENTATIONS
    # ========================================================================
    # These private methods implement each stage of the processing pipeline.
    # They are called in sequence by process_piece().

    def _run_api_identification(self,
                                identified_piece: IdentifiedPiece,
                                capture_package: CapturePackage) -> None:
        """
        Run API identification step of pipeline.

        Calls Brickognize API with the captured image and updates the
        IdentifiedPiece with element_id, name, and confidence.

        WHAT IT DOES:
        - Reads the image file from disk
        - Sends it to the Brickognize API
        - Gets back element_id, name, and confidence score
        - Updates the identified_piece with these results

        ERROR HANDLING:
        - If image file missing: Sets element_id to None
        - If API returns no results: Sets element_id to None
        - If network error: Sets element_id to None

        The pipeline continues even if this fails - the piece just
        won't have identification data.

        Called by: process_piece() as step 1

        Args:
            identified_piece: Piece being processed (updated in place)
            capture_package: Capture data with image path

        Side effects:
            Updates identified_piece.element_id, .name, .identification_confidence
        """
        try:
            logger.debug(f"Step 1: API identification for piece {identified_piece.piece_id}")

            # Call the API handler to identify the piece
            # This sends the image to Brickognize and waits for a response
            api_result = self.api_handler.identify_piece(capture_package.image_path)

            # Update the identified piece with the API results
            # This copies element_id, name, and confidence into our piece
            identified_piece.update_from_identification(api_result)

            logger.debug(
                f"API result: element_id={api_result.element_id}, "
                f"confidence={api_result.confidence:.2f}"
            )

        except FileNotFoundError as e:
            # The image file doesn't exist - critical error
            # This shouldn't happen if capture saved the file correctly
            logger.error(f"Image file not found: {e}")
            identified_piece.element_id = None

        except ValueError as e:
            # API returned no results or invalid data
            # This happens when API doesn't recognize the piece
            logger.warning(f"API identification failed: {e}")
            identified_piece.element_id = None

        except Exception as e:
            # Network error, timeout, or other API issue
            # Log the error and continue - piece will be marked as unknown
            logger.error(f"API error: {e}")
            identified_piece.element_id = None

    def _run_category_lookup(self, identified_piece: IdentifiedPiece) -> None:
        """
        Run category lookup step of pipeline.

        Looks up the element_id in the category database and updates the
        IdentifiedPiece with category hierarchy information.

        WHAT IT DOES:
        - Takes the element_id from API identification
        - Looks it up in our CSV category database
        - Gets the category hierarchy (primary/secondary/tertiary)
        - Updates the identified_piece with these categories

        EXAMPLE:
        - element_id: "3001"
        - Primary category: "Basic"
        - Secondary category: "Brick"
        - Tertiary category: "2x4"

        ERROR HANDLING:
        If lookup fails, sets primary_category to "Unknown"

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

            # Look up the element_id in our category database
            # This searches the CSV file for this element_id
            category_info = self.category_lookup.get_categories(
                identified_piece.element_id,
                piece_name=identified_piece.name
            )

            # Update the identified piece with category results
            # This copies primary/secondary/tertiary categories into our piece
            identified_piece.update_from_categories(category_info)

            logger.debug(
                f"Categories: {category_info.primary_category}/"
                f"{category_info.secondary_category}/"
                f"{category_info.tertiary_category}"
            )

        except Exception as e:
            # Category lookup should not fail, but catch just in case
            # If it does fail, mark as Unknown so it goes to overflow
            logger.error(f"Category lookup error: {e}")
            identified_piece.primary_category = "Unknown"

    def _run_bin_assignment(self, identified_piece: IdentifiedPiece) -> None:
        """
        Run bin assignment step of pipeline.

        Assigns a bin based on the piece's categories and configured
        sorting strategy.

        WHAT IT DOES:
        - Takes the categories from category lookup
        - Applies the sorting strategy (primary/secondary/tertiary)
        - Determines which physical bin this piece should go to
        - Updates the identified_piece with the bin number

        SORTING STRATEGIES:
        - Primary: Sort by primary category (Basic, Technic, etc.)
        - Secondary: Sort by secondary category within target primary
        - Tertiary: Sort by tertiary category within target primary+secondary

        EXAMPLE:
        - Strategy: Primary
        - Primary category: "Basic"
        - Bin number: 1 (if "Basic" is assigned to bin 1)

        ERROR HANDLING:
        If assignment fails, sets bin_number to 0 (overflow bin)

        Called by: process_piece() as step 3 (only if categories found)

        Args:
            identified_piece: Piece being processed (updated in place)

        Side effects:
            Updates identified_piece.bin_number
        """
        try:
            logger.debug(f"Step 3: Bin assignment for piece {identified_piece.piece_id}")

            # Create a CategoryInfo object from the identified piece data
            # This packages up all the category information for the bin assignment module
            category_info = CategoryInfo(
                element_id=identified_piece.element_id,
                primary_category=identified_piece.primary_category,
                secondary_category=identified_piece.secondary_category,
                tertiary_category=identified_piece.tertiary_category,
                found_in_database=identified_piece.found_in_database
            )

            # Ask the bin assignment module which bin this should go to
            # It will apply the strategy and return a bin number
            bin_assignment = self.bin_assignment.assign_bin(category_info)

            # Update the identified piece with the bin assignment result
            # This copies the bin_number into our piece
            identified_piece.update_from_bin_assignment(bin_assignment)

            logger.debug(f"Assigned to bin {bin_assignment.bin_number}")

        except Exception as e:
            # Bin assignment should not fail, but catch just in case
            # If it does fail, send to overflow bin (bin 0)
            logger.error(f"Bin assignment error: {e}")
            identified_piece.bin_number = 0  # Overflow bin

    # ========================================================================
    # STATISTICS AND MONITORING
    # ========================================================================

    def get_statistics(self) -> dict:
        """
        Get processing statistics.

        Returns basic statistics about pieces processed. Can be expanded
        to include more metrics if needed.

        WHAT IT RETURNS:
        - pieces_processed: How many pieces completed successfully
        - pieces_failed: How many pieces had errors
        - success_rate: Percentage that completed successfully

        Returns:
            Dictionary with statistics

        Example:
            stats = coordinator.get_statistics()
            print(f"Processed: {stats['pieces_processed']}")
            print(f"Success rate: {stats['success_rate']:.1%}")
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
# This function creates and returns a ProcessingCoordinator instance.
# It's called during application startup.

def create_processing_coordinator(
        api_handler: IdentificationAPIHandler,
        category_lookup: CategoryLookup,
        bin_assignment: BinAssignmentModule,
        config_manager: EnhancedConfigManager
) -> ProcessingCoordinator:
    """
    Factory function to create a processing coordinator.

    Called during application startup after all sub-modules are created.

    WHY USE A FACTORY FUNCTION?
    - Provides a consistent way to create instances
    - Can add initialization logic in one place
    - Makes testing easier (can mock the factory)
    - Matches the pattern used by other modules

    Args:
        api_handler: Initialized API handler for Brickognize
        category_lookup: Initialized category lookup module
        bin_assignment: Initialized bin assignment module
        config_manager: Configuration manager instance

    Returns:
        Initialized ProcessingCoordinator ready to use

    Example:
        # During application startup in LegoSorting008:
        config_manager = create_config_manager()
        category_service = create_category_hierarchy_service(config_manager)

        # Create all sub-modules
        api_handler = create_identification_api_handler(config_manager)
        category_lookup = create_category_lookup(config_manager)
        bin_assignment = create_bin_assignment_module(config_manager, category_service)

        # Create the coordinator with all sub-modules
        coordinator = create_processing_coordinator(
            api_handler,
            category_lookup,
            bin_assignment,
            config_manager
        )

        # Register GUI callback
        coordinator.register_identification_callback(gui.on_piece_identified)

        # Use in worker
        identified_piece = coordinator.process_piece(capture_package)
    """
    return ProcessingCoordinator(
        api_handler,
        category_lookup,
        bin_assignment,
        config_manager
    )