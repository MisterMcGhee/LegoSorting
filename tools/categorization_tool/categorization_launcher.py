#!/usr/bin/env python3
# tools/categorization_tool/categorization_launcher.py
"""
categorization_launcher.py - Entry point for the categorization tool

This is the ONLY way to launch the categorization tool. It handles:
- Logging setup
- Dependency initialization
- Error handling at startup
- Launching the GUI

USAGE:
    python -m tools.categorization_tool.categorization_launcher
    
    Or from project root:
    python tools/categorization_tool/categorization_launcher.py

FUTURE:
    Can add command-line arguments:
    --debug: Enable debug logging
    --config-path: Override config file path
    etc.
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def setup_logging(debug: bool = False):
    """
    Setup logging configuration.
    
    Args:
        debug: If True, set logging level to DEBUG
    """
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("LEGO Piece Categorization Tool")
    logger.info("=" * 60)
    logger.info(f"Logging level: {logging.getLevelName(level)}")
    
    return logger


def check_dependencies(logger):
    """
    Check if all required dependencies are available.
    
    Args:
        logger: Logger instance
    
    Returns:
        True if all dependencies available, False otherwise
    """
    logger.info("Checking dependencies...")
    
    missing = []
    
    # Check PyQt5
    try:
        from PyQt5.QtWidgets import QApplication
        logger.info("✓ PyQt5 available")
    except ImportError:
        logger.error("✗ PyQt5 not found")
        missing.append("PyQt5")
    
    # Check enhanced_config_manager
    try:
        from enhanced_config_manager import create_config_manager
        logger.info("✓ enhanced_config_manager available")
    except ImportError:
        logger.error("✗ enhanced_config_manager not found")
        missing.append("enhanced_config_manager")
    
    # Check category_hierarchy_service
    try:
        from processing.category_hierarchy_service import CategoryHierarchyService
        logger.info("✓ category_hierarchy_service available")
    except ImportError:
        logger.error("✗ category_hierarchy_service not found")
        missing.append("category_hierarchy_service")
    
    # Check our own modules
    try:
        from tools.categorization_tool.categorization_logic import create_categorization_controller
        logger.info("✓ categorization_logic available")
    except ImportError:
        logger.error("✗ categorization_logic not found")
        missing.append("categorization_logic")
    
    try:
        from tools.categorization_tool.categorization_gui import PieceCategorizationGUI
        logger.info("✓ categorization_gui available")
    except ImportError:
        logger.error("✗ categorization_gui not found")
        missing.append("categorization_gui")
    
    if missing:
        logger.error("=" * 60)
        logger.error("Missing dependencies:")
        for dep in missing:
            logger.error(f"  - {dep}")
        logger.error("=" * 60)
        return False
    
    logger.info("✓ All dependencies available")
    return True


def verify_files(logger):
    """
    Verify that required files exist.
    
    Args:
        logger: Logger instance
    
    Returns:
        True if files exist or can be handled, False if critical files missing
    """
    logger.info("Verifying files...")
    
    # Check for unknown_pieces.csv (not critical - will be handled by GUI)
    if os.path.exists('unknown_pieces.csv'):
        logger.info("✓ unknown_pieces.csv found")
    else:
        logger.warning("⚠ unknown_pieces.csv not found (GUI will handle this)")
    
    # Check for Lego_Categories.csv (critical)
    possible_paths = [
        'data/Lego_Categories.csv',
        'Lego_Categories.csv'
    ]
    
    found = False
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"✓ Lego_Categories.csv found at {path}")
            found = True
            break
    
    if not found:
        logger.warning("⚠ Lego_Categories.csv not found in standard locations")
        logger.warning("  Tool will check config for path")
    
    return True


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='LEGO Piece Categorization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Categorization Tool v1.0 (Phase 1)'
    )
    
    return parser.parse_args()


def launch_gui(logger):
    """
    Launch the categorization GUI.
    
    Args:
        logger: Logger instance
    
    Returns:
        Exit code (0 for success)
    """
    try:
        logger.info("Initializing application...")
        
        # Import Qt
        from PyQt5.QtWidgets import QApplication
        
        # Create Qt application
        app = QApplication(sys.argv)
        app.setApplicationName("LEGO Categorization Tool")
        
        logger.info("Creating configuration manager...")
        from enhanced_config_manager import create_config_manager
        config_manager = create_config_manager()
        
        logger.info("Creating category hierarchy service...")
        from processing.category_hierarchy_service import CategoryHierarchyService
        category_service = CategoryHierarchyService(config_manager)
        
        logger.info("Creating categorization controller...")
        from tools.categorization_tool.categorization_logic import create_categorization_controller
        controller = create_categorization_controller(config_manager, category_service)
        
        logger.info("Creating GUI...")
        from tools.categorization_tool.categorization_gui import PieceCategorizationGUI
        gui = PieceCategorizationGUI(controller)
        
        logger.info("✓ Initialization complete")
        logger.info("=" * 60)
        logger.info("Launching categorization tool...")
        logger.info("Close the window or press Ctrl+C to exit")
        logger.info("=" * 60)
        
        # Show GUI
        gui.show()
        
        # Run application
        exit_code = app.exec_()
        
        logger.info("=" * 60)
        logger.info("Categorization tool closed")
        logger.info("=" * 60)
        
        return exit_code
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("FATAL ERROR during launch:")
        logger.error(str(e))
        logger.error("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point for the categorization tool."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(debug=args.debug)
    
    try:
        # Check dependencies
        if not check_dependencies(logger):
            logger.error("\n❌ Dependency check failed")
            logger.error("Please install missing dependencies and try again")
            return 1
        
        # Verify files
        if not verify_files(logger):
            logger.error("\n❌ File verification failed")
            return 1
        
        logger.info("\n✓ Pre-flight checks passed")
        logger.info("")
        
        # Launch GUI
        exit_code = launch_gui(logger)
        
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠ Interrupted by user")
        logger.info("Exiting...")
        return 0
    
    except Exception as e:
        logger.error("\n❌ Unexpected error:", exc_info=True)
        return 1


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    This is the ONLY entry point for the categorization tool.
    
    Direct import and use of categorization_gui is discouraged.
    Always launch via this script to ensure proper initialization.
    """
    sys.exit(main())
