#!/usr/bin/env python3
"""
Simple test to verify imports work correctly
Run this from the bag_inventory_tool directory to test installation
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    print("  Importing config.config_manager...")
    from config.config_manager import create_config_manager, BagInventoryModule
    print("    ✓ Success")
except ImportError as e:
    print(f"    ✗ Failed: {e}")
    sys.exit(1)

try:
    print("  Importing models...")
    from models.piece_reference import PieceReference
    from models.reference_library import ReferenceLibrary
    print("    ✓ Success")
except ImportError as e:
    print(f"    ✗ Failed: {e}")
    sys.exit(1)

try:
    print("  Importing utils...")
    from utils.csv_handler import InventoryCSVHandler
    from utils.ocr_engine import OCREngine
    from utils import image_processing
    print("    ✓ Success")
except ImportError as e:
    print(f"    ✗ Failed: {e}")
    sys.exit(1)

try:
    print("  Importing core...")
    from core.pdf_processor import PDFProcessor
    from core.reference_builder import ReferenceBuilder
    print("    ✓ Success")
except ImportError as e:
    print(f"    ✗ Failed: {e}")
    sys.exit(1)

print("\n✓ All imports successful!")
print("\nYou can now run the tool with:")
print("  python main.py --pdf <path> --csv <path> --output <dir>")
