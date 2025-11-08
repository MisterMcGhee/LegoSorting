#!/usr/bin/env python3
"""
LEGO Bag Inventory Tool - Main Entry Point

Phase 0 + 1: Reference Library Extraction

Extracts piece reference library from LEGO instruction manual reference pages.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config_manager import create_config_manager, BagInventoryModule
from core.pdf_processor import PDFProcessor
from core.reference_builder import ReferenceBuilder
from utils.csv_handler import InventoryCSVHandler


def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging configuration.

    Args:
        verbose: Enable verbose (DEBUG) logging
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('bag_inventory_tool.log')
        ]
    )


def prompt_for_reference_pages(pdf_processor: PDFProcessor) -> list:
    """
    Prompt user to specify which pages contain the reference inventory.

    Args:
        pdf_processor: Initialized PDF processor

    Returns:
        List of page numbers (1-indexed)
    """
    total_pages = pdf_processor.get_page_count()

    print(f"\n{'='*60}")
    print(f"PDF has {total_pages} total pages")
    print(f"{'='*60}")
    print("\nThe reference inventory is typically on the last 1-3 pages of the manual.")
    print("It shows all pieces with their element IDs.\n")

    while True:
        user_input = input("Which page(s) contain the reference inventory? (e.g., '48' or '47,48,49'): ").strip()

        try:
            # Parse comma-separated page numbers
            page_numbers = [int(p.strip()) for p in user_input.split(',')]

            # Validate page numbers
            invalid = [p for p in page_numbers if p < 1 or p > total_pages]
            if invalid:
                print(f"❌ Invalid page numbers: {invalid} (must be 1-{total_pages})")
                continue

            # Confirm
            print(f"\n✓ Will process page(s): {', '.join(map(str, sorted(page_numbers)))}")
            confirm = input("Is this correct? (y/n): ").strip().lower()

            if confirm == 'y':
                return sorted(page_numbers)
            else:
                print("Let's try again.\n")

        except ValueError:
            print("❌ Invalid format. Please enter page numbers separated by commas (e.g., '47,48')\n")


def generate_report(library, inventory, output_dir: str) -> None:
    """
    Generate extraction report.

    Args:
        library: ReferenceLibrary
        inventory: InventoryCSVHandler
        output_dir: Output directory
    """
    stats = library.get_statistics()
    unvalidated = library.get_unvalidated_pieces()

    report_lines = [
        "="*70,
        "LEGO BAG INVENTORY TOOL - REFERENCE EXTRACTION REPORT",
        "="*70,
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Set: {library.set_number} - {library.set_name}",
        "",
        "-"*70,
        "EXTRACTION STATISTICS",
        "-"*70,
        f"Reference Pages: {', '.join(map(str, library.reference_pages))}",
        f"Total Pieces Extracted: {stats['total_pieces']}",
        f"Validated Against CSV: {stats['validated_pieces']} ({stats['validation_rate']:.1f}%)",
        f"Failed Validation: {stats['unvalidated_pieces']}",
        f"Average OCR Confidence: {stats['average_ocr_confidence']:.2f}",
        "",
        "-"*70,
        "CSV INVENTORY",
        "-"*70,
        f"Total Unique Pieces in CSV: {inventory.get_total_pieces()}",
        f"Total Quantity in CSV: {inventory.get_total_quantity()}",
        "",
    ]

    # Coverage analysis
    csv_element_ids = inventory.get_element_ids()
    extracted_element_ids = set(library.get_all_element_ids())

    in_csv_not_extracted = csv_element_ids - extracted_element_ids
    extracted_not_in_csv = extracted_element_ids - csv_element_ids

    report_lines.extend([
        "-"*70,
        "COVERAGE ANALYSIS",
        "-"*70,
        f"Pieces in CSV but NOT extracted: {len(in_csv_not_extracted)}",
    ])

    if in_csv_not_extracted:
        report_lines.append("\nMissing from extraction:")
        for element_id in sorted(list(in_csv_not_extracted))[:20]:  # Show first 20
            name = inventory.get_piece_name(element_id)
            report_lines.append(f"  - {element_id}: {name}")
        if len(in_csv_not_extracted) > 20:
            report_lines.append(f"  ... and {len(in_csv_not_extracted) - 20} more")

    report_lines.extend([
        "",
        f"Pieces extracted but NOT in CSV: {len(extracted_not_in_csv)}",
    ])

    if extracted_not_in_csv:
        report_lines.append("\nExtra pieces extracted:")
        for element_id in sorted(list(extracted_not_in_csv))[:20]:
            piece = library.get_piece(element_id)
            conf = piece.ocr_confidence if piece else 0
            report_lines.append(f"  - {element_id} (OCR conf: {conf:.2f})")
        if len(extracted_not_in_csv) > 20:
            report_lines.append(f"  ... and {len(extracted_not_in_csv) - 20} more")

    report_lines.extend([
        "",
        "-"*70,
        "VALIDATION FAILURES",
        "-"*70,
    ])

    if unvalidated:
        report_lines.append(f"\n{len(unvalidated)} pieces failed CSV validation:")
        for piece in unvalidated[:30]:  # Show first 30
            report_lines.append(f"  - {piece.element_id}: {piece.name} (conf: {piece.ocr_confidence:.2f})")
        if len(unvalidated) > 30:
            report_lines.append(f"  ... and {len(unvalidated) - 30} more")
    else:
        report_lines.append("\n✓ All extracted pieces validated against CSV!")

    report_lines.extend([
        "",
        "="*70,
        "END REPORT",
        "="*70,
    ])

    # Save report
    report_path = os.path.join(output_dir, "extraction_report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    # Print to console
    print('\n'.join(report_lines))
    print(f"\n✓ Report saved to: {report_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="LEGO Bag Inventory Tool - Reference Library Extraction (Phase 0+1)"
    )
    parser.add_argument('--pdf', required=True, help='Path to instruction manual PDF')
    parser.add_argument('--csv', required=True, help='Path to set inventory CSV')
    parser.add_argument('--output', default='output', help='Output directory (default: output)')
    parser.add_argument('--set-number', default='', help='LEGO set number (e.g., 31129)')
    parser.add_argument('--set-name', default='', help='LEGO set name')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--save-debug', action='store_true', help='Save debug visualizations')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("="*60)
    logger.info("LEGO Bag Inventory Tool - Reference Extraction")
    logger.info("Phase 0 + 1: Building Reference Library")
    logger.info("="*60)

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config_manager = create_config_manager()
        pdf_config = config_manager.get_module_config(BagInventoryModule.PDF.value)
        ref_config = config_manager.get_module_config(BagInventoryModule.REFERENCE.value)
        ocr_config = config_manager.get_module_config(BagInventoryModule.OCR.value)
        output_config = config_manager.get_module_config(BagInventoryModule.OUTPUT.value)

        # Load inventory CSV
        logger.info(f"Loading inventory CSV: {args.csv}")
        inventory = InventoryCSVHandler(args.csv)
        logger.info(str(inventory))

        # Open PDF
        logger.info(f"Opening PDF: {args.pdf}")
        pdf_processor = PDFProcessor(args.pdf, pdf_config)

        # Prompt for reference page numbers
        page_numbers = prompt_for_reference_pages(pdf_processor)

        # Extract reference pages
        logger.info(f"Extracting reference pages: {page_numbers}")
        reference_pages = pdf_processor.get_pages(page_numbers)

        if not reference_pages:
            logger.error("Failed to extract reference pages")
            return 1

        logger.info(f"Extracted {len(reference_pages)} reference page(s)")

        # Save debug visualization if requested
        if args.save_debug:
            debug_dir = os.path.join(args.output, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            for idx, page_img in enumerate(reference_pages):
                import cv2
                debug_path = os.path.join(debug_dir, f"reference_page_{page_numbers[idx]}.png")
                cv2.imwrite(debug_path, cv2.cvtColor(page_img, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved debug page: {debug_path}")

        # Build reference library
        logger.info("Building reference library...")
        builder = ReferenceBuilder(inventory, ref_config)
        library = builder.build_reference_library(
            reference_pages,
            page_numbers,
            ocr_config,
            args.set_number,
            args.set_name
        )

        logger.info(str(library))

        # Save reference library
        library_dir = os.path.join(args.output, "reference_library")
        builder.save_reference_library(library, library_dir)

        # Generate report
        logger.info("Generating extraction report...")
        generate_report(library, inventory, args.output)

        logger.info("\n" + "="*60)
        logger.info("✓ Reference library extraction complete!")
        logger.info(f"✓ Output directory: {args.output}")
        logger.info("="*60)

        return 0

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

    finally:
        # Cleanup
        if 'pdf_processor' in locals():
            pdf_processor.close()


if __name__ == "__main__":
    sys.exit(main())
