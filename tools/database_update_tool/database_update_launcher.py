#!/usr/bin/env python3
# tools/database_update_tool/database_update_launcher.py
"""
database_update_launcher.py - Entry point for the Database Update Tool

Generates element_id_lookup.csv by downloading Rebrickable data and
translating color IDs to BrickLink's system.

FIRST-TIME SETUP (one-time, requires free Rebrickable API key):
    1. Get a free API key at https://rebrickable.com/api/
    2. Fetch the color mapping:
           python tools/database_update_tool/database_update_launcher.py \\
               --build-color-map --api-key YOUR_KEY
    3. Download Rebrickable data and generate the lookup table:
           python tools/database_update_tool/database_update_launcher.py

SUBSEQUENT RUNS (no API key needed after step 2):
    python tools/database_update_tool/database_update_launcher.py
    python tools/database_update_tool/database_update_launcher.py --download
    python tools/database_update_tool/database_update_launcher.py --translate-only
    python tools/database_update_tool/database_update_launcher.py --report

Flags:
    --build-color-map     Fetch Rebrickable→BrickLink color mapping via API (one-time setup)
    --api-key KEY         Rebrickable API key (required with --build-color-map)
    (none)                Auto mode: download if raw files missing, else translate
    --download            Force re-download of Rebrickable CSVs before translating
    --translate-only      Use cached raw files; skip downloading
    --report              Print file status without downloading or writing output
    --debug               Enable verbose logging
    --raw-dir PATH        Directory for cached files (default: data/rebrickable_raw)
    --output PATH         Destination for element_id_lookup.csv (default: ./element_id_lookup.csv)
"""

import argparse
import logging
import os
import sys


def _setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _check_dependencies() -> bool:
    """Verify required modules are importable."""
    missing = []
    try:
        import csv       # noqa: F401
        import gzip      # noqa: F401
        import urllib    # noqa: F401
        import json      # noqa: F401
    except ImportError as e:
        missing.append(str(e))

    try:
        from tools.database_update_tool.database_update_logic import DatabaseUpdateLogic  # noqa: F401
    except ImportError as e:
        missing.append(f"database_update_logic: {e}")

    if missing:
        for m in missing:
            print(f"ERROR: Missing dependency — {m}", file=sys.stderr)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate element_id_lookup.csv from Rebrickable data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--build-color-map",
        action="store_true",
        help="Fetch Rebrickable→BrickLink color ID mapping via API (one-time setup step)",
    )
    parser.add_argument(
        "--api-key",
        metavar="KEY",
        help="Rebrickable API key (required with --build-color-map)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Force re-download of Rebrickable CSVs (overwrite cached files)",
    )
    parser.add_argument(
        "--translate-only",
        action="store_true",
        help="Skip downloading; translate using existing cached CSV files",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Show file status without downloading or writing output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/rebrickable_raw",
        metavar="PATH",
        help="Directory for cached Rebrickable CSV files (default: data/rebrickable_raw)",
    )
    parser.add_argument(
        "--output",
        default="element_id_lookup.csv",
        metavar="PATH",
        help="Output path for element_id_lookup.csv (default: ./element_id_lookup.csv)",
    )

    args = parser.parse_args()

    _setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    if not _check_dependencies():
        return 1

    from tools.database_update_tool.database_update_logic import create_database_update_logic

    logic = create_database_update_logic(
        raw_dir=args.raw_dir,
        output_path=args.output,
    )

    # --build-color-map: one-time API fetch for color mapping
    if args.build_color_map:
        if not args.api_key:
            print(
                "ERROR: --api-key is required with --build-color-map\n\n"
                "Get a free Rebrickable API key at: https://rebrickable.com/api/\n\n"
                "Then run:\n"
                "  python tools/database_update_tool/database_update_launcher.py "
                "--build-color-map --api-key YOUR_KEY",
                file=sys.stderr,
            )
            return 1
        logger.info("=== Building color map from Rebrickable API ===")
        try:
            saved = logic.fetch_color_map(args.api_key)
            print(f"\n  Color map saved: {saved} Rebrickable→BrickLink color entries")
            print(f"  Stored at: {os.path.join(args.raw_dir, 'color_map.csv')}")
            print(
                "\n  Next step — generate the element ID lookup table:\n"
                "    python tools/database_update_tool/database_update_launcher.py"
            )
            return 0
        except Exception as e:
            logger.error(f"Color map fetch failed: {e}", exc_info=args.debug)
            return 1

    # --report: just show file status
    if args.report:
        print(logic.generate_report())
        return 0

    # --translate-only: skip download, go straight to translation
    if args.translate_only:
        logger.info("=== Translate-only mode ===")
        try:
            stats = logic.translate_and_build_lookup()
            _print_stats(stats, args.output)
            return 0
        except FileNotFoundError as e:
            logger.error(str(e))
            return 1
        except Exception as e:
            logger.error(f"Translation failed: {e}", exc_info=args.debug)
            return 1

    # --download: force re-download then translate
    if args.download:
        logger.info("=== Full update (forced download) ===")
        try:
            stats = logic.run_full_update(force_download=True)
            _print_stats(stats, args.output)
            return 0
        except FileNotFoundError as e:
            logger.error(str(e))
            return 1
        except Exception as e:
            logger.error(f"Update failed: {e}", exc_info=args.debug)
            return 1

    # Auto mode: check if raw files exist (color_map.csv + bulk downloads)
    raw_dir = args.raw_dir
    color_map_present = os.path.exists(os.path.join(raw_dir, "color_map.csv"))
    bulk_files_present = all(
        os.path.exists(os.path.join(raw_dir, f"{name}.csv"))
        for name in ("elements", "part_relationships")
    )

    if not color_map_present:
        print(
            "\nSetup required: color_map.csv not found.\n\n"
            "The Rebrickable→BrickLink color mapping must be fetched once from the\n"
            "Rebrickable API.  Get a free API key at: https://rebrickable.com/api/\n\n"
            "Then run:\n"
            "  python tools/database_update_tool/database_update_launcher.py "
            "--build-color-map --api-key YOUR_KEY\n",
            file=sys.stderr,
        )
        return 1

    if bulk_files_present:
        logger.info("=== Auto mode: raw files found — running translation ===")
        try:
            stats = logic.translate_and_build_lookup()
            _print_stats(stats, args.output)
            return 0
        except FileNotFoundError as e:
            logger.error(str(e))
            return 1
        except Exception as e:
            logger.error(f"Translation failed: {e}", exc_info=args.debug)
            return 1
    else:
        logger.info("=== Auto mode: raw files missing — downloading first ===")
        try:
            stats = logic.run_full_update(force_download=False)
            _print_stats(stats, args.output)
            return 0
        except FileNotFoundError as e:
            logger.error(str(e))
            return 1
        except Exception as e:
            logger.error(f"Update failed: {e}", exc_info=args.debug)
            return 1


def _print_stats(stats: dict, output_path: str) -> None:
    total = stats["rows_written"] + stats["variant_rows_added"]
    print("\n" + "=" * 50)
    print("  element_id_lookup.csv generated")
    print("=" * 50)
    print(f"  Output path         : {output_path}")
    print(f"  Base rows           : {stats['rows_written']:>10,}")
    print(f"  Variant rows added  : {stats['variant_rows_added']:>10,}")
    print(f"  Total rows          : {total:>10,}")
    print(f"  Unique (part,color) : {stats['unique_keys']:>10,}")
    print(f"  Skipped (no color)  : {stats['rows_skipped_no_color']:>10,}")
    print(f"  Untranslatable clrs : {stats['untranslatable_colors']:>10,}")
    print("=" * 50)
    print("  Restart the sorting application to load the new table.")


if __name__ == "__main__":
    sys.exit(main())
