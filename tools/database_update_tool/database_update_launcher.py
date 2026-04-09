#!/usr/bin/env python3
# tools/database_update_tool/database_update_launcher.py
"""
database_update_launcher.py - Entry point for the Database Update Tool

Generates element_id_lookup.csv by downloading Rebrickable data and
translating Rebrickable color IDs to BrickLink color IDs using name matching.
No API key is required.

FIRST-TIME SETUP:
    1. Download Rebrickable CSVs:
           python tools/database_update_tool/database_update_launcher.py --download
       Or download manually from https://rebrickable.com/downloads/ and place
       colors.csv, elements.csv, and part_relationships.csv in data/rebrickable_raw/

    2. Generate the lookup table (auto mode — downloads if needed, then translates):
           python tools/database_update_tool/database_update_launcher.py

SUBSEQUENT RUNS:
    python tools/database_update_tool/database_update_launcher.py
    python tools/database_update_tool/database_update_launcher.py --download
    python tools/database_update_tool/database_update_launcher.py --translate-only
    python tools/database_update_tool/database_update_launcher.py --report

FLAGS:
    (none)                Auto mode: download raw files if missing, then translate
    --download            Force re-download of Rebrickable CSVs before translating
    --translate-only      Use cached raw files; skip downloading
    --rebuild-color-map   Regenerate color_map.csv from name matching (rarely needed)
    --report              Print file status without downloading or writing output
    --debug               Enable verbose logging
    --raw-dir PATH        Directory for cached files (default: <project_root>/data/rebrickable_raw)
    --output PATH         Destination for element_id_lookup.csv (default: <project_root>/data/element_id_lookup.csv)

COLOR ID TRANSLATION:
    Rebrickable and BrickLink use different integers for the same LEGO colors.
    Translation is done by matching LEGO official color names, which both systems
    share.  A bundled bricklink_colors.csv reference file (shipped with this tool)
    provides the BrickLink side.  color_map.csv is auto-generated on first run.
"""

import argparse
import logging
import os
import sys

# Resolve project root from this script's location (tools/database_update_tool/)
# so that default paths work regardless of the caller's working directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))


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
        import csv    # noqa: F401
        import gzip   # noqa: F401
        import urllib # noqa: F401
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
        "--rebuild-color-map",
        action="store_true",
        help="Regenerate color_map.csv from name matching (useful after updating bricklink_colors.csv)",
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
        default=os.path.join(_PROJECT_ROOT, "data", "rebrickable_raw"),
        metavar="PATH",
        help="Directory for cached Rebrickable CSV files (default: <project_root>/data/rebrickable_raw)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(_PROJECT_ROOT, "data", "element_id_lookup.csv"),
        metavar="PATH",
        help="Output path for element_id_lookup.csv (default: <project_root>/data/element_id_lookup.csv)",
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

    # --report: just show file status
    if args.report:
        print(logic.generate_report())
        return 0

    # --rebuild-color-map: force regenerate color_map.csv from name matching
    if args.rebuild_color_map:
        logger.info("=== Rebuilding color map from name matching ===")
        try:
            matched = logic.build_color_map_from_names()
            color_map_path = os.path.join(args.raw_dir, "color_map.csv")
            print(f"\n  Color map rebuilt: {matched} Rebrickable→BrickLink color entries")
            print(f"  Stored at: {color_map_path}")
            return 0
        except FileNotFoundError as e:
            logger.error(str(e))
            return 1
        except Exception as e:
            logger.error(f"Color map rebuild failed: {e}", exc_info=args.debug)
            return 1

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

    # Auto mode: download raw files if missing, then translate
    raw_dir = args.raw_dir
    bulk_files_present = all(
        os.path.exists(os.path.join(raw_dir, f"{name}.csv"))
        for name in ("colors", "elements", "part_relationships")
    )

    if bulk_files_present:
        logger.info("=== Auto mode: raw files found — running translation ===")
    else:
        logger.info("=== Auto mode: raw files missing — downloading first ===")

    try:
        stats = logic.run_full_update(force_download=not bulk_files_present)
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
