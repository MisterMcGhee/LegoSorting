# tools/database_update_tool/database_update_logic.py
"""
database_update_logic.py - Download and translate Rebrickable data for element ID lookup

Produces element_id_lookup.csv, the offline table used by
processing/element_id_lookup_module.py to resolve (design_id, BrickLink color_id)
pairs to LEGO element IDs.

SOURCE DATA (Rebrickable public downloads — no API key required):
  colors.csv.gz           — Rebrickable color list, includes bricklink_id column
  elements.csv.gz         — element_id → (part_num, rebrickable_color_id)
  part_relationships.csv.gz — variant/print/alternate relationships between parts

TRANSLATION STEPS:
  1. Build color mapping:  rebrickable_color_id → bricklink_color_id
     (from colors.csv bricklink_id column)

  2. Build variant groups: part_num → frozenset of related part_nums
     Only M (Mold Variant) and A (Alternate) relationship types are merged.
     P (Print) and T (Sub-part) relationships are intentionally excluded so that
     printed and decorated parts remain separate from their base design.

     Part-number suffix conventions used as a secondary sanity check:
       Single lowercase letter  (e.g. '3001a', '3001b') → Mold Variant
       'pb' + digits            (e.g. '3001pb001')       → Print  (excluded)
       'pr' + digits            (e.g. '3001pr0001')      → Print  (excluded)
       'c'  + digits            (e.g. '3001c01')         → Complete assembly
       No suffix                (e.g. '3001')            → Canonical base

  3. Translate elements.csv:
     a. Convert rebrickable_color_id to bricklink_color_id; skip row if no mapping
     b. Add translated row to output
     c. For each M/A variant in the same group, also add a row pointing to the
        same element_id — this pre-bakes variant expansion into the table so the
        lookup module needs no knowledge of variants

OUTPUT: element_id_lookup.csv
  Columns: element_id, part_num, bricklink_color_id
  Multiple rows with the same (part_num, bricklink_color_id) are intentional
  (doppelgänger element IDs — different physical batches of the same piece).
  The lookup module deduplicates per key when loading.

RE-RUN POLICY:
  Run this tool when new LEGO sets introduce parts or colors not yet in the
  table.  Raw files are cached locally so --translate-only skips re-downloading.
"""

import csv
import gzip
import io
import logging
import os
import re
import shutil
import urllib.request
from datetime import datetime, timezone
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rebrickable public download URLs (no authentication required)
# ---------------------------------------------------------------------------
REBRICKABLE_BASE = "https://cdn.rebrickable.com/media/downloads"
DOWNLOAD_URLS = {
    "colors":             f"{REBRICKABLE_BASE}/colors.csv.gz",
    "elements":           f"{REBRICKABLE_BASE}/elements.csv.gz",
    "part_relationships": f"{REBRICKABLE_BASE}/part_relationships.csv.gz",
}

# Relationship types to include in doppelgänger / variant groups
VARIANT_REL_TYPES = {"M", "A"}   # Mold Variant, Alternate
PRINT_REL_TYPES   = {"P"}         # Print — excluded from variant groups

# Suffix patterns for secondary classification (informational / sanity-check)
_SUFFIX_RE = re.compile(
    r"(?P<mold>[a-z])$"            # single letter  → mold variant
    r"|(?P<print_pb>pb\d+)"        # pb + digits     → print
    r"|(?P<print_pr>pr\d+)"        # pr + digits     → print (alt format)
    r"|(?P<complete>c\d+)"         # c  + digits     → complete assembly
    r"|(?P<pattern>pat\w+)",       # pat + anything  → pattern
    re.IGNORECASE,
)


def classify_part_suffix(part_num: str) -> str:
    """Return a string label for the part-number suffix type.

    Used for logging and sanity checks — not for lookup logic.
    The authoritative source for relationship type is part_relationships.csv.
    """
    # Strip leading numeric digits to isolate the suffix
    stripped = re.sub(r"^\d+", "", part_num)
    if not stripped:
        return "BASE"
    m = _SUFFIX_RE.match(stripped)
    if not m:
        return "UNKNOWN"
    if m.group("mold"):
        return "MOLD_VARIANT"
    if m.group("print_pb") or m.group("print_pr"):
        return "PRINT"
    if m.group("complete"):
        return "COMPLETE"
    if m.group("pattern"):
        return "PATTERN"
    return "UNKNOWN"


# ===========================================================================
# Core logic class
# ===========================================================================

class DatabaseUpdateLogic:
    """
    Orchestrates the download, translation, and output steps.

    Args:
        raw_dir:     Directory for cached Rebrickable CSV files.
        output_path: Destination path for element_id_lookup.csv.
    """

    def __init__(self, raw_dir: str = "data/rebrickable_raw",
                 output_path: str = "element_id_lookup.csv"):
        self.raw_dir = raw_dir
        self.output_path = output_path

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run_full_update(self, force_download: bool = False) -> dict:
        """Download (if needed) then translate.  Returns a stats dict."""
        self.download_rebrickable_files(force=force_download)
        return self.translate_and_build_lookup()

    def download_rebrickable_files(self, force: bool = False) -> None:
        """Download and decompress Rebrickable CSVs into raw_dir."""
        os.makedirs(self.raw_dir, exist_ok=True)

        for name, url in DOWNLOAD_URLS.items():
            dest = os.path.join(self.raw_dir, f"{name}.csv")
            if os.path.exists(dest) and not force:
                logger.info(f"  {name}.csv already present — skipping download")
                continue

            logger.info(f"  Downloading {name}.csv from Rebrickable…")
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    compressed = response.read()
                with gzip.open(io.BytesIO(compressed), "rt", encoding="utf-8") as gz:
                    content = gz.read()
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(f"  Saved {name}.csv ({len(content):,} bytes)")
            except Exception as e:
                raise RuntimeError(f"Failed to download {name}: {e}") from e

    def translate_and_build_lookup(self) -> dict:
        """
        Translate the cached Rebrickable CSVs and write element_id_lookup.csv.

        Returns a stats dict with keys:
          rows_written, rows_skipped_no_color, variant_rows_added,
          unique_keys, untranslatable_colors
        """
        colors_path    = os.path.join(self.raw_dir, "colors.csv")
        elements_path  = os.path.join(self.raw_dir, "elements.csv")
        rels_path      = os.path.join(self.raw_dir, "part_relationships.csv")

        for p in (colors_path, elements_path, rels_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Raw file missing: {p}\n"
                    f"Run with --download first."
                )

        # Step 1: Build BrickLink color map
        color_map = self._build_color_map(colors_path)
        logger.info(f"  Color map: {len(color_map)} Rebrickable → BrickLink entries")

        # Step 2: Build variant groups from part_relationships.csv
        variant_lookup = self._build_variant_groups(rels_path)
        logger.info(f"  Variant groups: {len(variant_lookup)} parts have M/A relatives")

        # Step 3: Translate elements.csv and expand variants
        rows_written          = 0
        rows_skipped_no_color = 0
        variant_rows_added    = 0
        seen_keys: Set[Tuple[str, str, str]] = set()  # (element_id, part_num, color_id)

        # Track which rebrickable colors have no BrickLink mapping
        untranslatable: Set[str] = set()

        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)

        with open(self.output_path, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            # Header — timestamp tells you when this was generated
            generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            writer.writerow([f"# Generated {generated_at} by tools/database_update_tool"])
            writer.writerow(["element_id", "part_num", "bricklink_color_id"])

            with open(elements_path, newline="", encoding="utf-8") as el_f:
                reader = csv.DictReader(el_f)
                for row in reader:
                    element_id    = row.get("element_id", "").strip()
                    part_num      = row.get("part_num", "").strip()
                    rb_color_id   = row.get("color_id", "").strip()

                    if not (element_id and part_num and rb_color_id):
                        continue

                    bl_color_id = color_map.get(rb_color_id)
                    if bl_color_id is None:
                        untranslatable.add(rb_color_id)
                        rows_skipped_no_color += 1
                        continue

                    # Write the canonical row
                    key = (element_id, part_num, bl_color_id)
                    if key not in seen_keys:
                        writer.writerow([element_id, part_num, bl_color_id])
                        seen_keys.add(key)
                        rows_written += 1

                    # Expand to all M/A variant relatives
                    relatives = variant_lookup.get(part_num, frozenset())
                    for rel_part in relatives:
                        if rel_part == part_num:
                            continue
                        rel_key = (element_id, rel_part, bl_color_id)
                        if rel_key not in seen_keys:
                            writer.writerow([element_id, rel_part, bl_color_id])
                            seen_keys.add(rel_key)
                            variant_rows_added += 1

        unique_keys = len({(r[1], r[2]) for r in seen_keys})

        stats = {
            "rows_written": rows_written,
            "variant_rows_added": variant_rows_added,
            "rows_skipped_no_color": rows_skipped_no_color,
            "unique_keys": unique_keys,
            "untranslatable_colors": len(untranslatable),
        }
        logger.info(
            f"  Output: {rows_written} base rows + {variant_rows_added} variant rows "
            f"→ {unique_keys} unique (part, color) keys\n"
            f"  Skipped {rows_skipped_no_color} rows (no BrickLink color mapping) "
            f"across {len(untranslatable)} untranslatable color IDs"
        )
        return stats

    def generate_report(self) -> str:
        """Return a text report of raw file status without writing any output."""
        lines = ["Database Update Tool — Status Report", "=" * 40]
        for name in DOWNLOAD_URLS:
            path = os.path.join(self.raw_dir, f"{name}.csv")
            if os.path.exists(path):
                size = os.path.getsize(path)
                mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime(
                    "%Y-%m-%d %H:%M"
                )
                lines.append(f"  {name:25s}  {size:>10,} bytes  last modified {mtime}")
            else:
                lines.append(f"  {name:25s}  NOT FOUND (run --download)")

        output_path = self.output_path
        if os.path.exists(output_path):
            size  = os.path.getsize(output_path)
            mtime = datetime.fromtimestamp(os.path.getmtime(output_path)).strftime(
                "%Y-%m-%d %H:%M"
            )
            lines.append(f"\n  element_id_lookup.csv   {size:>10,} bytes  generated {mtime}")
        else:
            lines.append(f"\n  element_id_lookup.csv   NOT FOUND (run translation)")
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_color_map(colors_path: str) -> Dict[str, str]:
        """
        Build Rebrickable color_id → BrickLink color_id mapping.

        Rebrickable's colors.csv includes a 'bricklink_id' column for this.
        Entries where bricklink_id is empty or '-1' are omitted — those colors
        have no BrickLink equivalent and their element entries will be skipped.
        """
        color_map: Dict[str, str] = {}
        with open(colors_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rb_id = row.get("id", "").strip()
                bl_id = row.get("bricklink_id", "").strip()
                if rb_id and bl_id and bl_id not in ("", "-1", "None"):
                    color_map[rb_id] = bl_id
        return color_map

    @staticmethod
    def _build_variant_groups(rels_path: str) -> Dict[str, FrozenSet[str]]:
        """
        Build a lookup: part_num → frozenset of all M/A related part numbers.

        Uses a path-compression union-find approach to correctly handle
        transitive chains (A is variant of B, B is variant of C → all three
        belong to the same group).

        Print (P) and Sub-part (T) relationships are explicitly excluded so
        decorated/printed parts remain separate from their base design.
        """
        parent: Dict[str, str] = {}

        def find_root(p: str) -> str:
            while p in parent:
                # Path compression: point directly to root
                parent[p] = parent.get(parent[p], parent[p])
                p = parent[p]
            return p

        def union(a: str, b: str) -> None:
            ra, rb = find_root(a), find_root(b)
            if ra != rb:
                parent[rb] = ra  # merge b's tree under a's root

        with open(rels_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_type = row.get("rel_type", "").strip().upper()
                child    = row.get("child_part_num", "").strip()
                par      = row.get("parent_part_num", "").strip()
                if rel_type in VARIANT_REL_TYPES and child and par:
                    union(child, par)

        # Build groups: root → {all members}
        groups: Dict[str, Set[str]] = {}
        for part in list(parent.keys()):
            root = find_root(part)
            groups.setdefault(root, {root}).add(part)

        # Build reverse: every member → frozenset of all group members
        member_to_group: Dict[str, FrozenSet[str]] = {}
        for members in groups.values():
            fs = frozenset(members)
            for m in fs:
                member_to_group[m] = fs

        return member_to_group


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_database_update_logic(
    raw_dir: str = "data/rebrickable_raw",
    output_path: str = "element_id_lookup.csv",
) -> DatabaseUpdateLogic:
    return DatabaseUpdateLogic(raw_dir=raw_dir, output_path=output_path)
