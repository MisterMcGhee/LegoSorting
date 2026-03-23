# Bag-Level Sorting — Desired Behaviors

Design document for the future bag sorting module.
Not yet implemented. Recorded here to preserve decisions made during planning.

---

## Overview

Bag sorting enables a user to sort loose LEGO pieces back into the numbered bags
they belong to for a specific set. This requires matching each physical piece to
a bag-level inventory derived from the set's instruction manual.

The identification pipeline must produce a complete `(design_id, color_id)` pair —
and ideally an `element_id` — to drive this mode. The category-based sorting
modules do not require color; bag sorting does.

---

## Identification Requirements

| Field        | Required? | Source                                      |
|--------------|-----------|---------------------------------------------|
| `design_id`  | Yes       | Brickognize API (always)                    |
| `color_id`   | Yes       | Brickognize API (`?predict_color=true`)     |
| `element_id` | Preferred | `element_id_lookup.csv` (design + color)    |

Color is requested from the API on every call regardless of sorting mode.
In non-color-sorting modes the color fields are simply unused.

---

## Color Confidence Threshold Behavior

| Sorting Mode        | Color below threshold         | Color above threshold     |
|---------------------|-------------------------------|---------------------------|
| Category (current)  | Ignored — sort by design_id   | Captured but unused       |
| Bag sorting         | Piece sent to overflow bin 0  | Used for bag matching     |
| Color sorting       | Piece sent to overflow bin 0  | Used for bin assignment   |

The threshold is configured via `color_confidence_threshold` in the API config
section of `enhanced_config_manager.py`. Default: 0.5.

---

## Element ID Matching Behavior

Each bag in a set inventory may specify pieces by `element_id`. Because LEGO
occasionally reissues a part in the same color with a new `element_id` (mold
revisions, packaging changes), a physical piece and its bag entry may have
different element IDs despite being functionally identical.

### Duplicate / Doppelgänger Handling

When the scanned piece's `element_id` does not directly match the bag's expected
`element_id`, the system should:

1. Check whether the scanned `(design_id, color_id)` pair matches the expected
   `(design_id, color_id)` pair of the bag entry.
2. If both match → treat as a valid match, route the piece to the correct bag,
   and flag it as an **element ID variant** in the session log.
3. If only `design_id` matches (color differs) → do not match; send to overflow.
4. If neither matches → send to overflow.

The flag allows post-session review without blocking the sort. Multiple element
IDs per `(design_id, color_id)` pair in the lookup table are a known data
reality and are explicitly handled by this doppelgänger check rather than by
forcing a single canonical element_id.

---

## Overflow Behavior Summary

Pieces are sent to bin 0 (overflow) when:
- `design_id` cannot be identified (API failure or low confidence)
- Color is below threshold **and** the active sorting mode requires color
- No bag in the target set inventory matches the identified piece
- `element_id` doppelgänger check fails (design matches but color does not)

Pieces are **not** sent to overflow solely because `element_id` is a variant
(doppelgänger) of the expected ID, provided `(design_id, color_id)` matches.

---

## Mold Variant and Alternate Handling

LEGO reissues parts periodically with updated molds (dimensional refinements,
stud logo changes, gate-mark repositioning).  Rebrickable tracks these as
**Mold Variant** (`M`) and **Alternate** (`A`) relationships.  For bag-sorting
purposes these are considered the same physical piece.

### Part-number suffix conventions

| Suffix pattern      | Example        | Type            | Grouped? |
|---------------------|----------------|-----------------|----------|
| Single letter       | `3001a`, `3001b` | Mold Variant  | **Yes**  |
| `pb` + digits       | `3001pb001`    | Print           | No       |
| `pr` + digits       | `3001pr0001`   | Print (alt)     | No       |
| `c` + digits        | `3001c01`      | Complete assy   | No       |
| No suffix           | `3001`         | Canonical base  | —        |

### Implementation

Variant expansion is **pre-baked into `element_id_lookup.csv`** at generation
time by `tools/database_update_tool`.  For every element entry under a canonical
part, the tool also writes rows for all M/A relatives of that part.  The lookup
module therefore has no runtime knowledge of variants — a plain dict lookup on
`(part_num, bricklink_color_id)` naturally returns the full set of element IDs
across all mold generations.

Printed parts (`P` relationship type, `pb`/`pr` suffix) are **excluded** from
variant groups.  A printed 2×4 brick (`3001pb001`) is not the same piece as a
plain 2×4 brick (`3001`) for bag-sorting purposes.

### Doppelgänger matching in bag sorting

When comparing a scanned piece against a bag inventory entry:
1. Compare scanned `element_id` against expected `element_id` — exact match.
2. If no exact match, compare `(design_id, color_id)` pairs.  If both match,
   the scanned piece is a valid doppelgänger (different mold generation).
   Route to the correct bag and flag in the session log.
3. If only `design_id` matches (color differs) — no match, overflow bin.

The full list of element IDs for a given `(design_id, color_id)` is available
via `ElementIDLookup.get_element_lookup().all_element_ids` for step-2 matching.

---

## Offline Lookup Table

`element_id_lookup.csv` — generated by `tools/database_update_tool`:

| Column              | Description                                     |
|---------------------|-------------------------------------------------|
| `element_id`        | LEGO element ID (opaque integer string)         |
| `part_num`          | Rebrickable part number / design_id             |
| `bricklink_color_id`| BrickLink color ID (matches Brickognize output) |

Multiple rows with the same `(part_num, bricklink_color_id)` are valid and
expected — they represent doppelgänger element IDs for the same piece across
mold generations.  The lookup module returns all of them per key.

When a `(design_id, color_id)` pair has no entry, `element_id` is left null.
Sorting by design_id continues normally.

Re-run `tools/database_update_tool` after downloading fresh Rebrickable CSVs.
Suggested trigger: when a null element_id lookup is observed for a piece you
expected to have one, or after a major new LEGO wave release.

---

## Future Module Location

The bag-sorting strategy should live in the existing sorting module pattern
alongside the current category-based strategies.  It receives the same
`IdentifiedPiece` that all other strategies receive — the element_id field is
already populated by the pipeline before bin assignment runs.
