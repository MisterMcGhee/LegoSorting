# LEGO Bag-Level Inventory Extraction Tool
## Design Document v1.0

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [System Architecture](#system-architecture)
4. [Component Specifications](#component-specifications)
5. [Data Flow](#data-flow)
6. [Development Phases](#development-phases)
7. [Testing Strategy](#testing-strategy)
8. [Technical Specifications](#technical-specifications)
9. [Future Enhancements](#future-enhancements)

---

## Executive Summary

### Purpose
This tool extracts bag-level piece inventories from LEGO instruction manuals and creates a structured database mapping which pieces belong in which numbered bags. This data is not available from LEGO or third-party sources and enables advanced sorting and inventory management for the LEGO Sorting Machine project.

### Key Innovation
By leveraging the visual reference inventory on the final pages of instruction manuals, we can create a self-contained system that requires no external piece identification databases. Each instruction manual contains everything needed to generate its own bag-level inventory.

### Primary Use Case
Enable the LEGO Sorting Machine to organize pieces by set and bag number, allowing users to:
- Sort a collection by which bags pieces belong to
- Validate bag completeness before building
- Organize loose pieces back into proper bags
- Generate missing bag inventories for incomplete sets

---

## Project Overview

### Problem Statement
LEGO sets contain pieces subdivided into numbered bags to streamline the building process. While total set inventories are readily available (BrickLink, Rebrickable), bag-level inventories are not publicly documented. Manually cataloging this data for each set is time-consuming and error-prone.

### Solution Approach
Extract bag-level inventory data directly from official LEGO instruction PDFs by:
1. Building a reference library from the manual's visual inventory pages
2. Detecting bag opening indicators to establish bag boundaries
3. Extracting piece images from "required pieces" boxes on each page
4. Matching pieces to the reference library using computer vision
5. Aggregating pieces by bag number and validating against total inventory

### Scope

**In Scope:**
- Parsing PDF instruction manuals
- OCR of element IDs from reference pages
- Template matching for piece identification
- Bag boundary detection
- Quantity extraction from required pieces boxes
- Validation against total inventory CSV
- Export to structured data format (JSON/CSV)

**Out of Scope:**
- Building a global piece image database
- Color identification (beyond what's in element_id)
- design_id/color_id extraction (handled by broader sorting system)
- Integration with the main LEGO Sorting Machine (separate project)
- Real-time processing (batch processing is acceptable)

### Success Criteria
- **Accuracy**: >95% correct piece identification per set
- **Validation**: Bag totals must reconcile with input inventory CSV
- **Usability**: Process one complete set with <5 minutes of manual intervention
- **Maintainability**: Modular architecture supporting future enhancements
- **Self-contained**: Each manual processing requires only PDF + CSV inputs

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  • Instruction Manual PDF                                   │
│  • Total Inventory CSV (element_id, quantity, name)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   REFERENCE EXTRACTION                      │
├─────────────────────────────────────────────────────────────┤
│  1. Extract final pages (visual inventory)                  │
│  2. Detect and crop individual piece images                 │
│  3. OCR element_id values                                   │
│  4. Validate against input CSV                              │
│  5. Build reference library: {image → element_id}           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  BAG BOUNDARY DETECTION                     │
├─────────────────────────────────────────────────────────────┤
│  1. Scan all instruction pages                              │
│  2. Detect bag opening symbols/indicators                   │
│  3. Establish page ranges for each bag                      │
│  4. Create bag → page_range mapping                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              REQUIRED PIECES BOX EXTRACTION                 │
├─────────────────────────────────────────────────────────────┤
│  For each page in each bag:                                 │
│  1. Locate "required pieces" boxes                          │
│  2. Extract individual piece images                         │
│  3. OCR quantity indicators (e.g., "2x")                    │
│  4. Crop and normalize piece images for matching            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  PIECE IDENTIFICATION                       │
├─────────────────────────────────────────────────────────────┤
│  1. Template match piece image against reference library    │
│  2. Apply confidence thresholds                             │
│  3. Cache successful matches for reuse                      │
│  4. Flag ambiguous matches for review                       │
│  5. Return element_id + confidence score                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  AGGREGATION & VALIDATION                   │
├─────────────────────────────────────────────────────────────┤
│  1. Aggregate pieces by bag number                          │
│  2. Sum quantities across all bags                          │
│  3. Compare against total inventory CSV                     │
│  4. Flag discrepancies                                      │
│  5. Generate confidence report                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                           │
├─────────────────────────────────────────────────────────────┤
│  • Bag-level inventory (JSON/CSV)                           │
│  • Validation report                                        │
│  • Manual review queue (ambiguous matches)                  │
│  • Processing log                                           │
└─────────────────────────────────────────────────────────────┘
```

### Module Structure

```
lego_bag_inventory/
│
├── core/
│   ├── __init__.py
│   ├── pdf_processor.py          # PDF extraction and page management
│   ├── reference_builder.py      # Build reference library from manual
│   ├── bag_detector.py           # Detect bag boundaries
│   ├── piece_extractor.py        # Extract pieces from required boxes
│   ├── piece_matcher.py          # Template matching engine
│   └── validator.py              # Inventory validation
│
├── utils/
│   ├── __init__.py
│   ├── image_processing.py       # Image manipulation utilities
│   ├── ocr_engine.py             # OCR wrapper and validation
│   ├── csv_handler.py            # CSV import/export
│   └── logging_config.py         # Structured logging
│
├── models/
│   ├── __init__.py
│   ├── reference_library.py      # Reference library data structure
│   ├── bag_inventory.py          # Bag inventory data structure
│   └── match_result.py           # Match result with confidence
│
├── config/
│   ├── __init__.py
│   └── settings.py               # Configuration parameters
│
└── tests/
    ├── test_reference_builder.py
    ├── test_bag_detector.py
    ├── test_piece_matcher.py
    └── test_validator.py
```

---

## Component Specifications

### 1. PDF Processor (`pdf_processor.py`)

**Purpose:** Extract pages from PDF instruction manuals as images for processing.

**Responsibilities:**
- Load PDF files
- Convert pages to images at appropriate resolution
- Provide page-by-page access
- Cache rendered pages for performance

**Key Functions:**
```python
class PDFProcessor:
    def __init__(self, pdf_path: str, dpi: int = 300):
        """Initialize with PDF path and rendering resolution"""
        
    def get_page_count(self) -> int:
        """Return total number of pages"""
        
    def get_page_image(self, page_num: int) -> np.ndarray:
        """Extract single page as image array"""
        
    def get_page_range(self, start: int, end: int) -> List[np.ndarray]:
        """Extract range of pages"""
        
    def get_last_n_pages(self, n: int = 3) -> List[np.ndarray]:
        """Get last N pages (typically reference inventory)"""
```

**Dependencies:**
- PyMuPDF (fitz) or pdf2image
- PIL/Pillow for image handling
- NumPy for array operations

**Configuration Parameters:**
- `dpi`: 300 (balance between quality and performance)
- `color_mode`: RGB
- `cache_enabled`: True

---

### 2. Reference Builder (`reference_builder.py`)

**Purpose:** Extract piece reference library from the manual's inventory pages.

**Responsibilities:**
- Process final inventory pages
- Detect individual piece image regions
- OCR element_id values
- Validate against input CSV
- Build searchable reference library

**Key Functions:**
```python
class ReferenceBuilder:
    def __init__(self, inventory_csv_path: str):
        """Load total inventory for validation"""
        
    def build_reference_library(self, reference_pages: List[np.ndarray]) -> ReferenceLibrary:
        """Main processing function"""
        
    def detect_piece_regions(self, page_image: np.ndarray) -> List[BoundingBox]:
        """Locate individual piece images on page"""
        
    def extract_element_id(self, region: np.ndarray) -> str:
        """OCR element_id from region"""
        
    def validate_element_id(self, element_id: str) -> bool:
        """Check if element_id exists in CSV"""
        
    def create_template(self, piece_image: np.ndarray) -> np.ndarray:
        """Prepare grayscale template for matching"""
```

**Processing Pipeline:**
```
Reference Page(s)
    ↓
[Detect piece regions] → Use contour detection + grid layout analysis
    ↓
[For each region]
    ├─ [Crop piece image]
    ├─ [Crop element_id text area]
    ├─ [OCR element_id] → Tesseract/EasyOCR
    ├─ [Validate against CSV]
    └─ [Store in library]
    ↓
Reference Library Complete
```

**Data Structure:**
```python
@dataclass
class PieceReference:
    element_id: str
    template_grayscale: np.ndarray
    original_image: np.ndarray
    name: str  # from CSV
    bounding_box: Tuple[int, int, int, int]
    
@dataclass
class ReferenceLibrary:
    pieces: Dict[str, PieceReference]
    set_number: str
    total_unique_pieces: int
    validation_status: Dict[str, bool]
```

**OCR Validation Strategy:**
- First pass: Standard OCR
- If validation fails: Retry with different preprocessing
- If still fails: Flag for manual review
- Build confidence score based on CSV match

**Edge Cases:**
- Multiple pieces per row (grid detection)
- Rotated or angled piece images (normalize orientation)
- Element_id partially obscured (bounding box adjustment)
- Duplicate element_ids on page (aggregate)

---

### 3. Bag Detector (`bag_detector.py`)

**Purpose:** Identify pages where new bags are opened, establishing bag boundaries.

**Responsibilities:**
- Scan all instruction pages
- Detect bag opening indicators
- Map page ranges to bag numbers
- Handle edge cases (sub-bags, optional bags)

**Key Functions:**
```python
class BagDetector:
    def __init__(self, bag_symbol_template: np.ndarray = None):
        """Initialize with optional custom bag symbol template"""
        
    def detect_bag_boundaries(self, pages: List[np.ndarray]) -> Dict[int, PageRange]:
        """Scan all pages and return bag → page_range mapping"""
        
    def find_bag_symbol(self, page: np.ndarray) -> Optional[BagIndicator]:
        """Locate bag opening symbol on a single page"""
        
    def extract_bag_number(self, indicator_region: np.ndarray) -> int:
        """OCR bag number from indicator region"""
```

**Detection Methods:**

**Method 1: Template Matching** (Primary)
```python
# Use a reference bag symbol image
bag_symbol_template = load_bag_symbol()  # from example manual
for page in pages:
    match = cv2.matchTemplate(page, bag_symbol_template, cv2.TM_CCOEFF_NORMED)
    if max(match) > threshold:
        # Found bag indicator
```

**Method 2: Symbol Recognition** (Fallback)
- Look for distinctive visual patterns (bag icon, numbered circle)
- Color detection (bag indicators often in distinctive colors)
- Position analysis (typically in corners or margins)

**Method 3: Text Detection** (Secondary validation)
- OCR for text like "BAG 1", "Bag 2", etc.
- Language-agnostic number detection

**Data Structure:**
```python
@dataclass
class BagIndicator:
    page_number: int
    bag_number: int
    position: Tuple[int, int]
    confidence: float

@dataclass
class BagBoundaries:
    bag_mappings: Dict[int, PageRange]  # {bag_number: (start_page, end_page)}
    total_bags: int
    confidence_scores: Dict[int, float]
```

**Edge Cases:**
- Bag 1 might not have explicit indicator (assume starts at page 1)
- Final bag ends at last instruction page (before reference pages)
- Some sets have sub-bags (1A, 1B) - treat as single bag initially
- Missing bag indicators - use heuristics (page count / estimated bags)

---

### 4. Piece Extractor (`piece_extractor.py`)

**Purpose:** Extract piece images and quantities from "required pieces" boxes on instruction pages.

**Responsibilities:**
- Locate required pieces boxes on each page
- Extract individual piece images
- OCR quantity indicators (e.g., "2x", "4x")
- Normalize images for matching

**Key Functions:**
```python
class PieceExtractor:
    def __init__(self, config: ExtractionConfig):
        """Initialize with extraction parameters"""
        
    def extract_from_page(self, page: np.ndarray) -> List[RequiredPiece]:
        """Extract all pieces from a single page"""
        
    def locate_required_boxes(self, page: np.ndarray) -> List[BoundingBox]:
        """Find required pieces boxes on page"""
        
    def extract_pieces_from_box(self, box_region: np.ndarray) -> List[RequiredPiece]:
        """Extract individual pieces from a box"""
        
    def extract_quantity(self, piece_region: np.ndarray) -> int:
        """OCR quantity indicator (e.g., "2x" → 2)"""
        
    def normalize_piece_image(self, piece_image: np.ndarray) -> np.ndarray:
        """Prepare image for template matching"""
```

**Box Detection Strategy:**

**Method 1: Blue Background Detection**
```python
# Required pieces boxes typically have light blue background
hsv = cv2.cvtColor(page, cv2.COLOR_BGR2HSV)
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Filter by size to get boxes
```

**Method 2: Rectangle Detection**
```python
# Look for rectangular regions in consistent positions
# Usually top-right or top-left of page
```

**Method 3: Text Detection + Layout Analysis**
- Detect text in consistent font/style
- Look for numbered step indicators nearby
- Use spatial relationships to locate boxes

**Piece Extraction Within Box:**
```python
def extract_pieces_from_box(box_region):
    # 1. Convert to grayscale
    gray = cv2.cvtColor(box_region, cv2.COLOR_BGR2GRAY)
    
    # 2. Detect piece contours (pieces are darker than background)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Filter contours by size (remove noise, text)
    piece_contours = [c for c in contours if min_size < cv2.contourArea(c) < max_size]
    
    # 4. For each contour:
    pieces = []
    for contour in piece_contours:
        x, y, w, h = cv2.boundingRect(contour)
        piece_image = box_region[y:y+h, x:x+w]
        
        # Look for quantity indicator nearby
        quantity = extract_quantity_near(box_region, x, y)
        
        pieces.append(RequiredPiece(piece_image, quantity))
    
    return pieces
```

**Quantity Extraction:**
```python
def extract_quantity(piece_region):
    # Look for pattern: digit + "x" (e.g., "2x", "10x")
    # OCR small region near piece
    # Parse number
    # Default to 1 if not found
```

**Image Normalization:**
```python
def normalize_piece_image(piece_image):
    # 1. Remove background (make white)
    # 2. Center piece in frame
    # 3. Standardize size (pad or resize)
    # 4. Convert to grayscale
    # 5. Apply contrast enhancement
    return normalized_image
```

**Data Structure:**
```python
@dataclass
class RequiredPiece:
    image: np.ndarray
    quantity: int
    position_in_box: Tuple[int, int]
    page_number: int
    confidence: float
```

---

### 5. Piece Matcher (`piece_matcher.py`)

**Purpose:** Match extracted piece images to reference library using template matching.

**Responsibilities:**
- Perform template matching
- Handle scale variations
- Implement caching for performance
- Return match confidence scores
- Flag ambiguous matches

**Key Functions:**
```python
class PieceMatcher:
    def __init__(self, reference_library: ReferenceLibrary):
        """Initialize with reference library"""
        
    def match_piece(self, piece_image: np.ndarray) -> MatchResult:
        """Find best match in reference library"""
        
    def multi_scale_match(self, piece_image: np.ndarray, template: np.ndarray) -> float:
        """Template match at multiple scales"""
        
    def get_cached_match(self, image_hash: str) -> Optional[MatchResult]:
        """Check cache for previous match"""
        
    def cache_match(self, image_hash: str, result: MatchResult):
        """Store successful match in cache"""
```

**Matching Algorithm:**

```python
def match_piece(piece_image):
    # 1. Compute image hash for cache lookup
    image_hash = compute_hash(piece_image)
    
    # 2. Check cache
    if image_hash in match_cache:
        return match_cache[image_hash]
    
    # 3. Prepare piece image
    piece_gray = cv2.cvtColor(piece_image, cv2.COLOR_BGR2GRAY)
    piece_normalized = normalize_size(piece_gray)
    
    # 4. Match against all templates
    matches = []
    for element_id, reference in reference_library.items():
        score = multi_scale_template_match(piece_normalized, reference.template)
        matches.append((element_id, score))
    
    # 5. Sort by score
    matches.sort(key=lambda x: x[1], reverse=True)
    
    # 6. Evaluate confidence
    best_match, best_score = matches[0]
    second_best_score = matches[1][1] if len(matches) > 1 else 0
    
    confidence_level = evaluate_confidence(best_score, second_best_score)
    
    result = MatchResult(
        element_id=best_match,
        confidence_score=best_score,
        confidence_level=confidence_level,
        runner_up=(matches[1][0], second_best_score) if len(matches) > 1 else None
    )
    
    # 7. Cache result
    match_cache[image_hash] = result
    
    return result
```

**Multi-Scale Template Matching:**
```python
def multi_scale_template_match(image, template):
    scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    best_score = 0
    
    for scale in scales:
        h, w = template.shape[:2]
        scaled_template = cv2.resize(template, (int(w * scale), int(h * scale)))
        
        # Ensure template not larger than image
        if scaled_template.shape[0] > image.shape[0] or scaled_template.shape[1] > image.shape[1]:
            continue
        
        result = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        best_score = max(best_score, max_val)
    
    return best_score
```

**Confidence Evaluation:**
```python
def evaluate_confidence(best_score, second_best_score):
    # Thresholds based on empirical testing
    if best_score >= 0.90:
        return "HIGH"
    elif best_score >= 0.80 and (best_score - second_best_score) >= 0.15:
        return "MEDIUM"
    elif best_score >= 0.70:
        return "LOW"
    else:
        return "NO_MATCH"
```

**Caching Strategy:**
```python
# Use perceptual hash for similar images
import imagehash
from PIL import Image

def compute_hash(image_array):
    image_pil = Image.fromarray(image_array)
    return str(imagehash.phash(image_pil))

# Cache structure
match_cache = {
    "hash_value": MatchResult(...),
    # ...
}

# Cache hits expected for common pieces
# Example: "3023" (Plate 1x2) might appear 15+ times in one set
# First match: ~50ms, subsequent: ~1ms
```

**Data Structure:**
```python
@dataclass
class MatchResult:
    element_id: str
    confidence_score: float
    confidence_level: str  # HIGH, MEDIUM, LOW, NO_MATCH
    runner_up: Optional[Tuple[str, float]]  # For ambiguous cases
    processing_time_ms: float
```

---

### 6. Validator (`validator.py`)

**Purpose:** Validate extracted bag inventory against total inventory and flag discrepancies.

**Responsibilities:**
- Aggregate pieces across all bags
- Compare totals to input CSV
- Identify missing or extra pieces
- Generate validation report
- Calculate overall confidence

**Key Functions:**
```python
class Validator:
    def __init__(self, total_inventory_csv: str):
        """Load expected inventory"""
        
    def validate_bag_inventory(self, bag_inventory: BagInventory) -> ValidationReport:
        """Compare bag totals to expected inventory"""
        
    def check_bag_consistency(self, bag_inventory: BagInventory) -> List[Issue]:
        """Check for internal consistency issues"""
        
    def generate_report(self, validation_results: ValidationResults) -> str:
        """Create human-readable validation report"""
```

**Validation Checks:**

**1. Quantity Reconciliation**
```python
def validate_quantities(bag_inventory, total_inventory):
    bag_totals = {}
    
    # Sum across all bags
    for bag in bag_inventory.bags:
        for piece in bag.pieces:
            bag_totals[piece.element_id] = bag_totals.get(piece.element_id, 0) + piece.quantity
    
    discrepancies = []
    
    # Compare to expected
    for element_id, expected_qty in total_inventory.items():
        actual_qty = bag_totals.get(element_id, 0)
        
        if actual_qty != expected_qty:
            discrepancies.append(Discrepancy(
                element_id=element_id,
                expected=expected_qty,
                actual=actual_qty,
                difference=actual_qty - expected_qty
            ))
    
    # Check for unexpected pieces
    for element_id in bag_totals:
        if element_id not in total_inventory:
            discrepancies.append(Discrepancy(
                element_id=element_id,
                expected=0,
                actual=bag_totals[element_id],
                difference=bag_totals[element_id],
                issue_type="UNEXPECTED_PIECE"
            ))
    
    return discrepancies
```

**2. Confidence Analysis**
```python
def analyze_confidence(bag_inventory):
    total_pieces = 0
    high_confidence = 0
    medium_confidence = 0
    low_confidence = 0
    no_match = 0
    
    for bag in bag_inventory.bags:
        for piece in bag.pieces:
            total_pieces += piece.quantity
            
            if piece.match_confidence == "HIGH":
                high_confidence += piece.quantity
            elif piece.match_confidence == "MEDIUM":
                medium_confidence += piece.quantity
            elif piece.match_confidence == "LOW":
                low_confidence += piece.quantity
            else:
                no_match += piece.quantity
    
    return ConfidenceReport(
        total_pieces=total_pieces,
        high_confidence_pct=(high_confidence / total_pieces) * 100,
        medium_confidence_pct=(medium_confidence / total_pieces) * 100,
        low_confidence_pct=(low_confidence / total_pieces) * 100,
        no_match_pct=(no_match / total_pieces) * 100
    )
```

**3. Spare Parts Handling**
```python
def identify_spares(total_inventory_csv):
    # Detect likely spares (quantity=1, common pieces)
    # Often listed at end of CSV
    # Common spare pieces: 3024 (1x1 plate), 54200 (1x1x2/3 slope), etc.
    
    spares = []
    main_inventory = []
    
    spare_threshold_line = find_spare_section_start(total_inventory_csv)
    
    for idx, row in enumerate(total_inventory_csv):
        if idx >= spare_threshold_line:
            spares.append(row)
        else:
            main_inventory.append(row)
    
    return main_inventory, spares
```

**Validation Report Structure:**
```python
@dataclass
class ValidationReport:
    overall_status: str  # PASS, PASS_WITH_WARNINGS, FAIL
    quantity_discrepancies: List[Discrepancy]
    confidence_analysis: ConfidenceReport
    missing_pieces: List[str]
    extra_pieces: List[str]
    ambiguous_matches: List[AmbiguousMatch]
    processing_summary: ProcessingSummary
    timestamp: datetime
```

**Report Generation:**
```
=== VALIDATION REPORT ===
Set: 31129 - Majestic Tiger
Processing Date: 2024-11-07 14:23:15

OVERALL STATUS: PASS_WITH_WARNINGS

QUANTITY RECONCILIATION:
✓ 145/150 element_ids match expected quantities (96.7%)
⚠ 5 discrepancies found:

  Element ID: 3023 (Orange Plate 1x2)
    Expected: 17  |  Found: 16  |  Difference: -1
    Bags: 1(4), 2(6), 3(6)
    
  Element ID: 54200 (White Slope 30 1x1x2/3)
    Expected: 8  |  Found: 9  |  Difference: +1
    Bags: 2(4), 4(5)

CONFIDENCE ANALYSIS:
  High Confidence:    342 pieces (89.3%)
  Medium Confidence:   28 pieces (7.3%)
  Low Confidence:       8 pieces (2.1%)
  No Match:             5 pieces (1.3%)

AMBIGUOUS MATCHES:
  • Bag 2, Step 5: Matched to 32952 (conf: 0.82)
    Runner-up: 32951 (conf: 0.79) - Manual review recommended

RECOMMENDATION:
Manual review of 5 low-confidence matches recommended.
Likely causes: Slight scale variation in required pieces boxes.

=== END REPORT ===
```

---

## Data Flow

### End-to-End Processing Flow

```
USER INPUT:
├─ instruction_manual.pdf
└─ set_inventory.csv

    ↓

STEP 1: INITIALIZATION
├─ Load PDF → PDFProcessor
├─ Load CSV → CSVHandler
└─ Configure processing parameters

    ↓

STEP 2: REFERENCE LIBRARY CONSTRUCTION
├─ Extract last 2-3 pages from PDF
├─ For each piece on reference pages:
│  ├─ Detect bounding box
│  ├─ Crop piece image → normalize → convert to grayscale template
│  ├─ OCR element_id
│  ├─ Validate element_id exists in CSV
│  └─ Store in ReferenceLibrary
└─ ReferenceLibrary: {element_id → template}

    ↓

STEP 3: BAG BOUNDARY DETECTION
├─ Scan pages 1-N (excluding reference pages)
├─ Detect bag opening symbols/indicators
├─ Extract bag numbers
└─ Create mapping: {bag_number → (start_page, end_page)}

    ↓

STEP 4: PIECE EXTRACTION (per bag)
For bag in bags:
  For page in bag.page_range:
    ├─ Locate required pieces boxes
    ├─ For each box:
    │  ├─ Extract piece images
    │  ├─ OCR quantities (e.g., "2x")
    │  └─ Add to extraction queue
    └─ Store: [(piece_image, quantity, page_num)]

    ↓

STEP 5: PIECE IDENTIFICATION
For each extracted_piece:
  ├─ Check cache (by image hash)
  ├─ If not cached:
  │  ├─ Multi-scale template match against ReferenceLibrary
  │  ├─ Select best match with confidence score
  │  └─ Cache result
  └─ Assign element_id + confidence_level

    ↓

STEP 6: AGGREGATION
├─ Group pieces by bag_number
├─ Sum quantities for duplicate pieces within bag
└─ Create BagInventory structure

    ↓

STEP 7: VALIDATION
├─ Sum all bags → total_found
├─ Compare to total_inventory.csv → discrepancies
├─ Analyze confidence scores
├─ Flag pieces needing manual review
└─ Generate ValidationReport

    ↓

STEP 8: OUTPUT GENERATION
├─ Export bag_inventory.json
├─ Export validation_report.txt
├─ Export manual_review_queue.json (if needed)
└─ Export processing_log.txt

    ↓

OUTPUT:
├─ bag_inventory.json: {bag_1: [...], bag_2: [...]}
├─ validation_report.txt: Human-readable summary
├─ manual_review_queue.json: Low-confidence matches
└─ processing_log.txt: Detailed execution log
```

### Data Structures

**ReferenceLibrary**
```json
{
  "set_number": "31129",
  "set_name": "Majestic Tiger",
  "pieces": {
    "3034": {
      "element_id": "3034",
      "name": "Red Plate 2 x 8",
      "template_hash": "a3b5c7d9...",
      "original_image_path": "ref_images/3034.png",
      "grayscale_template_path": "templates/3034_gray.png"
    },
    "32952": {
      "element_id": "32952",
      "name": "Blue Brick, Modified 1 x 1 x 1 2/3 with Studs on Side",
      "template_hash": "f4e2c1b8...",
      "original_image_path": "ref_images/32952.png",
      "grayscale_template_path": "templates/32952_gray.png"
    }
  },
  "total_unique_pieces": 150,
  "extraction_timestamp": "2024-11-07T14:15:00Z"
}
```

**BagInventory**
```json
{
  "set_number": "31129",
  "set_name": "Majestic Tiger",
  "total_bags": 6,
  "bags": [
    {
      "bag_number": 1,
      "page_range": [6, 18],
      "pieces": [
        {
          "element_id": "3034",
          "name": "Red Plate 2 x 8",
          "quantity": 1,
          "confidence": "HIGH",
          "confidence_score": 0.95,
          "pages_found": [7, 10]
        },
        {
          "element_id": "32952",
          "name": "Blue Brick, Modified...",
          "quantity": 2,
          "confidence": "HIGH",
          "confidence_score": 0.93,
          "pages_found": [12, 12]
        }
      ],
      "total_pieces_in_bag": 47,
      "unique_pieces_in_bag": 23
    },
    {
      "bag_number": 2,
      "page_range": [19, 35],
      "pieces": [...]
    }
  ],
  "processing_timestamp": "2024-11-07T14:23:15Z"
}
```

**ValidationReport**
```json
{
  "set_number": "31129",
  "overall_status": "PASS_WITH_WARNINGS",
  "summary": {
    "total_expected_pieces": 383,
    "total_found_pieces": 381,
    "matching_element_ids": 145,
    "total_element_ids": 150,
    "accuracy_percentage": 96.7
  },
  "discrepancies": [
    {
      "element_id": "3023",
      "name": "Orange Plate 1 x 2",
      "expected_quantity": 17,
      "found_quantity": 16,
      "difference": -1,
      "found_in_bags": [1, 2, 3],
      "possible_cause": "Extraction failure or OCR quantity error"
    }
  ],
  "confidence_breakdown": {
    "high_confidence_pieces": 342,
    "medium_confidence_pieces": 28,
    "low_confidence_pieces": 8,
    "no_match_pieces": 5
  },
  "manual_review_queue": [
    {
      "bag_number": 2,
      "page_number": 24,
      "piece_image_path": "review_queue/bag2_page24_piece3.png",
      "best_match": "32952",
      "confidence_score": 0.82,
      "runner_up": "32951",
      "runner_up_score": 0.79,
      "reason": "Ambiguous match - similar pieces"
    }
  ],
  "timestamp": "2024-11-07T14:23:15Z"
}
```

---

## Development Phases

### Phase 0: Setup & Environment (1-2 days)

**Objectives:**
- Establish project structure
- Configure development environment
- Set up testing framework
- Create sample datasets

**Deliverables:**
- Project directory structure
- requirements.txt with dependencies
- Basic configuration file
- Test data: 1 sample PDF + CSV

**Dependencies:**
```
opencv-python>=4.8.0
pytesseract>=0.3.10
PyMuPDF>=1.23.0  # or pdf2image
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
imagehash>=4.3.0
pytest>=7.4.0
```

**Tasks:**
1. Create module structure
2. Set up virtual environment
3. Install OCR engine (Tesseract)
4. Configure logging
5. Write basic configuration loader

**Success Criteria:**
- All imports work
- Can load sample PDF and CSV
- Basic logging functional

---

### Phase 1: Reference Library Extraction (1 week)

**Objectives:**
- Extract reference inventory pages from PDF
- Build piece image detection
- Implement OCR for element_ids
- Validate against input CSV
- Create reference library data structure

**Components to Build:**
- `pdf_processor.py`: Basic PDF loading and page extraction
- `reference_builder.py`: Complete implementation
- `ocr_engine.py`: OCR wrapper with validation
- `reference_library.py`: Data model

**Development Sequence:**

**Step 1.1: PDF Page Extraction**
```python
# Test: Can we extract the last 3 pages as images?
processor = PDFProcessor("sample_manual.pdf")
last_pages = processor.get_last_n_pages(3)
assert len(last_pages) == 3
assert last_pages[0].shape[2] == 3  # RGB
```

**Step 1.2: Piece Region Detection**
```python
# Test: Can we find individual piece bounding boxes?
builder = ReferenceBuilder()
regions = builder.detect_piece_regions(reference_page_image)
# Manually verify: Are all pieces detected?
# Visually inspect: Draw boxes on image
```

**Step 1.3: OCR Element IDs**
```python
# Test: Can we read element_ids correctly?
element_id = builder.extract_element_id(piece_region)
# Compare to manual ground truth
# Measure accuracy on 20-30 pieces
```

**Step 1.4: CSV Validation**
```python
# Test: Validate extracted IDs against CSV
csv_elements = load_csv("set_inventory.csv")
for element_id in extracted_ids:
    assert element_id in csv_elements
```

**Step 1.5: Template Creation**
```python
# Test: Can we create usable templates?
template = builder.create_template(piece_image)
assert template.ndim == 2  # Grayscale
assert template.dtype == np.uint8
```

**Testing Strategy:**
- Manual annotation of reference page (ground truth)
- Compare automated extraction to ground truth
- Target: >95% extraction accuracy

**Deliverables:**
- Working reference library builder
- ReferenceLibrary data structure
- Unit tests for each component
- Sample reference library from test PDF

**Success Criteria:**
- Extract all pieces from reference pages
- OCR accuracy >95% for element_ids
- All element_ids validate against CSV
- Processing time <30 seconds for reference pages

---

### Phase 2: Bag Boundary Detection (3-5 days)

**Objectives:**
- Detect bag opening symbols in instruction pages
- Extract bag numbers
- Map page ranges to bags
- Handle edge cases

**Components to Build:**
- `bag_detector.py`: Complete implementation

**Development Sequence:**

**Step 2.1: Bag Symbol Template Creation**
```python
# Manually crop bag symbol from sample page
# Create reference template for matching
bag_symbol = load_image("bag_symbol_template.png")
```

**Step 2.2: Template Matching for Symbols**
```python
# Test on pages known to have bag indicators
detector = BagDetector(bag_symbol_template)
indicator = detector.find_bag_symbol(test_page)
assert indicator is not None
assert indicator.confidence > 0.8
```

**Step 2.3: OCR Bag Numbers**
```python
# Extract number from indicator region
bag_number = detector.extract_bag_number(indicator_region)
assert isinstance(bag_number, int)
assert 1 <= bag_number <= 20  # Reasonable range
```

**Step 2.4: Full Manual Scan**
```python
# Process entire manual
boundaries = detector.detect_bag_boundaries(all_pages)
# Manually verify page ranges are correct
```

**Testing Strategy:**
- Manual annotation of bag boundaries in sample PDF
- Compare automated detection to ground truth
- Test on 2-3 different sets to verify generalization

**Deliverables:**
- Working bag detector
- BagBoundaries data structure
- Unit tests
- Documentation of bag symbol variations

**Success Criteria:**
- Detect all bag boundaries correctly
- Page ranges accurate
- Processing time <1 minute for full manual
- Handle edge cases (missing bag 1 indicator, etc.)

---

### Phase 3: Piece Extraction from Required Boxes (1 week)

**Objectives:**
- Locate required pieces boxes on instruction pages
- Extract individual piece images
- OCR quantities
- Normalize images for matching

**Components to Build:**
- `piece_extractor.py`: Complete implementation
- `image_processing.py`: Utility functions

**Development Sequence:**

**Step 3.1: Required Box Detection**
```python
# Test: Can we find the blue boxes?
extractor = PieceExtractor()
boxes = extractor.locate_required_boxes(instruction_page)
# Manually verify all boxes found
```

**Step 3.2: Piece Segmentation Within Box**
```python
# Test: Can we separate individual pieces?
pieces = extractor.extract_pieces_from_box(box_region)
# Visual inspection: Are all pieces separated?
```

**Step 3.3: Quantity OCR**
```python
# Test: Can we read "2x", "4x", etc.?
quantity = extractor.extract_quantity(piece_region)
# Compare to manual ground truth
```

**Step 3.4: Image Normalization**
```python
# Test: Normalize piece images consistently
normalized = extractor.normalize_piece_image(raw_piece_image)
assert normalized.shape[2] == 1  # Grayscale
# Visual check: Background removed, piece centered
```

**Step 3.5: Full Page Processing**
```python
# Test: Process complete instruction page
required_pieces = extractor.extract_from_page(page_image)
# Verify piece count and quantities
```

**Testing Strategy:**
- Manually annotate 5-10 instruction pages
- Compare automated extraction to ground truth
- Measure precision and recall

**Deliverables:**
- Working piece extractor
- Comprehensive image processing utilities
- Unit tests
- Sample extraction results from test pages

**Success Criteria:**
- Detect >95% of required pieces boxes
- Extract >90% of pieces correctly
- Quantity OCR >95% accurate
- Processing time <5 seconds per page

---

### Phase 4: Template Matching Engine (1 week)

**Objectives:**
- Implement robust template matching
- Handle scale variations
- Implement caching
- Confidence scoring

**Components to Build:**
- `piece_matcher.py`: Complete implementation
- `match_result.py`: Data model

**Development Sequence:**

**Step 4.1: Single-Scale Matching**
```python
# Test: Basic template matching
matcher = PieceMatcher(reference_library)
result = matcher.match_piece(test_piece_image)
assert result.confidence_score > 0.8
```

**Step 4.2: Multi-Scale Matching**
```python
# Test: Matching with scale variation
# Create test images at different scales
scaled_piece = cv2.resize(original, None, fx=1.2, fy=1.2)
result = matcher.match_piece(scaled_piece)
# Should still match correctly
```

**Step 4.3: Caching Implementation**
```python
# Test: Cache performance
result1 = matcher.match_piece(test_image)  # Cache miss
result2 = matcher.match_piece(test_image)  # Cache hit
assert result1.element_id == result2.element_id
assert result2.processing_time_ms < result1.processing_time_ms * 0.1
```

**Step 4.4: Confidence Evaluation**
```python
# Test: Confidence scoring logic
# Test with clear matches, ambiguous matches, non-matches
```

**Step 4.5: Full Reference Library Matching**
```python
# Test: Match all pieces from one bag
# Measure accuracy and performance
```

**Testing Strategy:**
- Create test set of 50 piece images with known element_ids
- Test matching accuracy
- Test with scale variations (±20%)
- Measure cache hit rate on repeated pieces

**Deliverables:**
- Working template matcher with multi-scale support
- Caching system
- MatchResult data structure
- Performance benchmarks

**Success Criteria:**
- Matching accuracy >95%
- Handle scale variations ±20%
- Cache hit rate >80% on typical sets
- Processing time <50ms per piece (first match), <5ms (cached)

---

### Phase 5: Validation & Reporting (3-5 days)

**Objectives:**
- Implement quantity reconciliation
- Confidence analysis
- Report generation
- Manual review queue

**Components to Build:**
- `validator.py`: Complete implementation
- Report templates

**Development Sequence:**

**Step 5.1: Quantity Validation**
```python
# Test: Compare bag totals to expected inventory
validator = Validator("set_inventory.csv")
report = validator.validate_bag_inventory(bag_inventory)
# Check discrepancies are accurately identified
```

**Step 5.2: Confidence Analysis**
```python
# Test: Aggregate confidence scores
confidence_report = validator.analyze_confidence(bag_inventory)
# Verify percentages sum to 100%
```

**Step 5.3: Report Generation**
```python
# Test: Generate human-readable report
report_text = validator.generate_report(validation_results)
# Manually review report clarity
```

**Step 5.4: Manual Review Queue**
```python
# Test: Identify pieces needing review
review_queue = validator.generate_review_queue(bag_inventory)
# Verify low-confidence pieces are flagged
```

**Testing Strategy:**
- Create test bag inventory with known discrepancies
- Verify all discrepancies detected
- Test report formatting

**Deliverables:**
- Working validator
- Report templates
- ValidationReport data structure
- Manual review queue system

**Success Criteria:**
- Detect all quantity discrepancies
- Accurate confidence analysis
- Clear, actionable reports
- Processing time <10 seconds

---

### Phase 6: Integration & End-to-End Testing (1 week)

**Objectives:**
- Connect all components
- Create main orchestrator
- End-to-end testing
- Performance optimization

**Components to Build:**
- `main.py`: Orchestrator script
- Integration tests

**Development Sequence:**

**Step 6.1: Main Orchestrator**
```python
def process_set(pdf_path, csv_path, output_dir):
    # 1. Initialize
    # 2. Build reference library
    # 3. Detect bag boundaries
    # 4. Extract and match pieces
    # 5. Validate
    # 6. Generate outputs
    pass
```

**Step 6.2: End-to-End Test**
```python
# Test: Process complete sample set
output = process_set(
    "sample_manual.pdf",
    "sample_inventory.csv",
    "output/"
)
# Manually verify all outputs correct
```

**Step 6.3: Performance Profiling**
```python
# Identify bottlenecks
# Optimize slow components
# Target: <5 minutes per set
```

**Step 6.4: Error Handling**
```python
# Test: Graceful handling of edge cases
# - Missing pages
# - OCR failures
# - No bag indicators
# - Template match failures
```

**Testing Strategy:**
- Process 3-5 complete sets end-to-end
- Verify outputs against manual annotation
- Measure accuracy and performance

**Deliverables:**
- Working end-to-end pipeline
- Main orchestrator script
- Integration tests
- Performance benchmarks

**Success Criteria:**
- Process complete set successfully
- Overall accuracy >90%
- Total processing time <5 minutes per set
- Graceful error handling

---

### Phase 7: User Interface & Tooling (1 week)

**Objectives:**
- Create command-line interface
- Manual review tool
- Batch processing script
- Documentation

**Components to Build:**
- CLI with argparse
- Review GUI (optional)
- Batch processor
- User documentation

**Development Sequence:**

**Step 7.1: CLI Development**
```bash
# Usage:
python lego_bag_tool.py \
    --pdf "sets/31129_instructions.pdf" \
    --csv "sets/31129_inventory.csv" \
    --output "output/31129/" \
    --verbose
```

**Step 7.2: Manual Review Tool**
```python
# Simple GUI or web interface
# Display low-confidence matches
# Allow user to correct element_id
# Update output files
```

**Step 7.3: Batch Processing**
```bash
# Process multiple sets
python batch_process.py \
    --input_dir "sets/" \
    --output_dir "output/" \
    --parallel 4
```

**Step 7.4: Documentation**
- User guide
- Installation instructions
- Troubleshooting guide
- API documentation

**Deliverables:**
- User-friendly CLI
- Optional manual review tool
- Batch processing capability
- Complete documentation

**Success Criteria:**
- Intuitive command-line interface
- Clear error messages
- Easy manual review process
- Comprehensive documentation

---

## Testing Strategy

### Unit Testing
- Test each module independently
- Mock dependencies
- Target: >80% code coverage

**Key Test Cases:**
```python
# reference_builder_test.py
def test_detect_piece_regions():
    # Given a reference page
    # When detecting regions
    # Then all pieces are found

def test_ocr_element_id():
    # Given a cropped piece region
    # When OCR is performed
    # Then correct element_id is extracted

# piece_matcher_test.py
def test_template_matching():
    # Given a piece image and reference library
    # When matching
    # Then correct element_id is returned

def test_multi_scale_matching():
    # Given scaled piece images
    # When matching
    # Then matches are scale-invariant

# validator_test.py
def test_quantity_validation():
    # Given bag inventory and total inventory
    # When validating
    # Then discrepancies are identified
```

### Integration Testing
- Test component interactions
- Use realistic data
- Verify end-to-end flow

**Key Integration Tests:**
```python
def test_reference_to_matching():
    # Build reference library
    # Extract piece from instruction page
    # Match piece to reference
    # Verify correct identification

def test_full_bag_processing():
    # Process one complete bag
    # Verify all pieces identified
    # Check quantities sum correctly
```

### End-to-End Testing
- Process complete sample sets
- Compare to manual ground truth
- Measure accuracy metrics

**Metrics:**
- Piece identification accuracy
- Quantity extraction accuracy
- Processing time per set
- Cache hit rate

### Manual Testing
- Visual inspection of intermediate results
- Review of validation reports
- Testing edge cases not covered by unit tests

---

## Technical Specifications

### Performance Targets

| Operation | Target Time |
|-----------|-------------|
| Reference library building | <30 seconds |
| Bag boundary detection | <1 minute |
| Single piece extraction | <1 second |
| Single piece matching (first) | <50ms |
| Single piece matching (cached) | <5ms |
| Full page processing | <10 seconds |
| Complete set processing | <5 minutes |
| Validation | <10 seconds |

### Accuracy Targets

| Metric | Target |
|--------|--------|
| Element ID OCR accuracy | >95% |
| Piece identification accuracy | >95% |
| Quantity OCR accuracy | >95% |
| Overall bag inventory accuracy | >90% |
| Bag boundary detection | >98% |

### Resource Requirements

**CPU:** Modern multi-core processor (4+ cores recommended)  
**RAM:** 8GB minimum, 16GB recommended  
**Storage:** ~500MB per set (intermediate images)  
**GPU:** Not required (CPU-based processing)

### Configuration Parameters

```python
# config/settings.py

PDF_PROCESSING:
    DPI = 300  # Resolution for PDF rendering
    COLOR_MODE = "RGB"
    CACHE_PAGES = True

REFERENCE_EXTRACTION:
    REFERENCE_PAGE_COUNT = 3  # Last N pages to check
    MIN_PIECE_SIZE = 1000  # pixels
    MAX_PIECE_SIZE = 50000  # pixels
    OCR_ENGINE = "tesseract"
    OCR_LANG = "eng"

BAG_DETECTION:
    TEMPLATE_MATCH_THRESHOLD = 0.7
    SYMBOL_SEARCH_MARGIN = 100  # pixels from page edge

PIECE_EXTRACTION:
    BLUE_BOX_HUE_RANGE = (100, 140)  # HSV hue for blue
    MIN_BOX_SIZE = 10000  # pixels
    QUANTITY_OCR_PATTERNS = [r'(\d+)x', r'x(\d+)']

TEMPLATE_MATCHING:
    SCALES = [0.8, 0.9, 1.0, 1.1, 1.2]
    MATCH_METHOD = cv2.TM_CCOEFF_NORMED
    HIGH_CONFIDENCE_THRESHOLD = 0.90
    MEDIUM_CONFIDENCE_THRESHOLD = 0.80
    LOW_CONFIDENCE_THRESHOLD = 0.70
    ENABLE_CACHING = True

VALIDATION:
    ALLOW_SPARE_PARTS = True
    SPARE_DETECTION_THRESHOLD = 0.95  # CSV line ratio
```

---

## Future Enhancements

### Phase 8+: Advanced Features

**1. Machine Learning Integration**
- Train CNN for piece classification (alternative to template matching)
- Learn to detect bag symbols automatically (eliminate manual template)
- Predict likely discrepancies based on patterns

**2. Color Recognition**
- Add color validation for ambiguous shape matches
- Build color palette from reference pages
- Support design_id → color_id → element_id mapping

**3. Multi-Language Support**
- OCR support for non-English manuals
- Internationalized bag symbols
- Localized report generation

**4. Web Interface**
- Upload PDF and CSV via browser
- Real-time processing status
- Interactive manual review
- Download results

**5. Database Integration**
- Store processed bag inventories in database
- Build searchable bag inventory repository
- API for LEGO Sorting Machine integration

**6. Advanced Validation**
- Cross-reference with community databases (Rebrickable)
- Statistical anomaly detection
- Suggest likely corrections for discrepancies

**7. Performance Optimization**
- GPU acceleration for template matching
- Parallel processing of pages
- Incremental processing (resume interrupted runs)

**8. Sorting Machine Integration**
- Direct export to sorting machine database format
- Add design_id and color_id lookup
- Support sorting profiles based on bag inventory

---

## Appendix

### Glossary

**element_id:** Unique identifier for a specific LEGO piece (shape + color combination)  
**design_id:** Identifier for a LEGO piece shape (independent of color)  
**color_id:** Two-digit code representing a LEGO color  
**Reference Library:** Collection of piece images and element_ids extracted from manual  
**Template Matching:** Computer vision technique to find similar images  
**Required Pieces Box:** Visual indicator in instructions showing pieces needed for a step  
**Bag Boundary:** Page range corresponding to a numbered bag  

### File Formats

**Input CSV Format:**
```csv
Quantity,ElementID,Name
14,36840,Black Bracket 1 x 1 - 1 x 1 Inverted
4,28802,Black Bracket 1 x 2 - 1 x 4 with Rounded Corners
```

**Output JSON Format:**
```json
{
  "set_number": "31129",
  "bags": [
    {
      "bag_number": 1,
      "pieces": [
        {"element_id": "3034", "quantity": 1, "confidence": "HIGH"}
      ]
    }
  ]
}
```

### References

- OpenCV Documentation: https://docs.opencv.org/
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- BrickLink API: https://www.bricklink.com/v3/api.page
- Rebrickable: https://rebrickable.com/

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-11-07 | Matthew & Claude | Initial design document |

---

## Approval

This design document serves as the blueprint for the LEGO Bag-Level Inventory Extraction Tool. Implementation will follow the phased approach outlined, with each phase requiring successful completion and testing before proceeding to the next.

**Next Steps:**
1. Review and approve design document
2. Set up development environment (Phase 0)
3. Begin Phase 1: Reference Library Extraction
