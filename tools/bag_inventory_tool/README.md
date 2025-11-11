# LEGO Bag-Level Inventory Extraction Tool

## Overview

This tool extracts bag-level piece inventories from LEGO instruction manuals by scraping reference pages and validating against set inventory CSVs.

**Current Implementation: Phase 0 + 1 (Reference Library Extraction)**

### What It Does

1. Loads your LEGO instruction manual PDF
2. Prompts you to identify which pages contain the reference inventory
3. Extracts individual piece images from those pages
4. Uses OCR to read element IDs
5. Validates against your inventory CSV (Quantity, ElementID, Name)
6. Creates a reference library with templates for future matching
7. Generates a detailed extraction report

## Installation

### Prerequisites

- Python 3.8 or higher
- macOS, Linux, or Windows

### Step 1: Install System Dependencies

**macOS:**
```bash
# Install Python if needed
brew install python3
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip
```

### Step 2: Create Virtual Environment

```bash
cd tools/bag_inventory_tool

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** The first time EasyOCR runs, it will download language models (~100MB). This is normal.

## Usage

### Basic Usage

```bash
python main.py \
  --pdf "path/to/instructions.pdf" \
  --csv "path/to/inventory.csv" \
  --output "output/"
```

### With Set Information

```bash
python main.py \
  --pdf "31129_instructions.pdf" \
  --csv "31129_inventory.csv" \
  --set-number "31129" \
  --set-name "Majestic Tiger" \
  --output "output/31129/" \
  --verbose
```

### Command-Line Options

| Option | Required | Description |
|--------|----------|-------------|
| `--pdf` | Yes | Path to instruction manual PDF |
| `--csv` | Yes | Path to set inventory CSV |
| `--output` | No | Output directory (default: `output`) |
| `--set-number` | No | LEGO set number (e.g., "31129") |
| `--set-name` | No | LEGO set name (e.g., "Majestic Tiger") |
| `--verbose` / `-v` | No | Enable verbose logging |
| `--save-debug` | No | Save debug visualizations |

## CSV Format

Your inventory CSV must have these columns (tab or comma-separated):

```
Quantity	ElementID	Name
1	87994	Black Bar 3L (Bar Arrow)
6	87747	Black Barb / Claw / Horn / Tooth - Medium
4	53451	Black Barb / Claw / Horn / Tooth - Small
2	3062	Black Brick, Round 1 x 1
```

- **Quantity**: Number of pieces
- **ElementID**: Unique element identifier (can include letters, e.g., "4081b")
- **Name**: Human-readable piece name

## Interactive Prompts

When you run the tool, it will prompt you to identify reference pages:

```
==============================================================
PDF has 156 total pages
==============================================================

The reference inventory is typically on the last 1-3 pages of the manual.
It shows all pieces with their element IDs.

Which page(s) contain the reference inventory? (e.g., '48' or '47,48,49'): 155,156

✓ Will process page(s): 155, 156
Is this correct? (y/n): y
```

**Tips for Finding Reference Pages:**
- Usually the last 1-3 pages of the manual
- Shows all pieces in a grid or column layout
- Each piece has an element ID beneath it
- Often has a title like "Bill of Materials" or piece inventory

## Output

The tool creates the following output structure:

```
output/
├── reference_library/
│   ├── templates/           # Grayscale templates (for future matching)
│   │   ├── 87994.png
│   │   ├── 87747.png
│   │   └── ...
│   ├── originals/          # Original color images
│   │   ├── 87994.png
│   │   ├── 87747.png
│   │   └── ...
│   └── library.json        # Metadata and validation status
├── extraction_report.txt   # Detailed extraction report
└── bag_inventory_tool.log  # Processing log
```

### Extraction Report

The report includes:
- **Extraction Statistics**: Number of pieces extracted, validation rate
- **Coverage Analysis**: Which pieces from CSV were found/missing
- **Validation Failures**: Pieces that didn't match CSV
- **OCR Confidence Scores**: Quality metrics

Example report section:
```
EXTRACTION STATISTICS
----------------------------------------------------------------------
Reference Pages: 155, 156
Total Pieces Extracted: 148
Validated Against CSV: 142 (95.9%)
Failed Validation: 6
Average OCR Confidence: 0.87

COVERAGE ANALYSIS
----------------------------------------------------------------------
Pieces in CSV but NOT extracted: 2
Pieces extracted but NOT in CSV: 0
```

## Configuration

The tool creates a `bag_inventory_config.json` file with default settings. You can modify:

### PDF Settings
```json
"pdf": {
  "dpi": 300,              // Higher = better quality, slower
  "color_mode": "RGB",
  "cache_enabled": true
}
```

### Reference Extraction
```json
"reference": {
  "min_piece_area": 500,   // Minimum piece size (pixels)
  "max_piece_area": 50000, // Maximum piece size
  "binary_threshold": 200  // Adjust for different manual styles
}
```

### OCR Settings
```json
"ocr": {
  "confidence_threshold": 0.3,  // Lower = more lenient
  "gpu": false,                 // Set true if you have CUDA GPU
  "languages": ["en"]
}
```

## Troubleshooting

### Issue: "EasyOCR not installed"
**Solution:** Ensure you've installed requirements: `pip install -r requirements.txt`

### Issue: "Failed to extract reference pages"
**Solution:**
- Check PDF file path is correct
- Ensure PDF is not password-protected
- Try a lower DPI in config (e.g., 200)

### Issue: Low validation rate (<80%)
**Possible causes:**
1. Wrong reference pages selected
2. Poor PDF scan quality
3. OCR threshold too high

**Solutions:**
- Double-check you selected the correct pages
- Lower `confidence_threshold` in OCR config
- Try `--save-debug` to see what's being detected
- Adjust `binary_threshold` in reference config

### Issue: "No valid element ID found" for many pieces
**Solution:**
- Element IDs might be in unusual positions
- Try adjusting `search_margin` in code (default: 100 pixels)
- Check if manual uses non-standard layout

### Issue: Out of memory
**Solution:**
- Lower PDF DPI to 200 or 150
- Process one reference page at a time
- Disable template caching in config

## Validation Tips

To ensure accurate extraction:

1. **Check the Report**: Look at validation rate and coverage analysis
2. **Review Debug Images**: Use `--save-debug` to see detected pieces
3. **Verify Element IDs**: Spot-check a few extracted templates against originals
4. **Compare Totals**: Number extracted should roughly match CSV unique pieces

**Good extraction:** 95%+ validation rate, <5% missing from CSV

**Needs review:** <90% validation rate, >10% missing

## Next Steps (Future Phases)

This Phase 0+1 implementation validates the core extraction. Future phases will add:

- **Phase 2**: Bag boundary detection (finding where bags start/end)
- **Phase 3**: Piece extraction from instruction pages
- **Phase 4**: Template matching to identify pieces
- **Phase 5**: Validation and reporting
- **Phase 6**: End-to-end integration

## Project Structure

```
bag_inventory_tool/
├── config/              # Configuration management
├── core/                # Core processing (PDF, reference builder)
├── utils/               # Utilities (CSV, OCR, image processing)
├── models/              # Data models
├── tests/               # Test suite
├── main.py             # CLI entry point
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Development

### Running Tests
```bash
pytest tests/
```

### Adding Debug Logging
```bash
python main.py --pdf ... --csv ... --verbose
```

### Configuration Style

This tool follows the same configuration pattern as `enhanced_config_manager.py` in the parent project but remains standalone for now.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the extraction report for specific errors
3. Enable `--verbose` logging for detailed output
4. Use `--save-debug` to see intermediate results

## License

Part of the LEGO Sorting Machine project.
