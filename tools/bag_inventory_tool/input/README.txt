LEGO BAG INVENTORY TOOL - INPUT FILES
======================================

Place your input files in this directory:

1. INSTRUCTION MANUAL PDF
   - The LEGO instruction booklet for your set (PDF format)
   - Example: 31129_instructions.pdf
   - The program will ask you which pages contain the reference inventory
     (usually the last 1-3 pages showing all pieces)

2. SET INVENTORY CSV
   - A CSV file listing all pieces in the set
   - Required columns: element_id, design_id, part_name, quantity
   - Example: 31129_inventory.csv

EXAMPLE STRUCTURE:
------------------
input/
  ├── 31129_instructions.pdf
  └── 31129_inventory.csv

HOW TO GET THESE FILES:
-----------------------
- PDF: Download from LEGO's website or scan your physical manual
- CSV: Export from BrickLink, Rebrickable, or create manually

After placing files here, configure PyCharm to point to them in the
Run Configuration parameters (see setup instructions).
