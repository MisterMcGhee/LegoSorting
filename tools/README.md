# Tools Directory

This directory contains auxiliary tools that support the main LEGO Sorting System but operate independently.

## Design Philosophy

Tools in this directory are:
- **Standalone**: Can run without the main sorting system
- **Independent**: Separate from core sorting pipeline
- **Auxiliary**: Support the system but aren't required for operation
- **Modular**: Easy to maintain and extend separately

## Available Tools

### Categorization Tool
**Location**: `tools/categorization_tool/`

Tool for categorizing LEGO pieces that have been identified by the Brickognize API but are not yet in the Lego_Categories.csv database.

**Launch**:
```bash
python tools/categorization_tool/categorization_launcher.py
```

**Features**:
- Load unknown pieces from `unknown_pieces.csv`
- Display piece information (ID, name, encounters, dates)
- Tiered dropdown categories (Primary/Secondary/Tertiary)
- Inline text entry for adding new categories
- Auto-reload category hierarchy
- Immediate save to database
- Skip difficult pieces
- Progress tracking

**Structure**:
```
categorization_tool/
├── __init__.py
├── categorization_logic.py      # Business logic (no GUI)
├── categorization_gui.py         # PyQt5 GUI components
└── categorization_launcher.py   # Entry point (run this)
```

## Adding New Tools

When creating a new tool, follow this structure:

```
tools/
└── your_tool/
    ├── __init__.py
    ├── your_tool_logic.py      # Business logic
    ├── your_tool_gui.py        # GUI components (if applicable)
    └── your_tool_launcher.py   # Entry point
```

**Guidelines**:
1. **Separation of Concerns**: Keep business logic separate from GUI
2. **Launcher Pattern**: Always use a launcher as the entry point
3. **Independent**: Don't depend on main sorting system modules unless necessary
4. **Documentation**: Include comprehensive docstrings
5. **Testing**: Add test functions in logic modules

## Tool Architecture

### Three-Layer Pattern

Each tool should follow this architecture:

**1. Logic Layer** (`*_logic.py`)
- Pure Python business logic
- NO GUI dependencies
- Handles data operations
- Can be tested independently
- Example: CSV operations, validation, data models

**2. GUI Layer** (`*_gui.py`)
- PyQt5 GUI components only
- Delegates all business logic to logic layer
- Handles user interaction
- Example: Windows, widgets, layouts

**3. Launcher Layer** (`*_launcher.py`)
- Entry point for the tool
- Initialization and setup
- Dependency checking
- Error handling
- Can add CLI arguments

### Benefits

✅ **Clean Separation**: Logic can be tested without GUI  
✅ **Maintainability**: Changes to GUI don't affect logic  
✅ **Reusability**: Logic layer can be used in different contexts  
✅ **Professional**: Industry-standard architecture  
✅ **Future-Proof**: Easy to integrate into unified launcher  

## Integration with Main System

Tools are designed to be:
- **Separate executables** during development
- **Easily integrated** into a unified launcher later

### Current (Development):
```bash
# Main system
python main_sorting_system.py

# Categorization tool
python tools/categorization_tool/categorization_launcher.py
```

### Future (Production):
```bash
# Unified launcher with menu
python lego_sorting_launcher.py

# Or still run separately
python tools/categorization_tool/categorization_launcher.py
```

## Dependencies

Tools may depend on:
- `enhanced_config_manager`: Configuration management
- `processing/category_hierarchy_service`: Category data
- PyQt5: GUI framework

Tools should NOT depend on:
- Camera module
- Arduino module
- Main sorting pipeline
- Queue manager

## Running Tools

### From Project Root:
```bash
python tools/categorization_tool/categorization_launcher.py
```

### As Module:
```bash
python -m tools.categorization_tool.categorization_launcher
```

### With Arguments (if supported):
```bash
python tools/categorization_tool/categorization_launcher.py --debug
```

## Development Guidelines

### Creating a New Tool

1. **Create directory**: `tools/new_tool/`
2. **Add __init__.py**: Package initialization
3. **Create logic layer**: `new_tool_logic.py`
4. **Create GUI layer**: `new_tool_gui.py` (if needed)
5. **Create launcher**: `new_tool_launcher.py`
6. **Document**: Add to this README
7. **Test**: Add test functions to logic layer

### File Naming Convention

- `*_logic.py`: Business logic and data operations
- `*_gui.py`: GUI components (PyQt5)
- `*_launcher.py`: Entry point script

### Coding Standards

- **Docstrings**: Required for all classes and functions
- **Type hints**: Encouraged for public APIs
- **Logging**: Use Python logging module
- **Error handling**: Graceful degradation
- **Comments**: Explain why, not what

## Example Tools (Future)

Potential tools to add:

- **Database Viewer**: Browse and edit Lego_Categories.csv
- **Calibration Tool**: Camera and detection calibration
- **Statistics Dashboard**: View sorting statistics
- **Export Tool**: Export data to various formats
- **Backup Manager**: Backup and restore configuration
- **Test Suite**: Run system tests and diagnostics

## File Structure Overview

```
tools/
├── __init__.py                              # Package init
├── README.md                                # This file
│
└── categorization_tool/                     # Piece categorization tool
    ├── __init__.py
    ├── categorization_logic.py              # Business logic
    ├── categorization_gui.py                # PyQt5 GUI
    └── categorization_launcher.py           # Entry point
```

## Support

For issues or questions about tools:
1. Check tool-specific documentation
2. Review logs (set --debug flag)
3. Verify dependencies are installed
4. Ensure required files exist

---

**Version**: 1.0.0  
**Last Updated**: 2024-10-23  
**Maintainer**: LEGO Sorting Team
