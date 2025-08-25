"""
test_enhanced_config_manager.py - Comprehensive test suite

This test suite validates that the enhanced_config_manager does everything it needs to do.
Run with: python -m pytest test_enhanced_config_manager.py -v

WHAT TESTS DO:
- Verify that all expected functionality works correctly
- Catch bugs before they reach your main application
- Ensure changes don't break existing functionality
- Validate edge cases and error conditions
"""

import pytest
import json
import os
import tempfile
import threading
import time
from unittest.mock import patch, mock_open
from enhanced_config_manager import (
    EnhancedConfigManager,
    ConfigSchema,
    ConfigValidator,
    ModuleConfig,
    create_config_manager
)


class TestConfigSchema:
    """Test the configuration schema and defaults"""

    def test_all_modules_have_schemas(self):
        """Verify every module in ModuleConfig has a schema definition"""
        for module in ModuleConfig:
            schema = ConfigSchema.get_module_schema(module.value)
            assert isinstance(schema, dict), f"Schema for {module.value} should be a dict"
            assert len(schema) > 0, f"Schema for {module.value} should not be empty"

    def test_camera_schema_completeness(self):
        """Verify camera schema has all expected fields with correct types"""
        schema = ConfigSchema.get_module_schema(ModuleConfig.CAMERA.value)

        # Check required fields exist
        assert "device_id" in schema
        assert "width" in schema
        assert "height" in schema
        assert "fps" in schema

        # Check types are correct
        assert isinstance(schema["device_id"], int)
        assert isinstance(schema["width"], int)
        assert isinstance(schema["height"], int)
        assert isinstance(schema["fps"], int)
        assert isinstance(schema["auto_exposure"], bool)

    def test_sorting_schema_completeness(self):
        """Verify sorting schema has all expected fields"""
        schema = ConfigSchema.get_module_schema(ModuleConfig.SORTING.value)

        expected_fields = [
            "strategy", "target_primary_category", "target_secondary_category",
            "max_bins", "overflow_bin", "confidence_threshold", "pre_assignments"
        ]

        for field in expected_fields:
            assert field in schema, f"Sorting schema missing field: {field}"

    def test_required_fields_defined(self):
        """Verify required fields are properly defined for critical modules"""
        camera_required = ConfigSchema.get_required_fields(ModuleConfig.CAMERA.value)
        assert "device_id" in camera_required

        sorting_required = ConfigSchema.get_required_fields(ModuleConfig.SORTING.value)
        assert "strategy" in sorting_required
        assert "max_bins" in sorting_required


class TestConfigValidator:
    """Test configuration validation logic"""

    def test_valid_camera_config_passes(self):
        """Valid camera configuration should pass validation"""
        valid_config = {
            "device_id": 0,
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "auto_exposure": True
        }

        is_valid, errors = ConfigValidator.validate_module(ModuleConfig.CAMERA.value, valid_config)
        assert is_valid, f"Valid camera config should pass, got errors: {errors}"
        assert len(errors) == 0

    def test_invalid_camera_config_fails(self):
        """Invalid camera configuration should fail validation"""
        invalid_config = {
            "device_id": -1,  # Invalid: negative device ID
            "fps": 200,  # Invalid: fps too high
            "width": 1920,
            "height": 1080
        }

        is_valid, errors = ConfigValidator.validate_module(ModuleConfig.CAMERA.value, invalid_config)
        assert not is_valid, "Invalid camera config should fail validation"
        assert len(errors) > 0

        # Check specific error messages
        error_text = " ".join(errors)
        assert "device_id" in error_text
        assert "fps" in error_text

    def test_missing_required_fields_fails(self):
        """Configuration missing required fields should fail"""
        incomplete_config = {
            "width": 1920,
            "height": 1080
            # Missing required "device_id"
        }

        is_valid, errors = ConfigValidator.validate_module(ModuleConfig.CAMERA.value, incomplete_config)
        assert not is_valid
        assert any("device_id" in error for error in errors)

    def test_sorting_strategy_validation(self):
        """Sorting strategy validation should work correctly"""
        # Valid primary strategy
        valid_primary = {
            "strategy": "primary",
            "max_bins": 5,
            "overflow_bin": 0,
            "confidence_threshold": 0.7,
            "pre_assignments": {}
        }
        is_valid, errors = ConfigValidator.validate_module(ModuleConfig.SORTING.value, valid_primary)
        assert is_valid, f"Valid primary strategy should pass: {errors}"

        # Invalid strategy
        invalid_strategy = {
            "strategy": "invalid_strategy",
            "max_bins": 5,
            "overflow_bin": 0
        }
        is_valid, errors = ConfigValidator.validate_module(ModuleConfig.SORTING.value, invalid_strategy)
        assert not is_valid
        assert any("Invalid sorting strategy" in error for error in errors)

    def test_secondary_sorting_requires_primary_category(self):
        """Secondary sorting should require target_primary_category"""
        secondary_without_target = {
            "strategy": "secondary",
            "max_bins": 5,
            "overflow_bin": 0,
            "target_primary_category": "",  # Empty string = missing
            "confidence_threshold": 0.7
        }

        is_valid, errors = ConfigValidator.validate_module(ModuleConfig.SORTING.value, secondary_without_target)
        assert not is_valid
        assert any("target_primary_category" in error for error in errors)


class TestEnhancedConfigManager:
    """Test the main configuration manager functionality"""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing"""
        # Create a temporary file
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)

        yield path  # This is what the test function receives

        # Cleanup after test
        if os.path.exists(path):
            os.remove(path)

    @pytest.fixture
    def sample_config_data(self):
        """Sample configuration data for testing"""
        return {
            "camera": {
                "device_id": 1,
                "width": 1280,
                "height": 720,
                "fps": 15
            },
            "sorting": {
                "strategy": "secondary",
                "target_primary_category": "Basic",
                "max_bins": 7
            }
        }

    def test_create_config_manager(self, temp_config_file):
        """Test creating a config manager instance"""
        manager = create_config_manager(temp_config_file)
        assert isinstance(manager, EnhancedConfigManager)
        assert manager.config_path == temp_config_file

    def test_creates_default_config_when_file_missing(self, temp_config_file):
        """Should create default config when file doesn't exist"""
        # Remove the temp file so it doesn't exist
        os.remove(temp_config_file)

        manager = EnhancedConfigManager(temp_config_file)

        # Should have created the file with defaults
        assert os.path.exists(temp_config_file)

        # Should have all modules configured
        for module in ModuleConfig:
            config = manager.get_module_config(module.value)
            assert isinstance(config, dict)
            assert len(config) > 0

    def test_loads_existing_config_file(self, temp_config_file, sample_config_data):
        """Should load and use existing configuration file"""
        # Write sample config to file
        with open(temp_config_file, 'w') as f:
            json.dump(sample_config_data, f)

        manager = EnhancedConfigManager(temp_config_file)

        # Should use values from file
        camera_config = manager.get_module_config(ModuleConfig.CAMERA.value)
        assert camera_config["device_id"] == 1  # From file
        assert camera_config["width"] == 1280  # From file
        assert camera_config["auto_exposure"] == True  # From defaults (not in file)

    def test_get_module_config_returns_complete_config(self, temp_config_file):
        """get_module_config should return complete config with all defaults"""
        manager = EnhancedConfigManager(temp_config_file)

        camera_config = manager.get_module_config(ModuleConfig.CAMERA.value)

        # Should have all expected fields from schema
        schema = ConfigSchema.get_module_schema(ModuleConfig.CAMERA.value)
        for field in schema:
            assert field in camera_config, f"Missing field {field} in returned config"

    def test_update_module_config_validation(self, temp_config_file):
        """update_module_config should validate changes"""
        manager = EnhancedConfigManager(temp_config_file)

        # Valid update should succeed
        valid_update = {"device_id": 2, "fps": 25}
        result = manager.update_module_config(ModuleConfig.CAMERA.value, valid_update)
        assert result == True

        # Check the update was applied
        camera_config = manager.get_module_config(ModuleConfig.CAMERA.value)
        assert camera_config["device_id"] == 2
        assert camera_config["fps"] == 25

        # Invalid update should fail
        invalid_update = {"device_id": -5, "fps": 300}
        result = manager.update_module_config(ModuleConfig.CAMERA.value, invalid_update, validate=True)
        assert result == False

    def test_save_and_load_roundtrip(self, temp_config_file):
        """Configuration should survive save/load roundtrip"""
        # Create manager and make changes
        manager1 = EnhancedConfigManager(temp_config_file)
        manager1.update_module_config(ModuleConfig.CAMERA.value, {"device_id": 99})

        # Create new manager from same file
        manager2 = EnhancedConfigManager(temp_config_file)

        # Should have the saved changes
        camera_config = manager2.get_module_config(ModuleConfig.CAMERA.value)
        assert camera_config["device_id"] == 99

    def test_validation_report(self, temp_config_file):
        """get_validation_report should provide comprehensive validation info"""
        manager = EnhancedConfigManager(temp_config_file)

        report = manager.get_validation_report()

        assert "valid" in report
        assert "modules" in report
        assert "timestamp" in report

        # Should have report for each module
        for module in ModuleConfig:
            assert module.value in report["modules"]
            module_report = report["modules"][module.value]
            assert "valid" in module_report
            assert "errors" in module_report


class TestCategoryParsing:
    """Test CSV category parsing functionality"""

    @pytest.fixture
    def sample_csv_content(self):
        """Sample CSV content for testing"""
        return """element_id,name,primary_category,secondary_category,tertiary_category
3001,Brick 2x4,Basic,Brick,Standard
3002,Brick 1x2,Basic,Brick,Standard
3003,Brick 2x2,Basic,Brick,Standard
2456,Brick 2x6,Basic,Brick,Large
6002,Tire,Vehicle,Wheel,Tire
6003,Rim,Vehicle,Wheel,Rim"""

    @pytest.fixture
    def temp_csv_file(self, sample_csv_content):
        """Create temporary CSV file for testing"""
        fd, path = tempfile.mkstemp(suffix='.csv')
        with os.fdopen(fd, 'w') as f:
            f.write(sample_csv_content)

        yield path

        if os.path.exists(path):
            os.remove(path)

    def test_parse_categories_from_csv(self, temp_config_file, temp_csv_file):
        """Should correctly parse category hierarchy from CSV"""
        # Create config manager with custom CSV path
        manager = EnhancedConfigManager(temp_config_file)

        # Parse the CSV
        hierarchy = manager.parse_categories_from_csv(temp_csv_file)

        # Check structure
        assert "primary" in hierarchy
        assert "primary_to_secondary" in hierarchy
        assert "secondary_to_tertiary" in hierarchy
        assert "all_categories" in hierarchy

        # Check primary categories
        assert "Basic" in hierarchy["primary"]
        assert "Vehicle" in hierarchy["primary"]

        # Check secondary categories
        assert "Brick" in hierarchy["primary_to_secondary"]["Basic"]
        assert "Wheel" in hierarchy["primary_to_secondary"]["Vehicle"]

        # Check tertiary categories
        assert "Standard" in hierarchy["secondary_to_tertiary"][("Basic", "Brick")]
        assert "Large" in hierarchy["secondary_to_tertiary"][("Basic", "Brick")]

        # Check individual pieces
        assert "3001" in hierarchy["all_categories"]
        assert hierarchy["all_categories"]["3001"]["name"] == "Brick 2x4"

    def test_get_categories_for_strategy(self, temp_config_file, temp_csv_file):
        """Should return correct categories for each strategy type"""
        manager = EnhancedConfigManager(temp_config_file)
        manager.parse_categories_from_csv(temp_csv_file)

        # Primary strategy
        primary_cats = manager.get_categories_for_strategy("primary")
        assert "Basic" in primary_cats
        assert "Vehicle" in primary_cats

        # Secondary strategy
        secondary_cats = manager.get_categories_for_strategy("secondary", "Basic")
        assert "Brick" in secondary_cats

        # Tertiary strategy
        tertiary_cats = manager.get_categories_for_strategy("tertiary", "Basic", "Brick")
        assert "Standard" in tertiary_cats
        assert "Large" in tertiary_cats

    def test_category_caching(self, temp_config_file, temp_csv_file):
        """Category parsing should cache results"""
        manager = EnhancedConfigManager(temp_config_file)

        # First call should parse CSV
        hierarchy1 = manager.parse_categories_from_csv(temp_csv_file)

        # Second call should return cached result
        hierarchy2 = manager.parse_categories_from_csv(temp_csv_file)

        # Should be the same object (cached)
        assert hierarchy1 is hierarchy2


class TestThreadSafety:
    """Test thread safety of configuration manager"""

    def test_concurrent_config_access(self, temp_config_file):
        """Multiple threads should be able to access config safely"""
        manager = EnhancedConfigManager(temp_config_file)
        results = []
        errors = []

        def config_reader():
            try:
                for i in range(10):
                    config = manager.get_module_config(ModuleConfig.CAMERA.value)
                    assert "device_id" in config
                    results.append(config["device_id"])
                    time.sleep(0.001)  # Small delay to encourage race conditions
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=config_reader)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)

        # Should have no errors and expected number of results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 50  # 5 threads × 10 reads each


class TestObserverPattern:
    """Test configuration change notifications"""

    def test_observer_notifications(self, temp_config_file):
        """Observers should be notified of configuration changes"""
        manager = EnhancedConfigManager(temp_config_file)

        notifications = []

        def observer(module_name, changes):
            notifications.append((module_name, changes))

        # Register observer
        manager.register_observer(observer)

        # Make a change
        manager.update_module_config(ModuleConfig.CAMERA.value, {"device_id": 5})

        # Should have received notification
        assert len(notifications) == 1
        module_name, changes = notifications[0]
        assert module_name == ModuleConfig.CAMERA.value
        assert "device_id" in changes
        assert changes["device_id"] == 5


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_json_file_creates_defaults(self, temp_config_file):
        """Invalid JSON file should fall back to creating defaults"""
        # Write invalid JSON to file
        with open(temp_config_file, 'w') as f:
            f.write("{ invalid json content")

        # Should handle gracefully and create defaults
        manager = EnhancedConfigManager(temp_config_file)

        # Should be able to get valid config
        camera_config = manager.get_module_config(ModuleConfig.CAMERA.value)
        assert isinstance(camera_config, dict)
        assert "device_id" in camera_config

    def test_missing_csv_file_raises_error(self, temp_config_file):
        """Missing CSV file should raise appropriate error"""
        manager = EnhancedConfigManager(temp_config_file)

        with pytest.raises(FileNotFoundError):
            manager.parse_categories_from_csv("nonexistent_file.csv")

    def test_unknown_module_returns_empty_schema(self):
        """Unknown module should return empty schema gracefully"""
        schema = ConfigSchema.get_module_schema("nonexistent_module")
        assert schema == {}

        required = ConfigSchema.get_required_fields("nonexistent_module")
        assert required == []


# Integration tests that test multiple components together
class TestIntegration:
    """Test integration between different components"""

    def test_full_workflow(self, temp_config_file, temp_csv_file):
        """Test complete workflow from file loading to category parsing"""
        # Create manager
        manager = create_config_manager(temp_config_file)

        # Update piece identifier to use our test CSV
        manager.update_module_config(
            ModuleConfig.PIECE_IDENTIFIER.value,
            {"csv_path": temp_csv_file}
        )

        # Parse categories
        hierarchy = manager.get_category_hierarchy()

        # Get categories for different strategies
        primary_cats = manager.get_categories_for_strategy("primary")
        secondary_cats = manager.get_categories_for_strategy("secondary", primary_cats[0])

        # Validate results
        assert len(primary_cats) > 0
        assert len(secondary_cats) > 0

        # Get validation report
        report = manager.get_validation_report()
        assert report["valid"] == True


# Performance tests
class TestPerformance:
    """Test performance characteristics"""

    def test_config_access_performance(self, temp_config_file):
        """Config access should be reasonably fast"""
        manager = EnhancedConfigManager(temp_config_file)

        import time
        start_time = time.time()

        # Access config many times
        for _ in range(1000):
            config = manager.get_module_config(ModuleConfig.CAMERA.value)
            assert config["device_id"] is not None

        elapsed = time.time() - start_time

        # Should be fast (less than 1 second for 1000 accesses)
        assert elapsed < 1.0, f"Config access too slow: {elapsed:.3f}s for 1000 accesses"


if __name__ == "__main__":
    """
    Run tests directly with: python test_enhanced_config_manager.py
    Or use pytest for better output: python -m pytest test_enhanced_config_manager.py -v
    """
    import sys

    # Simple test runner if pytest is not available
    print("Running Enhanced Config Manager Tests...")
    print("=" * 50)

    try:
        # Try to run with pytest if available
        import pytest

        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        print("pytest not available, running basic tests...")

        # Run a few basic tests manually
        test_schema = TestConfigSchema()
        test_schema.test_all_modules_have_schemas()
        test_schema.test_camera_schema_completeness()

        test_validator = TestConfigValidator()
        test_validator.test_valid_camera_config_passes()

        print("✅ Basic tests passed!")
        print("Install pytest for comprehensive testing: pip install pytest")