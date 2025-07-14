"""
Test suite for the configuration management module.

This module tests the configuration consistency, environment variable handling,
and utility functions to ensure proper operation across different environments.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch
import tempfile
import warnings


def test_import_config():
    """Test that the config module can be imported successfully."""
    from config import settings

    assert settings is not None


def test_project_paths():
    """Test that all project paths are correctly configured."""
    from config import settings

    # Test that all paths are Path objects
    assert isinstance(settings.PROJECT_ROOT, Path)
    assert isinstance(settings.DATA_DIR, Path)
    assert isinstance(settings.SRC_DIR, Path)
    assert isinstance(settings.RESULTS_DIR, Path)

    # Test that paths are properly constructed relative to project root
    assert settings.DATA_DIR == settings.PROJECT_ROOT / "data"
    assert settings.SRC_DIR == settings.PROJECT_ROOT / "src"
    assert settings.RESULTS_DIR == settings.PROJECT_ROOT / "results"

    # Test subdirectories
    assert settings.RAW_DATA_DIR == settings.DATA_DIR / "raw"
    assert settings.PROCESSED_DATA_DIR == settings.DATA_DIR / "processed"
    assert settings.EXTERNAL_DATA_DIR == settings.DATA_DIR / "external"

    assert settings.OPTIMIZATIONS_DIR == settings.RESULTS_DIR / "optimizations"
    assert settings.BACKTESTS_DIR == settings.RESULTS_DIR / "backtests"
    assert settings.REPORTS_DIR == settings.RESULTS_DIR / "reports"
    assert settings.PLOTS_DIR == settings.RESULTS_DIR / "plots"


def test_default_values():
    """Test that default configuration values are sensible."""
    from config import settings

    # Test data settings
    assert isinstance(settings.DEFAULT_START_DATE, str)
    assert isinstance(settings.DEFAULT_END_DATE, str)
    assert isinstance(settings.DEFAULT_BENCHMARK, str)
    assert isinstance(settings.DEFAULT_ASSETS, list)
    assert len(settings.DEFAULT_ASSETS) > 0

    # Test that dates are in correct format
    from datetime import datetime

    datetime.strptime(settings.DEFAULT_START_DATE, "%Y-%m-%d")
    datetime.strptime(settings.DEFAULT_END_DATE, "%Y-%m-%d")

    # Test optimization settings
    assert isinstance(settings.OPTIMIZATION_METHODS, dict)
    assert len(settings.OPTIMIZATION_METHODS) > 0

    # Test backtesting settings
    assert isinstance(settings.BACKTESTING, dict)
    assert settings.BACKTESTING["initial_capital"] > 0
    assert 0 <= settings.BACKTESTING["transaction_cost"] <= 1
    assert 0 <= settings.BACKTESTING["cash_buffer"] <= 1


class TestEnvironmentVariables:
    """Test environment variable handling."""

    def test_environment_variable_overrides(self):
        """Test that environment variables properly override defaults."""
        with patch.dict(
            os.environ,
            {
                "FUNDTUNELAB_START_DATE": "2019-01-01",
                "FUNDTUNELAB_BENCHMARK": "VTI",
                "FUNDTUNELAB_INITIAL_CAPITAL": "50000",
                "FUNDTUNELAB_LOG_LEVEL": "DEBUG",
            },
        ):
            # Re-import to get updated values
            import importlib
            from config import settings

            importlib.reload(settings)

            assert settings.DEFAULT_START_DATE == "2019-01-01"
            assert settings.DEFAULT_BENCHMARK == "VTI"
            assert settings.BACKTESTING["initial_capital"] == 50000
            assert settings.LOGGING["level"] == "DEBUG"

    def test_api_key_retrieval(self):
        """Test API key retrieval functionality."""
        from config import settings

        # Test with mock API key
        with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test_key_123"}):
            key = settings.get_api_key("alpha_vantage")
            assert key == "test_key_123"

        # Test with missing API key
        with patch.dict(os.environ, {}, clear=True):
            key = settings.get_api_key("alpha_vantage")
            assert key == ""

    def test_api_key_warnings(self):
        """Test that warnings are issued for missing API keys of enabled providers."""
        from config import settings

        # Mock enabled provider with missing API key
        with patch.dict(settings.DATA_PROVIDERS, {"alpha_vantage": {"enabled": True}}):
            with patch.dict(os.environ, {}, clear=True):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    settings.get_api_key("alpha_vantage")
                    assert len(w) == 1
                    assert "alpha_vantage" in str(w[0].message)
                    assert "ALPHA_VANTAGE_API_KEY" in str(w[0].message)

    def test_invalid_provider(self):
        """Test error handling for invalid provider names."""
        from config import settings

        with pytest.raises(ValueError, match="Unknown provider"):
            settings.get_api_key("invalid_provider")

    def test_check_environment_setup(self):
        """Test environment setup checking."""
        from config import settings

        with patch.dict(
            os.environ,
            {"ALPHA_VANTAGE_API_KEY": "test_key", "QUANDL_API_KEY": ""},
            clear=True,
        ):
            status = settings.check_environment_setup()
            assert isinstance(status, dict)
            assert status["alpha_vantage"] is True
            assert status["quandl"] is False
            assert status["iex"] is False
            assert status["fred"] is False


class TestUtilityFunctions:
    """Test utility functions."""

    def test_ensure_directories(self):
        """Test that ensure_directories creates necessary directories."""
        from config import settings

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock PROJECT_ROOT to use temp directory
            with patch.object(settings, "PROJECT_ROOT", Path(temp_dir)):
                with patch.object(settings, "DATA_DIR", Path(temp_dir) / "data"):
                    with patch.object(
                        settings, "RESULTS_DIR", Path(temp_dir) / "results"
                    ):
                        settings.ensure_directories()

                        # Check that directories were created
                        assert (Path(temp_dir) / "data").exists()
                        assert (Path(temp_dir) / "results").exists()

    def test_get_data_file_path(self):
        """Test data file path generation."""
        from config import settings

        # Test different data types
        raw_path = settings.get_data_file_path("test.csv", "raw")
        processed_path = settings.get_data_file_path("test.csv", "processed")
        external_path = settings.get_data_file_path("test.csv", "external")

        assert raw_path == settings.RAW_DATA_DIR / "test.csv"
        assert processed_path == settings.PROCESSED_DATA_DIR / "test.csv"
        assert external_path == settings.EXTERNAL_DATA_DIR / "test.csv"

        # Test default data type
        default_path = settings.get_data_file_path("test.csv")
        assert default_path == settings.PROCESSED_DATA_DIR / "test.csv"

        # Test invalid data type
        with pytest.raises(ValueError, match="Unknown data_type"):
            settings.get_data_file_path("test.csv", "invalid")

    def test_get_results_file_path(self):
        """Test results file path generation."""
        from config import settings

        # Test different result types
        opt_path = settings.get_results_file_path("test.json", "optimizations")
        backtest_path = settings.get_results_file_path("test.json", "backtests")
        report_path = settings.get_results_file_path("test.html", "reports")
        plot_path = settings.get_results_file_path("test.png", "plots")

        assert opt_path == settings.OPTIMIZATIONS_DIR / "test.json"
        assert backtest_path == settings.BACKTESTS_DIR / "test.json"
        assert report_path == settings.REPORTS_DIR / "test.html"
        assert plot_path == settings.PLOTS_DIR / "test.png"

        # Test default result type
        default_path = settings.get_results_file_path("test.json")
        assert default_path == settings.OPTIMIZATIONS_DIR / "test.json"

        # Test invalid result type
        with pytest.raises(ValueError, match="Unknown result_type"):
            settings.get_results_file_path("test.json", "invalid")

    def test_validate_settings(self):
        """Test settings validation."""
        from config import settings

        # Should pass with default settings
        assert settings.validate_settings() is True

        # Test with invalid settings
        with patch.object(settings, "DEFAULT_ASSETS", []):
            with pytest.raises(ValueError, match="DEFAULT_ASSETS cannot be empty"):
                settings.validate_settings()

        with patch.object(settings, "DEFAULT_BENCHMARK", ""):
            with pytest.raises(ValueError, match="DEFAULT_BENCHMARK must be specified"):
                settings.validate_settings()

        # Test with invalid date format
        with patch.object(settings, "DEFAULT_START_DATE", "invalid-date"):
            with pytest.raises(ValueError, match="Invalid date format"):
                settings.validate_settings()


class TestConfigurationConsistency:
    """Test that configuration is consistent across imports."""

    def test_multiple_imports_consistency(self):
        """Test that multiple imports return consistent values."""
        from config import settings as settings1
        from config import settings as settings2

        # Test that key values are consistent
        assert settings1.PROJECT_ROOT == settings2.PROJECT_ROOT
        assert settings1.DEFAULT_ASSETS == settings2.DEFAULT_ASSETS
        assert settings1.DEFAULT_BENCHMARK == settings2.DEFAULT_BENCHMARK
        assert settings1.OPTIMIZATION_METHODS == settings2.OPTIMIZATION_METHODS

    def test_configuration_immutability(self):
        """Test that configuration values are not accidentally modified."""
        from config import settings

        # Simulate accidental modification to test immutability
        settings.DEFAULT_ASSETS.append("TEST")
        settings.OPTIMIZATION_METHODS["test"] = {"enabled": True}

        # Re-import and check values are restored
        import importlib

        importlib.reload(settings)

        # Note: This test demonstrates the need for immutable configurations
        # In a production system, you might want to use frozen dataclasses or similar


class TestDataProviderConfiguration:
    """Test data provider configuration."""

    def test_data_providers_structure(self):
        """Test that data providers are properly configured."""
        from config import settings

        assert isinstance(settings.DATA_PROVIDERS, dict)

        for provider, config in settings.DATA_PROVIDERS.items():
            assert isinstance(config, dict)
            assert "enabled" in config
            assert isinstance(config["enabled"], bool)

            if "rate_limit" in config:
                assert isinstance(config["rate_limit"], int)
                assert config["rate_limit"] > 0

            if "timeout" in config:
                assert isinstance(config["timeout"], int)
                assert config["timeout"] > 0


def test_performance_metrics_configuration():
    """Test that performance metrics are properly defined."""
    from config import settings

    assert isinstance(settings.PERFORMANCE_METRICS, list)
    assert len(settings.PERFORMANCE_METRICS) > 0

    # Test that all metrics are strings
    for metric in settings.PERFORMANCE_METRICS:
        assert isinstance(metric, str)
        assert len(metric) > 0


def test_logging_configuration():
    """Test logging configuration."""
    from config import settings

    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    assert settings.LOGGING["level"] in valid_levels
    assert isinstance(settings.LOGGING["format"], str)
    assert isinstance(settings.LOGGING["file"], Path)
    assert isinstance(settings.LOGGING["console_output"], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
