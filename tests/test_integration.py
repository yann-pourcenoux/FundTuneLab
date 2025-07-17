"""
Comprehensive End-to-End Integration Tests for FundTuneLab

This module contains tests that validate the complete workflow from data collection
to report generation, ensuring all components work together correctly.
"""

import pytest
import tempfile
import shutil
import json
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import from the installed package
from src.orchestrator import FundTuneLabOrchestrator, run_orchestrator
from src.unified_reporting import UnifiedReportGenerator


class TestEndToEndWorkflow:
    """Test the complete end-to-end workflow."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp(prefix="fundtunelab_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_results_dir(self, temp_workspace, monkeypatch):
        """Mock the RESULTS_DIR to use temporary workspace."""
        test_results_dir = temp_workspace / "results"
        test_results_dir.mkdir(exist_ok=True, parents=True)

        # Create subdirectories
        (test_results_dir / "portfolios").mkdir(exist_ok=True)
        (test_results_dir / "reports").mkdir(exist_ok=True)
        (test_results_dir / "plots").mkdir(exist_ok=True)
        (test_results_dir / "backtests").mkdir(exist_ok=True)
        (test_results_dir / "logs").mkdir(exist_ok=True)

        monkeypatch.setattr("src.orchestrator.RESULTS_DIR", test_results_dir)
        monkeypatch.setattr(
            "src.orchestrator.REPORTS_DIR", test_results_dir / "reports"
        )
        monkeypatch.setattr("src.orchestrator.PLOTS_DIR", test_results_dir / "plots")
        monkeypatch.setattr(
            "src.orchestrator.BACKTESTS_DIR", test_results_dir / "backtests"
        )
        monkeypatch.setattr(
            "src.orchestrator.PORTFOLIOS_DIR", test_results_dir / "portfolios"
        )

        return test_results_dir

    def test_orchestrator_initialization(self, mock_results_dir):
        """Test that the orchestrator initializes correctly."""
        orchestrator = FundTuneLabOrchestrator()

        assert orchestrator is not None
        assert orchestrator.logger is not None
        assert len(orchestrator.stages) == 8
        assert orchestrator.execution_state["start_time"] is None
        assert orchestrator.execution_state["stages_completed"] == []

    def test_orchestrator_stage_definition(self):
        """Test that all expected stages are defined."""
        orchestrator = FundTuneLabOrchestrator()

        expected_stages = [
            "data_collection",
            "data_preprocessing",
            "pypfopt_optimization",
            "eiten_optimization",
            "riskfolio_optimization",
            "portfolio_comparison",
            "backtesting",
            "report_generation",
        ]

        actual_stages = [stage[0] for stage in orchestrator.stages]
        assert actual_stages == expected_stages

    @patch("src.orchestrator.download_default_assets")
    def test_data_collection_stage(self, mock_download, mock_results_dir):
        """Test the data collection stage."""
        # Mock successful data collection
        mock_download.return_value = {
            "SPY": pd.DataFrame({"Close": [100, 101, 102]}),
            "BND": pd.DataFrame({"Close": [50, 51, 52]}),
            "VTI": None,  # Simulate one failure
        }

        orchestrator = FundTuneLabOrchestrator()
        result = orchestrator._run_data_collection()

        assert result["total_assets"] == 3
        assert result["successful_downloads"] == 2
        assert result["success_rate"] == 2 / 3
        assert "SPY" in result["assets"]
        mock_download.assert_called_once()

    @patch("src.orchestrator.preprocess_all_data")
    def test_data_preprocessing_stage(self, mock_preprocess, mock_results_dir):
        """Test the data preprocessing stage."""
        # Mock successful preprocessing
        mock_preprocess.return_value = {
            "files_found": 5,
            "files_successful": 4,
            "files_failed": 1,
            "processing_time": 10.5,
        }

        orchestrator = FundTuneLabOrchestrator()
        result = orchestrator._run_data_preprocessing()

        assert result["files_found"] == 5
        assert result["files_successful"] == 4
        mock_preprocess.assert_called_once()

    @patch("src.orchestrator.optimize_portfolio_from_data")
    def test_pypfopt_optimization_stage(self, mock_optimize, mock_results_dir):
        """Test the PyPortfolioOpt optimization stage."""
        # Mock successful optimization
        mock_optimize.side_effect = [
            {"success": True, "method": "max_sharpe"},
            {"success": True, "method": "min_volatility"},
            {
                "success": False,
                "error": "Convergence failed",
                "method": "efficient_return",
            },
        ]

        orchestrator = FundTuneLabOrchestrator()
        result = orchestrator._run_pypfopt_optimization()

        assert result["total_methods"] == 3
        assert result["successful_methods"] == 2
        assert mock_optimize.call_count == 3

    @patch("src.orchestrator.generate_unified_report")
    def test_report_generation_stage(self, mock_generate, mock_results_dir):
        """Test the report generation stage."""
        # Mock successful report generation
        mock_generate.return_value = {
            "json": str(mock_results_dir / "reports" / "unified_report.json"),
            "html": str(mock_results_dir / "reports" / "unified_report.html"),
            "markdown": str(mock_results_dir / "reports" / "unified_report.md"),
        }

        orchestrator = FundTuneLabOrchestrator()
        result = orchestrator._run_report_generation()

        assert result["success"] is True
        assert result["total_reports"] == 4  # 3 unified + 1 master summary
        assert "generated_files" in result
        mock_generate.assert_called_once()

    def test_orchestrator_skip_stages(self, mock_results_dir):
        """Test that stages can be skipped correctly."""
        with patch.multiple(
            "src.orchestrator",
            download_default_assets=MagicMock(return_value={}),
            preprocess_all_data=MagicMock(
                return_value={"files_found": 0, "files_successful": 0}
            ),
            generate_unified_report=MagicMock(return_value={}),
        ):
            orchestrator = FundTuneLabOrchestrator()
            result = orchestrator.run_full_workflow(
                skip_stages=["pypfopt_optimization", "eiten_optimization"]
            )

            # Should complete without the skipped stages
            assert "pypfopt_optimization" not in result["stages_completed"]
            assert "eiten_optimization" not in result["stages_completed"]
            assert "data_collection" in result["stages_completed"]
            assert len(result["stages_failed"]) == 0

    def test_orchestrator_error_handling(self, mock_results_dir):
        """Test error handling in the orchestrator."""
        with patch(
            "src.orchestrator.download_default_assets",
            side_effect=Exception("Network error"),
        ):
            orchestrator = FundTuneLabOrchestrator()
            result = orchestrator.run_full_workflow()

            # Should stop after data collection failure (critical stage)
            assert "data_collection" in result["stages_failed"]
            assert len(result["stages_completed"]) == 0
            assert len(result["errors"]) > 0

    def test_execution_state_tracking(self, mock_results_dir):
        """Test that execution state is tracked correctly."""
        with patch.multiple(
            "src.orchestrator",
            download_default_assets=MagicMock(return_value={}),
            preprocess_all_data=MagicMock(
                return_value={"files_found": 0, "files_successful": 0}
            ),
            generate_unified_report=MagicMock(return_value={}),
        ):
            orchestrator = FundTuneLabOrchestrator()
            result = orchestrator.run_full_workflow(
                skip_stages=[
                    "pypfopt_optimization",
                    "eiten_optimization",
                    "riskfolio_optimization",
                    "portfolio_comparison",
                    "backtesting",
                ]
            )

            assert result["start_time"] is not None
            assert result["end_time"] is not None
            assert result["total_duration"] is not None
            assert result["total_duration"] > 0


class TestUnifiedReporting:
    """Test the unified reporting functionality."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp(prefix="reporting_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_portfolio_data(self, temp_workspace):
        """Create sample portfolio data files."""
        portfolios_dir = temp_workspace / "portfolios"
        portfolios_dir.mkdir(exist_ok=True, parents=True)

        # Create sample JSON portfolio data
        portfolio_data = {
            "optimizer": "pypfopt",
            "method": "max_sharpe",
            "performance": {
                "expected_return": 0.15,
                "volatility": 0.20,
                "sharpe_ratio": 0.75,
            },
            "weights": {"SPY": 0.6, "BND": 0.4},
        }

        json_file = portfolios_dir / "portfolio_optimization_test.json"
        with open(json_file, "w") as f:
            json.dump(portfolio_data, f)

        # Create sample CSV weights file
        weights_df = pd.DataFrame({"Asset": ["SPY", "BND"], "Weight": [0.6, 0.4]})
        csv_file = portfolios_dir / "portfolio_optimization_test_weights.csv"
        weights_df.to_csv(csv_file, index=False)

        return temp_workspace

    def test_unified_report_generator_initialization(self, temp_workspace):
        """Test UnifiedReportGenerator initialization."""
        output_dir = temp_workspace / "reports"
        generator = UnifiedReportGenerator(output_dir)

        assert generator.output_dir == output_dir
        assert output_dir.exists()
        assert generator.portfolio_data == {}
        assert generator.comparison_data == {}
        assert generator.backtest_data == {}

    def test_portfolio_data_collection(self, sample_portfolio_data):
        """Test collection of portfolio data."""
        # Mock the RESULTS_DIR to point to our test data
        with patch("src.unified_reporting.RESULTS_DIR", sample_portfolio_data):
            generator = UnifiedReportGenerator()
            generator._collect_portfolio_data()

            assert len(generator.portfolio_data) > 0
            assert "pypfopt" in generator.portfolio_data

    def test_unified_data_structure_creation(self, sample_portfolio_data):
        """Test creation of unified data structure."""
        with patch("src.unified_reporting.RESULTS_DIR", sample_portfolio_data):
            generator = UnifiedReportGenerator()
            generator._collect_portfolio_data()

            unified_data = generator._create_unified_data_structure()

            assert "metadata" in unified_data
            assert "executive_summary" in unified_data
            assert "portfolio_optimization" in unified_data
            assert "portfolio_comparison" in unified_data
            assert "backtesting" in unified_data
            assert "visualizations" in unified_data

    def test_json_report_generation(self, sample_portfolio_data, temp_workspace):
        """Test JSON report generation."""
        output_dir = temp_workspace / "reports"

        with patch("src.unified_reporting.RESULTS_DIR", sample_portfolio_data):
            generator = UnifiedReportGenerator(output_dir)
            generator._collect_portfolio_data()

            unified_data = generator._create_unified_data_structure()
            file_path = generator._generate_json_report(unified_data, "test")

            assert file_path.exists()
            assert file_path.suffix == ".json"

            # Verify JSON is valid
            with open(file_path, "r") as f:
                loaded_data = json.load(f)
                assert "metadata" in loaded_data

    def test_html_report_generation(self, sample_portfolio_data, temp_workspace):
        """Test HTML report generation."""
        output_dir = temp_workspace / "reports"

        with patch("src.unified_reporting.RESULTS_DIR", sample_portfolio_data):
            generator = UnifiedReportGenerator(output_dir)
            generator._collect_portfolio_data()

            unified_data = generator._create_unified_data_structure()
            file_path = generator._generate_html_report(
                unified_data, "test", include_plots=False
            )

            assert file_path.exists()
            assert file_path.suffix == ".html"

            # Verify HTML contains expected sections
            content = file_path.read_text()
            assert "FundTuneLab Unified Portfolio Analysis Report" in content
            assert "Executive Summary" in content

    def test_markdown_report_generation(self, sample_portfolio_data, temp_workspace):
        """Test Markdown report generation."""
        output_dir = temp_workspace / "reports"

        with patch("src.unified_reporting.RESULTS_DIR", sample_portfolio_data):
            generator = UnifiedReportGenerator(output_dir)
            generator._collect_portfolio_data()

            unified_data = generator._create_unified_data_structure()
            file_path = generator._generate_markdown_report(
                unified_data, "test", include_plots=False
            )

            assert file_path.exists()
            assert file_path.suffix == ".md"

            # Verify Markdown contains expected sections
            content = file_path.read_text()
            assert "# FundTuneLab Unified Portfolio Analysis Report" in content
            assert "## Executive Summary" in content

    def test_comprehensive_report_generation(
        self, sample_portfolio_data, temp_workspace
    ):
        """Test comprehensive report generation with multiple formats."""
        output_dir = temp_workspace / "reports"

        with patch("src.unified_reporting.RESULTS_DIR", sample_portfolio_data):
            generator = UnifiedReportGenerator(output_dir)
            result = generator.generate_comprehensive_report(
                include_plots=False, format_types=["json", "html", "markdown"]
            )

            assert len(result) == 3
            assert "json" in result
            assert "html" in result
            assert "markdown" in result

            # Verify all files exist
            for file_path in result.values():
                assert Path(file_path).exists()


class TestWorkflowValidation:
    """Test end-to-end workflow validation."""

    def test_cli_argument_validation(self):
        """Test CLI argument validation."""
        from src.cli import validate_skip_stages

        # Valid stages should pass
        valid_stages = ["data_collection", "backtesting"]
        result = validate_skip_stages(valid_stages)
        assert result == valid_stages

        # Invalid stages should raise SystemExit
        with pytest.raises(SystemExit):
            validate_skip_stages(["invalid_stage"])

    def test_convenience_function(self):
        """Test the run_orchestrator convenience function."""
        with patch(
            "src.orchestrator.FundTuneLabOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.run_full_workflow.return_value = {"success": True}
            mock_orchestrator_class.return_value = mock_orchestrator

            run_orchestrator(skip_stages=["backtesting"])

            mock_orchestrator_class.assert_called_once_with()
            mock_orchestrator.run_full_workflow.assert_called_once_with(
                skip_stages=["backtesting"]
            )


class TestDataIntegrity:
    """Test data integrity and validation."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp(prefix="fundtunelab_integrity_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    """Test data integrity and validation."""

    def test_empty_results_handling(self, temp_workspace):
        """Test handling of empty results directories."""
        # More explicit setup for clarity
        results_dir = temp_workspace / "results"
        portfolios_dir = results_dir / "portfolios"
        reports_dir = results_dir / "reports"

        results_dir.mkdir(exist_ok=True)
        portfolios_dir.mkdir(exist_ok=True)
        reports_dir.mkdir(exist_ok=True)

        with patch("src.unified_reporting.RESULTS_DIR", results_dir):
            # Pass the explicit reports_dir to the constructor
            generator = UnifiedReportGenerator(output_dir=reports_dir)
            generator._collect_portfolio_data()
            generator._collect_comparison_data()
            generator._collect_backtest_data()

            unified_data = generator._create_unified_data_structure()

            assert (
                unified_data["executive_summary"]["total_portfolios_generated"] == 0
            ), "Should be 0 portfolios"
            assert not unified_data["executive_summary"][
                "comparison_analyses_available"
            ], "Should be no comparison data"
            assert not unified_data["executive_summary"]["backtests_available"], (
                "Should be no backtest data"
            )

    def test_corrupted_file_handling(self, temp_workspace):
        """Test handling of corrupted data files."""
        results_dir = temp_workspace / "results"
        portfolios_dir = results_dir / "portfolios"
        reports_dir = results_dir / "reports"

        results_dir.mkdir(exist_ok=True)
        portfolios_dir.mkdir(exist_ok=True, parents=True)
        reports_dir.mkdir(exist_ok=True)

        # Create corrupted JSON file
        corrupted_file = portfolios_dir / "corrupted.json"
        corrupted_file.write_text("{ invalid json content")

        with patch("src.unified_reporting.RESULTS_DIR", results_dir):
            generator = UnifiedReportGenerator(output_dir=reports_dir)
            generator._collect_portfolio_data()

            # Should handle corrupted files gracefully
            assert len(generator.metadata["errors"]) > 0, (
                "Should have recorded an error"
            )
            assert any(
                "corrupted.json" in error for error in generator.metadata["errors"]
            ), "Error message should mention the corrupted file"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
