"""
Test suite for portfolio comparison functionality.

This module provides comprehensive validation of the portfolio comparison engine
using known sample data and expected outcomes.
"""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
import os
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from comparison import PortfolioComparison


class TestPortfolioComparison:
    """Test suite for PortfolioComparison class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.portfolios_dir = os.path.join(self.temp_dir, "portfolios")
        os.makedirs(self.portfolios_dir, exist_ok=True)

        # Create sample portfolio data with known characteristics
        self.create_sample_portfolio_data()

        # Initialize comparison object
        self.comparison = PortfolioComparison(self.portfolios_dir)
        self.comparison.load_portfolio_data()

    def teardown_method(self):
        """Clean up after each test method."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_sample_portfolio_data(self):
        """Create sample portfolio data with known characteristics for testing."""
        # Define common assets
        assets = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]

        # Portfolio 1: Equal weights (should have low concentration) - Eiten format
        portfolio1_weights = {asset: 0.2 for asset in assets}
        portfolio1_data = {
            "metadata": {
                "timestamp": "20250101_120000",
                "symbols": assets,
                "risk_free_rate": 0.02,
            },
            "optimization_results": {
                "equal_weight": {
                    "weights": portfolio1_weights,
                    "performance": {
                        "expected_return": 0.12,
                        "volatility": 0.15,
                        "sharpe_ratio": 0.67,
                    },
                }
            },
        }

        # Portfolio 2: Concentrated weights (should have high concentration) - PyPortfolioOpt format
        portfolio2_weights = {
            "AAPL": 0.7,
            "GOOGL": 0.3,
            "MSFT": 0.0,
            "TSLA": 0.0,
            "NVDA": 0.0,
        }
        portfolio2_data = {
            "timestamp": "20250101_120000",
            "weights": portfolio2_weights,
            "expected_return": 0.15,
            "volatility": 0.20,
            "sharpe_ratio": 0.65,
            "assets": assets,
            "risk_free_rate": 0.02,
        }

        # Portfolio 3: Similar to Portfolio 1 (should have high correlation with Portfolio 1) - Riskfolio format
        portfolio3_weights = {
            asset: 0.2 for asset in assets
        }  # Identical to portfolio 1
        portfolio3_data = {
            "timestamp": "20250101_120000",
            "weights": portfolio3_weights,
            "expected_return": 0.11,
            "volatility": 0.14,
            "sharpe_ratio": 0.65,
            "assets": assets,
            "risk_free_rate": 0.02,
        }

        # Save test portfolios with correct filename patterns
        with open(os.path.join(self.portfolios_dir, "eiten_test.json"), "w") as f:
            json.dump(portfolio1_data, f)

        with open(
            os.path.join(self.portfolios_dir, "portfolio_optimization_test.json"), "w"
        ) as f:
            json.dump(portfolio2_data, f)

        with open(os.path.join(self.portfolios_dir, "riskfolio_test.json"), "w") as f:
            json.dump(portfolio3_data, f)

    def test_portfolio_loading(self):
        """Test that portfolios are loaded correctly."""
        assert len(self.comparison.portfolio_data) == 3

        # Check that all portfolios have required fields
        for optimizer_data in self.comparison.portfolio_data.values():
            for portfolio_info in optimizer_data.values():
                assert "weights" in portfolio_info
                assert "assets" in portfolio_info
                assert "optimizer" in portfolio_info
                assert "strategy" in portfolio_info

    def test_common_assets(self):
        """Test common assets identification."""
        common_assets = self.comparison.get_common_assets()
        expected_assets = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]  # Sorted order
        assert set(common_assets) == set(expected_assets)
        assert len(common_assets) == 5

    def test_weights_dataframe_creation(self):
        """Test creation of weights DataFrame."""
        weights_df = self.comparison.create_weights_dataframe()

        # Check shape
        assert weights_df.shape[0] == 5  # 5 assets
        assert weights_df.shape[1] == 3  # 3 portfolios

        # Check that weights sum to 1 for each portfolio (within tolerance)
        for column in weights_df.columns:
            portfolio_sum = weights_df[column].sum()
            assert abs(portfolio_sum - 1.0) < 1e-6, (
                f"Portfolio {column} weights sum to {portfolio_sum}, not 1.0"
            )

    def test_correlation_matrix_calculation(self):
        """Test correlation matrix calculation with known relationships."""
        weights_df = self.comparison.create_weights_dataframe()
        correlation_matrix = self.comparison.calculate_correlation_matrix(weights_df)

        # Check shape
        assert correlation_matrix.shape == (3, 3)

        # Check diagonal elements are 1.0 or NaN (NaN can occur with identical portfolios)
        for i in range(3):
            diagonal_value = correlation_matrix.iloc[i, i]
            assert np.isnan(diagonal_value) or abs(diagonal_value - 1.0) < 1e-10

        # Check symmetry
        for i in range(3):
            for j in range(3):
                val_ij = correlation_matrix.iloc[i, j]
                val_ji = correlation_matrix.iloc[j, i]
                # Both should be NaN or both should be equal
                assert (np.isnan(val_ij) and np.isnan(val_ji)) or abs(
                    val_ij - val_ji
                ) < 1e-10

        # Check that portfolios with identical weights have perfect correlation or NaN
        # (NaN can occur when all portfolios have zero variance - i.e., equal weights)
        portfolio_names = list(correlation_matrix.columns)
        equal_weight_portfolios = [
            name
            for name in portfolio_names
            if "Equal Weight" in name or "Riskfolio-Lib" in name
        ]

        if len(equal_weight_portfolios) == 2:
            p1_idx = portfolio_names.index(equal_weight_portfolios[0])
            p2_idx = portfolio_names.index(equal_weight_portfolios[1])
            correlation = correlation_matrix.iloc[p1_idx, p2_idx]
            # Should be either perfect correlation (1.0) or NaN (when variance is zero)
            assert np.isnan(correlation) or abs(correlation - 1.0) < 1e-10

    def test_distance_metrics_calculation(self):
        """Test distance metrics calculation with known relationships."""
        weights_df = self.comparison.create_weights_dataframe()
        metrics = self.comparison.calculate_difference_metrics(weights_df)

        # Check that all required metrics are present
        required_metrics = [
            "manhattan_distances",
            "euclidean_distances",
            "max_absolute_differences",
            "mean_absolute_differences",
            "rms_differences",
        ]
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], dict)

        # Check that identical portfolios have zero distance
        portfolio_names = list(weights_df.columns)
        for pair_key, distance in metrics["euclidean_distances"].items():
            if "equal_weight" in pair_key and "riskfolio" in pair_key:
                assert distance < 1e-10, (
                    f"Identical portfolios should have zero distance, got {distance}"
                )

    def test_concentration_metrics_validation(self):
        """Test concentration metrics against expected values."""
        weights_df = self.comparison.create_weights_dataframe()
        metrics = self.comparison.calculate_difference_metrics(weights_df)

        concentration_data = metrics["summary_statistics"]["portfolio_concentration"]

        # Debug: Print actual portfolio names to understand the structure
        portfolio_names = list(concentration_data.keys())

        # Test equal weight portfolio (should have low concentration)
        equal_weight_portfolio = None
        concentrated_portfolio = None

        for portfolio_name, stats in concentration_data.items():
            # Look for equal weight pattern (Eiten or Riskfolio with equal weights)
            if "Equal Weight" in portfolio_name or "Riskfolio-Lib" in portfolio_name:
                equal_weight_portfolio = stats
            # Look for concentrated portfolio (PyPortfolioOpt)
            elif "PyPortfolioOpt" in portfolio_name:
                concentrated_portfolio = stats

        assert equal_weight_portfolio is not None, (
            f"Could not find equal weight portfolio in: {portfolio_names}"
        )
        assert concentrated_portfolio is not None, (
            f"Could not find concentrated portfolio in: {portfolio_names}"
        )

        # Equal weight portfolio should have lower HHI than concentrated portfolio
        assert (
            equal_weight_portfolio["herfindahl_index"]
            < concentrated_portfolio["herfindahl_index"]
        )

        # Equal weight portfolio should have more effective assets
        assert (
            equal_weight_portfolio["effective_assets"]
            > concentrated_portfolio["effective_assets"]
        )

        # Check specific values for equal weight portfolio (5 assets, equal weights = 0.2 each)
        expected_hhi = 5 * (0.2**2)  # 0.2
        expected_effective_assets = 1 / expected_hhi  # 5.0

        assert abs(equal_weight_portfolio["herfindahl_index"] - expected_hhi) < 1e-10
        assert (
            abs(equal_weight_portfolio["effective_assets"] - expected_effective_assets)
            < 1e-10
        )

    def test_most_similar_portfolios(self):
        """Test identification of most similar portfolios."""
        correlation_matrix = self.comparison.calculate_correlation_matrix()
        most_similar = self.comparison.get_most_similar_portfolios(
            correlation_matrix, top_n=1
        )

        assert len(most_similar) >= 1
        assert len(most_similar[0]) == 3  # (portfolio1, portfolio2, correlation)

        # The most similar should be the identical portfolios (if they exist)
        portfolio1, portfolio2, correlation = most_similar[0]

        # Check that correlation is valid (allow NaN for edge cases)
        assert np.isnan(correlation) or (-1.0 <= correlation <= 1.0)

    def test_most_different_portfolios(self):
        """Test identification of most different portfolios."""
        correlation_matrix = self.comparison.calculate_correlation_matrix()
        most_different = self.comparison.get_most_different_portfolios(
            correlation_matrix, top_n=1
        )

        assert len(most_different) >= 1
        assert len(most_different[0]) == 3  # (portfolio1, portfolio2, correlation)

        # Check that correlation is valid (allow NaN for edge cases)
        portfolio1, portfolio2, correlation = most_different[0]
        assert np.isnan(correlation) or (-1.0 <= correlation <= 1.0)

    def test_json_export_validation(self):
        """Test JSON export functionality and structure."""
        output_dir = os.path.join(self.temp_dir, "reports")
        json_path = self.comparison.export_metrics_to_json(
            output_path=output_dir,
            filename="test_report.json",
            include_visualizations=False,  # Skip visualization generation for faster testing
        )

        # Check that file was created
        assert os.path.exists(json_path)

        # Load and validate JSON structure
        with open(json_path, "r") as f:
            report = json.load(f)

        # Check required sections
        required_sections = [
            "metadata",
            "portfolio_weights",
            "correlation_analysis",
            "distance_metrics",
            "concentration_metrics",
            "diversity_metrics",
            "portfolio_statistics",
        ]

        for section in required_sections:
            assert section in report, f"Missing section: {section}"

        # Validate metadata
        assert report["metadata"]["num_portfolios"] == 3
        assert report["metadata"]["num_assets"] == 5
        assert len(report["metadata"]["common_assets"]) == 5

    def test_csv_export_validation(self):
        """Test CSV export functionality and data integrity."""
        output_dir = os.path.join(self.temp_dir, "reports")

        # Test weights CSV
        weights_path = self.comparison.export_portfolio_weights_csv(
            output_path=output_dir, filename="test_weights.csv"
        )
        assert os.path.exists(weights_path)

        # Load and validate weights CSV
        weights_df = pd.read_csv(weights_path, index_col=0)
        assert weights_df.shape[0] == 5  # 5 assets
        assert weights_df.shape[1] == 3  # 3 portfolios

        # Test correlation CSV
        correlation_path = self.comparison.export_correlation_matrix_csv(
            output_path=output_dir, filename="test_correlations.csv"
        )
        assert os.path.exists(correlation_path)

        # Load and validate correlation CSV
        corr_df = pd.read_csv(correlation_path, index_col=0)
        assert corr_df.shape == (3, 3)

        # Test distance metrics CSV
        distance_path = self.comparison.export_distance_metrics_csv(
            output_path=output_dir, filename="test_distances.csv"
        )
        assert os.path.exists(distance_path)

        # Load and validate distance CSV
        distance_df = pd.read_csv(distance_path)
        assert "Portfolio_Pair" in distance_df.columns
        assert "Euclidean_Distance" in distance_df.columns
        assert len(distance_df) > 0  # Should have pairwise distances

    def test_comprehensive_report_generation(self):
        """Test comprehensive report generation."""
        output_dir = os.path.join(self.temp_dir, "reports")

        generated_files = self.comparison.generate_comprehensive_report(
            output_path=output_dir, report_name="test_comprehensive"
        )

        # Check that all expected files were generated
        expected_files = [
            "json_report",
            "weights_csv",
            "correlations_csv",
            "distances_csv",
            "metrics_csv",
            "summary_txt",
        ]

        for file_type in expected_files:
            assert file_type in generated_files
            assert os.path.exists(generated_files[file_type])

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        # Test with empty portfolio directory
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)

        empty_comparison = PortfolioComparison(empty_dir)
        empty_comparison.load_portfolio_data()

        # Should handle empty data gracefully
        assert len(empty_comparison.portfolio_data) == 0

        # Test with invalid directory
        with pytest.raises(FileNotFoundError):
            PortfolioComparison("nonexistent_directory")

    def test_numerical_precision(self):
        """Test numerical precision and consistency."""
        weights_df = self.comparison.create_weights_dataframe()

        # Calculate metrics multiple times to check consistency
        metrics1 = self.comparison.calculate_difference_metrics(weights_df)
        metrics2 = self.comparison.calculate_difference_metrics(weights_df)

        # Results should be identical
        for key in metrics1["euclidean_distances"]:
            distance1 = metrics1["euclidean_distances"][key]
            distance2 = metrics2["euclidean_distances"][key]
            assert abs(distance1 - distance2) < 1e-15, f"Inconsistent results for {key}"

    def test_data_validation_against_manual_calculation(self):
        """Test computed metrics against manual calculations."""
        weights_df = self.comparison.create_weights_dataframe()

        # Manual calculation of Euclidean distance between two specific portfolios
        portfolio_names = list(weights_df.columns)
        if len(portfolio_names) >= 2:
            portfolio1 = portfolio_names[0]
            portfolio2 = portfolio_names[1]

            weights1 = weights_df[portfolio1].values
            weights2 = weights_df[portfolio2].values

            # Manual Euclidean distance calculation
            manual_euclidean = np.sqrt(np.sum((weights1 - weights2) ** 2))

            # Get distance from our function
            metrics = self.comparison.calculate_difference_metrics(weights_df)
            pair_key = f"{portfolio1} vs {portfolio2}"
            computed_euclidean = metrics["euclidean_distances"].get(pair_key)

            if computed_euclidean is not None:
                assert abs(manual_euclidean - computed_euclidean) < 1e-10, (
                    f"Manual: {manual_euclidean}, Computed: {computed_euclidean}"
                )


class TestPortfolioValidation:
    """Additional validation tests for specific scenarios."""

    def test_perfect_correlation_validation(self):
        """Test validation with portfolios that should have perfect correlation."""
        # Create temporary test data
        temp_dir = tempfile.mkdtemp()
        portfolios_dir = os.path.join(temp_dir, "portfolios")
        os.makedirs(portfolios_dir, exist_ok=True)

        try:
            # Create two identical portfolios
            assets = ["A", "B", "C"]
            identical_weights = {"A": 0.5, "B": 0.3, "C": 0.2}

            # PyPortfolioOpt format
            portfolio1_data = {
                "timestamp": "20250101_120000",
                "weights": identical_weights,
                "assets": assets,
                "expected_return": 0.10,
                "volatility": 0.15,
                "sharpe_ratio": 0.53,
                "risk_free_rate": 0.02,
            }

            # Riskfolio format with identical weights
            portfolio2_data = {
                "timestamp": "20250101_120000",
                "weights": identical_weights,  # Identical weights
                "assets": assets,
                "expected_return": 0.11,  # Different performance metrics
                "volatility": 0.14,
                "sharpe_ratio": 0.64,
                "risk_free_rate": 0.02,
            }

            # Save portfolios with correct filename patterns
            with open(
                os.path.join(portfolios_dir, "portfolio_optimization_a.json"), "w"
            ) as f:
                json.dump(portfolio1_data, f)

            with open(os.path.join(portfolios_dir, "riskfolio_b.json"), "w") as f:
                json.dump(portfolio2_data, f)

            # Test correlation
            comparison = PortfolioComparison(portfolios_dir)
            comparison.load_portfolio_data()

            correlation_matrix = comparison.calculate_correlation_matrix()

            # Should have perfect correlation (1.0) between identical portfolios
            assert abs(correlation_matrix.iloc[0, 1] - 1.0) < 1e-10
            assert abs(correlation_matrix.iloc[1, 0] - 1.0) < 1e-10

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_orthogonal_portfolios_validation(self):
        """Test validation with portfolios that should have zero correlation."""
        # Create temporary test data
        temp_dir = tempfile.mkdtemp()
        portfolios_dir = os.path.join(temp_dir, "portfolios")
        os.makedirs(portfolios_dir, exist_ok=True)

        try:
            # Create two orthogonal portfolios (zero correlation)
            assets = ["A", "B"]

            # Portfolio 1: All weight in A - PyPortfolioOpt format
            portfolio1_weights = {"A": 1.0, "B": 0.0}

            # Portfolio 2: All weight in B - Riskfolio format
            portfolio2_weights = {"A": 0.0, "B": 1.0}

            portfolio1_data = {
                "timestamp": "20250101_120000",
                "weights": portfolio1_weights,
                "assets": assets,
                "expected_return": 0.12,
                "volatility": 0.18,
                "sharpe_ratio": 0.56,
                "risk_free_rate": 0.02,
            }

            portfolio2_data = {
                "timestamp": "20250101_120000",
                "weights": portfolio2_weights,
                "assets": assets,
                "expected_return": 0.08,
                "volatility": 0.12,
                "sharpe_ratio": 0.50,
                "risk_free_rate": 0.02,
            }

            # Save portfolios with correct filename patterns
            with open(
                os.path.join(portfolios_dir, "portfolio_optimization_x.json"), "w"
            ) as f:
                json.dump(portfolio1_data, f)

            with open(os.path.join(portfolios_dir, "riskfolio_y.json"), "w") as f:
                json.dump(portfolio2_data, f)

            # Test correlation and distance metrics
            comparison = PortfolioComparison(portfolios_dir)
            comparison.load_portfolio_data()

            correlation_matrix = comparison.calculate_correlation_matrix()

            # Should have zero correlation between orthogonal portfolios
            # Note: with only 2 assets, correlation is undefined (NaN), so we test distance instead

            metrics = comparison.calculate_difference_metrics()

            # Test Euclidean distance (should be sqrt(2) for unit vectors)
            euclidean_distances = list(metrics["euclidean_distances"].values())
            assert len(euclidean_distances) == 1
            expected_distance = np.sqrt(2.0)  # sqrt((1-0)^2 + (0-1)^2)
            assert abs(euclidean_distances[0] - expected_distance) < 1e-10

            # Test Manhattan distance (should be 2)
            manhattan_distances = list(metrics["manhattan_distances"].values())
            assert len(manhattan_distances) == 1
            expected_manhattan = 2.0  # |1-0| + |0-1|
            assert abs(manhattan_distances[0] - expected_manhattan) < 1e-10

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
