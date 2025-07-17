"""
PyPortfolioOpt Optimizer Module for FundTuneLab

This module provides portfolio optimization functionality using PyPortfolioOpt library.
It loads preprocessed data, applies mean-variance optimization, calculates efficient frontier,
and generates portfolio weights in standardized JSON/CSV format.
"""

from loguru import logger
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import numpy as np
from config.settings import PROCESSED_DATA_DIR, RESULTS_DIR, ensure_directories

# PyPortfolioOpt imports
from pypfopt import EfficientFrontier, expected_returns, risk_models


class PortfolioOptimizerError(Exception):
    """Custom exception for portfolio optimization errors."""

    pass


class DataLoadError(PortfolioOptimizerError):
    """Exception raised when data loading fails."""

    pass


class OptimizationError(PortfolioOptimizerError):
    """Exception raised when optimization fails."""

    pass


class PyPortfolioOptOptimizer:
    """
    Main class for portfolio optimization using PyPortfolioOpt.

    Handles data loading, optimization, and result formatting.
    """

    def __init__(
        self,
        processed_data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        risk_free_rate: float = 0.02,
    ):
        """
        Initialize the optimizer.

        Args:
            processed_data_dir: Directory containing preprocessed data
            results_dir: Directory to save optimization results
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            log_level: Logging level
        """
        self.processed_data_dir = processed_data_dir or PROCESSED_DATA_DIR
        self.results_dir = results_dir or RESULTS_DIR / "portfolios"
        self.risk_free_rate = risk_free_rate

        # Ensure directories exist
        ensure_directories()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.prices_df = None
        self.returns_df = None
        self.expected_returns = None
        self.cov_matrix = None
        self.assets = []

        # Optimization results
        self.weights = None
        self.performance = None
        self.efficient_frontier_data = None

        logger.info("PyPortfolioOpt Optimizer initialized")

    def load_preprocessed_data(self, date_pattern: str = "20250714") -> pd.DataFrame:
        """
        Load preprocessed data from CSV files.

        Args:
            date_pattern: Date pattern to match in filenames

        Returns:
            Combined price data DataFrame

        Raises:
            DataLoadError: If data loading fails
        """
        try:
            logger.info(f"Loading preprocessed data from {self.processed_data_dir}")

            # Find all processed CSV files
            csv_files = list(
                self.processed_data_dir.glob(f"*processed_{date_pattern}.csv")
            )

            if not csv_files:
                raise DataLoadError(
                    f"No processed CSV files found with pattern *processed_{date_pattern}.csv"
                )

            logger.info(f"Found {len(csv_files)} processed files")

            # Load and combine data
            all_data = []
            symbols = []

            for csv_file in csv_files:
                logger.debug(f"Loading {csv_file}")
                df = pd.read_csv(csv_file, parse_dates=["Date"])

                # Extract symbol from filename if not in data
                if "Symbol" not in df.columns:
                    symbol = csv_file.stem.split("_")[0]  # Extract symbol from filename
                    df["Symbol"] = symbol

                # Validate required columns
                required_columns = ["Date", "Close", "Symbol"]
                missing_columns = [
                    col for col in required_columns if col not in df.columns
                ]
                if missing_columns:
                    raise DataLoadError(
                        f"Missing required columns in {csv_file}: {missing_columns}"
                    )

                symbol = df["Symbol"].iloc[0]
                symbols.append(symbol)
                all_data.append(df[["Date", "Close", "Symbol"]])

            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)

            # Pivot to get prices by symbol
            prices_df = combined_df.pivot(
                index="Date", columns="Symbol", values="Close"
            )
            prices_df = prices_df.dropna()  # Remove rows with missing data

            # Sort by date
            prices_df = prices_df.sort_index()

            self.prices_df = prices_df
            self.assets = list(prices_df.columns)

            logger.info(f"Loaded data for {len(self.assets)} assets: {self.assets}")
            logger.info(
                f"Date range: {prices_df.index.min()} to {prices_df.index.max()}"
            )
            logger.info(f"Total observations: {len(prices_df)}")

            # Validate data integrity
            self._validate_price_data()

            return prices_df

        except Exception as e:
            raise DataLoadError(f"Failed to load preprocessed data: {str(e)}")

    def _validate_price_data(self):
        """Validate the loaded price data."""
        if self.prices_df is None or self.prices_df.empty:
            raise DataLoadError("No price data loaded")

        # Check for negative or zero prices
        if (self.prices_df <= 0).any().any():
            logger.warning("Found negative or zero prices in data")

        # Check for missing data
        missing_pct = (self.prices_df.isnull().sum() / len(self.prices_df)) * 100
        if missing_pct.any():
            logger.warning(
                f"Missing data percentage by asset:\n{missing_pct[missing_pct > 0]}"
            )

        # Check minimum number of observations
        min_obs = 20  # Minimum for meaningful statistics
        if len(self.prices_df) < min_obs:
            raise DataLoadError(
                f"Insufficient data: {len(self.prices_df)} observations (minimum {min_obs})"
            )

        logger.info("Price data validation completed successfully")

    def calculate_returns_and_statistics(self, frequency: int = 252):
        """
        Calculate returns and risk-return statistics.

        Args:
            frequency: Number of trading periods per year (default: 252 for daily data)
        """
        if self.prices_df is None:
            raise DataLoadError(
                "No price data loaded. Call load_preprocessed_data() first."
            )

        logger.info("Calculating returns and statistics")

        # Use PyPortfolioOpt functions
        self.expected_returns = expected_returns.mean_historical_return(
            self.prices_df, frequency=frequency
        )
        self.cov_matrix = risk_models.sample_cov(self.prices_df, frequency=frequency)
        self.returns_df = expected_returns.returns_from_prices(self.prices_df)

        logger.info("Returns and statistics calculated successfully")
        logger.info(f"Expected returns:\n{self.expected_returns}")

        return self.expected_returns, self.cov_matrix

    def optimize_portfolio(
        self,
        method: str = "max_sharpe",
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        weight_bounds: Tuple[float, float] = (0, 1),
        **kwargs,
    ) -> Dict[str, float]:
        """
        Optimize portfolio using specified method.

        Args:
            method: Optimization method ('max_sharpe', 'min_volatility', 'efficient_return', 'efficient_risk')
            target_return: Target return for efficient_return method
            target_risk: Target risk for efficient_risk method
            weight_bounds: Weight bounds for assets
            **kwargs: Additional parameters for optimization

        Returns:
            Dictionary of optimized weights

        Raises:
            OptimizationError: If optimization fails
        """
        if self.expected_returns is None or self.cov_matrix is None:
            raise OptimizationError(
                "Returns and covariance not calculated. Call calculate_returns_and_statistics() first."
            )

        logger.info(f"Optimizing portfolio using method: {method}")

        try:
            # Use PyPortfolioOpt
            ef = EfficientFrontier(
                self.expected_returns, self.cov_matrix, weight_bounds=weight_bounds
            )

            if method == "max_sharpe":
                weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            elif method == "min_volatility":
                weights = ef.min_volatility()
            elif method == "efficient_return":
                if target_return is None:
                    raise ValueError(
                        "target_return required for efficient_return method"
                    )
                weights = ef.efficient_return(target_return)
            elif method == "efficient_risk":
                if target_risk is None:
                    raise ValueError("target_risk required for efficient_risk method")
                weights = ef.efficient_risk(target_risk)
            else:
                raise ValueError(f"Unknown optimization method: {method}")

            # Clean weights (remove tiny positions)
            weights = ef.clean_weights()

            # Calculate performance
            self.performance = ef.portfolio_performance(
                verbose=False, risk_free_rate=self.risk_free_rate
            )

            self.weights = weights
            logger.info("Portfolio optimization completed successfully")
            logger.info(f"Optimized weights:\n{weights}")

            if self.performance:
                exp_ret, volatility, sharpe = self.performance
                logger.info(f"Expected Return: {exp_ret:.4f}")
                logger.info(f"Volatility: {volatility:.4f}")
                logger.info(f"Sharpe Ratio: {sharpe:.4f}")

            return weights

        except Exception as e:
            raise OptimizationError(f"Portfolio optimization failed: {str(e)}")

    def calculate_efficient_frontier(self, num_points: int = 100) -> pd.DataFrame:
        """
        Calculate efficient frontier points.

        Args:
            num_points: Number of points to calculate on the frontier

        Returns:
            DataFrame with efficient frontier data
        """
        if self.expected_returns is None or self.cov_matrix is None:
            raise OptimizationError("Returns and covariance not calculated")

        logger.info(f"Calculating efficient frontier with {num_points} points")

        try:
            # Use PyPortfolioOpt implementation
            # Calculate range of returns
            ef_min = EfficientFrontier(self.expected_returns, self.cov_matrix)
            ef_min.min_volatility()
            min_ret = ef_min.portfolio_performance()[0]

            max_ret = self.expected_returns.max()

            # Generate return targets
            ret_range = np.linspace(min_ret, max_ret * 0.99, num_points)

            # Calculate efficient portfolios
            frontier_data = []
            for target_ret in ret_range:
                try:
                    ef = EfficientFrontier(self.expected_returns, self.cov_matrix)
                    ef.efficient_return(target_ret)
                    exp_ret, volatility, sharpe = ef.portfolio_performance(
                        risk_free_rate=self.risk_free_rate
                    )

                    frontier_data.append(
                        {
                            "return": exp_ret,
                            "volatility": volatility,
                            "sharpe_ratio": sharpe,
                        }
                    )
                except Exception:
                    continue  # Skip infeasible points

            self.efficient_frontier_data = pd.DataFrame(frontier_data)
            logger.info(
                f"Calculated {len(self.efficient_frontier_data)} efficient frontier points"
            )

        except Exception as e:
            raise OptimizationError(f"Efficient frontier calculation failed: {str(e)}")

        return self.efficient_frontier_data

    def save_results(
        self, filename_prefix: str = "portfolio_optimization"
    ) -> Dict[str, Path]:
        """
        Save optimization results to JSON and CSV files.

        Args:
            filename_prefix: Prefix for output filenames

        Returns:
            Dictionary of saved file paths
        """
        if self.weights is None:
            raise OptimizationError("No optimization results to save")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_prefix}_{timestamp}"

        saved_files = {}

        try:
            # Prepare results data
            results = {
                "timestamp": timestamp,
                "assets": self.assets,
                "weights": self.weights,
                "performance": {
                    "expected_return": self.performance[0]
                    if self.performance
                    else None,
                    "volatility": self.performance[1] if self.performance else None,
                    "sharpe_ratio": self.performance[2] if self.performance else None,
                }
                if self.performance
                else None,
                "risk_free_rate": self.risk_free_rate,
            }

            # Save JSON
            json_path = self.results_dir / f"{base_filename}.json"
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            saved_files["json"] = json_path
            logger.info(f"Results saved to JSON: {json_path}")

            # Save CSV (weights)
            csv_path = self.results_dir / f"{base_filename}_weights.csv"
            weights_df = pd.DataFrame(
                list(self.weights.items()), columns=["Asset", "Weight"]
            )
            weights_df.to_csv(csv_path, index=False)
            saved_files["csv"] = csv_path
            logger.info(f"Weights saved to CSV: {csv_path}")

            # Save efficient frontier if available
            if (
                self.efficient_frontier_data is not None
                and not self.efficient_frontier_data.empty
            ):
                ef_path = self.results_dir / f"{base_filename}_efficient_frontier.csv"
                self.efficient_frontier_data.to_csv(ef_path, index=False)
                saved_files["efficient_frontier"] = ef_path
                logger.info(f"Efficient frontier saved to CSV: {ef_path}")

            return saved_files

        except Exception as e:
            raise OptimizationError(f"Failed to save results: {str(e)}")


def optimize_portfolio_from_data(
    processed_data_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    method: str = "max_sharpe",
    risk_free_rate: float = 0.02,
    calculate_frontier: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to run complete portfolio optimization workflow.

    Args:
        processed_data_dir: Directory containing preprocessed data
        results_dir: Directory to save results
        method: Optimization method
        risk_free_rate: Risk-free rate
        calculate_frontier: Whether to calculate efficient frontier
        **kwargs: Additional optimization parameters

    Returns:
        Dictionary containing optimization results
    """
    try:
        # Initialize optimizer
        optimizer = PyPortfolioOptOptimizer(
            processed_data_dir=processed_data_dir,
            results_dir=results_dir,
            risk_free_rate=risk_free_rate,
        )

        # Load data
        optimizer.load_preprocessed_data()

        # Calculate returns and statistics
        expected_returns, cov_matrix = optimizer.calculate_returns_and_statistics()

        # Optimize portfolio
        weights = optimizer.optimize_portfolio(method=method, **kwargs)

        # Calculate efficient frontier if requested
        efficient_frontier = None
        if calculate_frontier:
            efficient_frontier = optimizer.calculate_efficient_frontier()

        # Save results
        saved_files = optimizer.save_results()

        return {
            "success": True,
            "weights": weights,
            "performance": optimizer.performance,
            "efficient_frontier": efficient_frontier,
            "saved_files": saved_files,
            "assets": optimizer.assets,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


if __name__ == "__main__":
    # Example usage
    results = optimize_portfolio_from_data(
        method="max_sharpe", risk_free_rate=0.02, calculate_frontier=True
    )

    if results["success"]:
        print("Portfolio optimization completed successfully!")
        print(f"Assets: {results['assets']}")
        print(f"Weights: {results['weights']}")
        if results["performance"]:
            exp_ret, vol, sharpe = results["performance"]
            print(f"Expected Return: {exp_ret:.4f}")
            print(f"Volatility: {vol:.4f}")
            print(f"Sharpe Ratio: {sharpe:.4f}")
        print(f"Files saved: {list(results['saved_files'].keys())}")
    else:
        print(f"Optimization failed: {results['error']}")
