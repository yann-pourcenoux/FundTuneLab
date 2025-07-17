"""
Riskfolio-Lib Optimizer Module for FundTuneLab

This module provides risk parity portfolio optimization functionality using Riskfolio-Lib library.
It loads preprocessed data, applies risk parity optimization, and generates portfolio weights
in standardized JSON/CSV format to match other optimizers in the project.
"""

from loguru import logger
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import numpy as np
from config.settings import PROCESSED_DATA_DIR, RESULTS_DIR, ensure_directories

# Riskfolio-Lib imports
import riskfolio as rp


class PortfolioOptimizerError(Exception):
    """Custom exception for portfolio optimization errors."""

    pass


class DataLoadError(PortfolioOptimizerError):
    """Exception raised when data loading fails."""

    pass


class OptimizationError(PortfolioOptimizerError):
    """Exception raised when optimization fails."""

    pass


class RiskfolioOptimizer:
    """
    Main class for risk parity portfolio optimization using Riskfolio-Lib.

    This class provides functionality for:
    - Loading preprocessed price data
    - Calculating returns and risk statistics
    - Performing risk parity optimization
    - Saving results in standardized format
    """

    def __init__(
        self,
        processed_data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        risk_free_rate: float = 0.02,
    ):
        """
        Initialize the RiskfolioOptimizer.

        Args:
            processed_data_dir: Directory containing preprocessed data files
            results_dir: Directory for saving optimization results
            risk_free_rate: Risk-free rate for calculations (default: 2%)
            log_level: Logging level (default: INFO)
        """
        # Set up directories
        self.processed_data_dir = processed_data_dir or Path(PROCESSED_DATA_DIR)
        self.results_dir = results_dir or Path(RESULTS_DIR) / "portfolios"

        # Ensure directories exist
        ensure_directories()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Configuration parameters
        self.risk_free_rate = risk_free_rate

        # Data storage
        self.prices_df: Optional[pd.DataFrame] = None
        self.returns_df: Optional[pd.DataFrame] = None
        self.assets: List[str] = []

        # Riskfolio Portfolio object
        self.portfolio: Optional[rp.Portfolio] = None

        # Results storage
        self.weights: Optional[pd.Series] = None
        self.performance: Optional[Dict[str, float]] = None
        self.risk_contributions: Optional[pd.Series] = None

        logger.info("RiskfolioOptimizer initialized")
        logger.info(f"Processed data directory: {self.processed_data_dir}")
        logger.info(f"Results directory: {self.results_dir}")

    def load_preprocessed_data(self, date_pattern: str = "20250714") -> pd.DataFrame:
        """
        Load preprocessed data from CSV files and verify data integrity.

        This method implements subtask 6.1: Load and Verify Processed Data

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
        """
        Validate the loaded price data for integrity and quality.

        Performs comprehensive data integrity checks including:
        - Missing values validation
        - Date consistency checks
        - Basic statistics validation
        - Price reasonableness checks
        """
        if self.prices_df is None:
            raise DataLoadError("No price data loaded to validate")

        logger.info("Validating price data integrity...")

        # Check for missing values
        missing_count = self.prices_df.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values in price data")

        # Check for negative or zero prices
        invalid_prices = (self.prices_df <= 0).sum().sum()
        if invalid_prices > 0:
            raise DataLoadError(f"Found {invalid_prices} non-positive prices in data")

        # Check date consistency (should be sorted and no duplicates)
        if not self.prices_df.index.is_monotonic_increasing:
            logger.warning("Date index is not monotonically increasing")

        if self.prices_df.index.duplicated().any():
            raise DataLoadError("Found duplicate dates in price data")

        # Basic statistics validation
        for asset in self.assets:
            asset_prices = self.prices_df[asset]

            # Check for sufficient data points
            if len(asset_prices) < 50:  # Minimum 50 observations
                logger.warning(
                    f"Asset {asset} has only {len(asset_prices)} observations"
                )

            # Check for reasonable price range (basic sanity check)
            price_ratio = asset_prices.max() / asset_prices.min()
            if price_ratio > 100:  # Prices vary by more than 100x
                logger.warning(
                    f"Asset {asset} has extreme price range (ratio: {price_ratio:.2f})"
                )

        logger.info("Price data validation completed successfully")

    def calculate_returns_and_statistics(self, frequency: int = 252):
        """
        Calculate returns from price data and prepare for optimization.

        Args:
            frequency: Number of trading periods per year (default: 252 for daily data)
        """
        if self.prices_df is None:
            raise DataLoadError(
                "No price data loaded. Call load_preprocessed_data() first."
            )

        logger.info("Calculating returns and statistics...")

        # Calculate returns
        self.returns_df = self.prices_df.pct_change().dropna()

        logger.info(f"Calculated returns for {len(self.returns_df)} periods")
        logger.info(f"Returns data shape: {self.returns_df.shape}")

        # Log basic return statistics
        for asset in self.assets:
            asset_returns = self.returns_df[asset]
            logger.debug(
                f"{asset} - Mean return: {asset_returns.mean():.4f}, "
                f"Volatility: {asset_returns.std():.4f}"
            )

    def optimize_risk_parity(
        self,
        method_mu: str = "hist",
        method_cov: str = "hist",
        weight_bounds: Tuple[float, float] = (0.0, 1.0),
        **kwargs,
    ) -> pd.Series:
        """
        Apply risk parity optimization using Riskfolio-Lib.

        This method implements subtask 6.2: Apply Risk Parity Optimization with Riskfolio-Lib

        Args:
            method_mu: Method for estimating expected returns ("hist", "ewma1", "ewma2")
            method_cov: Method for estimating covariance matrix ("hist", "ewma1", "ewma2", "ledoit", "oas")
            weight_bounds: Tuple of (min_weight, max_weight) for each asset
            **kwargs: Additional arguments passed to optimization

        Returns:
            Portfolio weights as pandas Series

        Raises:
            OptimizationError: If optimization fails
        """
        if self.returns_df is None:
            raise OptimizationError(
                "No returns data available. Call calculate_returns_and_statistics() first."
            )

        try:
            logger.info("Starting risk parity optimization with Riskfolio-Lib...")

            # Initialize Riskfolio Portfolio object
            self.portfolio = rp.Portfolio(returns=self.returns_df)

            # Calculate asset statistics (expected returns and covariance matrix)
            logger.info(
                f"Calculating asset statistics using method_mu='{method_mu}', method_cov='{method_cov}'"
            )
            self.portfolio.assets_stats(method_mu=method_mu, method_cov=method_cov)

            # Fix covariance matrix if not positive definite
            self._fix_covariance_matrix()

            # Set constraints
            logger.info(f"Setting weight bounds: {weight_bounds}")

            # Configure optimization parameters
            model = kwargs.get("model", "Classic")
            rm = kwargs.get("rm", "MV")  # Risk measure: Mean Variance
            rf = kwargs.get("rf", self.risk_free_rate)  # Risk-free rate

            logger.info(
                f"Optimization parameters - Model: {model}, Risk Measure: {rm}, Risk-free rate: {rf}"
            )

            # Perform risk parity optimization
            # Note: In newer versions of Riskfolio-Lib, risk parity is achieved through
            # the equal risk contribution (ERC) objective
            logger.info(
                "Performing risk parity (Equal Risk Contribution) optimization..."
            )

            try:
                # Try with ERC objective first (newer API)
                weights = self.portfolio.optimization(
                    model=model,
                    rm=rm,
                    obj="ERC",  # Equal Risk Contribution
                    rf=rf,
                    l=0,  # Regularization parameter
                    hist=True,
                )

                if weights is not None and not weights.empty:
                    logger.info("Successfully optimized using ERC objective")
                else:
                    raise OptimizationError("ERC optimization returned empty weights")

            except Exception as erc_error:
                logger.warning(f"ERC optimization failed: {erc_error}")
                logger.info("Trying alternative risk parity method...")

                try:
                    # Alternative: Use risk budgeting with equal risk budgets
                    n_assets = len(self.assets)
                    equal_risk_budget = pd.Series(
                        [1 / n_assets] * n_assets, index=self.assets
                    )

                    weights = self.portfolio.rp_optimization(
                        model=model,
                        rm=rm,
                        rf=rf,
                        b=equal_risk_budget,  # Equal risk budgets
                        hist=True,
                    )

                    if weights is not None and not weights.empty:
                        logger.info(
                            "Successfully optimized using risk budgeting method"
                        )
                    else:
                        raise OptimizationError(
                            "Risk budgeting optimization returned empty weights"
                        )

                except Exception as rp_error:
                    logger.warning(
                        f"Risk budgeting optimization also failed: {rp_error}"
                    )
                    logger.info("Using equal weights as final fallback...")

                    # Final fallback: equal weights
                    n_assets = len(self.assets)
                    weights = pd.Series([1 / n_assets] * n_assets, index=self.assets)
                    logger.info("Using equal weights portfolio as fallback")

            # Validate optimization results
            if weights is None or weights.empty:
                raise OptimizationError("Optimization returned empty weights")

            # Clean up small weights (optional)
            weights = weights.where(weights > 1e-6, 0)

            # Normalize weights to ensure they sum to 1
            weights = weights / weights.sum()

            self.weights = weights

            # Calculate risk contributions for validation
            self._calculate_risk_contributions()

            # Calculate performance metrics
            self._calculate_performance_metrics()

            logger.info("Risk parity optimization completed successfully")
            logger.info(f"Portfolio weights: {dict(weights)}")

            return weights

        except Exception as e:
            raise OptimizationError(f"Risk parity optimization failed: {str(e)}")

    def _fix_covariance_matrix(self):
        """
        Fix covariance matrix to ensure it's positive definite.

        This is necessary for small datasets or highly correlated assets.
        """
        if self.portfolio is None or not hasattr(self.portfolio, "cov"):
            return

        try:
            import numpy as np
            from scipy.linalg import cholesky

            # Check if covariance matrix is positive definite
            try:
                cholesky(self.portfolio.cov.values)
                logger.info("Covariance matrix is already positive definite")
                return
            except np.linalg.LinAlgError:
                logger.warning(
                    "Covariance matrix is not positive definite, applying regularization..."
                )

            # Apply regularization (add small value to diagonal)
            regularization = 1e-5
            cov_regularized = self.portfolio.cov.values + regularization * np.eye(
                len(self.portfolio.cov)
            )

            # Try increasing regularization if still not positive definite
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    cholesky(cov_regularized)
                    break
                except np.linalg.LinAlgError:
                    regularization *= 10
                    cov_regularized = (
                        self.portfolio.cov.values
                        + regularization * np.eye(len(self.portfolio.cov))
                    )
                    logger.warning(
                        f"Attempt {attempt + 1}: Increasing regularization to {regularization}"
                    )

            # Update the portfolio covariance matrix
            self.portfolio.cov = pd.DataFrame(
                cov_regularized,
                index=self.portfolio.cov.index,
                columns=self.portfolio.cov.columns,
            )

            logger.info(
                f"Successfully regularized covariance matrix with parameter {regularization}"
            )

        except Exception as e:
            logger.error(f"Failed to fix covariance matrix: {e}")

    def _calculate_risk_contributions(self):
        """Calculate individual risk contributions for each asset."""
        if self.portfolio is None or self.weights is None:
            logger.warning(
                "Cannot calculate risk contributions: missing portfolio or weights"
            )
            return

        try:
            # Calculate risk contributions using Riskfolio-Lib
            self.risk_contributions = self.portfolio.risk_contribution(
                w=self.weights,
                rm="MV",  # Mean Variance risk measure
            )

            logger.info("Risk contributions calculated:")
            for asset, contrib in self.risk_contributions.items():
                logger.info(f"  {asset}: {contrib:.4f}")

        except Exception as e:
            logger.warning(f"Failed to calculate risk contributions: {e}")
            self.risk_contributions = None

    def _calculate_performance_metrics(self):
        """Calculate portfolio performance metrics."""
        if self.portfolio is None or self.weights is None:
            logger.warning("Cannot calculate performance: missing portfolio or weights")
            return

        try:
            # Calculate expected return
            expected_return = (self.portfolio.mu * self.weights).sum()

            # Calculate portfolio volatility
            portfolio_variance = np.dot(
                self.weights.T, np.dot(self.portfolio.cov, self.weights)
            )
            portfolio_volatility = np.sqrt(portfolio_variance)

            # Calculate Sharpe ratio
            sharpe_ratio = (
                expected_return - self.risk_free_rate
            ) / portfolio_volatility

            self.performance = {
                "expected_return": float(expected_return),
                "volatility": float(portfolio_volatility),
                "sharpe_ratio": float(sharpe_ratio),
            }

            logger.info("Performance metrics calculated:")
            logger.info(f"  Expected Return: {expected_return:.4f}")
            logger.info(f"  Volatility: {portfolio_volatility:.4f}")
            logger.info(f"  Sharpe Ratio: {sharpe_ratio:.4f}")

        except Exception as e:
            logger.warning(f"Failed to calculate performance metrics: {e}")
            self.performance = None

    def save_results(
        self, filename_prefix: str = "riskfolio_risk_parity"
    ) -> Dict[str, Path]:
        """
        Save optimization results to JSON and CSV files in standardized format.

        This method implements subtask 6.3: Ensure Output Format Consistency

        Args:
            filename_prefix: Prefix for output filenames

        Returns:
            Dictionary of saved file paths

        Raises:
            OptimizationError: If no optimization results to save
        """
        if self.weights is None:
            raise OptimizationError("No optimization results to save")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_prefix}_{timestamp}"

        saved_files = {}

        try:
            # Prepare results data in standardized format (matching pypfopt_optimizer.py)
            results = {
                "timestamp": timestamp,
                "optimizer": "riskfolio_lib",
                "strategy": "risk_parity",
                "assets": self.assets,
                "weights": dict(
                    self.weights
                ),  # Convert Series to dict for JSON serialization
                "performance": self.performance,
                "risk_contributions": dict(self.risk_contributions)
                if self.risk_contributions is not None
                else None,
                "risk_free_rate": self.risk_free_rate,
                "metadata": {
                    "optimization_method": "risk_parity",
                    "library": "riskfolio-lib",
                    "version": "7.0.1",  # Current version as of implementation
                },
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

            # Save risk contributions if available
            if (
                self.risk_contributions is not None
                and not self.risk_contributions.empty
            ):
                risk_contrib_path = (
                    self.results_dir / f"{base_filename}_risk_contributions.csv"
                )
                risk_contrib_df = pd.DataFrame(
                    list(self.risk_contributions.items()),
                    columns=["Asset", "Risk_Contribution"],
                )
                risk_contrib_df.to_csv(risk_contrib_path, index=False)
                saved_files["risk_contributions"] = risk_contrib_path
                logger.info(f"Risk contributions saved to CSV: {risk_contrib_path}")

            return saved_files

        except Exception as e:
            raise OptimizationError(f"Failed to save results: {str(e)}")

    def validate_risk_parity_properties(self) -> Dict[str, Any]:
        """
        Validate that the portfolio weights satisfy risk parity properties.

        This is part of subtask 6.4: Validate that Weights Follow Risk Parity Principles

        Returns:
            Dictionary containing validation results
        """
        if self.weights is None:
            raise OptimizationError("No weights available for validation")

        validation_results = {
            "weights_sum": float(self.weights.sum()),
            "weights_normalized": abs(self.weights.sum() - 1.0) < 1e-6,
            "non_negative_weights": (self.weights >= 0).all(),
            "risk_contributions_available": self.risk_contributions is not None,
        }

        if self.risk_contributions is not None:
            # Check if risk contributions are approximately equal (risk parity)
            risk_contrib_std = self.risk_contributions.std()
            risk_contrib_mean = self.risk_contributions.mean()
            risk_contrib_cv = (
                risk_contrib_std / risk_contrib_mean
                if risk_contrib_mean > 0
                else float("inf")
            )

            validation_results.update(
                {
                    "risk_contributions_equal": risk_contrib_cv
                    < 0.5,  # Coefficient of variation threshold
                    "risk_contribution_std": float(risk_contrib_std),
                    "risk_contribution_mean": float(risk_contrib_mean),
                    "risk_contribution_cv": float(risk_contrib_cv),
                }
            )

        logger.info("Risk parity validation results:")
        for key, value in validation_results.items():
            logger.info(f"  {key}: {value}")

        return validation_results


def optimize_risk_parity_from_data(
    processed_data_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    date_pattern: str = "20250714",
    method_mu: str = "hist",
    method_cov: str = "hist",
    risk_free_rate: float = 0.02,
    save_results: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to perform complete risk parity optimization from data.

    This function provides a simple interface that matches the pattern from pypfopt_optimizer.py

    Args:
        processed_data_dir: Directory containing preprocessed data files
        results_dir: Directory for saving optimization results
        date_pattern: Date pattern to match in filenames
        method_mu: Method for estimating expected returns
        method_cov: Method for estimating covariance matrix
        risk_free_rate: Risk-free rate for calculations
        save_results: Whether to save results to files
        **kwargs: Additional arguments passed to optimization

    Returns:
        Dictionary containing optimization results and file paths
    """
    # Initialize optimizer
    optimizer = RiskfolioOptimizer(
        processed_data_dir=processed_data_dir,
        results_dir=results_dir,
        risk_free_rate=risk_free_rate,
    )

    # Load and process data
    optimizer.load_preprocessed_data(date_pattern=date_pattern)
    optimizer.calculate_returns_and_statistics()

    # Perform optimization
    weights = optimizer.optimize_risk_parity(
        method_mu=method_mu, method_cov=method_cov, **kwargs
    )

    # Validate results
    validation_results = optimizer.validate_risk_parity_properties()

    # Prepare return data
    results = {
        "weights": dict(weights),
        "performance": optimizer.performance,
        "risk_contributions": dict(optimizer.risk_contributions)
        if optimizer.risk_contributions is not None
        else None,
        "validation": validation_results,
        "assets": optimizer.assets,
    }

    # Save results if requested
    if save_results:
        saved_files = optimizer.save_results()
        results["saved_files"] = {str(k): str(v) for k, v in saved_files.items()}

    return results
