"""
Eiten Optimizer Module for FundTuneLab

This module provides portfolio optimization functionality using statistical methods
inspired by the Eiten library. It implements Eigen Portfolios, Minimum Variance,
Maximum Sharpe Ratio, and Genetic Algorithm-based portfolio optimization.
"""

import logging
import warnings
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import numpy as np
from scipy import linalg
from scipy.optimize import minimize, differential_evolution
from config.settings import PROCESSED_DATA_DIR, RESULTS_DIR, ensure_directories

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class PortfolioOptimizerError(Exception):
    """Custom exception for portfolio optimization errors."""

    pass


class DataLoadError(PortfolioOptimizerError):
    """Exception raised when data loading fails."""

    pass


class OptimizationError(PortfolioOptimizerError):
    """Exception raised when optimization fails."""

    pass


class EitenOptimizer:
    """
    Main class for portfolio optimization using Eiten-inspired statistical methods.

    This class provides functionality for:
    - Eigen Portfolio construction using eigenvalue decomposition
    - Minimum Variance Portfolio optimization
    - Maximum Sharpe Ratio Portfolio optimization
    - Genetic Algorithm-based portfolio optimization
    - Noise filtering using Random Matrix Theory
    """

    def __init__(
        self,
        processed_data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        risk_free_rate: float = 0.02,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the Eiten Optimizer.

        Args:
            processed_data_dir: Directory containing processed data files
            results_dir: Directory to save optimization results
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
            log_level: Logging level
        """
        self.processed_data_dir = processed_data_dir or PROCESSED_DATA_DIR
        self.results_dir = results_dir or RESULTS_DIR
        self.risk_free_rate = risk_free_rate

        # Ensure directories exist
        ensure_directories()

        # Setup logging
        self.logger = self._setup_logging(log_level)

        # Initialize data containers
        self.price_data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.filtered_cov_matrix = None
        self.symbols = None

        # Results storage
        self.optimization_results = {}

        self.logger.info("EitenOptimizer initialized successfully")

    def _setup_logging(self, log_level: int) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        logger.setLevel(log_level)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def load_preprocessed_data(self, date_pattern: str = "20250714") -> pd.DataFrame:
        """
        Load preprocessed data files and combine into a single DataFrame.

        Args:
            date_pattern: Date pattern to identify processed files

        Returns:
            Combined DataFrame with price data for all assets

        Raises:
            DataLoadError: If no data files found or loading fails
        """
        try:
            self.logger.info(f"Loading preprocessed data with pattern: {date_pattern}")

            # Find all processed CSV files matching the date pattern
            data_files = list(
                self.processed_data_dir.glob(f"*_processed_{date_pattern}.csv")
            )

            if not data_files:
                raise DataLoadError(
                    f"No processed data files found with pattern: {date_pattern}"
                )

            self.logger.info(f"Found {len(data_files)} data files")

            # Load and combine data
            combined_data = []
            symbols = []

            for file_path in data_files:
                self.logger.debug(f"Loading file: {file_path}")
                df = pd.read_csv(file_path)

                # Extract symbol from filename or Symbol column
                if "Symbol" in df.columns:
                    symbol = df["Symbol"].iloc[0]
                else:
                    symbol = file_path.stem.split("_")[0]  # Extract from filename

                symbols.append(symbol)

                # Ensure Date column is datetime
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")

                # Select close price and rename column to symbol
                close_prices = df["Close"].rename(symbol)
                combined_data.append(close_prices)

            # Combine all data
            self.price_data = pd.concat(combined_data, axis=1)
            self.symbols = symbols

            # Remove any rows with NaN values
            initial_rows = len(self.price_data)
            self.price_data = self.price_data.dropna()
            final_rows = len(self.price_data)

            if final_rows < initial_rows:
                self.logger.warning(
                    f"Removed {initial_rows - final_rows} rows with missing data"
                )

            if len(self.price_data) < 10:
                raise DataLoadError(
                    "Insufficient data points after cleaning (< 10 observations)"
                )

            self.logger.info(f"Successfully loaded data for {len(self.symbols)} assets")
            self.logger.info(
                f"Date range: {self.price_data.index.min()} to {self.price_data.index.max()}"
            )
            self.logger.info(f"Total observations: {len(self.price_data)}")

            return self.price_data

        except Exception as e:
            error_msg = f"Failed to load preprocessed data: {str(e)}"
            self.logger.error(error_msg)
            raise DataLoadError(error_msg)

    def _validate_price_data(self):
        """Validate that price data is loaded and valid."""
        if self.price_data is None:
            raise DataLoadError(
                "No price data loaded. Call load_preprocessed_data() first."
            )

        if self.price_data.empty:
            raise DataLoadError("Price data is empty")

        if len(self.price_data.columns) < 2:
            raise DataLoadError("Need at least 2 assets for portfolio optimization")

    def calculate_returns_and_statistics(self, frequency: int = 252):
        """
        Calculate returns and statistical measures.

        Args:
            frequency: Number of trading periods per year (252 for daily data)
        """
        self._validate_price_data()

        try:
            self.logger.info("Calculating returns and statistics")

            # Calculate returns
            self.returns = self.price_data.pct_change().dropna()

            # Calculate mean returns (annualized)
            self.mean_returns = self.returns.mean() * frequency

            # Calculate covariance matrix (annualized)
            self.cov_matrix = self.returns.cov() * frequency

            self.logger.info(f"Calculated returns for {len(self.symbols)} assets")
            self.logger.info(
                f"Mean returns range: {self.mean_returns.min():.4f} to {self.mean_returns.max():.4f}"
            )

        except Exception as e:
            error_msg = f"Failed to calculate returns and statistics: {str(e)}"
            self.logger.error(error_msg)
            raise OptimizationError(error_msg)

    def apply_noise_filtering(self, apply_filtering: bool = True):
        """
        Apply noise filtering to covariance matrix using Random Matrix Theory.

        Args:
            apply_filtering: Whether to apply noise filtering
        """
        if not apply_filtering:
            self.filtered_cov_matrix = self.cov_matrix.copy()
            return

        try:
            self.logger.info("Applying noise filtering using Random Matrix Theory")

            # Get eigenvalues and eigenvectors
            eigenvals, eigenvecs = linalg.eigh(self.cov_matrix.values)

            # Sort eigenvalues in descending order
            idx = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]

            # Calculate theoretical bounds for random eigenvalues
            N = len(self.symbols)  # Number of assets
            T = len(self.returns)  # Number of observations
            q = N / T  # Ratio

            # Marcenko-Pastur distribution bounds
            lambda_max = (1 + np.sqrt(q)) ** 2

            # Filter out noise eigenvalues
            signal_eigenvals = eigenvals[eigenvals > lambda_max]
            noise_eigenvals = eigenvals[eigenvals <= lambda_max]

            # Replace noise eigenvalues with their mean
            if len(noise_eigenvals) > 0:
                noise_mean = np.mean(noise_eigenvals)
                eigenvals[eigenvals <= lambda_max] = noise_mean

            # Reconstruct filtered covariance matrix
            filtered_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

            # Ensure positive definiteness
            min_eigenval = np.min(linalg.eigvals(filtered_cov))
            if min_eigenval < 1e-8:
                filtered_cov += np.eye(N) * (1e-8 - min_eigenval)

            self.filtered_cov_matrix = pd.DataFrame(
                filtered_cov,
                index=self.cov_matrix.index,
                columns=self.cov_matrix.columns,
            )

            self.logger.info(
                f"Noise filtering complete. Signal eigenvalues: {len(signal_eigenvals)}"
            )

        except Exception as e:
            self.logger.warning(
                f"Noise filtering failed: {str(e)}. Using original covariance matrix."
            )
            self.filtered_cov_matrix = self.cov_matrix.copy()

    def calculate_eigen_portfolio(self, portfolio_number: int = 2) -> Dict[str, float]:
        """
        Calculate Eigen Portfolio using eigenvalue decomposition.

        Args:
            portfolio_number: Which eigen portfolio to use (1-5, where 1 is market portfolio)

        Returns:
            Dictionary with asset weights
        """
        try:
            self.logger.info(f"Calculating Eigen Portfolio #{portfolio_number}")

            if self.filtered_cov_matrix is None:
                self.apply_noise_filtering(True)

            # Get eigenvalues and eigenvectors
            eigenvals, eigenvecs = linalg.eigh(self.filtered_cov_matrix.values)

            # Sort in descending order
            idx = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]

            # Select the specified eigen portfolio
            if portfolio_number > len(eigenvals):
                raise OptimizationError(
                    f"Portfolio number {portfolio_number} exceeds available eigenvectors"
                )

            weights = eigenvecs[:, portfolio_number - 1]

            # Normalize weights to sum to 1 and make long-only
            weights = np.abs(weights)  # Make long-only
            weights = weights / np.sum(weights)  # Normalize

            # Create results dictionary
            portfolio_weights = dict(zip(self.symbols, weights))

            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, self.mean_returns)
            portfolio_vol = np.sqrt(
                np.dot(weights, np.dot(self.filtered_cov_matrix, weights))
            )
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol

            results = {
                "weights": portfolio_weights,
                "expected_return": portfolio_return,
                "volatility": portfolio_vol,
                "sharpe_ratio": sharpe_ratio,
                "portfolio_number": portfolio_number,
                "eigenvalue": eigenvals[portfolio_number - 1],
            }

            self.optimization_results["eigen_portfolio"] = results

            self.logger.info(f"Eigen Portfolio #{portfolio_number} completed")
            self.logger.info(f"Expected Return: {portfolio_return:.4f}")
            self.logger.info(f"Volatility: {portfolio_vol:.4f}")
            self.logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")

            return portfolio_weights

        except Exception as e:
            error_msg = f"Failed to calculate Eigen Portfolio: {str(e)}"
            self.logger.error(error_msg)
            raise OptimizationError(error_msg)

    def optimize_minimum_variance(
        self, weight_bounds: Tuple[float, float] = (0, 1)
    ) -> Dict[str, float]:
        """
        Optimize for minimum variance portfolio.

        Args:
            weight_bounds: Bounds for individual asset weights

        Returns:
            Dictionary with asset weights
        """
        try:
            self.logger.info("Optimizing Minimum Variance Portfolio")

            if self.filtered_cov_matrix is None:
                self.apply_noise_filtering(True)

            n_assets = len(self.symbols)

            # Objective function: minimize portfolio variance
            def objective(weights):
                return np.dot(weights, np.dot(self.filtered_cov_matrix.values, weights))

            # Constraints: weights sum to 1
            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

            # Bounds for weights
            bounds = [weight_bounds] * n_assets

            # Initial guess: equal weights
            x0 = np.array([1 / n_assets] * n_assets)

            # Optimize
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )

            if not result.success:
                raise OptimizationError(f"Optimization failed: {result.message}")

            weights = result.x
            portfolio_weights = dict(zip(self.symbols, weights))

            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, self.mean_returns)
            portfolio_vol = np.sqrt(objective(weights))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol

            results = {
                "weights": portfolio_weights,
                "expected_return": portfolio_return,
                "volatility": portfolio_vol,
                "sharpe_ratio": sharpe_ratio,
            }

            self.optimization_results["minimum_variance"] = results

            self.logger.info("Minimum Variance Portfolio optimization completed")
            self.logger.info(f"Expected Return: {portfolio_return:.4f}")
            self.logger.info(f"Volatility: {portfolio_vol:.4f}")
            self.logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")

            return portfolio_weights

        except Exception as e:
            error_msg = f"Failed to optimize Minimum Variance Portfolio: {str(e)}"
            self.logger.error(error_msg)
            raise OptimizationError(error_msg)

    def optimize_max_sharpe(
        self, weight_bounds: Tuple[float, float] = (0, 1)
    ) -> Dict[str, float]:
        """
        Optimize for maximum Sharpe ratio portfolio.

        Args:
            weight_bounds: Bounds for individual asset weights

        Returns:
            Dictionary with asset weights
        """
        try:
            self.logger.info("Optimizing Maximum Sharpe Ratio Portfolio")

            if self.filtered_cov_matrix is None:
                self.apply_noise_filtering(True)

            n_assets = len(self.symbols)

            # Objective function: minimize negative Sharpe ratio
            def objective(weights):
                portfolio_return = np.dot(weights, self.mean_returns)
                portfolio_vol = np.sqrt(
                    np.dot(weights, np.dot(self.filtered_cov_matrix.values, weights))
                )
                if portfolio_vol == 0:
                    return -np.inf
                return -(portfolio_return - self.risk_free_rate) / portfolio_vol

            # Constraints: weights sum to 1
            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

            # Bounds for weights
            bounds = [weight_bounds] * n_assets

            # Initial guess: equal weights
            x0 = np.array([1 / n_assets] * n_assets)

            # Optimize
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )

            if not result.success:
                raise OptimizationError(f"Optimization failed: {result.message}")

            weights = result.x
            portfolio_weights = dict(zip(self.symbols, weights))

            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, self.mean_returns)
            portfolio_vol = np.sqrt(
                np.dot(weights, np.dot(self.filtered_cov_matrix.values, weights))
            )
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol

            results = {
                "weights": portfolio_weights,
                "expected_return": portfolio_return,
                "volatility": portfolio_vol,
                "sharpe_ratio": sharpe_ratio,
            }

            self.optimization_results["max_sharpe"] = results

            self.logger.info("Maximum Sharpe Ratio Portfolio optimization completed")
            self.logger.info(f"Expected Return: {portfolio_return:.4f}")
            self.logger.info(f"Volatility: {portfolio_vol:.4f}")
            self.logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")

            return portfolio_weights

        except Exception as e:
            error_msg = f"Failed to optimize Maximum Sharpe Ratio Portfolio: {str(e)}"
            self.logger.error(error_msg)
            raise OptimizationError(error_msg)

    def optimize_genetic_algorithm(
        self,
        weight_bounds: Tuple[float, float] = (0, 1),
        maxiter: int = 1000,
        popsize: int = 15,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Optimize portfolio using Genetic Algorithm for maximum Sharpe ratio.

        Args:
            weight_bounds: Bounds for individual asset weights
            maxiter: Maximum number of iterations
            popsize: Population size for genetic algorithm
            seed: Random seed for reproducibility

        Returns:
            Dictionary with asset weights
        """
        try:
            self.logger.info("Optimizing portfolio using Genetic Algorithm")

            if self.filtered_cov_matrix is None:
                self.apply_noise_filtering(True)

            n_assets = len(self.symbols)

            # Objective function: minimize negative Sharpe ratio
            def objective(weights):
                # Normalize weights to sum to 1
                weights = weights / np.sum(weights)

                portfolio_return = np.dot(weights, self.mean_returns)
                portfolio_vol = np.sqrt(
                    np.dot(weights, np.dot(self.filtered_cov_matrix.values, weights))
                )

                if portfolio_vol == 0:
                    return 1e6  # Large penalty for zero volatility

                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
                return -sharpe  # Minimize negative Sharpe ratio

            # Bounds for all weights
            bounds = [weight_bounds] * n_assets

            # Use differential evolution (genetic algorithm variant)
            result = differential_evolution(
                objective,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                seed=seed,
                disp=False,
            )

            if not result.success:
                self.logger.warning(
                    f"Genetic algorithm optimization warning: {result.message}"
                )

            # Normalize weights
            weights = result.x / np.sum(result.x)
            portfolio_weights = dict(zip(self.symbols, weights))

            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, self.mean_returns)
            portfolio_vol = np.sqrt(
                np.dot(weights, np.dot(self.filtered_cov_matrix.values, weights))
            )
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol

            results = {
                "weights": portfolio_weights,
                "expected_return": portfolio_return,
                "volatility": portfolio_vol,
                "sharpe_ratio": sharpe_ratio,
                "iterations": result.nit,
                "function_evaluations": result.nfev,
            }

            self.optimization_results["genetic_algorithm"] = results

            self.logger.info("Genetic Algorithm optimization completed")
            self.logger.info(f"Expected Return: {portfolio_return:.4f}")
            self.logger.info(f"Volatility: {portfolio_vol:.4f}")
            self.logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
            self.logger.info(
                f"Iterations: {result.nit}, Function evaluations: {result.nfev}"
            )

            return portfolio_weights

        except Exception as e:
            error_msg = f"Failed to optimize using Genetic Algorithm: {str(e)}"
            self.logger.error(error_msg)
            raise OptimizationError(error_msg)

    def save_results(self, filename_prefix: str = "eiten_optimizer") -> Dict[str, Path]:
        """
        Save optimization results to JSON and CSV files.

        Args:
            filename_prefix: Prefix for output filenames

        Returns:
            Dictionary with paths to saved files
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Prepare results for saving
            save_data = {
                "metadata": {
                    "timestamp": timestamp,
                    "optimizer": "EitenOptimizer",
                    "risk_free_rate": self.risk_free_rate,
                    "symbols": self.symbols,
                    "data_points": len(self.price_data)
                    if self.price_data is not None
                    else 0,
                },
                "optimization_results": self.optimization_results,
            }

            # Save JSON file
            json_filename = f"{filename_prefix}_{timestamp}.json"
            json_path = self.results_dir / "portfolios" / json_filename

            with open(json_path, "w") as f:
                json.dump(save_data, f, indent=2, default=str)

            self.logger.info(f"Results saved to {json_path}")

            # Save CSV files for each optimization method
            csv_paths = {}

            for method, results in self.optimization_results.items():
                if "weights" in results:
                    csv_filename = f"{filename_prefix}_{method}_{timestamp}_weights.csv"
                    csv_path = self.results_dir / "portfolios" / csv_filename

                    # Create DataFrame with weights
                    weights_df = pd.DataFrame.from_dict(
                        results["weights"], orient="index", columns=["Weight"]
                    )
                    weights_df.index.name = "Symbol"
                    weights_df["Method"] = method.replace("_", " ").title()

                    # Add performance metrics as additional columns
                    for metric in ["expected_return", "volatility", "sharpe_ratio"]:
                        if metric in results:
                            weights_df[metric.replace("_", " ").title()] = results[
                                metric
                            ]

                    weights_df.to_csv(csv_path)
                    csv_paths[method] = csv_path

                    self.logger.info(f"Weights for {method} saved to {csv_path}")

            return {"json": json_path, **csv_paths}

        except Exception as e:
            error_msg = f"Failed to save results: {str(e)}"
            self.logger.error(error_msg)
            raise PortfolioOptimizerError(error_msg)


def optimize_eiten_portfolios(
    processed_data_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    date_pattern: str = "20250714",
    risk_free_rate: float = 0.02,
    apply_noise_filtering: bool = True,
    eigen_portfolio_number: int = 2,
    weight_bounds: Tuple[float, float] = (0, 1),
    genetic_algorithm_params: Optional[Dict] = None,
    save_results: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to run all Eiten optimization methods.

    Args:
        processed_data_dir: Directory containing processed data files
        results_dir: Directory to save optimization results
        date_pattern: Date pattern to identify processed files
        risk_free_rate: Risk-free rate for Sharpe ratio calculations
        apply_noise_filtering: Whether to apply noise filtering to covariance matrix
        eigen_portfolio_number: Which eigen portfolio to use (1-5)
        weight_bounds: Bounds for individual asset weights
        genetic_algorithm_params: Parameters for genetic algorithm optimization
        save_results: Whether to save results to files
        **kwargs: Additional arguments

    Returns:
        Dictionary containing all optimization results
    """

    # Initialize optimizer
    optimizer = EitenOptimizer(
        processed_data_dir=processed_data_dir,
        results_dir=results_dir,
        risk_free_rate=risk_free_rate,
    )

    # Load data and calculate statistics
    optimizer.load_preprocessed_data(date_pattern)
    optimizer.calculate_returns_and_statistics()
    optimizer.apply_noise_filtering(apply_noise_filtering)

    # Set default genetic algorithm parameters
    if genetic_algorithm_params is None:
        genetic_algorithm_params = {"maxiter": 1000, "popsize": 15, "seed": 42}

    # Run all optimization methods
    results = {}

    try:
        # Eigen Portfolio
        results["eigen_portfolio"] = optimizer.calculate_eigen_portfolio(
            eigen_portfolio_number
        )

        # Minimum Variance Portfolio
        results["minimum_variance"] = optimizer.optimize_minimum_variance(weight_bounds)

        # Maximum Sharpe Ratio Portfolio
        results["max_sharpe"] = optimizer.optimize_max_sharpe(weight_bounds)

        # Genetic Algorithm Portfolio
        results["genetic_algorithm"] = optimizer.optimize_genetic_algorithm(
            weight_bounds=weight_bounds, **genetic_algorithm_params
        )

        # Save results if requested
        if save_results:
            file_paths = optimizer.save_results()
            results["file_paths"] = file_paths

        # Add summary information
        results["summary"] = {
            "optimization_methods": list(optimizer.optimization_results.keys()),
            "symbols": optimizer.symbols,
            "data_points": len(optimizer.price_data),
            "date_range": {
                "start": str(optimizer.price_data.index.min()),
                "end": str(optimizer.price_data.index.max()),
            },
        }

        return results

    except Exception as e:
        error_msg = f"Portfolio optimization failed: {str(e)}"
        optimizer.logger.error(error_msg)
        raise OptimizationError(error_msg)
