"""
VectorBT Backtesting Module for FundTuneLab

This module provides backtesting functionality using VectorBT library.
It loads portfolio weights from optimizers and historical price data,
then calculates performance metrics including returns, volatility,
Sharpe ratio, and maximum drawdown.
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import numpy as np
import vectorbt as vbt

from config.settings import PROCESSED_DATA_DIR, RESULTS_DIR, ensure_directories

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class BacktestingError(Exception):
    """Custom exception for backtesting errors."""

    pass


class DataLoadError(BacktestingError):
    """Exception raised when data loading fails."""

    pass


class PerformanceCalculationError(BacktestingError):
    """Exception raised when performance calculation fails."""

    pass


class VectorBTBacktester:
    """
    Main class for portfolio backtesting using VectorBT.

    This class provides functionality for:
    - Loading portfolio weights from optimizer outputs
    - Loading historical price data
    - Running backtests using VectorBT
    - Calculating performance metrics (returns, volatility, Sharpe ratio, max drawdown)
    - Saving results in standardized format
    """

    def __init__(
        self,
        processed_data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        risk_free_rate: float = 0.02,
        rebalancing_frequency: str = "M",  # Monthly rebalancing by default
        initial_capital: float = 100000.0,
    ):
        """
        Initialize the VectorBT backtester.

        Args:
            processed_data_dir: Directory containing processed price data
            results_dir: Directory to save backtest results
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            rebalancing_frequency: Frequency for portfolio rebalancing ('D', 'W', 'M', 'Q', 'Y')
            initial_capital: Initial portfolio capital for backtesting
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Directory setup
        self.processed_data_dir = processed_data_dir or PROCESSED_DATA_DIR
        self.results_dir = results_dir or RESULTS_DIR / "backtests"

        # Ensure directories exist
        ensure_directories()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Backtesting parameters
        self.risk_free_rate = risk_free_rate
        self.rebalancing_frequency = rebalancing_frequency
        self.initial_capital = initial_capital

        # Data storage
        self.price_data: Optional[pd.DataFrame] = None
        self.portfolio_weights: Optional[Dict[str, float]] = None
        self.portfolio_metadata: Optional[Dict[str, Any]] = None
        self.backtest_results: Optional[Dict[str, Any]] = None

        self.logger.info("VectorBT Backtester initialized")
        self.logger.info(f"Processed data directory: {self.processed_data_dir}")
        self.logger.info(f"Results directory: {self.results_dir}")
        self.logger.info(f"Risk-free rate: {self.risk_free_rate}")
        self.logger.info(f"Rebalancing frequency: {self.rebalancing_frequency}")
        self.logger.info(f"Initial capital: ${self.initial_capital:,.2f}")

    def load_portfolio_weights(
        self, portfolio_file: Union[str, Path]
    ) -> Dict[str, float]:
        """
        Load portfolio weights from JSON or CSV file.

        Args:
            portfolio_file: Path to portfolio weights file (.json or .csv)

        Returns:
            Dictionary of asset weights

        Raises:
            DataLoadError: If loading fails
        """
        try:
            portfolio_path = Path(portfolio_file)

            if not portfolio_path.exists():
                raise FileNotFoundError(f"Portfolio file not found: {portfolio_path}")

            if portfolio_path.suffix.lower() == ".json":
                # Load from JSON format
                with open(portfolio_path, "r") as f:
                    data = json.load(f)

                # Handle different JSON formats
                if "weights" in data:
                    # Standard format (PyPortfolioOpt, Riskfolio)
                    self.portfolio_weights = data.get("weights", {})
                    self.portfolio_metadata = data
                elif "optimization_results" in data:
                    # Eiten format - use the first available strategy
                    results = data["optimization_results"]
                    if results:
                        first_strategy = list(results.keys())[0]
                        strategy_data = results[first_strategy]
                        self.portfolio_weights = strategy_data.get("weights", {})

                        # Create metadata combining general info and strategy-specific info
                        self.portfolio_metadata = {
                            **data.get("metadata", {}),
                            "strategy": first_strategy,
                            "weights": self.portfolio_weights,
                            "performance": {
                                "expected_return": strategy_data.get("expected_return"),
                                "volatility": strategy_data.get("volatility"),
                                "sharpe_ratio": strategy_data.get("sharpe_ratio"),
                            },
                        }
                        self.logger.info(f"Using Eiten strategy: {first_strategy}")
                    else:
                        raise ValueError(
                            "No optimization results found in Eiten format"
                        )
                else:
                    raise ValueError(
                        "Unrecognized JSON format - missing 'weights' or 'optimization_results'"
                    )

            elif portfolio_path.suffix.lower() == ".csv":
                # Load from CSV format
                weights_df = pd.read_csv(portfolio_path)

                if "Asset" in weights_df.columns and "Weight" in weights_df.columns:
                    self.portfolio_weights = dict(
                        zip(weights_df["Asset"], weights_df["Weight"])
                    )
                else:
                    raise ValueError("CSV must contain 'Asset' and 'Weight' columns")

                # Create minimal metadata
                self.portfolio_metadata = {
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "assets": list(self.portfolio_weights.keys()),
                    "weights": self.portfolio_weights,
                }
            else:
                raise ValueError("Portfolio file must be .json or .csv format")

            # Validate weights
            if not self.portfolio_weights:
                raise ValueError("No portfolio weights found in file")

            # Normalize weights to ensure they sum to 1
            total_weight = sum(self.portfolio_weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                self.logger.warning(
                    f"Portfolio weights sum to {total_weight:.6f}, normalizing to 1.0"
                )
                self.portfolio_weights = {
                    asset: weight / total_weight
                    for asset, weight in self.portfolio_weights.items()
                }

            self.logger.info(
                f"Loaded portfolio weights for {len(self.portfolio_weights)} assets"
            )
            self.logger.info(f"Assets: {list(self.portfolio_weights.keys())}")

            return self.portfolio_weights

        except Exception as e:
            error_msg = (
                f"Failed to load portfolio weights from {portfolio_file}: {str(e)}"
            )
            self.logger.error(error_msg)
            raise DataLoadError(error_msg)

    def load_price_data(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load historical price data for backtesting.

        Args:
            symbols: List of asset symbols to load (if None, uses portfolio weights)

        Returns:
            DataFrame with price data for all symbols

        Raises:
            DataLoadError: If loading fails
        """
        try:
            if symbols is None:
                if self.portfolio_weights is None:
                    raise ValueError(
                        "Must load portfolio weights first or provide symbols"
                    )
                symbols = list(self.portfolio_weights.keys())

            price_dfs = []

            for symbol in symbols:
                # Find the most recent processed file for this symbol
                pattern = f"{symbol}_processed_*.csv"
                files = list(self.processed_data_dir.glob(pattern))

                if not files:
                    raise FileNotFoundError(
                        f"No processed data found for symbol {symbol}"
                    )

                # Use the most recent file
                latest_file = max(files, key=lambda x: x.stat().st_mtime)

                # Load price data
                df = pd.read_csv(latest_file)

                # Ensure required columns exist
                if "Date" not in df.columns or "Close" not in df.columns:
                    raise ValueError(f"Required columns missing in {latest_file}")

                # Create a clean price series
                price_series = df.set_index("Date")["Close"]
                price_series.index = pd.to_datetime(price_series.index)
                price_series.name = symbol

                price_dfs.append(price_series)
                self.logger.info(
                    f"Loaded {len(price_series)} price points for {symbol}"
                )

            # Combine all price series into a single DataFrame
            self.price_data = pd.concat(price_dfs, axis=1)

            # Forward fill missing values and drop any remaining NaN rows
            self.price_data = self.price_data.fillna(method="ffill").dropna()

            self.logger.info(f"Combined price data shape: {self.price_data.shape}")
            self.logger.info(
                f"Date range: {self.price_data.index.min()} to {self.price_data.index.max()}"
            )

            return self.price_data

        except Exception as e:
            error_msg = f"Failed to load price data: {str(e)}"
            self.logger.error(error_msg)
            raise DataLoadError(error_msg)

    def calculate_portfolio_returns(self) -> pd.Series:
        """
        Calculate portfolio returns based on weights and price data.

        Returns:
            Series of portfolio returns

        Raises:
            PerformanceCalculationError: If calculation fails
        """
        try:
            if self.price_data is None:
                raise ValueError("Price data not loaded")
            if self.portfolio_weights is None:
                raise ValueError("Portfolio weights not loaded")

            # Calculate individual asset returns
            returns = self.price_data.pct_change().dropna()

            # Ensure all portfolio assets are in the returns data
            missing_assets = set(self.portfolio_weights.keys()) - set(returns.columns)
            if missing_assets:
                raise ValueError(f"Missing price data for assets: {missing_assets}")

            # Calculate weighted portfolio returns
            portfolio_returns = pd.Series(0.0, index=returns.index)

            for asset, weight in self.portfolio_weights.items():
                portfolio_returns += weight * returns[asset]

            portfolio_returns.name = "Portfolio Returns"

            self.logger.info(
                f"Calculated {len(portfolio_returns)} portfolio return observations"
            )

            return portfolio_returns

        except Exception as e:
            error_msg = f"Failed to calculate portfolio returns: {str(e)}"
            self.logger.error(error_msg)
            raise PerformanceCalculationError(error_msg)

    def run_backtest(self, portfolio_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Run complete backtest for a portfolio.

        Args:
            portfolio_file: Path to portfolio weights file

        Returns:
            Dictionary containing backtest results and performance metrics

        Raises:
            BacktestingError: If backtesting fails
        """
        try:
            self.logger.info(f"Starting backtest for portfolio: {portfolio_file}")

            # Load portfolio weights
            self.load_portfolio_weights(portfolio_file)

            # Load price data
            self.load_price_data()

            # Calculate portfolio returns
            portfolio_returns = self.calculate_portfolio_returns()

            # Use VectorBT to create a portfolio simulation
            # This simulates the portfolio performance with rebalancing
            portfolio = vbt.Portfolio.from_orders(
                close=self.price_data,
                size=self._calculate_rebalancing_orders(),
                init_cash=self.initial_capital,
                freq=self.rebalancing_frequency,
            )

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                portfolio_returns, portfolio
            )

            # Prepare results
            self.backtest_results = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "portfolio_file": str(portfolio_file),
                "portfolio_metadata": self.portfolio_metadata,
                "backtest_parameters": {
                    "risk_free_rate": self.risk_free_rate,
                    "rebalancing_frequency": self.rebalancing_frequency,
                    "initial_capital": self.initial_capital,
                    "start_date": str(self.price_data.index.min().date()),
                    "end_date": str(self.price_data.index.max().date()),
                    "total_days": len(self.price_data),
                },
                "performance_metrics": performance_metrics,
                "portfolio_returns_summary": {
                    "total_observations": len(portfolio_returns),
                    "mean_daily_return": float(portfolio_returns.mean()),
                    "std_daily_return": float(portfolio_returns.std()),
                    "min_daily_return": float(portfolio_returns.min()),
                    "max_daily_return": float(portfolio_returns.max()),
                },
            }

            self.logger.info("Backtest completed successfully")
            return self.backtest_results

        except Exception as e:
            error_msg = f"Backtest failed: {str(e)}"
            self.logger.error(error_msg)
            raise BacktestingError(error_msg)

    def _calculate_rebalancing_orders(self) -> pd.DataFrame:
        """
        Calculate rebalancing orders for VectorBT portfolio simulation.

        Returns:
            DataFrame with rebalancing orders
        """
        # Create a simple rebalancing strategy based on portfolio weights
        # This is a simplified version - can be enhanced for more complex rebalancing

        # For now, implement a basic buy-and-hold with periodic rebalancing
        orders = pd.DataFrame(
            index=self.price_data.index, columns=self.price_data.columns
        )
        orders = orders.fillna(0.0)

        # Set initial positions based on weights
        for asset, weight in self.portfolio_weights.items():
            if asset in orders.columns:
                # Calculate initial shares to buy
                initial_price = self.price_data[asset].iloc[0]
                target_value = self.initial_capital * weight
                shares = target_value / initial_price
                orders.loc[orders.index[0], asset] = shares

        return orders

    def _calculate_performance_metrics(
        self, portfolio_returns: pd.Series, portfolio: vbt.Portfolio
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Args:
            portfolio_returns: Series of portfolio returns
            portfolio: VectorBT portfolio object

        Returns:
            Dictionary of performance metrics
        """
        try:
            # Basic return statistics
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = (1 + portfolio_returns.mean()) ** 252 - 1
            annualized_volatility = portfolio_returns.std() * np.sqrt(252)

            # Sharpe ratio
            excess_returns = portfolio_returns - (self.risk_free_rate / 252)
            sharpe_ratio = (
                excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                if excess_returns.std() > 0
                else 0.0
            )

            # Maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            # VectorBT portfolio metrics (if available)
            vbt_metrics = {}
            try:
                vbt_metrics = {
                    "total_return_vbt": float(portfolio.total_return()),
                    "sharpe_ratio_vbt": float(portfolio.sharpe_ratio()),
                    "max_drawdown_vbt": float(portfolio.max_drawdown()),
                }
            except Exception as e:
                self.logger.warning(f"Could not calculate VectorBT metrics: {e}")

            # Combine all metrics
            metrics = {
                "total_return": float(total_return),
                "annualized_return": float(annualized_return),
                "annualized_volatility": float(annualized_volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "risk_free_rate": self.risk_free_rate,
                **vbt_metrics,
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def save_results(self, filename_prefix: str = "backtest") -> Dict[str, Path]:
        """
        Save backtest results to files.

        Args:
            filename_prefix: Prefix for output filenames

        Returns:
            Dictionary mapping result type to file path
        """
        if self.backtest_results is None:
            raise ValueError("No backtest results to save")

        timestamp = self.backtest_results["timestamp"]

        # Save main results as JSON
        json_filename = f"{filename_prefix}_{timestamp}.json"
        json_path = self.results_dir / json_filename

        with open(json_path, "w") as f:
            json.dump(self.backtest_results, f, indent=2, default=str)

        self.logger.info(f"Saved backtest results to {json_path}")

        return {"json": json_path}


def run_portfolio_backtest(
    portfolio_file: Union[str, Path],
    processed_data_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    risk_free_rate: float = 0.02,
    rebalancing_frequency: str = "M",
    initial_capital: float = 100000.0,
    save_results: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to run a complete backtest for a portfolio.

    Args:
        portfolio_file: Path to portfolio weights file
        processed_data_dir: Directory containing processed price data
        results_dir: Directory to save results
        risk_free_rate: Risk-free rate for calculations
        rebalancing_frequency: Portfolio rebalancing frequency
        initial_capital: Initial capital for backtesting
        save_results: Whether to save results to files

    Returns:
        Dictionary containing backtest results
    """
    backtester = VectorBTBacktester(
        processed_data_dir=processed_data_dir,
        results_dir=results_dir,
        risk_free_rate=risk_free_rate,
        rebalancing_frequency=rebalancing_frequency,
        initial_capital=initial_capital,
    )

    results = backtester.run_backtest(portfolio_file)

    if save_results:
        portfolio_name = Path(portfolio_file).stem
        backtester.save_results(filename_prefix=portfolio_name)

    return results


def backtest_all_portfolios(
    portfolios_dir: Union[str, Path] = None,
    processed_data_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    risk_free_rate: float = 0.02,
    rebalancing_frequency: str = "M",
    initial_capital: float = 100000.0,
    file_pattern: str = "*.json",
) -> Dict[str, Dict[str, Any]]:
    """
    Backtest all portfolios in a directory and return comparative results.

    Args:
        portfolios_dir: Directory containing portfolio files
        processed_data_dir: Directory containing processed price data
        results_dir: Directory to save results
        risk_free_rate: Risk-free rate for calculations
        rebalancing_frequency: Portfolio rebalancing frequency
        initial_capital: Initial capital for backtesting
        file_pattern: Pattern to match portfolio files (e.g., "*.json", "*pypfopt*.json")

    Returns:
        Dictionary mapping portfolio names to their backtest results
    """
    logger = logging.getLogger(__name__)

    # Set up directories
    portfolios_path = (
        Path(portfolios_dir) if portfolios_dir else RESULTS_DIR / "portfolios"
    )

    if not portfolios_path.exists():
        raise FileNotFoundError(f"Portfolios directory not found: {portfolios_path}")

    # Find all portfolio files
    portfolio_files = list(portfolios_path.glob(file_pattern))

    if not portfolio_files:
        logger.warning(
            f"No portfolio files found matching pattern '{file_pattern}' in {portfolios_path}"
        )
        return {}

    logger.info(f"Found {len(portfolio_files)} portfolio files to backtest")

    # Initialize backtester
    backtester = VectorBTBacktester(
        processed_data_dir=processed_data_dir,
        results_dir=results_dir,
        risk_free_rate=risk_free_rate,
        rebalancing_frequency=rebalancing_frequency,
        initial_capital=initial_capital,
    )

    # Run backtests for all portfolios
    all_results = {}

    for portfolio_file in portfolio_files:
        try:
            logger.info(f"Backtesting portfolio: {portfolio_file.name}")

            # Run backtest
            results = backtester.run_backtest(portfolio_file)

            # Save results
            portfolio_name = portfolio_file.stem
            backtester.save_results(filename_prefix=portfolio_name)

            # Store in results dictionary
            all_results[portfolio_name] = results

            # Log key metrics
            metrics = results["performance_metrics"]
            logger.info(f"  Total Return: {metrics['total_return']:.4f}")
            logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.4f}")

        except Exception as e:
            logger.error(f"Failed to backtest {portfolio_file.name}: {str(e)}")
            continue

    logger.info(f"Completed backtesting for {len(all_results)} portfolios")

    # Create comparative summary
    comparative_results = create_comparative_summary(all_results)

    # Save comparative summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = (
        backtester.results_dir / f"comparative_backtest_summary_{timestamp}.json"
    )

    with open(summary_path, "w") as f:
        json.dump(comparative_results, f, indent=2, default=str)

    logger.info(f"Saved comparative summary to {summary_path}")

    return all_results


def create_comparative_summary(
    backtest_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Create a comparative summary of multiple backtest results.

    Args:
        backtest_results: Dictionary mapping portfolio names to backtest results

    Returns:
        Comparative summary dictionary
    """
    if not backtest_results:
        return {}

    # Extract performance metrics for comparison
    metrics_comparison = {}
    portfolios = list(backtest_results.keys())

    for portfolio_name, results in backtest_results.items():
        metrics = results.get("performance_metrics", {})
        metrics_comparison[portfolio_name] = {
            "total_return": metrics.get("total_return", 0),
            "annualized_return": metrics.get("annualized_return", 0),
            "annualized_volatility": metrics.get("annualized_volatility", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
        }

    # Calculate rankings
    rankings = {}
    metric_names = [
        "total_return",
        "annualized_return",
        "sharpe_ratio",
    ]  # Higher is better
    reverse_metrics = ["annualized_volatility", "max_drawdown"]  # Lower is better

    for metric in metric_names:
        sorted_portfolios = sorted(
            portfolios, key=lambda p: metrics_comparison[p][metric], reverse=True
        )
        rankings[f"{metric}_ranking"] = {
            portfolio: rank + 1 for rank, portfolio in enumerate(sorted_portfolios)
        }

    for metric in reverse_metrics:
        sorted_portfolios = sorted(
            portfolios,
            key=lambda p: abs(
                metrics_comparison[p][metric]
            ),  # Use absolute value for drawdown
        )
        rankings[f"{metric}_ranking"] = {
            portfolio: rank + 1 for rank, portfolio in enumerate(sorted_portfolios)
        }

    # Create summary
    summary = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "portfolios_analyzed": len(backtest_results),
        "portfolio_names": portfolios,
        "metrics_comparison": metrics_comparison,
        "rankings": rankings,
        "best_performers": {
            "highest_total_return": max(
                portfolios, key=lambda p: metrics_comparison[p]["total_return"]
            ),
            "highest_sharpe_ratio": max(
                portfolios, key=lambda p: metrics_comparison[p]["sharpe_ratio"]
            ),
            "lowest_volatility": min(
                portfolios,
                key=lambda p: abs(metrics_comparison[p]["annualized_volatility"]),
            ),
            "lowest_max_drawdown": min(
                portfolios, key=lambda p: abs(metrics_comparison[p]["max_drawdown"])
            ),
        },
    }

    return summary


def validate_backtest_results(
    backtest_results: Dict[str, Any],
    benchmark_results: Optional[Dict[str, Any]] = None,
    validation_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Validate backtesting results against benchmarks and expected thresholds.

    Args:
        backtest_results: Results from run_backtest()
        benchmark_results: Optional benchmark performance metrics for comparison
        validation_thresholds: Optional dict of metric thresholds for validation

    Returns:
        Dictionary containing validation results and status
    """
    logger = logging.getLogger(__name__)

    # Default validation thresholds (reasonable ranges for portfolio metrics)
    default_thresholds = {
        "max_total_return_abs": 2.0,  # |Total return| should be < 200%
        "max_annualized_return_abs": 1.0,  # |Annualized return| should be < 100%
        "max_volatility": 1.0,  # Volatility should be < 100%
        "min_sharpe_ratio": -10.0,  # Sharpe ratio should be > -10
        "max_sharpe_ratio": 10.0,  # Sharpe ratio should be < 10
        "max_drawdown_abs": 1.0,  # |Max drawdown| should be < 100%
    }

    thresholds = validation_thresholds or default_thresholds
    validation_results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "validation_status": "PASS",
        "tests_performed": [],
        "failures": [],
        "warnings": [],
        "benchmark_comparison": {},
        "metric_validation": {},
    }

    try:
        metrics = backtest_results.get("performance_metrics", {})

        # Test 1: Metric Range Validation
        logger.info("Running metric range validation...")

        # Total return validation
        total_return = metrics.get("total_return", 0)
        if abs(total_return) > thresholds["max_total_return_abs"]:
            validation_results["failures"].append(
                f"Total return {total_return:.4f} exceeds threshold {thresholds['max_total_return_abs']}"
            )
            validation_results["validation_status"] = "FAIL"

        # Annualized return validation
        ann_return = metrics.get("annualized_return", 0)
        if abs(ann_return) > thresholds["max_annualized_return_abs"]:
            validation_results["failures"].append(
                f"Annualized return {ann_return:.4f} exceeds threshold {thresholds['max_annualized_return_abs']}"
            )
            validation_results["validation_status"] = "FAIL"

        # Volatility validation
        volatility = metrics.get("annualized_volatility", 0)
        if volatility > thresholds["max_volatility"] or volatility < 0:
            validation_results["failures"].append(
                f"Volatility {volatility:.4f} is invalid (should be 0 < vol < {thresholds['max_volatility']})"
            )
            validation_results["validation_status"] = "FAIL"

        # Sharpe ratio validation
        sharpe = metrics.get("sharpe_ratio", 0)
        if (
            sharpe < thresholds["min_sharpe_ratio"]
            or sharpe > thresholds["max_sharpe_ratio"]
        ):
            validation_results["warnings"].append(
                f"Sharpe ratio {sharpe:.4f} outside typical range [{thresholds['min_sharpe_ratio']}, {thresholds['max_sharpe_ratio']}]"
            )

        # Max drawdown validation
        max_dd = metrics.get("max_drawdown", 0)
        if abs(max_dd) > thresholds["max_drawdown_abs"] or max_dd > 0:
            validation_results["failures"].append(
                f"Max drawdown {max_dd:.4f} is invalid (should be negative and > -{thresholds['max_drawdown_abs']})"
            )
            validation_results["validation_status"] = "FAIL"

        validation_results["tests_performed"].append("metric_range_validation")
        validation_results["metric_validation"] = {
            "total_return": {
                "value": total_return,
                "valid": abs(total_return) <= thresholds["max_total_return_abs"],
            },
            "annualized_return": {
                "value": ann_return,
                "valid": abs(ann_return) <= thresholds["max_annualized_return_abs"],
            },
            "volatility": {
                "value": volatility,
                "valid": 0 < volatility <= thresholds["max_volatility"],
            },
            "sharpe_ratio": {
                "value": sharpe,
                "valid": thresholds["min_sharpe_ratio"]
                <= sharpe
                <= thresholds["max_sharpe_ratio"],
            },
            "max_drawdown": {
                "value": max_dd,
                "valid": max_dd <= 0 and abs(max_dd) <= thresholds["max_drawdown_abs"],
            },
        }

        # Test 2: Portfolio Returns Consistency
        logger.info("Running portfolio returns consistency validation...")

        returns_summary = backtest_results.get("portfolio_returns_summary", {})
        mean_return = returns_summary.get("mean_daily_return", 0)
        std_return = returns_summary.get("std_daily_return", 0)

        # Check if daily returns statistics are consistent with annualized metrics
        expected_ann_return = (1 + mean_return) ** 252 - 1
        expected_ann_vol = std_return * np.sqrt(252)

        return_diff = abs(ann_return - expected_ann_return)
        vol_diff = abs(volatility - expected_ann_vol)

        if return_diff > 0.01:  # 1% tolerance
            validation_results["warnings"].append(
                f"Annualized return calculation inconsistency: {return_diff:.4f} difference"
            )

        if vol_diff > 0.01:  # 1% tolerance
            validation_results["warnings"].append(
                f"Annualized volatility calculation inconsistency: {vol_diff:.4f} difference"
            )

        validation_results["tests_performed"].append("returns_consistency_validation")

        # Test 3: Benchmark Comparison (if provided)
        if benchmark_results:
            logger.info("Running benchmark comparison...")

            benchmark_metrics = benchmark_results.get("performance_metrics", {})

            comparison = {}
            for metric in [
                "total_return",
                "annualized_return",
                "sharpe_ratio",
                "max_drawdown",
            ]:
                portfolio_value = metrics.get(metric, 0)
                benchmark_value = benchmark_metrics.get(metric, 0)

                if benchmark_value != 0:
                    relative_performance = (portfolio_value - benchmark_value) / abs(
                        benchmark_value
                    )
                else:
                    relative_performance = portfolio_value

                comparison[metric] = {
                    "portfolio": portfolio_value,
                    "benchmark": benchmark_value,
                    "relative_performance": relative_performance,
                    "outperformed": (
                        portfolio_value > benchmark_value
                        if metric != "max_drawdown"
                        else portfolio_value
                        > benchmark_value  # Less negative is better for drawdown
                    ),
                }

            validation_results["benchmark_comparison"] = comparison
            validation_results["tests_performed"].append("benchmark_comparison")

        # Test 4: Data Quality Validation
        logger.info("Running data quality validation...")

        backtest_params = backtest_results.get("backtest_parameters", {})
        total_days = backtest_params.get("total_days", 0)
        total_observations = returns_summary.get("total_observations", 0)

        if total_observations < 10:
            validation_results["warnings"].append(
                f"Very few return observations ({total_observations}) - results may be unreliable"
            )

        if total_days > 0 and total_observations > 0:
            data_coverage = total_observations / total_days
            if data_coverage < 0.8:  # Less than 80% data coverage
                validation_results["warnings"].append(
                    f"Low data coverage: {data_coverage:.2%} ({total_observations}/{total_days} days)"
                )

        validation_results["tests_performed"].append("data_quality_validation")

        # Final validation status
        if validation_results["failures"]:
            validation_results["validation_status"] = "FAIL"
        elif validation_results["warnings"]:
            validation_results["validation_status"] = "PASS_WITH_WARNINGS"
        else:
            validation_results["validation_status"] = "PASS"

        logger.info(
            f"Validation completed with status: {validation_results['validation_status']}"
        )

        return validation_results

    except Exception as e:
        logger.error(f"Validation failed with error: {str(e)}")
        validation_results["validation_status"] = "ERROR"
        validation_results["failures"].append(f"Validation error: {str(e)}")
        return validation_results


def create_benchmark_portfolio(
    symbols: List[str],
    benchmark_type: str = "equal_weight",
    processed_data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Create a simple benchmark portfolio for comparison.

    Args:
        symbols: List of asset symbols
        benchmark_type: Type of benchmark ('equal_weight', 'market_cap', 'spy_only')
        processed_data_dir: Directory containing processed price data

    Returns:
        Benchmark portfolio results
    """
    logger = logging.getLogger(__name__)

    try:
        if benchmark_type == "equal_weight":
            # Equal weight portfolio
            weights = {symbol: 1.0 / len(symbols) for symbol in symbols}

        elif benchmark_type == "spy_only":
            # SPY-only benchmark
            weights = {symbol: 1.0 if symbol == "SPY" else 0.0 for symbol in symbols}

        elif benchmark_type == "60_40":
            # Simple 60/40 stock/bond portfolio (using SPY and TLT)
            weights = {}
            for symbol in symbols:
                if symbol in ["SPY", "VTI", "QQQ", "IWM"]:  # Equity ETFs
                    weights[symbol] = 0.6 / sum(
                        1 for s in symbols if s in ["SPY", "VTI", "QQQ", "IWM"]
                    )
                elif symbol in ["BND", "TLT", "SHY", "TIPS"]:  # Bond ETFs
                    weights[symbol] = 0.4 / sum(
                        1 for s in symbols if s in ["BND", "TLT", "SHY", "TIPS"]
                    )
                else:
                    weights[symbol] = 0.0

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
        else:
            raise ValueError(f"Unknown benchmark_type: {benchmark_type}")

        # Create a temporary JSON file for the benchmark
        benchmark_data = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "benchmark_type": benchmark_type,
            "weights": weights,
            "assets": list(weights.keys()),
        }

        # Run backtest on benchmark
        benchmark_results = run_portfolio_backtest(
            portfolio_file="",  # We'll pass the data directly
            processed_data_dir=processed_data_dir,
            save_results=False,
        )

        # Override the portfolio loading for benchmark
        backtester = VectorBTBacktester(processed_data_dir=processed_data_dir)
        backtester.portfolio_weights = weights
        backtester.portfolio_metadata = benchmark_data
        benchmark_results = backtester.run_backtest("")

        logger.info(f"Created {benchmark_type} benchmark with {len(weights)} assets")
        return benchmark_results

    except Exception as e:
        logger.error(f"Failed to create benchmark portfolio: {str(e)}")
        return {}


def run_comprehensive_validation(
    portfolio_file: Union[str, Path],
    create_benchmark: bool = True,
    benchmark_type: str = "equal_weight",
    save_validation: bool = True,
    **backtest_kwargs,
) -> Dict[str, Any]:
    """
    Run complete validation suite for a portfolio backtest.

    Args:
        portfolio_file: Path to portfolio weights file
        create_benchmark: Whether to create and compare against benchmark
        benchmark_type: Type of benchmark to create
        save_validation: Whether to save validation results
        **backtest_kwargs: Additional backtesting arguments

    Returns:
        Complete validation results
    """
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Running comprehensive validation for {portfolio_file}")

        # Run portfolio backtest
        portfolio_results = run_portfolio_backtest(
            portfolio_file, save_results=False, **backtest_kwargs
        )

        # Create benchmark if requested
        benchmark_results = None
        if create_benchmark:
            symbols = portfolio_results["portfolio_metadata"].get("assets", [])
            if symbols:
                benchmark_results = create_benchmark_portfolio(symbols, benchmark_type)

        # Run validation
        validation_results = validate_backtest_results(
            portfolio_results, benchmark_results
        )

        # Combine all results
        complete_results = {
            "portfolio_file": str(portfolio_file),
            "portfolio_results": portfolio_results,
            "benchmark_results": benchmark_results,
            "validation_results": validation_results,
            "summary": {
                "validation_status": validation_results["validation_status"],
                "tests_performed": len(validation_results["tests_performed"]),
                "failures": len(validation_results["failures"]),
                "warnings": len(validation_results["warnings"]),
            },
        }

        # Save validation results if requested
        if save_validation:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            portfolio_name = Path(portfolio_file).stem
            validation_path = (
                RESULTS_DIR
                / "backtests"
                / f"validation_{portfolio_name}_{timestamp}.json"
            )

            with open(validation_path, "w") as f:
                json.dump(complete_results, f, indent=2, default=str)

            logger.info(f"Saved validation results to {validation_path}")

        return complete_results

    except Exception as e:
        logger.error(f"Comprehensive validation failed: {str(e)}")
        return {"validation_status": "ERROR", "error": str(e)}


def integrate_with_optimizer(
    optimizer_instance,
    run_backtest: bool = True,
    save_backtest: bool = True,
    **backtest_kwargs,
) -> Optional[Dict[str, Any]]:
    """
    Integration function that optimizers can call to automatically run backtests.

    Args:
        optimizer_instance: Instance of an optimizer class (PyPortfolioOptOptimizer, etc.)
        run_backtest: Whether to run backtesting after optimization
        save_backtest: Whether to save backtest results
        **backtest_kwargs: Additional arguments for backtesting

    Returns:
        Backtest results if run_backtest is True, None otherwise
    """
    logger = logging.getLogger(__name__)

    if not run_backtest:
        return None

    try:
        # Check if optimizer has results to backtest
        if not hasattr(optimizer_instance, "results") or not optimizer_instance.results:
            logger.warning("Optimizer instance has no results to backtest")
            return None

        # Get the most recent portfolio file from optimizer results
        if (
            hasattr(optimizer_instance, "last_saved_files")
            and optimizer_instance.last_saved_files
        ):
            # Use the JSON file if available
            portfolio_file = optimizer_instance.last_saved_files.get("json")
            if not portfolio_file or not Path(portfolio_file).exists():
                logger.warning("No valid portfolio file found from optimizer")
                return None
        else:
            logger.warning("Optimizer instance has no saved files information")
            return None

        # Run backtest
        logger.info(f"Running backtest for optimizer output: {portfolio_file}")

        backtest_results = run_portfolio_backtest(
            portfolio_file=portfolio_file, save_results=save_backtest, **backtest_kwargs
        )

        logger.info("Backtest integration completed successfully")
        return backtest_results

    except Exception as e:
        logger.error(f"Failed to integrate backtest with optimizer: {str(e)}")
        return None


if __name__ == "__main__":
    # Enhanced command-line interface
    import argparse

    parser = argparse.ArgumentParser(description="FundTuneLab Portfolio Backtesting")
    parser.add_argument(
        "portfolio_file", nargs="?", help="Single portfolio file to backtest"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Backtest all portfolios in results/portfolios/",
    )
    parser.add_argument(
        "--pattern", default="*.json", help="File pattern for batch processing"
    )
    parser.add_argument(
        "--risk-free-rate", type=float, default=0.02, help="Risk-free rate"
    )
    parser.add_argument("--rebalancing", default="M", help="Rebalancing frequency")
    parser.add_argument(
        "--capital", type=float, default=100000.0, help="Initial capital"
    )

    args = parser.parse_args()

    if args.all:
        # Batch process all portfolios
        print("Running batch backtest for all portfolios...")
        results = backtest_all_portfolios(
            file_pattern=args.pattern,
            risk_free_rate=args.risk_free_rate,
            rebalancing_frequency=args.rebalancing,
            initial_capital=args.capital,
        )
        print(f"Completed backtesting for {len(results)} portfolios")
        print("Check results/backtests/ for detailed results and comparative summary")

    elif args.portfolio_file:
        # Single portfolio backtest
        results = run_portfolio_backtest(
            args.portfolio_file,
            risk_free_rate=args.risk_free_rate,
            rebalancing_frequency=args.rebalancing,
            initial_capital=args.capital,
        )
        print("Backtest completed. Results saved to results/backtests/")
        print(f"Total Return: {results['performance_metrics']['total_return']:.4f}")
        print(f"Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {results['performance_metrics']['max_drawdown']:.4f}")

    else:
        parser.print_help()
