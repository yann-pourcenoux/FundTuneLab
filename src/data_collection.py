"""
Data Collection Module for FundTuneLab

This module handles the collection of financial data using yfinance and other data providers.
It includes robust error handling, rate limiting, and comprehensive logging.
"""

import time
import logging
import warnings
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
import yfinance as yf
from requests.exceptions import RequestException, Timeout, ConnectionError

from config.settings import (
    DEFAULT_ASSETS,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DATA_PROVIDERS,
    RAW_DATA_DIR,
    ensure_directories,
)

# Suppress yfinance warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")


class DataCollectionError(Exception):
    """Custom exception for data collection errors."""

    pass


class RateLimitError(DataCollectionError):
    """Exception raised when rate limits are exceeded."""

    pass


class APIConnectionError(DataCollectionError):
    """Exception raised when API connection fails persistently."""

    pass


class DataValidationError(DataCollectionError):
    """Exception raised when downloaded data fails validation."""

    pass


class DataCollector:
    """
    Main class for collecting financial data from various providers.

    Handles rate limiting, error recovery, and data validation.
    """

    def __init__(self, provider: str = "yahoo"):
        """
        Initialize the DataCollector.

        Args:
            provider (str): Data provider to use ('yahoo', 'alpha_vantage', 'quandl')
        """
        self.provider = provider
        self.provider_config = DATA_PROVIDERS.get(provider, {})

        if not self.provider_config.get("enabled", False):
            raise ValueError(f"Provider '{provider}' is not enabled or configured")

        self.rate_limit = self.provider_config.get("rate_limit", 1000)
        self.timeout = self.provider_config.get("timeout", 30)
        self.last_request_time = 0
        self.request_count = 0
        self.request_timestamps = []

        # Circuit breaker pattern for API resilience
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = 0
        self.circuit_breaker_threshold = 5  # Max consecutive failures
        self.circuit_breaker_timeout = 300  # 5 minutes cooldown

        # Error statistics tracking
        self.error_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "network_errors": 0,
            "rate_limit_hits": 0,
            "validation_errors": 0,
        }

        # Setup logging
        self.logger = self._setup_logging()

        # Ensure data directories exist
        ensure_directories()

        self.logger.info(f"DataCollector initialized with provider: {provider}")

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the data collector."""
        logger = logging.getLogger(f"data_collector_{self.provider}")
        logger.setLevel(logging.INFO)

        # Create file handler if it doesn't exist
        if not logger.handlers:
            log_file = RAW_DATA_DIR / "data_collection.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Create formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add handlers to logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger

    def _enforce_rate_limit(self):
        """Enforce rate limiting based on provider configuration."""
        current_time = time.time()

        # Clean old timestamps (older than 1 hour for most providers)
        one_hour_ago = current_time - 3600
        self.request_timestamps = [
            ts for ts in self.request_timestamps if ts > one_hour_ago
        ]

        # Check if we're approaching rate limits
        if len(self.request_timestamps) >= self.rate_limit * 0.9:  # 90% of limit
            sleep_time = 3601 - (current_time - min(self.request_timestamps))
            if sleep_time > 0:
                self.logger.warning(
                    f"Rate limit approaching, sleeping for {sleep_time:.1f} seconds"
                )
                time.sleep(sleep_time)

        # Add minimum delay between requests (1 second for Yahoo Finance)
        time_since_last = current_time - self.last_request_time
        if time_since_last < 1.0:
            sleep_time = 1.0 - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_timestamps.append(self.last_request_time)

    def _check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker is open (blocking requests).

        Returns:
            bool: True if requests are allowed, False if circuit is open
        """
        current_time = time.time()

        # Reset circuit breaker after timeout
        if (
            self.circuit_breaker_failures >= self.circuit_breaker_threshold
            and current_time - self.circuit_breaker_last_failure
            > self.circuit_breaker_timeout
        ):
            self.logger.info("Circuit breaker reset - allowing requests again")
            self.circuit_breaker_failures = 0
            self.circuit_breaker_last_failure = 0
            return True

        # Check if circuit is open
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            remaining_time = self.circuit_breaker_timeout - (
                current_time - self.circuit_breaker_last_failure
            )
            self.logger.warning(
                f"Circuit breaker OPEN - blocking requests for {remaining_time:.1f} more seconds"
            )
            return False

        return True

    def _record_success(self):
        """Record a successful API call."""
        self.error_stats["total_requests"] += 1
        self.error_stats["successful_requests"] += 1
        # Reset circuit breaker on success
        if self.circuit_breaker_failures > 0:
            self.logger.info("Resetting circuit breaker after successful request")
            self.circuit_breaker_failures = 0
            self.circuit_breaker_last_failure = 0

    def _record_failure(self, error_type: str = "general"):
        """Record a failed API call and update circuit breaker."""
        self.error_stats["total_requests"] += 1
        self.error_stats["failed_requests"] += 1

        if error_type == "network":
            self.error_stats["network_errors"] += 1
        elif error_type == "rate_limit":
            self.error_stats["rate_limit_hits"] += 1
        elif error_type == "validation":
            self.error_stats["validation_errors"] += 1

        # Update circuit breaker
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = time.time()

        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            self.logger.error(
                f"Circuit breaker OPENED after {self.circuit_breaker_failures} consecutive failures"
            )

    def _validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists and has data available.

        Args:
            symbol (str): Stock/ETF symbol to validate

        Returns:
            bool: True if symbol is valid and has data
        """
        # Check circuit breaker
        if not self._check_circuit_breaker():
            raise APIConnectionError(
                "Circuit breaker is open - too many recent failures"
            )

        try:
            self._enforce_rate_limit()
            ticker = yf.Ticker(symbol)

            # Try to get basic info
            info = ticker.info
            if not info or info.get("regularMarketPrice") is None:
                self._record_failure("validation")
                return False

            # Try to get a small amount of historical data
            hist = ticker.history(period="5d", timeout=self.timeout)
            if hist.empty:
                self._record_failure("validation")
                return False

            self._record_success()
            return True

        except (RequestException, Timeout, ConnectionError) as e:
            self._record_failure("network")
            self.logger.warning(
                f"Network error during symbol validation for {symbol}: {str(e)}"
            )
            return False
        except Exception as e:
            self._record_failure("general")
            self.logger.warning(f"Symbol validation failed for {symbol}: {str(e)}")
            return False

    def download_symbol_data(
        self,
        symbol: str,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        save_to_file: bool = True,
        max_retries: int = 3,
    ) -> Optional[pd.DataFrame]:
        """
        Download historical data for a single symbol.

        Args:
            symbol (str): Stock/ETF symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            save_to_file (bool): Whether to save data to CSV file
            max_retries (int): Maximum number of retry attempts

        Returns:
            Optional[pd.DataFrame]: Historical data or None if failed
        """
        retries = 0
        last_error = None

        while retries <= max_retries:
            # Check circuit breaker before each attempt
            if not self._check_circuit_breaker():
                raise APIConnectionError(
                    "Circuit breaker is open - too many recent failures"
                )

            try:
                self._enforce_rate_limit()
                self.logger.info(
                    f"Downloading data for {symbol} (attempt {retries + 1})"
                )

                # Create ticker object
                ticker = yf.Ticker(symbol)

                # Download historical data
                data = ticker.history(
                    start=start_date, end=end_date, timeout=self.timeout
                )

                if data.empty:
                    raise DataValidationError(f"No data returned for symbol {symbol}")

                # Validate data quality
                if len(data) < 10:  # Minimum reasonable amount of data
                    raise DataValidationError(
                        f"Insufficient data for {symbol}: only {len(data)} rows"
                    )

                # Clean and prepare data
                data = self._clean_data(data, symbol)

                # Save to file if requested
                if save_to_file:
                    self._save_data_to_csv(data, symbol)

                self._record_success()
                self.logger.info(
                    f"Successfully downloaded {len(data)} rows for {symbol}"
                )
                return data

            except (RequestException, Timeout, ConnectionError) as e:
                self._record_failure("network")
                last_error = e
                retries += 1
                if retries <= max_retries:
                    wait_time = min(2**retries, 30)  # Exponential backoff, max 30s
                    self.logger.warning(
                        f"Network error downloading {symbol} (attempt {retries}): {str(e)}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(
                        f"Failed to download {symbol} after {max_retries} retries: {str(e)}"
                    )

            except (DataValidationError, DataCollectionError) as e:
                self._record_failure("validation")
                last_error = e
                retries += 1
                if retries <= max_retries:
                    wait_time = min(2**retries, 30)
                    self.logger.warning(
                        f"Data validation error downloading {symbol} (attempt {retries}): {str(e)}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(
                        f"Failed to download {symbol} after {max_retries} retries: {str(e)}"
                    )

            except Exception as e:
                self._record_failure("general")
                last_error = e
                retries += 1
                if retries <= max_retries:
                    wait_time = min(2**retries, 30)
                    self.logger.warning(
                        f"Error downloading {symbol} (attempt {retries}): {str(e)}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(
                        f"Failed to download {symbol} after {max_retries} retries: {str(e)}"
                    )

        # All retries exhausted
        self.logger.error(f"Unable to download data for {symbol}: {str(last_error)}")
        return None

    def _clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean and validate downloaded data.

        Args:
            data (pd.DataFrame): Raw data from yfinance
            symbol (str): Symbol name for logging

        Returns:
            pd.DataFrame: Cleaned data
        """
        original_length = len(data)

        # Remove rows with all NaN values
        data = data.dropna(how="all")

        # Ensure we have the essential columns
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DataCollectionError(
                f"Missing required columns for {symbol}: {missing_columns}"
            )

        # Remove rows where Close price is NaN or <= 0
        data = data[data["Close"].notna() & (data["Close"] > 0)]

        if len(data) == 0:
            raise DataCollectionError(
                f"No valid price data for {symbol} after cleaning"
            )

        # Log data quality information
        rows_removed = original_length - len(data)
        if rows_removed > 0:
            self.logger.info(f"Removed {rows_removed} invalid rows from {symbol} data")

        # Sort by date to ensure chronological order
        data = data.sort_index()

        # Add symbol column for easy identification
        data["Symbol"] = symbol

        return data

    def _save_data_to_csv(self, data: pd.DataFrame, symbol: str):
        """
        Save data to CSV file in the raw data directory.

        Args:
            data (pd.DataFrame): Data to save
            symbol (str): Symbol name for filename
        """
        try:
            filename = f"{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
            file_path = RAW_DATA_DIR / filename

            # Save with date index
            data.to_csv(file_path, index=True)
            self.logger.info(f"Saved {symbol} data to {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to save data for {symbol}: {str(e)}")
            raise DataCollectionError(f"Could not save data for {symbol}: {str(e)}")

    def download_multiple_symbols(
        self,
        symbols: List[str],
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        save_to_file: bool = True,
        validate_symbols: bool = True,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Download historical data for multiple symbols.

        Args:
            symbols (List[str]): List of stock/ETF symbols
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            save_to_file (bool): Whether to save data to CSV files
            validate_symbols (bool): Whether to validate symbols before downloading

        Returns:
            Dict[str, Optional[pd.DataFrame]]: Dictionary of symbol -> data
        """
        self.logger.info(f"Starting download for {len(symbols)} symbols")

        # Validate symbols if requested
        if validate_symbols:
            self.logger.info("Validating symbols...")
            valid_symbols = []
            for symbol in symbols:
                if self._validate_symbol(symbol):
                    valid_symbols.append(symbol)
                    self.logger.info(f"✓ {symbol} validated")
                else:
                    self.logger.warning(f"✗ {symbol} failed validation, skipping")
            symbols = valid_symbols

        results = {}
        failed_symbols = []

        for i, symbol in enumerate(symbols, 1):
            self.logger.info(f"Processing {symbol} ({i}/{len(symbols)})")

            try:
                data = self.download_symbol_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    save_to_file=save_to_file,
                )
                results[symbol] = data

                if data is not None:
                    self.logger.info(f"✓ Successfully downloaded {symbol}")
                else:
                    failed_symbols.append(symbol)
                    self.logger.error(f"✗ Failed to download {symbol}")

            except Exception as e:
                self.logger.error(f"✗ Exception downloading {symbol}: {str(e)}")
                failed_symbols.append(symbol)
                results[symbol] = None

        # Summary
        successful_count = sum(1 for data in results.values() if data is not None)
        self.logger.info(
            f"Download complete: {successful_count}/{len(symbols)} symbols successful"
        )

        if failed_symbols:
            self.logger.warning(f"Failed symbols: {', '.join(failed_symbols)}")

        return results

    def get_latest_data(
        self, symbols: List[str], period: str = "1d"
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Get the latest data for symbols (useful for daily updates).

        Args:
            symbols (List[str]): List of symbols
            period (str): Period for latest data ('1d', '5d', '1mo', etc.)

        Returns:
            Dict[str, Optional[pd.DataFrame]]: Latest data for each symbol
        """
        self.logger.info(f"Getting latest data for {len(symbols)} symbols")

        results = {}

        for symbol in symbols:
            try:
                self._enforce_rate_limit()
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, timeout=self.timeout)

                if not data.empty:
                    data = self._clean_data(data, symbol)
                    results[symbol] = data
                    self.logger.info(f"✓ Got latest data for {symbol}")
                else:
                    results[symbol] = None
                    self.logger.warning(f"✗ No latest data for {symbol}")

            except Exception as e:
                self.logger.error(f"✗ Error getting latest data for {symbol}: {str(e)}")
                results[symbol] = None

        return results

    def get_error_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive error statistics and circuit breaker status.

        Returns:
            Dict[str, any]: Error statistics and system status
        """
        total_requests = self.error_stats["total_requests"]
        success_rate = (
            self.error_stats["successful_requests"] / total_requests * 100
            if total_requests > 0
            else 0
        )

        current_time = time.time()
        circuit_breaker_status = "CLOSED"
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            remaining_time = self.circuit_breaker_timeout - (
                current_time - self.circuit_breaker_last_failure
            )
            if remaining_time > 0:
                circuit_breaker_status = f"OPEN ({remaining_time:.1f}s remaining)"
            else:
                circuit_breaker_status = "READY_TO_RESET"

        return {
            "provider": self.provider,
            "total_requests": total_requests,
            "successful_requests": self.error_stats["successful_requests"],
            "failed_requests": self.error_stats["failed_requests"],
            "success_rate_percent": round(success_rate, 2),
            "error_breakdown": {
                "network_errors": self.error_stats["network_errors"],
                "validation_errors": self.error_stats["validation_errors"],
                "rate_limit_hits": self.error_stats["rate_limit_hits"],
                "other_errors": (
                    self.error_stats["failed_requests"]
                    - self.error_stats["network_errors"]
                    - self.error_stats["validation_errors"]
                    - self.error_stats["rate_limit_hits"]
                ),
            },
            "circuit_breaker": {
                "status": circuit_breaker_status,
                "consecutive_failures": self.circuit_breaker_failures,
                "threshold": self.circuit_breaker_threshold,
                "timeout_seconds": self.circuit_breaker_timeout,
            },
            "rate_limiting": {
                "requests_last_hour": len(self.request_timestamps),
                "rate_limit": self.rate_limit,
            },
        }


def download_default_assets(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    save_to_file: bool = True,
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Convenience function to download data for all default assets.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        save_to_file (bool): Whether to save data to CSV files

    Returns:
        Dict[str, Optional[pd.DataFrame]]: Data for each asset
    """
    collector = DataCollector(provider="yahoo")
    return collector.download_multiple_symbols(
        symbols=DEFAULT_ASSETS,
        start_date=start_date,
        end_date=end_date,
        save_to_file=save_to_file,
    )


if __name__ == "__main__":
    # Example usage
    print("FundTuneLab Data Collection Module")
    print("=" * 50)

    # Download data for default assets
    print(f"Downloading data for {len(DEFAULT_ASSETS)} default assets...")
    results = download_default_assets()

    # Print summary
    successful = sum(1 for data in results.values() if data is not None)
    print("\nDownload Summary:")
    print(f"Successful: {successful}/{len(DEFAULT_ASSETS)}")
    print(f"Files saved to: {RAW_DATA_DIR}")
