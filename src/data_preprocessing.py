"""
Data Preprocessing Module for FundTuneLab

This module handles comprehensive data cleaning and preprocessing for financial data,
including missing value handling, outlier detection and removal, date standardization,
and data integrity validation.
"""

from loguru import logger
from typing import Dict, Optional, Any
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from config.settings import RAW_DATA_DIR, ensure_directories


class DataPreprocessingError(Exception):
    """Custom exception for data preprocessing errors."""

    pass


class DataIntegrityError(DataPreprocessingError):
    """Exception raised when data integrity checks fail."""

    pass


class DataPreprocessor:
    """
    Main class for preprocessing financial data.

    Handles missing values, outlier detection, date standardization,
    and data integrity validation.
    """

    def __init__(
        self,
        raw_data_dir: Optional[Path] = None,
        processed_data_dir: Optional[Path] = None,
    ):
        """
        Initialize the DataPreprocessor.

        Args:
            raw_data_dir: Directory containing raw data files
            processed_data_dir: Directory to save processed data
            log_level: Logging level
        """
        self.raw_data_dir = raw_data_dir or RAW_DATA_DIR
        self.processed_data_dir = processed_data_dir or (Path("data") / "processed")

        # Ensure directories exist
        ensure_directories()
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Data quality statistics
        self.quality_stats = {
            "files_processed": 0,
            "total_rows_input": 0,
            "total_rows_output": 0,
            "missing_values_found": 0,
            "missing_values_handled": 0,
            "outliers_detected": 0,
            "outliers_removed": 0,
            "date_format_issues": 0,
            "integrity_checks_failed": 0,
        }

        # Required columns for financial data
        self.required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        self.price_columns = ["Open", "High", "Low", "Close"]

        logger.info("DataPreprocessor initialized")

    def identify_missing_values(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Identify and analyze missing values in the dataset.

        Args:
            df: DataFrame to analyze
            symbol: Symbol name for logging

        Returns:
            Dictionary with missing value analysis
        """
        logger.info(f"Analyzing missing values for {symbol}")

        # Count missing values per column
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100

        # Identify rows with any missing values
        rows_with_missing = df.isnull().any(axis=1).sum()

        # Check for missing values in critical columns
        critical_missing = {}
        for col in self.price_columns:
            if col in df.columns and missing_counts[col] > 0:
                critical_missing[col] = missing_counts[col]

        analysis = {
            "total_rows": len(df),
            "rows_with_missing": rows_with_missing,
            "missing_by_column": missing_counts.to_dict(),
            "missing_percentages": missing_percentages.to_dict(),
            "critical_missing": critical_missing,
            "has_missing_data": missing_counts.sum() > 0,
        }

        # Update global stats
        self.quality_stats["missing_values_found"] += missing_counts.sum()

        if analysis["has_missing_data"]:
            logger.warning(
                f"{symbol}: Found {missing_counts.sum()} missing values across {missing_counts[missing_counts > 0].count()} columns"
            )
            for col, count in critical_missing.items():
                logger.warning(
                    f"{symbol}: Critical column '{col}' has {count} missing values ({missing_percentages[col]:.2f}%)"
                )
        else:
            logger.info(f"{symbol}: No missing values found")

        return analysis

    def handle_missing_values(
        self, df: pd.DataFrame, symbol: str, strategy: str = "auto"
    ) -> pd.DataFrame:
        """
        Handle missing values using appropriate strategies.

        Args:
            df: DataFrame to process
            symbol: Symbol name for logging
            strategy: Strategy for handling missing values
                     ("auto", "drop", "forward_fill", "interpolate")

        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"Handling missing values for {symbol} using strategy: {strategy}")

        original_length = len(df)
        df_cleaned = df.copy()

        # First, handle completely empty rows
        empty_rows = df_cleaned.isnull().all(axis=1).sum()
        if empty_rows > 0:
            df_cleaned = df_cleaned.dropna(how="all")
            logger.info(f"{symbol}: Removed {empty_rows} completely empty rows")

        # Handle missing values in price columns
        for col in self.price_columns:
            if col not in df_cleaned.columns:
                continue

            missing_count = df_cleaned[col].isnull().sum()
            if missing_count == 0:
                continue

            logger.info(f"{symbol}: Handling {missing_count} missing values in {col}")

            if strategy == "auto":
                # Auto strategy: use forward fill for small gaps, drop for large gaps
                if missing_count / len(df_cleaned) > 0.1:  # More than 10% missing
                    logger.warning(
                        f"{symbol}: High missing rate ({missing_count / len(df_cleaned) * 100:.1f}%) in {col}, using interpolation"
                    )
                    df_cleaned[col] = df_cleaned[col].interpolate(
                        method="linear", limit_direction="both"
                    )
                else:
                    # Forward fill small gaps
                    df_cleaned[col] = df_cleaned[col].fillna(method="ffill")
                    # Backward fill any remaining (at the beginning)
                    df_cleaned[col] = df_cleaned[col].fillna(method="bfill")

            elif strategy == "drop":
                df_cleaned = df_cleaned.dropna(subset=[col])

            elif strategy == "forward_fill":
                df_cleaned[col] = df_cleaned[col].fillna(method="ffill")
                df_cleaned[col] = df_cleaned[col].fillna(method="bfill")

            elif strategy == "interpolate":
                df_cleaned[col] = df_cleaned[col].interpolate(
                    method="linear", limit_direction="both"
                )

        # Handle missing Volume data (less critical)
        if "Volume" in df_cleaned.columns and df_cleaned["Volume"].isnull().sum() > 0:
            missing_vol = df_cleaned["Volume"].isnull().sum()
            # Fill with median volume
            median_volume = df_cleaned["Volume"].median()
            df_cleaned["Volume"] = df_cleaned["Volume"].fillna(median_volume)
            logger.info(
                f"{symbol}: Filled {missing_vol} missing Volume values with median ({median_volume:,.0f})"
            )

        # Final cleanup: remove any rows that still have missing critical data
        before_final_cleanup = len(df_cleaned)
        df_cleaned = df_cleaned.dropna(subset=self.price_columns, how="any")
        removed_final = before_final_cleanup - len(df_cleaned)

        if removed_final > 0:
            logger.warning(
                f"{symbol}: Removed {removed_final} rows with remaining missing critical data"
            )

        rows_removed = original_length - len(df_cleaned)
        self.quality_stats["missing_values_handled"] += rows_removed

        logger.info(
            f"{symbol}: Missing value handling complete. Removed {rows_removed} rows ({rows_removed / original_length * 100:.2f}%)"
        )

        return df_cleaned

    def detect_outliers(
        self, df: pd.DataFrame, symbol: str, method: str = "iqr", threshold: float = 3.0
    ) -> Dict[str, Any]:
        """
        Detect outliers in financial data using statistical methods.

        Args:
            df: DataFrame to analyze
            symbol: Symbol name for logging
            method: Method for outlier detection ("iqr", "zscore", "modified_zscore")
            threshold: Threshold for outlier detection

        Returns:
            Dictionary with outlier analysis
        """
        logger.info(f"Detecting outliers for {symbol} using {method} method")

        outlier_info = {
            "method": method,
            "threshold": threshold,
            "outliers_by_column": {},
            "total_outliers": 0,
            "outlier_indices": set(),
        }

        for col in self.price_columns:
            if col not in df.columns:
                continue

            values = df[col].dropna()
            if len(values) == 0:
                continue

            outliers = []

            if method == "iqr":
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[
                    (df[col] < lower_bound) | (df[col] > upper_bound)
                ].index.tolist()

            elif method == "zscore":
                # Manual z-score calculation using numpy
                mean = values.mean()
                std = values.std()
                z_scores = np.abs((values - mean) / std)
                outlier_mask = z_scores > threshold
                outliers = df[df[col].isin(values[outlier_mask])].index.tolist()

            elif method == "modified_zscore":
                median = values.median()
                mad = np.median(np.abs(values - median))
                if mad == 0:  # Handle case where MAD is 0
                    modified_z_scores = np.zeros_like(values)
                else:
                    modified_z_scores = 0.6745 * (values - median) / mad
                outlier_mask = np.abs(modified_z_scores) > threshold
                outliers = df[df[col].isin(values[outlier_mask])].index.tolist()

            outlier_info["outliers_by_column"][col] = {
                "count": len(outliers),
                "indices": outliers,
                "percentage": len(outliers) / len(df) * 100,
            }

            outlier_info["outlier_indices"].update(outliers)

            if len(outliers) > 0:
                logger.info(
                    f"{symbol}: Found {len(outliers)} outliers in {col} ({len(outliers) / len(df) * 100:.2f}%)"
                )

        outlier_info["total_outliers"] = len(outlier_info["outlier_indices"])
        self.quality_stats["outliers_detected"] += outlier_info["total_outliers"]

        return outlier_info

    def filter_outliers(
        self,
        df: pd.DataFrame,
        symbol: str,
        method: str = "iqr",
        threshold: float = 3.0,
        action: str = "remove",
    ) -> pd.DataFrame:
        """
        Filter out outliers from the dataset.

        Args:
            df: DataFrame to process
            symbol: Symbol name for logging
            method: Method for outlier detection
            threshold: Threshold for outlier detection
            action: Action to take ("remove", "cap", "log")

        Returns:
            DataFrame with outliers handled
        """
        logger.info(f"Filtering outliers for {symbol}")

        outlier_info = self.detect_outliers(df, symbol, method, threshold)

        if outlier_info["total_outliers"] == 0:
            logger.info(f"{symbol}: No outliers detected")
            return df

        df_filtered = df.copy()

        if action == "remove":
            # Remove rows with outliers in any price column
            outlier_indices = list(outlier_info["outlier_indices"])
            df_filtered = df_filtered.drop(outlier_indices)
            removed_count = len(outlier_indices)

            logger.info(
                f"{symbol}: Removed {removed_count} rows with outliers ({removed_count / len(df) * 100:.2f}%)"
            )
            self.quality_stats["outliers_removed"] += removed_count

        elif action == "cap":
            # Cap outliers to reasonable bounds
            for col in self.price_columns:
                if col not in df_filtered.columns:
                    continue

                Q1 = df_filtered[col].quantile(0.25)
                Q3 = df_filtered[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                capped_count = (
                    (df_filtered[col] < lower_bound) | (df_filtered[col] > upper_bound)
                ).sum()

                df_filtered[col] = df_filtered[col].clip(
                    lower=lower_bound, upper=upper_bound
                )

                if capped_count > 0:
                    logger.info(f"{symbol}: Capped {capped_count} outliers in {col}")

        return df_filtered

    def standardize_date_formats(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Standardize date formats to ensure consistency.

        Args:
            df: DataFrame to process
            symbol: Symbol name for logging

        Returns:
            DataFrame with standardized dates
        """
        logger.info(f"Standardizing date formats for {symbol}")

        df_standardized = df.copy()

        # Handle different date column scenarios
        date_column = None
        if "Date" in df_standardized.columns:
            date_column = "Date"
        elif (
            df_standardized.index.name == "Date"
            or "date" in str(df_standardized.index.name).lower()
        ):
            # Date is in the index
            df_standardized = df_standardized.reset_index()
            date_column = df_standardized.columns[
                0
            ]  # Assume first column after reset is date

        if date_column is None:
            logger.error(f"{symbol}: No date column found")
            self.quality_stats["date_format_issues"] += 1
            raise DataPreprocessingError(f"No date column found in {symbol} data")

        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df_standardized[date_column]):
                df_standardized[date_column] = pd.to_datetime(
                    df_standardized[date_column]
                )

            # Remove timezone information for consistency
            if df_standardized[date_column].dt.tz is not None:
                df_standardized[date_column] = df_standardized[
                    date_column
                ].dt.tz_localize(None)
                logger.info(f"{symbol}: Removed timezone information from dates")

            # Sort by date
            df_standardized = df_standardized.sort_values(date_column)

            # Set date as index for easier time series operations
            df_standardized = df_standardized.set_index(date_column)

            # Check for duplicate dates
            duplicate_dates = df_standardized.index.duplicated().sum()
            if duplicate_dates > 0:
                logger.warning(
                    f"{symbol}: Found {duplicate_dates} duplicate dates, keeping last occurrence"
                )
                df_standardized = df_standardized[
                    ~df_standardized.index.duplicated(keep="last")
                ]

            # Check for gaps in dates (for daily data)
            date_diff = df_standardized.index.to_series().diff()
            large_gaps = (
                date_diff > pd.Timedelta(days=7)
            ).sum()  # More than a week gap
            if large_gaps > 0:
                logger.warning(
                    f"{symbol}: Found {large_gaps} large gaps (>7 days) in date sequence"
                )

            logger.info(
                f"{symbol}: Date standardization complete. Date range: {df_standardized.index.min()} to {df_standardized.index.max()}"
            )

        except Exception as e:
            logger.error(f"{symbol}: Error standardizing dates: {str(e)}")
            self.quality_stats["date_format_issues"] += 1
            raise DataPreprocessingError(
                f"Date standardization failed for {symbol}: {str(e)}"
            )

        return df_standardized

    def validate_data_integrity(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive data integrity checks.

        Args:
            df: DataFrame to validate
            symbol: Symbol name for logging

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating data integrity for {symbol}")

        validation_results = {
            "symbol": symbol,
            "total_rows": len(df),
            "checks": {},
            "warnings": [],
            "errors": [],
            "passed": True,
        }

        # Check 1: Required columns present
        missing_cols = [
            col
            for col in self.required_columns
            if col not in df.columns and col != "Date"
        ]
        if missing_cols:
            validation_results["errors"].append(
                f"Missing required columns: {missing_cols}"
            )
            validation_results["passed"] = False
        validation_results["checks"]["required_columns"] = len(missing_cols) == 0

        # Check 2: No missing values in critical columns
        for col in self.price_columns:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    validation_results["errors"].append(
                        f"Missing values in {col}: {missing_count}"
                    )
                    validation_results["passed"] = False
                validation_results["checks"][f"{col}_no_missing"] = missing_count == 0

        # Check 3: Positive values for prices and volume
        for col in self.price_columns + ["Volume"]:
            if col in df.columns:
                negative_count = (df[col] <= 0).sum()
                if negative_count > 0:
                    validation_results["warnings"].append(
                        f"Non-positive values in {col}: {negative_count}"
                    )
                validation_results["checks"][f"{col}_positive"] = negative_count == 0

        # Check 4: OHLC relationship (Open, High, Low, Close)
        if all(col in df.columns for col in self.price_columns):
            # High should be >= max(Open, Close)
            high_violations = (df["High"] < df[["Open", "Close"]].max(axis=1)).sum()
            # Low should be <= min(Open, Close)
            low_violations = (df["Low"] > df[["Open", "Close"]].min(axis=1)).sum()

            if high_violations > 0:
                validation_results["warnings"].append(
                    f"OHLC violations - High too low: {high_violations}"
                )
            if low_violations > 0:
                validation_results["warnings"].append(
                    f"OHLC violations - Low too high: {low_violations}"
                )

            validation_results["checks"]["ohlc_valid"] = (
                high_violations + low_violations
            ) == 0

        # Check 5: Date continuity (no major gaps for daily data)
        if hasattr(df.index, "to_series"):
            date_gaps = df.index.to_series().diff()
            large_gaps = (date_gaps > pd.Timedelta(days=14)).sum()  # More than 2 weeks
            if large_gaps > 0:
                validation_results["warnings"].append(
                    f"Large date gaps detected: {large_gaps}"
                )
            validation_results["checks"]["date_continuity"] = large_gaps == 0

        # Check 6: Reasonable value ranges
        for col in self.price_columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()

                # Check for extremely high or low values (potential data errors)
                if min_val < 0.01:  # Less than 1 cent
                    validation_results["warnings"].append(
                        f"{col} has very low values (min: {min_val:.6f})"
                    )
                if max_val > 100000:  # More than $100,000
                    validation_results["warnings"].append(
                        f"{col} has very high values (max: {max_val:.2f})"
                    )

                validation_results["checks"][f"{col}_reasonable_range"] = (
                    0.01 <= min_val <= max_val <= 100000
                )

        # Check 7: Symbol consistency
        if "Symbol" in df.columns:
            unique_symbols = df["Symbol"].unique()
            if len(unique_symbols) > 1:
                validation_results["warnings"].append(
                    f"Multiple symbols in data: {unique_symbols}"
                )
            elif len(unique_symbols) == 1 and unique_symbols[0] != symbol:
                validation_results["warnings"].append(
                    f"Symbol mismatch: expected {symbol}, found {unique_symbols[0]}"
                )
            validation_results["checks"]["symbol_consistency"] = (
                len(unique_symbols) == 1 and unique_symbols[0] == symbol
            )

        # Summary
        total_checks = len(validation_results["checks"])
        passed_checks = sum(validation_results["checks"].values())
        validation_results["pass_rate"] = (
            passed_checks / total_checks if total_checks > 0 else 0
        )

        if validation_results["errors"]:
            self.quality_stats["integrity_checks_failed"] += 1
            logger.error(
                f"{symbol}: Data integrity validation FAILED with {len(validation_results['errors'])} errors"
            )
        elif validation_results["warnings"]:
            logger.warning(
                f"{symbol}: Data integrity validation passed with {len(validation_results['warnings'])} warnings"
            )
        else:
            logger.info(f"{symbol}: Data integrity validation PASSED (100%)")

        return validation_results

    def save_processed_data(
        self, df: pd.DataFrame, symbol: str, suffix: str = "processed"
    ) -> Path:
        """
        Save processed data with integrity checks.

        Args:
            df: DataFrame to save
            symbol: Symbol name
            suffix: Suffix for filename

        Returns:
            Path to saved file
        """
        logger.info(f"Saving processed data for {symbol}")

        # Perform final integrity check
        validation_results = self.validate_data_integrity(df, symbol)

        if not validation_results["passed"]:
            raise DataIntegrityError(
                f"Data integrity check failed for {symbol}: {validation_results['errors']}"
            )

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{symbol}_{suffix}_{timestamp}.csv"
        file_path = self.processed_data_dir / filename

        try:
            # Save data with date index
            df.to_csv(file_path, index=True)

            # Save metadata
            metadata = {
                "symbol": symbol,
                "processing_date": datetime.now().isoformat(),
                "total_rows": len(df),
                "date_range": {
                    "start": df.index.min().isoformat()
                    if hasattr(df.index, "min")
                    else None,
                    "end": df.index.max().isoformat()
                    if hasattr(df.index, "max")
                    else None,
                },
                "validation_results": validation_results,
                "columns": df.columns.tolist(),
            }

            metadata_path = (
                self.processed_data_dir / f"{symbol}_{suffix}_{timestamp}_metadata.json"
            )
            import json

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.info(f"Saved processed data: {file_path}")
            logger.info(f"Saved metadata: {metadata_path}")

            self.quality_stats["files_processed"] += 1

            return file_path

        except Exception as e:
            logger.error(f"Failed to save processed data for {symbol}: {str(e)}")
            raise DataPreprocessingError(
                f"Could not save processed data for {symbol}: {str(e)}"
            )

    def process_single_file(
        self,
        file_path: Path,
        missing_strategy: str = "auto",
        outlier_method: str = "iqr",
        outlier_action: str = "remove",
    ) -> Dict[str, Any]:
        """
        Process a single raw data file through the complete pipeline.

        Args:
            file_path: Path to raw data file
            missing_strategy: Strategy for handling missing values
            outlier_method: Method for outlier detection
            outlier_action: Action for outlier handling

        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing file: {file_path}")

        # Extract symbol from filename
        symbol = file_path.stem.split("_")[0]

        try:
            # Load raw data
            df = pd.read_csv(file_path)
            original_rows = len(df)
            self.quality_stats["total_rows_input"] += original_rows

            logger.info(f"Loaded {symbol}: {original_rows} rows, {df.shape[1]} columns")

            # Step 1: Handle missing values
            missing_analysis = self.identify_missing_values(df, symbol)
            df = self.handle_missing_values(df, symbol, missing_strategy)

            # Step 2: Filter outliers
            df = self.filter_outliers(df, symbol, outlier_method, action=outlier_action)

            # Step 3: Standardize date formats
            df = self.standardize_date_formats(df, symbol)

            # Step 4: Save processed data with integrity checks
            output_path = self.save_processed_data(df, symbol)

            final_rows = len(df)
            self.quality_stats["total_rows_output"] += final_rows

            processing_results = {
                "symbol": symbol,
                "status": "success",
                "input_file": str(file_path),
                "output_file": str(output_path),
                "original_rows": original_rows,
                "final_rows": final_rows,
                "rows_removed": original_rows - final_rows,
                "removal_percentage": (original_rows - final_rows)
                / original_rows
                * 100,
                "missing_analysis": missing_analysis,
                "processing_date": datetime.now().isoformat(),
            }

            logger.info(
                f"Successfully processed {symbol}: {original_rows} → {final_rows} rows ({processing_results['removal_percentage']:.2f}% removed)"
            )

            return processing_results

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return {
                "symbol": symbol,
                "status": "error",
                "input_file": str(file_path),
                "error": str(e),
                "processing_date": datetime.now().isoformat(),
            }

    def process_all_files(
        self,
        file_pattern: str = "*.csv",
        missing_strategy: str = "auto",
        outlier_method: str = "iqr",
        outlier_action: str = "remove",
    ) -> Dict[str, Any]:
        """
        Process all raw data files in the directory.

        Args:
            file_pattern: Pattern to match files
            missing_strategy: Strategy for handling missing values
            outlier_method: Method for outlier detection
            outlier_action: Action for outlier handling

        Returns:
            Dictionary with batch processing results
        """
        logger.info(f"Starting batch processing of files matching: {file_pattern}")

        # Find all matching files
        raw_files = list(self.raw_data_dir.glob(file_pattern))

        if not raw_files:
            logger.warning(f"No files found matching pattern: {file_pattern}")
            return {"status": "no_files", "files_processed": 0}

        logger.info(f"Found {len(raw_files)} files to process")

        # Reset stats for this batch
        batch_start_time = datetime.now()
        results = {
            "batch_start_time": batch_start_time.isoformat(),
            "files_found": len(raw_files),
            "files_processed": 0,
            "files_successful": 0,
            "files_failed": 0,
            "processing_results": [],
            "summary_stats": {},
            "errors": [],
        }

        # Process each file
        for file_path in raw_files:
            try:
                result = self.process_single_file(
                    file_path,
                    missing_strategy=missing_strategy,
                    outlier_method=outlier_method,
                    outlier_action=outlier_action,
                )

                results["processing_results"].append(result)
                results["files_processed"] += 1

                if result["status"] == "success":
                    results["files_successful"] += 1
                else:
                    results["files_failed"] += 1
                    results["errors"].append(result.get("error", "Unknown error"))

            except Exception as e:
                logger.error(f"Unexpected error processing {file_path}: {str(e)}")
                results["files_failed"] += 1
                results["errors"].append(str(e))

        # Calculate summary statistics
        batch_end_time = datetime.now()
        results["batch_end_time"] = batch_end_time.isoformat()
        results["total_processing_time"] = str(batch_end_time - batch_start_time)
        results["success_rate"] = (
            results["files_successful"] / len(raw_files) * 100 if raw_files else 0
        )
        results["quality_stats"] = self.quality_stats.copy()

        # Save batch report
        report_path = (
            self.processed_data_dir
            / f"batch_processing_report_{batch_start_time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        try:
            import json

            with open(report_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Saved batch processing report: {report_path}")
        except Exception as e:
            logger.error(f"Failed to save batch report: {str(e)}")

        logger.info(
            f"Batch processing complete: {results['files_successful']}/{len(raw_files)} files successful ({results['success_rate']:.1f}%)"
        )

        return results

    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get comprehensive quality statistics from processing."""
        return {
            "quality_stats": self.quality_stats.copy(),
            "data_quality_rate": (
                (
                    self.quality_stats["total_rows_output"]
                    / self.quality_stats["total_rows_input"]
                )
                * 100
                if self.quality_stats["total_rows_input"] > 0
                else 0
            ),
            "missing_value_rate": (
                (
                    self.quality_stats["missing_values_found"]
                    / self.quality_stats["total_rows_input"]
                )
                * 100
                if self.quality_stats["total_rows_input"] > 0
                else 0
            ),
            "outlier_rate": (
                (
                    self.quality_stats["outliers_detected"]
                    / self.quality_stats["total_rows_input"]
                )
                * 100
                if self.quality_stats["total_rows_input"] > 0
                else 0
            ),
        }


def preprocess_all_data(
    raw_data_dir: Optional[Path] = None,
    processed_data_dir: Optional[Path] = None,
    missing_strategy: str = "auto",
    outlier_method: str = "iqr",
    outlier_action: str = "remove",
) -> Dict[str, Any]:
    """
    Convenience function to preprocess all raw data files.

    Args:
        raw_data_dir: Directory containing raw data files
        processed_data_dir: Directory to save processed data
        missing_strategy: Strategy for handling missing values
        outlier_method: Method for outlier detection
        outlier_action: Action for outlier handling

    Returns:
        Dictionary with processing results
    """
    preprocessor = DataPreprocessor(raw_data_dir, processed_data_dir)
    return preprocessor.process_all_files(
        missing_strategy=missing_strategy,
        outlier_method=outlier_method,
        outlier_action=outlier_action,
    )


if __name__ == "__main__":
    # Example usage
    results = preprocess_all_data()
    print(
        f"Processing complete: {results['files_successful']}/{results['files_found']} files processed successfully"
    )
