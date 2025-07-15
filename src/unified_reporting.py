"""
Unified Reporting Module for FundTuneLab

This module integrates outputs from data collection, preprocessing, optimization,
comparison, and backtesting into comprehensive, unified reports in multiple formats.
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import warnings

from config.settings import (
    RESULTS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)


class UnifiedReportGenerator:
    """
    Generates comprehensive unified reports by aggregating outputs from all
    FundTuneLab modules into cohesive formats.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the unified report generator.

        Args:
            output_dir: Directory for saving unified reports (defaults to REPORTS_DIR)
        """
        self.output_dir = output_dir or REPORTS_DIR
        self.logger = logging.getLogger(__name__)

        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Data containers
        self.portfolio_data = {}
        self.comparison_data = {}
        self.backtest_data = {}
        self.metadata = {
            "generation_time": datetime.now(),
            "data_sources": [],
            "files_processed": [],
            "errors": [],
        }

    def generate_comprehensive_report(
        self,
        include_plots: bool = True,
        format_types: List[str] = ["json", "html", "csv"],
    ) -> Dict[str, str]:
        """
        Generate a comprehensive unified report in multiple formats.

        Args:
            include_plots: Whether to include plot references in reports
            format_types: List of output formats to generate ("json", "html", "csv", "markdown")

        Returns:
            Dictionary mapping format types to generated file paths
        """
        self.logger.info("Starting comprehensive unified report generation...")

        # Step 1: Collect all data
        self._collect_portfolio_data()
        self._collect_comparison_data()
        self._collect_backtest_data()

        # Step 2: Generate unified data structure
        unified_data = self._create_unified_data_structure()

        # Step 3: Generate reports in requested formats
        generated_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for format_type in format_types:
            try:
                if format_type == "json":
                    file_path = self._generate_json_report(unified_data, timestamp)
                elif format_type == "html":
                    file_path = self._generate_html_report(
                        unified_data, timestamp, include_plots
                    )
                elif format_type == "csv":
                    file_path = self._generate_csv_reports(unified_data, timestamp)
                elif format_type == "markdown":
                    file_path = self._generate_markdown_report(
                        unified_data, timestamp, include_plots
                    )
                else:
                    self.logger.warning(f"Unsupported format type: {format_type}")
                    continue

                generated_files[format_type] = str(file_path)
                self.logger.info(f"Generated {format_type} report: {file_path}")

            except Exception as e:
                self.logger.error(f"Failed to generate {format_type} report: {e}")
                self.metadata["errors"].append(f"{format_type}: {str(e)}")

        return generated_files

    def _collect_portfolio_data(self):
        """Collect all portfolio optimization results."""
        portfolios_dir = RESULTS_DIR / "portfolios"

        if not portfolios_dir.exists():
            self.logger.warning("Portfolios directory not found")
            return

        # Collect JSON files (main portfolio data)
        json_files = list(portfolios_dir.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                # Extract optimizer name and timestamp from filename
                filename = json_file.stem
                if "eiten" in filename.lower():
                    optimizer = "eiten"
                elif "riskfolio" in filename.lower():
                    optimizer = "riskfolio"
                elif "portfolio_optimization" in filename.lower():
                    optimizer = "pypfopt"
                else:
                    optimizer = "unknown"

                if optimizer not in self.portfolio_data:
                    self.portfolio_data[optimizer] = {}

                self.portfolio_data[optimizer][filename] = data
                self.metadata["files_processed"].append(str(json_file))

            except Exception as e:
                self.logger.error(
                    f"Failed to load portfolio data from {json_file}: {e}"
                )
                self.metadata["errors"].append(f"Portfolio data: {str(e)}")

        # Collect CSV weight files
        csv_files = list(portfolios_dir.glob("*_weights.csv"))

        for csv_file in csv_files:
            try:
                weights_df = pd.read_csv(csv_file, index_col=0)

                # Link CSV to corresponding JSON data
                base_name = csv_file.stem.replace("_weights", "")

                # Find corresponding optimizer and add weights
                for optimizer, data in self.portfolio_data.items():
                    for key, portfolio_data in data.items():
                        if base_name in key:
                            portfolio_data["weights_dataframe"] = weights_df.to_dict()
                            break

                self.metadata["files_processed"].append(str(csv_file))

            except Exception as e:
                self.logger.error(f"Failed to load weights from {csv_file}: {e}")

    def _collect_comparison_data(self):
        """Collect portfolio comparison analysis results."""
        reports_dir = RESULTS_DIR / "reports"

        if not reports_dir.exists():
            self.logger.warning("Reports directory not found")
            return

        # Look for comparison JSON files
        comparison_files = list(reports_dir.glob("portfolio_analysis_*.json"))

        for comp_file in comparison_files:
            try:
                with open(comp_file, "r") as f:
                    data = json.load(f)

                timestamp = comp_file.stem.split("_")[-1]
                self.comparison_data[timestamp] = data
                self.metadata["files_processed"].append(str(comp_file))

            except Exception as e:
                self.logger.error(
                    f"Failed to load comparison data from {comp_file}: {e}"
                )
                self.metadata["errors"].append(f"Comparison data: {str(e)}")

        # Collect comparison CSV files
        csv_patterns = [
            "*_correlations.csv",
            "*_distances.csv",
            "*_metrics.csv",
            "*_weights.csv",
        ]

        for pattern in csv_patterns:
            csv_files = list(reports_dir.glob(pattern))

            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file, index_col=0)

                    # Extract timestamp and data type
                    parts = csv_file.stem.split("_")
                    timestamp = parts[-2] + "_" + parts[-1]  # Get timestamp
                    data_type = parts[-3]  # correlations, distances, metrics, weights

                    if timestamp not in self.comparison_data:
                        self.comparison_data[timestamp] = {}

                    self.comparison_data[timestamp][f"{data_type}_dataframe"] = (
                        df.to_dict()
                    )
                    self.metadata["files_processed"].append(str(csv_file))

                except Exception as e:
                    self.logger.error(
                        f"Failed to load CSV comparison data from {csv_file}: {e}"
                    )

    def _collect_backtest_data(self):
        """Collect backtesting results."""
        backtests_dir = RESULTS_DIR / "backtests"

        if not backtests_dir.exists():
            self.logger.warning("Backtests directory not found")
            return

        # Collect backtest JSON files
        backtest_files = list(backtests_dir.glob("*.json"))

        for bt_file in backtest_files:
            try:
                with open(bt_file, "r") as f:
                    data = json.load(f)

                filename = bt_file.stem
                self.backtest_data[filename] = data
                self.metadata["files_processed"].append(str(bt_file))

            except Exception as e:
                self.logger.error(f"Failed to load backtest data from {bt_file}: {e}")
                self.metadata["errors"].append(f"Backtest data: {str(e)}")

    def _create_unified_data_structure(self) -> Dict[str, Any]:
        """Create a unified data structure from all collected data."""

        # Calculate summary statistics
        portfolio_summary = self._calculate_portfolio_summary()
        comparison_summary = self._calculate_comparison_summary()
        backtest_summary = self._calculate_backtest_summary()

        # Get available plots
        available_plots = self._get_available_plots()

        unified_data = {
            "metadata": {
                "generation_time": self.metadata["generation_time"].isoformat(),
                "total_files_processed": len(self.metadata["files_processed"]),
                "files_processed": self.metadata["files_processed"],
                "errors": self.metadata["errors"],
                "data_sources": {
                    "portfolios": len(self.portfolio_data),
                    "comparison_analyses": len(self.comparison_data),
                    "backtests": len(self.backtest_data),
                    "plots": len(available_plots),
                },
            },
            "executive_summary": {
                "total_optimizers_used": len(self.portfolio_data),
                "optimizers": list(self.portfolio_data.keys()),
                "total_portfolios_generated": portfolio_summary["total_portfolios"],
                "comparison_analyses_available": len(self.comparison_data) > 0,
                "backtests_available": len(self.backtest_data) > 0,
                "plots_available": len(available_plots) > 0,
            },
            "portfolio_optimization": {
                "summary": portfolio_summary,
                "detailed_results": self.portfolio_data,
            },
            "portfolio_comparison": {
                "summary": comparison_summary,
                "detailed_results": self.comparison_data,
            },
            "backtesting": {
                "summary": backtest_summary,
                "detailed_results": self.backtest_data,
            },
            "visualizations": {
                "available_plots": available_plots,
                "plots_directory": str(PLOTS_DIR),
            },
        }

        return unified_data

    def _calculate_portfolio_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for portfolio optimization results."""
        total_portfolios = 0
        optimizer_portfolios = {}

        for optimizer, data in self.portfolio_data.items():
            optimizer_portfolios[optimizer] = len(data)
            total_portfolios += len(data)

        return {
            "total_portfolios": total_portfolios,
            "portfolios_by_optimizer": optimizer_portfolios,
            "optimizers_used": list(self.portfolio_data.keys()),
        }

    def _calculate_comparison_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for comparison analysis."""
        if not self.comparison_data:
            return {"available": False}

        # Get the most recent comparison analysis
        latest_timestamp = (
            max(self.comparison_data.keys()) if self.comparison_data else None
        )
        latest_data = (
            self.comparison_data.get(latest_timestamp, {}) if latest_timestamp else {}
        )

        return {
            "available": True,
            "total_analyses": len(self.comparison_data),
            "latest_analysis": latest_timestamp,
            "has_correlations": "correlations_dataframe" in latest_data,
            "has_distances": "distances_dataframe" in latest_data,
            "has_metrics": "metrics_dataframe" in latest_data,
        }

    def _calculate_backtest_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for backtest results."""
        if not self.backtest_data:
            return {"available": False}

        return {
            "available": True,
            "total_backtests": len(self.backtest_data),
            "backtest_files": list(self.backtest_data.keys()),
        }

    def _get_available_plots(self) -> List[Dict[str, str]]:
        """Get list of available plot files."""
        plots = []

        if PLOTS_DIR.exists():
            plot_files = (
                list(PLOTS_DIR.glob("*.png"))
                + list(PLOTS_DIR.glob("*.jpg"))
                + list(PLOTS_DIR.glob("*.svg"))
            )

            for plot_file in plot_files:
                plots.append(
                    {
                        "filename": plot_file.name,
                        "path": str(plot_file),
                        "type": self._identify_plot_type(plot_file.name),
                    }
                )

        return plots

    def _identify_plot_type(self, filename: str) -> str:
        """Identify the type of plot based on filename."""
        filename_lower = filename.lower()

        if "correlation" in filename_lower:
            return "correlation_analysis"
        elif "distance" in filename_lower:
            return "distance_analysis"
        elif "weights" in filename_lower:
            return "weights_comparison"
        elif "concentration" in filename_lower:
            return "concentration_analysis"
        else:
            return "other"

    def _generate_json_report(
        self, unified_data: Dict[str, Any], timestamp: str
    ) -> Path:
        """Generate comprehensive JSON report."""
        file_path = self.output_dir / f"unified_report_{timestamp}.json"

        with open(file_path, "w") as f:
            json.dump(unified_data, f, indent=2, default=str)

        return file_path

    def _generate_csv_reports(
        self, unified_data: Dict[str, Any], timestamp: str
    ) -> Path:
        """Generate CSV reports for tabular data."""
        csv_dir = self.output_dir / "csv_exports" / timestamp
        csv_dir.mkdir(exist_ok=True, parents=True)

        # Export portfolio weights if available
        portfolio_weights = []

        for optimizer, data in unified_data["portfolio_optimization"][
            "detailed_results"
        ].items():
            for portfolio_name, portfolio_data in data.items():
                if "weights_dataframe" in portfolio_data:
                    weights_df = pd.DataFrame(portfolio_data["weights_dataframe"])
                    weights_df["optimizer"] = optimizer
                    weights_df["portfolio"] = portfolio_name
                    portfolio_weights.append(weights_df)

        if portfolio_weights:
            combined_weights = pd.concat(portfolio_weights, ignore_index=True)
            weights_file = csv_dir / "all_portfolio_weights.csv"
            combined_weights.to_csv(weights_file, index=False)

        # Export comparison data if available
        for timestamp_key, comp_data in unified_data["portfolio_comparison"][
            "detailed_results"
        ].items():
            for data_type, df_data in comp_data.items():
                if "_dataframe" in data_type and isinstance(df_data, dict):
                    df = pd.DataFrame(df_data)
                    csv_file = csv_dir / f"comparison_{data_type}_{timestamp_key}.csv"
                    df.to_csv(csv_file)

        return csv_dir

    def _generate_html_report(
        self, unified_data: Dict[str, Any], timestamp: str, include_plots: bool
    ) -> Path:
        """Generate HTML report with rich formatting."""
        file_path = self.output_dir / f"unified_report_{timestamp}.html"

        html_content = self._create_html_content(unified_data, include_plots)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return file_path

    def _generate_markdown_report(
        self, unified_data: Dict[str, Any], timestamp: str, include_plots: bool
    ) -> Path:
        """Generate Markdown report."""
        file_path = self.output_dir / f"unified_report_{timestamp}.md"

        markdown_content = self._create_markdown_content(unified_data, include_plots)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        return file_path

    def _create_html_content(
        self, unified_data: Dict[str, Any], include_plots: bool
    ) -> str:
        """Create HTML content for the unified report."""

        generation_time = unified_data["metadata"]["generation_time"]
        summary = unified_data["executive_summary"]

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FundTuneLab Unified Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .section h3 {{ color: #34495e; }}
        .metric {{ display: inline-block; background-color: #3498db; color: white; padding: 10px 15px; margin: 5px; border-radius: 5px; }}
        .error {{ background-color: #e74c3c; color: white; padding: 10px; border-radius: 5px; margin: 5px 0; }}
        .success {{ background-color: #27ae60; color: white; padding: 10px; border-radius: 5px; margin: 5px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ border: 1px solid #bdc3c7; padding: 12px; text-align: left; }}
        th {{ background-color: #34495e; color: white; }}
        .plot-gallery {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .plot-item {{ border: 1px solid #bdc3c7; border-radius: 8px; padding: 15px; max-width: 300px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>FundTuneLab Unified Portfolio Analysis Report</h1>
        <p>Generated on: {generation_time}</p>
    </div>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <div class="metric">Optimizers Used: {summary["total_optimizers_used"]}</div>
        <div class="metric">Portfolios Generated: {summary["total_portfolios_generated"]}</div>
        <div class="metric">Comparison Available: {"Yes" if summary["comparison_analyses_available"] else "No"}</div>
        <div class="metric">Backtests Available: {"Yes" if summary["backtests_available"] else "No"}</div>
        <div class="metric">Plots Available: {"Yes" if summary["plots_available"] else "No"}</div>
        
        <h3>Optimizers Used:</h3>
        <ul>
        """

        for optimizer in summary["optimizers"]:
            html += f"<li>{optimizer.title()}</li>"

        html += """
        </ul>
    </div>
    """

        # Portfolio Optimization Section
        portfolio_data = unified_data["portfolio_optimization"]
        html += f"""
    <div class="section">
        <h2>Portfolio Optimization Results</h2>
        <p>Total portfolios generated: {portfolio_data["summary"]["total_portfolios"]}</p>
        """

        for optimizer, count in portfolio_data["summary"][
            "portfolios_by_optimizer"
        ].items():
            html += f"<div class='metric'>{optimizer.title()}: {count} portfolios</div>"

        html += """
    </div>
    """

        # Comparison Analysis Section
        comparison_data = unified_data["portfolio_comparison"]
        if comparison_data["summary"]["available"]:
            html += f"""
    <div class="section">
        <h2>Portfolio Comparison Analysis</h2>
        <div class="success">Comparison analysis completed successfully</div>
        <p>Latest analysis: {comparison_data["summary"]["latest_analysis"]}</p>
        <ul>
            <li>Correlations: {"Available" if comparison_data["summary"]["has_correlations"] else "Not Available"}</li>
            <li>Distance Metrics: {"Available" if comparison_data["summary"]["has_distances"] else "Not Available"}</li>
            <li>Portfolio Metrics: {"Available" if comparison_data["summary"]["has_metrics"] else "Not Available"}</li>
        </ul>
    </div>
    """
        else:
            html += """
    <div class="section">
        <h2>Portfolio Comparison Analysis</h2>
        <div class="error">No comparison analysis data available</div>
    </div>
    """

        # Backtesting Section
        backtest_data = unified_data["backtesting"]
        if backtest_data["summary"]["available"]:
            html += f"""
    <div class="section">
        <h2>Backtesting Results</h2>
        <div class="success">Backtesting completed successfully</div>
        <p>Total backtests: {backtest_data["summary"]["total_backtests"]}</p>
        <ul>
        """
            for backtest in backtest_data["summary"]["backtest_files"]:
                html += f"<li>{backtest}</li>"

            html += """
        </ul>
    </div>
    """
        else:
            html += """
    <div class="section">
        <h2>Backtesting Results</h2>
        <div class="error">No backtesting data available</div>
    </div>
    """

        # Visualizations Section
        if include_plots and unified_data["visualizations"]["available_plots"]:
            html += """
    <div class="section">
        <h2>Visualizations</h2>
        <div class="plot-gallery">
        """

            for plot in unified_data["visualizations"]["available_plots"]:
                html += f"""
        <div class="plot-item">
            <h4>{plot["filename"]}</h4>
            <p>Type: {plot["type"]}</p>
            <p>Path: {plot["path"]}</p>
        </div>
        """

            html += """
        </div>
    </div>
    """

        # Errors Section
        if unified_data["metadata"]["errors"]:
            html += """
    <div class="section">
        <h2>Errors and Warnings</h2>
        """
            for error in unified_data["metadata"]["errors"]:
                html += f"<div class='error'>{error}</div>"

            html += """
    </div>
    """

        html += """
</body>
</html>
        """

        return html

    def _create_markdown_content(
        self, unified_data: Dict[str, Any], include_plots: bool
    ) -> str:
        """Create Markdown content for the unified report."""

        generation_time = unified_data["metadata"]["generation_time"]
        summary = unified_data["executive_summary"]

        markdown = f"""# FundTuneLab Unified Portfolio Analysis Report

**Generated on:** {generation_time}

## Executive Summary

- **Optimizers Used:** {summary["total_optimizers_used"]}
- **Portfolios Generated:** {summary["total_portfolios_generated"]}
- **Comparison Available:** {"Yes" if summary["comparison_analyses_available"] else "No"}
- **Backtests Available:** {"Yes" if summary["backtests_available"] else "No"}
- **Plots Available:** {"Yes" if summary["plots_available"] else "No"}

### Optimizers Used:
"""

        for optimizer in summary["optimizers"]:
            markdown += f"- {optimizer.title()}\n"

        # Portfolio Optimization Section
        portfolio_data = unified_data["portfolio_optimization"]
        markdown += f"""
## Portfolio Optimization Results

Total portfolios generated: {portfolio_data["summary"]["total_portfolios"]}

### Portfolios by Optimizer:
"""

        for optimizer, count in portfolio_data["summary"][
            "portfolios_by_optimizer"
        ].items():
            markdown += f"- **{optimizer.title()}:** {count} portfolios\n"

        # Comparison Analysis Section
        comparison_data = unified_data["portfolio_comparison"]
        if comparison_data["summary"]["available"]:
            markdown += f"""
## Portfolio Comparison Analysis

✅ **Comparison analysis completed successfully**

- **Latest analysis:** {comparison_data["summary"]["latest_analysis"]}
- **Correlations:** {"Available" if comparison_data["summary"]["has_correlations"] else "Not Available"}
- **Distance Metrics:** {"Available" if comparison_data["summary"]["has_distances"] else "Not Available"}
- **Portfolio Metrics:** {"Available" if comparison_data["summary"]["has_metrics"] else "Not Available"}
"""
        else:
            markdown += """
## Portfolio Comparison Analysis

❌ **No comparison analysis data available**
"""

        # Backtesting Section
        backtest_data = unified_data["backtesting"]
        if backtest_data["summary"]["available"]:
            markdown += f"""
## Backtesting Results

✅ **Backtesting completed successfully**

- **Total backtests:** {backtest_data["summary"]["total_backtests"]}

### Backtest Files:
"""
            for backtest in backtest_data["summary"]["backtest_files"]:
                markdown += f"- {backtest}\n"
        else:
            markdown += """
## Backtesting Results

❌ **No backtesting data available**
"""

        # Visualizations Section
        if include_plots and unified_data["visualizations"]["available_plots"]:
            markdown += """
## Visualizations

### Available Plots:
"""

            for plot in unified_data["visualizations"]["available_plots"]:
                markdown += f"""
#### {plot["filename"]}
- **Type:** {plot["type"]}
- **Path:** {plot["path"]}
"""

        # Errors Section
        if unified_data["metadata"]["errors"]:
            markdown += """
## Errors and Warnings

"""
            for error in unified_data["metadata"]["errors"]:
                markdown += f"❌ {error}\n\n"

        markdown += f"""
---

**Report generated by FundTuneLab Unified Reporting Module**  
**Files processed:** {len(unified_data["metadata"]["files_processed"])}  
**Data sources:** Portfolios: {len(unified_data["portfolio_optimization"]["detailed_results"])}, Comparisons: {len(unified_data["portfolio_comparison"]["detailed_results"])}, Backtests: {len(unified_data["backtesting"]["detailed_results"])}
"""

        return markdown


def generate_unified_report(
    output_dir: Optional[Path] = None,
    include_plots: bool = True,
    format_types: List[str] = ["json", "html", "markdown"],
) -> Dict[str, str]:
    """
    Convenience function to generate unified reports.

    Args:
        output_dir: Directory for saving reports
        include_plots: Whether to include plot references
        format_types: List of output formats to generate

    Returns:
        Dictionary mapping format types to generated file paths
    """
    generator = UnifiedReportGenerator(output_dir)
    return generator.generate_comprehensive_report(include_plots, format_types)


if __name__ == "__main__":
    # Generate sample unified report
    print("FundTuneLab Unified Reporting")
    print("=" * 40)

    generated_files = generate_unified_report()

    print("Generated unified reports:")
    for format_type, file_path in generated_files.items():
        print(f"  {format_type}: {file_path}")
