"""
FundTuneLab Master Orchestration Script

This module provides end-to-end workflow orchestration for the FundTuneLab project,
sequentially executing data collection, preprocessing, optimization modules for all
three libraries, comparison engine, and backtesting analysis.
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import traceback

# Import all required modules
from .data_collection import download_default_assets
from .data_preprocessing import preprocess_all_data
from .pypfopt_optimizer import optimize_portfolio_from_data
from .eiten_optimizer import optimize_eiten_portfolios
from .riskfolio_optimizer import optimize_risk_parity_from_data
from .comparison import load_portfolios
from .backtesting import run_comprehensive_validation
from .unified_reporting import generate_unified_report
from config.settings import (
    RESULTS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    BACKTESTS_DIR,
    ensure_directories,
)

# Define the portfolios directory (not in settings.py)
PORTFOLIOS_DIR = RESULTS_DIR / "portfolios"


class FundTuneLabOrchestrator:
    """
    Master orchestrator for the FundTuneLab workflow.

    Coordinates the execution of all components in the correct sequence:
    1. Data Collection
    2. Data Preprocessing
    3. Portfolio Optimization (all three libraries)
    4. Portfolio Comparison
    5. Backtesting and Validation
    6. Report Generation
    """

    def __init__(self, log_level: int = logging.INFO):
        """
        Initialize the orchestrator.

        Args:
            log_level: Logging level for the orchestrator
        """
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)

        # Ensure all required directories exist
        ensure_directories()

        # Track execution state
        self.execution_state = {
            "start_time": None,
            "end_time": None,
            "total_duration": None,
            "stages_completed": [],
            "stages_failed": [],
            "results": {},
            "errors": [],
        }

        # Define the execution stages
        self.stages = [
            ("data_collection", "Data Collection", self._run_data_collection),
            ("data_preprocessing", "Data Preprocessing", self._run_data_preprocessing),
            (
                "pypfopt_optimization",
                "PyPortfolioOpt Optimization",
                self._run_pypfopt_optimization,
            ),
            ("eiten_optimization", "Eiten Optimization", self._run_eiten_optimization),
            (
                "riskfolio_optimization",
                "Riskfolio-Lib Optimization",
                self._run_riskfolio_optimization,
            ),
            (
                "portfolio_comparison",
                "Portfolio Comparison",
                self._run_portfolio_comparison,
            ),
            ("backtesting", "Backtesting & Validation", self._run_backtesting),
            (
                "report_generation",
                "Final Report Generation",
                self._run_report_generation,
            ),
        ]

    def setup_logging(self, log_level: int):
        """Set up comprehensive logging for the orchestrator."""
        # Create logs directory if it doesn't exist
        logs_dir = RESULTS_DIR / "logs"
        logs_dir.mkdir(exist_ok=True, parents=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"orchestrator_{timestamp}.log"

        # Configure logging format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(log_level)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    def run_full_workflow(
        self, skip_stages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete FundTuneLab workflow.

        Args:
            skip_stages: List of stage names to skip (useful for partial reruns)

        Returns:
            Dictionary containing execution results and metadata
        """
        self.logger.info("=" * 70)
        self.logger.info("STARTING FUNDTUNELAB FULL WORKFLOW")
        self.logger.info("=" * 70)

        self.execution_state["start_time"] = datetime.now()
        skip_stages = skip_stages or []

        for stage_id, stage_name, stage_func in self.stages:
            if stage_id in skip_stages:
                self.logger.info(f"SKIPPING: {stage_name}")
                continue

            self.logger.info(f"STARTING: {stage_name}")
            stage_start = time.time()

            try:
                result = stage_func()
                stage_duration = time.time() - stage_start

                self.execution_state["stages_completed"].append(stage_id)
                self.execution_state["results"][stage_id] = {
                    "success": True,
                    "duration": stage_duration,
                    "result": result,
                }

                self.logger.info(f"COMPLETED: {stage_name} in {stage_duration:.2f}s")

            except Exception as e:
                stage_duration = time.time() - stage_start
                error_msg = f"Failed in {stage_name}: {str(e)}"

                self.execution_state["stages_failed"].append(stage_id)
                self.execution_state["errors"].append(error_msg)
                self.execution_state["results"][stage_id] = {
                    "success": False,
                    "duration": stage_duration,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }

                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())

                # Continue with next stage unless it's a critical dependency
                if stage_id in ["data_collection", "data_preprocessing"]:
                    self.logger.error("Critical stage failed. Stopping workflow.")
                    break

        self.execution_state["end_time"] = datetime.now()
        self.execution_state["total_duration"] = (
            self.execution_state["end_time"] - self.execution_state["start_time"]
        ).total_seconds()

        self._save_execution_summary()
        self._print_final_summary()

        return self.execution_state

    def _run_data_collection(self) -> Dict[str, Any]:
        """Execute data collection stage."""
        self.logger.info("Downloading financial data for default assets...")
        results = download_default_assets()

        successful = sum(1 for data in results.values() if data is not None)
        total = len(results)

        self.logger.info(
            f"Data collection: {successful}/{total} assets downloaded successfully"
        )

        return {
            "total_assets": total,
            "successful_downloads": successful,
            "success_rate": successful / total if total > 0 else 0,
            "assets": list(results.keys()),
        }

    def _run_data_preprocessing(self) -> Dict[str, Any]:
        """Execute data preprocessing stage."""
        self.logger.info("Preprocessing downloaded data...")
        results = preprocess_all_data()

        self.logger.info(
            f"Data preprocessing: {results['files_successful']}/{results['files_found']} files processed"
        )

        return results

    def _run_pypfopt_optimization(self) -> Dict[str, Any]:
        """Execute PyPortfolioOpt optimization stage."""
        self.logger.info("Running PyPortfolioOpt optimization...")

        # Run multiple optimization methods
        methods = ["max_sharpe", "min_volatility", "efficient_return"]
        results = {}

        for method in methods:
            try:
                result = optimize_portfolio_from_data(
                    method=method, risk_free_rate=0.02, calculate_frontier=True
                )
                results[method] = result

                if result["success"]:
                    self.logger.info(
                        f"PyPortfolioOpt {method} optimization completed successfully"
                    )
                else:
                    self.logger.warning(
                        f"PyPortfolioOpt {method} optimization failed: {result.get('error', 'Unknown error')}"
                    )

            except Exception as e:
                self.logger.error(
                    f"PyPortfolioOpt {method} optimization error: {str(e)}"
                )
                results[method] = {"success": False, "error": str(e)}

        successful_methods = sum(1 for r in results.values() if r.get("success", False))

        return {
            "total_methods": len(methods),
            "successful_methods": successful_methods,
            "methods": results,
        }

    def _run_eiten_optimization(self) -> Dict[str, Any]:
        """Execute Eiten optimization stage."""
        self.logger.info("Running Eiten optimization...")

        try:
            results = optimize_eiten_portfolios()

            if results.get("success", False):
                self.logger.info("Eiten optimization completed successfully")
            else:
                self.logger.warning(
                    f"Eiten optimization failed: {results.get('error', 'Unknown error')}"
                )

            return results

        except Exception as e:
            self.logger.error(f"Eiten optimization error: {str(e)}")
            return {"success": False, "error": str(e)}

    def _run_riskfolio_optimization(self) -> Dict[str, Any]:
        """Execute Riskfolio-Lib optimization stage."""
        self.logger.info("Running Riskfolio-Lib optimization...")

        try:
            results = optimize_risk_parity_from_data()

            if results.get("success", False):
                self.logger.info("Riskfolio-Lib optimization completed successfully")
            else:
                self.logger.warning(
                    f"Riskfolio-Lib optimization failed: {results.get('error', 'Unknown error')}"
                )

            return results

        except Exception as e:
            self.logger.error(f"Riskfolio-Lib optimization error: {str(e)}")
            return {"success": False, "error": str(e)}

    def _run_portfolio_comparison(self) -> Dict[str, Any]:
        """Execute portfolio comparison stage."""
        self.logger.info("Running portfolio comparison analysis...")

        try:
            # Load portfolios and run comparison
            comparison = load_portfolios()

            # Generate comprehensive report
            generated_reports = comparison.generate_comprehensive_report()

            # Generate visualizations
            saved_plots = comparison.generate_all_visualizations()

            self.logger.info(
                f"Portfolio comparison completed: {len(generated_reports)} reports, {len(saved_plots)} plots generated"
            )

            return {
                "success": True,
                "reports_generated": len(generated_reports),
                "plots_generated": len(saved_plots),
                "report_files": generated_reports,
                "plot_files": saved_plots,
            }

        except Exception as e:
            self.logger.error(f"Portfolio comparison error: {str(e)}")
            return {"success": False, "error": str(e)}

    def _run_backtesting(self) -> Dict[str, Any]:
        """Execute backtesting and validation stage."""
        self.logger.info("Running comprehensive backtesting and validation...")

        try:
            results = run_comprehensive_validation()

            if results.get("success", False):
                self.logger.info("Backtesting completed successfully")
            else:
                self.logger.warning(
                    f"Backtesting failed: {results.get('error', 'Unknown error')}"
                )

            return results

        except Exception as e:
            self.logger.error(f"Backtesting error: {str(e)}")
            return {"success": False, "error": str(e)}

    def _run_report_generation(self) -> Dict[str, Any]:
        """Generate final consolidated reports."""
        self.logger.info("Generating final consolidated reports...")

        try:
            # Generate unified reports in multiple formats
            generated_files = generate_unified_report(
                output_dir=REPORTS_DIR,
                include_plots=True,
                format_types=["json", "html", "markdown", "csv"],
            )

            # Create master summary report (legacy format)
            summary_report = self._create_master_summary()

            # Save the summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = REPORTS_DIR / f"master_summary_{timestamp}.json"

            with open(summary_file, "w") as f:
                json.dump(summary_report, f, indent=2, default=str)

            # Add summary to generated files
            generated_files["master_summary"] = str(summary_file)

            self.logger.info(f"Generated {len(generated_files)} report files:")
            for format_type, file_path in generated_files.items():
                self.logger.info(f"  {format_type}: {file_path}")

            return {
                "success": True,
                "generated_files": generated_files,
                "summary": summary_report,
                "total_reports": len(generated_files),
            }

        except Exception as e:
            self.logger.error(f"Report generation error: {str(e)}")
            return {"success": False, "error": str(e)}

    def _create_master_summary(self) -> Dict[str, Any]:
        """Create a master summary of all workflow results."""
        return {
            "workflow_metadata": {
                "execution_time": self.execution_state["start_time"],
                "total_duration_seconds": self.execution_state["total_duration"],
                "stages_completed": self.execution_state["stages_completed"],
                "stages_failed": self.execution_state["stages_failed"],
            },
            "stage_results": self.execution_state["results"],
            "errors": self.execution_state["errors"],
            "files_generated": {
                "reports_dir": str(REPORTS_DIR),
                "plots_dir": str(PLOTS_DIR),
                "backtests_dir": str(BACKTESTS_DIR),
            },
        }

    def _save_execution_summary(self):
        """Save execution summary to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = RESULTS_DIR / "logs" / f"execution_summary_{timestamp}.json"

        with open(summary_file, "w") as f:
            json.dump(self.execution_state, f, indent=2, default=str)

        self.logger.info(f"Execution summary saved to {summary_file}")

    def _print_final_summary(self):
        """Print final execution summary."""
        self.logger.info("=" * 70)
        self.logger.info("WORKFLOW EXECUTION SUMMARY")
        self.logger.info("=" * 70)

        total_stages = len(self.stages)
        completed = len(self.execution_state["stages_completed"])
        failed = len(self.execution_state["stages_failed"])

        self.logger.info(
            f"Total Duration: {self.execution_state['total_duration']:.2f} seconds"
        )
        self.logger.info(f"Stages Completed: {completed}/{total_stages}")
        self.logger.info(f"Stages Failed: {failed}/{total_stages}")

        if self.execution_state["stages_completed"]:
            self.logger.info("Completed Stages:")
            for stage in self.execution_state["stages_completed"]:
                duration = self.execution_state["results"][stage]["duration"]
                self.logger.info(f"  ✓ {stage} ({duration:.2f}s)")

        if self.execution_state["stages_failed"]:
            self.logger.info("Failed Stages:")
            for stage in self.execution_state["stages_failed"]:
                self.logger.info(f"  ✗ {stage}")

        self.logger.info("=" * 70)


def run_orchestrator(
    skip_stages: Optional[List[str]] = None, log_level: int = logging.INFO
) -> Dict[str, Any]:
    """
    Convenience function to run the full FundTuneLab workflow.

    Args:
        skip_stages: List of stage names to skip
        log_level: Logging level

    Returns:
        Dictionary containing execution results
    """
    orchestrator = FundTuneLabOrchestrator(log_level=log_level)
    return orchestrator.run_full_workflow(skip_stages=skip_stages)


if __name__ == "__main__":
    # Run the full workflow
    print("FundTuneLab Master Orchestration")
    print("=" * 50)

    results = run_orchestrator()

    # Print final status
    if results["stages_failed"]:
        print(
            f"\nWorkflow completed with {len(results['stages_failed'])} failed stages."
        )
        print("Check the logs for detailed error information.")
    else:
        print("\nWorkflow completed successfully!")
        print("All results have been saved to the results/ directory.")
