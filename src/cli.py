"""
Command-Line Interface for FundTuneLab

This module provides a CLI to invoke the master orchestration script,
allowing users to pass configuration options and control workflow execution.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

try:
    from .orchestrator import run_orchestrator
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from orchestrator import run_orchestrator


def setup_cli_parser() -> argparse.ArgumentParser:
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="FundTuneLab Portfolio Optimization Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run full workflow
  %(prog)s --skip data_collection             # Skip data collection stage
  %(prog)s --verbose                          # Run with verbose logging
  %(prog)s --skip data_collection preprocessing  # Skip multiple stages
  %(prog)s --quiet --skip backtesting         # Run quietly, skip backtesting

Available stages to skip:
  data_collection, data_preprocessing, pypfopt_optimization,
  eiten_optimization, riskfolio_optimization, portfolio_comparison,
  backtesting, report_generation
        """,
    )

    # Main execution options
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        metavar="STAGE",
        help="Skip one or more workflow stages (space-separated list)",
    )

    # Logging options
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )
    log_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Enable quiet logging (WARNING level only)",
    )

    # Information options
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List all available workflow stages and exit",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running (not implemented yet)",
    )

    # Output options
    parser.add_argument(
        "--output-dir", type=Path, help="Override default output directory (results/)"
    )

    return parser


def validate_skip_stages(skip_stages: List[str]) -> List[str]:
    """
    Validate that the skip stages are valid stage names.

    Args:
        skip_stages: List of stage names to validate

    Returns:
        List of validated stage names

    Raises:
        SystemExit: If any invalid stage names are provided
    """
    valid_stages = {
        "data_collection",
        "data_preprocessing",
        "pypfopt_optimization",
        "eiten_optimization",
        "riskfolio_optimization",
        "portfolio_comparison",
        "backtesting",
        "report_generation",
    }

    invalid_stages = [stage for stage in skip_stages if stage not in valid_stages]

    if invalid_stages:
        print(f"Error: Invalid stage names: {', '.join(invalid_stages)}")
        print(f"Valid stages are: {', '.join(sorted(valid_stages))}")
        sys.exit(1)

    return skip_stages


def print_stages_info():
    """Print information about all available workflow stages."""
    stages_info = [
        ("data_collection", "Download financial data for default assets"),
        ("data_preprocessing", "Clean and preprocess downloaded data"),
        ("pypfopt_optimization", "Run PyPortfolioOpt optimization methods"),
        ("eiten_optimization", "Run Eiten portfolio optimization"),
        ("riskfolio_optimization", "Run Riskfolio-Lib optimization"),
        ("portfolio_comparison", "Compare and analyze all portfolio results"),
        ("backtesting", "Run comprehensive backtesting and validation"),
        ("report_generation", "Generate final consolidated reports"),
    ]

    print("FundTuneLab Workflow Stages:")
    print("=" * 60)

    for stage_id, description in stages_info:
        print(f"{stage_id:25} {description}")

    print("\nUse --skip <stage_name> to skip specific stages.")


def main():
    """Main CLI entry point."""
    parser = setup_cli_parser()
    args = parser.parse_args()

    # Handle information requests
    if args.list_stages:
        print_stages_info()
        sys.exit(0)

    # Set up logging level
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO

    # Validate skip stages
    skip_stages = validate_skip_stages(args.skip)

    # Print execution info
    print("FundTuneLab Portfolio Optimization Workflow")
    print("=" * 50)

    if skip_stages:
        print(f"Skipping stages: {', '.join(skip_stages)}")

    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
        # TODO: Implement output directory override

    print("")  # Empty line for readability

    try:
        # Run the orchestrator
        results = run_orchestrator(skip_stages=skip_stages, log_level=log_level)

        # Print final status
        completed = len(results["stages_completed"])
        failed = len(results["stages_failed"])
        total = completed + failed

        print("\n" + "=" * 50)
        print("EXECUTION SUMMARY")
        print("=" * 50)

        if failed == 0:
            print(f"✓ All {completed} stages completed successfully!")
            print("All results have been saved to the results/ directory.")
            sys.exit(0)
        else:
            print("⚠ Workflow completed with issues:")
            print(f"  ✓ Completed: {completed}/{total} stages")
            print(f"  ✗ Failed: {failed}/{total} stages")
            print("\nCheck the logs in results/logs/ for detailed error information.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
