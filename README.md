# FundTuneLab

A personal exploration of Python libraries for investment portfolio optimization, applied to real-world fund allocation.

## Overview

FundTuneLab is a comprehensive portfolio optimization platform that integrates three leading Python optimization libraries:

- **PyPortfolioOpt**: Modern portfolio theory and risk models
- **Riskfolio-Lib**: Advanced risk parity and risk budgeting
- **Eiten**: Genetic algorithms and eigenvalue-based optimization

The platform provides end-to-end workflow automation from data collection to report generation, with comprehensive comparison and backtesting capabilities.

## Features

### 🎯 **End-to-End Workflow Automation**

- **Master Orchestration**: Automated workflow from data collection to final reporting
- **Command-Line Interface**: Easy-to-use CLI for running complete analyses
- **Modular Architecture**: Skip stages or run partial workflows as needed

### 📊 **Multi-Library Optimization**

- **PyPortfolioOpt**: Maximum Sharpe ratio, minimum volatility, efficient frontier
- **Riskfolio-Lib**: Risk parity, hierarchical risk parity, CVaR optimization  
- **Eiten**: Genetic algorithms, eigenvalue portfolios, minimum variance

### 🔍 **Advanced Analysis & Comparison**

- **Portfolio Comparison**: Side-by-side analysis of different optimization approaches
- **Performance Metrics**: Comprehensive risk-return analysis
- **Correlation Analysis**: Understanding relationships between strategies
- **Distance Metrics**: Quantitative similarity measurements

### 📈 **Comprehensive Backtesting**

- **Historical Performance**: Out-of-sample validation
- **Risk Metrics**: Drawdown analysis, volatility tracking
- **Benchmark Comparison**: Performance vs market indices

### 📋 **Rich Reporting**

- **Multiple Formats**: JSON, HTML, Markdown, and CSV exports
- **Interactive Visualizations**: Correlation heatmaps, weight comparisons
- **Executive Summaries**: High-level insights and recommendations

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd FundTuneLab

# Install dependencies using uv
uv sync
```

### 2. Running the Complete Workflow

```bash
# Run the full end-to-end workflow
uv run python src/cli.py

# Run with verbose logging
uv run python src/cli.py --verbose

# Skip specific stages (e.g., skip backtesting)
uv run python src/cli.py --skip backtesting

# See all available options
uv run python src/cli.py --help
```

### 3. Alternative: Using the CLI Entry Point

```bash
# Install as a package (optional)
uv pip install -e .

# Run using the installed command
fundtunelab

# With options
fundtunelab --skip data_collection --verbose
```

## Workflow Stages

The FundTuneLab workflow consists of 8 sequential stages:

| Stage | Description | Duration |
|-------|-------------|----------|
| `data_collection` | Download financial data for default assets | ~30s |
| `data_preprocessing` | Clean and validate financial data | ~15s |
| `pypfopt_optimization` | Run PyPortfolioOpt optimizations | ~45s |
| `eiten_optimization` | Run Eiten optimizations | ~60s |
| `riskfolio_optimization` | Run Riskfolio-Lib optimizations | ~30s |
| `portfolio_comparison` | Compare and analyze all portfolios | ~20s |
| `backtesting` | Run comprehensive backtesting | ~90s |
| `report_generation` | Generate unified reports | ~10s |

### Skipping Stages

You can skip any stage to speed up development or focus on specific analyses:

```bash
# Skip data collection and preprocessing (use existing data)
uv run python src/cli.py --skip data_collection data_preprocessing

# Skip all optimizations (focus on reporting)
uv run python src/cli.py --skip pypfopt_optimization eiten_optimization riskfolio_optimization

# Skip backtesting for faster iteration
uv run python src/cli.py --skip backtesting
```

## Usage Examples

### Basic Analysis

```bash
# Run complete analysis with default settings
uv run python src/cli.py

# Results will be saved to:
# - results/portfolios/     # Individual portfolio files
# - results/reports/        # Unified reports (HTML, JSON, Markdown)
# - results/plots/          # Visualization files
# - results/backtests/      # Backtesting results
```

### Development Workflow

```bash
# Quick iteration: skip data collection and backtesting
uv run python src/cli.py --skip data_collection backtesting

# Debug mode: verbose logging, skip expensive operations
uv run python src/cli.py --verbose --skip eiten_optimization backtesting

# Reporting only: generate reports from existing data
uv run python src/cli.py --skip data_collection data_preprocessing pypfopt_optimization eiten_optimization riskfolio_optimization portfolio_comparison backtesting
```

## Output Structure

After running the workflow, results are organized as follows:

```
results/
├── portfolios/           # Individual optimization results
│   ├── *.json           # Portfolio metadata and performance
│   └── *_weights.csv    # Asset allocation weights
├── reports/             # Unified analysis reports
│   ├── unified_report_*.html      # Rich HTML report
│   ├── unified_report_*.json      # Complete data export
│   ├── unified_report_*.md        # Markdown summary
│   └── csv_exports/              # Tabular data exports
├── plots/               # Visualization outputs
│   ├── correlation_heatmap_*.png
│   ├── weights_comparison_*.png
│   └── concentration_analysis_*.png
├── backtests/           # Backtesting results
│   └── *.json          # Performance analysis
└── logs/                # Execution logs
    ├── orchestrator_*.log
    └── execution_summary_*.json
```

## Advanced Usage

### Programmatic API

```python
from src.orchestrator import run_orchestrator

# Run with custom configuration
results = run_orchestrator(
    skip_stages=["backtesting"],
    log_level=logging.DEBUG
)

# Access results
print(f"Completed: {len(results['stages_completed'])} stages")
print(f"Failed: {len(results['stages_failed'])} stages")
```

### Custom Reporting

```python
from src.unified_reporting import generate_unified_report

# Generate custom reports
reports = generate_unified_report(
    output_dir="custom_reports/",
    include_plots=True,
    format_types=["html", "json"]
)
```

## Configuration

### Asset Universe

Default assets are configured in `config/settings.py`:

```python
DEFAULT_ASSETS = [
    # US Equity ETFs
    "SPY", "VTI", "QQQ", "IWM",
    # International Equity ETFs  
    "VEA", "VWO", "EFA",
    # Bond ETFs
    "BND", "TLT", "SHY", "TIPS",
    # Alternative Assets
    "VNQ", "GLD", "DBC"
]
```

### Optimization Parameters

Each optimizer can be configured in `config/settings.py`:

```python
OPTIMIZATION_METHODS = {
    "mean_variance": {
        "enabled": True,
        "risk_aversion": 1.0,
        "max_weight": 0.4,
    },
    "risk_parity": {
        "enabled": True,
        "max_weight": 0.3,
    },
    # ... more methods
}
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
uv run pytest

# Run integration tests only
uv run pytest tests/test_integration.py -v

# Run with coverage
uv run pytest --cov=src tests/
```

## Development

### Project Structure

```
FundTuneLab/
├── src/                     # Source code
│   ├── orchestrator.py      # Master workflow orchestration
│   ├── cli.py              # Command-line interface
│   ├── unified_reporting.py # Multi-format report generation
│   ├── data_collection.py   # Financial data download
│   ├── data_preprocessing.py # Data cleaning and validation
│   ├── pypfopt_optimizer.py # PyPortfolioOpt integration
│   ├── eiten_optimizer.py   # Eiten integration
│   ├── riskfolio_optimizer.py # Riskfolio-Lib integration
│   ├── comparison.py        # Portfolio comparison engine
│   └── backtesting.py       # Performance validation
├── config/                  # Configuration settings
├── tests/                   # Test suite
├── data/                    # Data storage
├── results/                 # Analysis outputs
└── notebooks/               # Jupyter notebooks
```

### Adding New Optimizers

1. Create a new optimizer module in `src/`
2. Implement the standard interface:

   ```python
   def optimize_from_data(**kwargs):
       # Implementation
       return {
           "success": True,
           "weights": {...},
           "performance": {...}
       }
   ```

3. Add the optimizer to `orchestrator.py`
4. Update the CLI help text

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Dependencies

- **Core**: NumPy, Pandas, SciPy
- **Optimization**: PyPortfolioOpt, Riskfolio-Lib, scikit-learn
- **Data**: yfinance, python-dotenv
- **Visualization**: Matplotlib, Seaborn
- **Backtesting**: vectorbt
- **Development**: pytest, pytest-mock, ruff

## Acknowledgments

- PyPortfolioOpt team for excellent portfolio optimization tools
- Riskfolio-Lib developers for advanced risk management capabilities  
- Eiten project for genetic algorithm implementations
- yfinance for reliable financial data access
