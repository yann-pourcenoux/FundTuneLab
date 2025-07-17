"""
Configuration settings for FundTuneLab portfolio optimization project.

This module provides centralized configuration for data sources, optimization parameters,
file paths, and other project settings. All modules should import settings from here
to ensure consistency across the project.
"""

from pathlib import Path
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Root directory of the project
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Source code directory
SRC_DIR = PROJECT_ROOT / "src"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
OPTIMIZATIONS_DIR = RESULTS_DIR / "optimizations"
BACKTESTS_DIR = RESULTS_DIR / "backtests"
REPORTS_DIR = RESULTS_DIR / "reports"
PLOTS_DIR = RESULTS_DIR / "plots"

# Configuration directory
CONFIG_DIR = PROJECT_ROOT / "config"

# Notebooks directory
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ============================================================================
# DATA SETTINGS
# ============================================================================

# Default date range for historical data
DEFAULT_START_DATE = os.getenv("FUNDTUNELAB_START_DATE", "2020-01-01")
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")

# Data providers and sources
DATA_PROVIDERS = {
    "yahoo": {
        "enabled": True,
        "rate_limit": 2000,  # requests per hour
        "timeout": 30,  # seconds
    }
}

# Default asset universe for optimization
DEFAULT_ASSETS = [
    # US Equity ETFs
    "SPY",  # S&P 500
    "VTI",  # Total Stock Market
    "QQQ",  # NASDAQ 100
    "IWM",  # Russell 2000
    # International Equity ETFs
    "VEA",  # Developed Markets
    "VWO",  # Emerging Markets
    "EFA",  # EAFE
    # Bond ETFs
    "BND",  # Total Bond Market
    "TLT",  # 20+ Year Treasury
    "SHY",  # 1-3 Year Treasury
    "TIPS",  # Treasury Inflation-Protected Securities
    # Alternative Assets
    "VNQ",  # Real Estate
    "GLD",  # Gold
    "DBC",  # Commodities
]

# Benchmark for performance comparison
DEFAULT_BENCHMARK = os.getenv("FUNDTUNELAB_BENCHMARK", "SPY")

# ============================================================================
# OPTIMIZATION SETTINGS
# ============================================================================

# Portfolio optimization methods
OPTIMIZATION_METHODS = {
    "mean_variance": {
        "enabled": True,
        "risk_aversion": 1.0,
        "max_weight": 0.4,
        "min_weight": 0.0,
    },
    "risk_parity": {
        "enabled": True,
        "max_weight": 0.3,
        "min_weight": 0.05,
    },
    "hierarchical_risk_parity": {
        "enabled": True,
        "distance_metric": "correlation",
        "linkage_method": "ward",
    },
    "black_litterman": {
        "enabled": False,
        "tau": 0.05,
        "confidence": 0.25,
    },
    "cvar": {
        "enabled": True,
        "alpha": 0.05,  # 95% CVaR
        "max_weight": 0.4,
        "min_weight": 0.0,
    },
}

# Risk models
RISK_MODELS = {
    "sample_covariance": {
        "enabled": True,
        "frequency": "daily",  # daily, weekly, monthly
        "lookback_days": 252,  # 1 year of trading days
    },
    "ledoit_wolf": {
        "enabled": True,
        "frequency": "daily",
        "lookback_days": 252,
    },
    "exponential_covariance": {
        "enabled": True,
        "frequency": "daily",
        "lookback_days": 252,
        "alpha": 0.94,  # decay factor
    },
}

# Expected returns models
EXPECTED_RETURNS_MODELS = {
    "mean_historical_return": {
        "enabled": True,
        "frequency": "daily",
        "lookback_days": 252,
    },
    "capm_return": {
        "enabled": True,
        "frequency": "daily",
        "lookback_days": 252,
        "market_proxy": "SPY",
    },
    "ema_historical_return": {
        "enabled": True,
        "frequency": "daily",
        "lookback_days": 252,
        "alpha": 0.94,
    },
}

# ============================================================================
# BACKTESTING SETTINGS
# ============================================================================

# Backtesting parameters
BACKTESTING = {
    "initial_capital": int(
        os.getenv("FUNDTUNELAB_INITIAL_CAPITAL", "100000")
    ),  # $100,000
    "rebalance_frequency": "monthly",  # daily, weekly, monthly, quarterly
    "transaction_cost": 0.001,  # 0.1% per trade
    "min_trade_size": 100,  # minimum trade in dollars
    "cash_buffer": 0.02,  # 2% cash buffer
    "slippage": 0.0005,  # 0.05% slippage
}

# Performance metrics to calculate
PERFORMANCE_METRICS = [
    "total_return",
    "annualized_return",
    "volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "information_ratio",
    "alpha",
    "beta",
    "var_95",
    "cvar_95",
]

# ============================================================================
# PLOTTING AND VISUALIZATION SETTINGS
# ============================================================================

# Plot settings
PLOT_SETTINGS = {
    "style": "seaborn-v0_8",
    "figsize": (12, 8),
    "dpi": 300,
    "save_format": "png",
    "color_palette": "Set2",
    "font_size": 12,
    "title_font_size": 14,
    "save_plots": True,
}

# Report generation settings
REPORT_SETTINGS = {
    "format": "html",  # html, pdf, markdown
    "include_plots": True,
    "template": "default",
    "auto_save": True,
}

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

# Logging configuration
LOGGING = {
    "level": os.getenv(
        "FUNDTUNELAB_LOG_LEVEL", "INFO"
    ),  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": RESULTS_DIR / "logs" / "fundtunelab.log",
    "max_file_size": "10MB",
    "backup_count": 5,
    "console_output": True,
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        RESULTS_DIR,
        OPTIMIZATIONS_DIR,
        BACKTESTS_DIR,
        REPORTS_DIR,
        PLOTS_DIR,
        RESULTS_DIR / "logs",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_data_file_path(filename: str, data_type: str = "processed") -> Path:
    """Get the full path for a data file."""
    if data_type == "raw":
        return RAW_DATA_DIR / filename
    elif data_type == "processed":
        return PROCESSED_DATA_DIR / filename
    elif data_type == "external":
        return EXTERNAL_DATA_DIR / filename
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def get_results_file_path(filename: str, result_type: str = "optimizations") -> Path:
    """Get the full path for a results file."""
    if result_type == "optimizations":
        return OPTIMIZATIONS_DIR / filename
    elif result_type == "backtests":
        return BACKTESTS_DIR / filename
    elif result_type == "reports":
        return REPORTS_DIR / filename
    elif result_type == "plots":
        return PLOTS_DIR / filename
    else:
        raise ValueError(f"Unknown result_type: {result_type}")


def validate_settings():
    """Validate that all settings are properly configured."""
    errors = []

    # Check that required directories exist or can be created
    try:
        ensure_directories()
    except Exception as e:
        errors.append(f"Cannot create directories: {e}")

    # Validate asset lists
    if not DEFAULT_ASSETS:
        errors.append("DEFAULT_ASSETS cannot be empty")

    if not DEFAULT_BENCHMARK:
        errors.append("DEFAULT_BENCHMARK must be specified")

    # Validate date ranges
    try:
        from datetime import datetime

        datetime.strptime(DEFAULT_START_DATE, "%Y-%m-%d")
        datetime.strptime(DEFAULT_END_DATE, "%Y-%m-%d")
    except ValueError as e:
        errors.append(f"Invalid date format: {e}")

    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    return True


# Auto-create directories when module is imported
ensure_directories()
