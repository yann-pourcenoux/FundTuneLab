# FundTuneLab

A personal exploration of Python libraries for investment portfolio optimization, applied to real-world fund allocation. This project tests, compares, and documents various tools and techniques for building optimized investment strategies.

## Features

- Portfolio optimization using various Python libraries
- Real-world fund allocation strategies
- Performance backtesting and analysis
- Comparative analysis of optimization techniques
- Research-driven approach with Jupyter notebooks

## Installation

### Prerequisites

- Python 3.8+ (Python 3.13+ recommended)
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd FundTuneLab
   ```

2. Install dependencies using uv:

   ```bash
   uv sync
   ```

3. Activate the virtual environment:

   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

### Alternative: Direct execution with uv

You can run scripts directly without activating the virtual environment:

```bash
uv run python src/main.py
uv run jupyter notebook notebooks/
```

## Project Structure

```
FundTuneLab/
├── src/                    # Source code
├── data/                   # Data files
│   ├── raw/               # Original data files
│   ├── processed/         # Cleaned datasets
│   └── external/          # Third-party data
├── config/                # Configuration files
├── notebooks/             # Jupyter notebooks
├── results/               # Analysis outputs
│   ├── optimizations/     # Portfolio optimization results
│   ├── backtests/         # Performance backtests
│   ├── reports/           # Generated reports
│   └── plots/             # Visualizations
├── pyproject.toml         # Project dependencies
└── README.md              # This file
```

## Usage

### Running the Project

Start by exploring the Jupyter notebooks:

```bash
uv run jupyter notebook notebooks/
```

### Adding Dependencies

Add new Python packages using uv:

```bash
uv add numpy pandas matplotlib
uv add --dev pytest jupyter
```

### Development

1. Follow the project structure for organizing code
2. Use notebooks for exploration and research
3. Place reusable code in the `src/` directory
4. Store results and outputs in the `results/` directory

## Contributing

This is a personal research project, but suggestions and discussions are welcome!

## License

See LICENSE file for details.
