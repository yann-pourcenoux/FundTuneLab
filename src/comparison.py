"""
Portfolio Comparison Engine for FundTuneLab

This module provides functionality to compare and analyze portfolio allocations
from different optimization modules (PyPortfolioOpt, Riskfolio-Lib, Eiten).
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from loguru import logger


class PortfolioComparison:
    """
    A class to handle comparison and analysis of portfolio allocations
    from different optimization engines.
    """

    def __init__(self, portfolios_dir: str = "results/portfolios"):
        """
        Initialize the PortfolioComparison object.

        Args:
            portfolios_dir: Directory containing portfolio results
        """
        self.portfolios_dir = Path(portfolios_dir)
        self.portfolio_data = {}
        self.comparison_results = {}

        if not self.portfolios_dir.exists():
            raise FileNotFoundError(f"Portfolio directory {portfolios_dir} not found")

    def load_portfolio_data(self) -> Dict[str, Dict]:
        """
        Load portfolio data from all available optimization results.

        Returns:
            Dictionary containing loaded portfolio data organized by optimizer and strategy
        """
        logger.info(f"Loading portfolio data from {self.portfolios_dir}")

        # Find all JSON files (metadata files)
        json_files = list(self.portfolios_dir.glob("*.json"))

        for json_file in json_files:
            try:
                portfolio_info = self._load_single_portfolio(json_file)
                if portfolio_info:
                    # Handle special case for multiple strategies (Eiten)
                    if isinstance(portfolio_info, dict) and portfolio_info.get(
                        "multiple_strategies"
                    ):
                        for strategy_info in portfolio_info["strategies"]:
                            optimizer_name = strategy_info["optimizer"]
                            strategy = strategy_info["strategy"]
                            timestamp = strategy_info["timestamp"]

                            # Create nested structure: optimizer -> strategy -> data
                            if optimizer_name not in self.portfolio_data:
                                self.portfolio_data[optimizer_name] = {}

                            key = f"{strategy}_{timestamp}"
                            self.portfolio_data[optimizer_name][key] = strategy_info

                            logger.info(
                                f"Loaded {optimizer_name} - {strategy} ({timestamp})"
                            )
                    else:
                        # Handle single strategy portfolios
                        optimizer_name = portfolio_info["optimizer"]
                        strategy = portfolio_info["strategy"]
                        timestamp = portfolio_info["timestamp"]

                        # Create nested structure: optimizer -> strategy -> data
                        if optimizer_name not in self.portfolio_data:
                            self.portfolio_data[optimizer_name] = {}

                        key = f"{strategy}_{timestamp}"
                        self.portfolio_data[optimizer_name][key] = portfolio_info

                        logger.info(
                            f"Loaded {optimizer_name} - {strategy} ({timestamp})"
                        )

            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                continue

        logger.info(f"Successfully loaded {len(self.portfolio_data)} optimizer results")
        return self.portfolio_data

    def _load_single_portfolio(self, json_file: Path) -> Optional[Dict]:
        """
        Load a single portfolio from JSON metadata file.

        Args:
            json_file: Path to the JSON metadata file

        Returns:
            Dictionary containing portfolio information
        """
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Determine optimizer type and extract relevant information
            filename = json_file.stem

            if filename.startswith("portfolio_optimization"):
                # PyPortfolioOpt format
                portfolio_info = {
                    "optimizer": "PyPortfolioOpt",
                    "strategy": "max_sharpe",  # Default strategy inferred
                    "timestamp": data.get("timestamp", ""),
                    "weights": data.get("weights", {}),
                    "performance": {
                        "expected_return": data.get("expected_return"),
                        "volatility": data.get("volatility"),
                        "sharpe_ratio": data.get("sharpe_ratio"),
                    },
                    "assets": data.get("assets", []),
                    "risk_free_rate": data.get("risk_free_rate", 0.02),
                    "metadata": data,
                }

            elif filename.startswith("riskfolio"):
                # Riskfolio-Lib format
                portfolio_info = {
                    "optimizer": "Riskfolio-Lib",
                    "strategy": "risk_parity",
                    "timestamp": data.get("timestamp", ""),
                    "weights": data.get("weights", {}),
                    "performance": {
                        "expected_return": data.get("expected_return"),
                        "volatility": data.get("volatility"),
                        "sharpe_ratio": data.get("sharpe_ratio"),
                    },
                    "assets": data.get("assets", []),
                    "risk_free_rate": data.get("risk_free_rate", 0.02),
                    "metadata": data,
                }

            elif filename.startswith("eiten"):
                # Eiten format (multiple strategies in one file)
                strategies = []
                base_info = {
                    "optimizer": "Eiten",
                    "timestamp": data.get("metadata", {}).get("timestamp", ""),
                    "assets": data.get("metadata", {}).get("symbols", []),
                    "risk_free_rate": data.get("metadata", {}).get(
                        "risk_free_rate", 0.02
                    ),
                    "metadata": data,
                }

                # Extract individual strategies
                optimization_results = data.get("optimization_results", {})
                for strategy_name, strategy_data in optimization_results.items():
                    strategy_info = base_info.copy()
                    strategy_info.update(
                        {
                            "strategy": strategy_name.replace("_", " ").title(),
                            "weights": strategy_data.get("weights", {}),
                            "performance": strategy_data.get("performance", {}),
                        }
                    )
                    strategies.append(strategy_info)

                # Return all strategies for Eiten
                if strategies:
                    return {"multiple_strategies": True, "strategies": strategies}
                else:
                    return None
            else:
                logger.warning(f"Unknown portfolio format: {filename}")
                return None

            return portfolio_info

        except Exception as e:
            logger.error(f"Error parsing {json_file}: {e}")
            return None

    def get_common_assets(self) -> List[str]:
        """
        Get the list of assets that are common across all portfolios.

        Returns:
            List of common asset symbols
        """
        all_assets = []

        for optimizer_data in self.portfolio_data.values():
            for portfolio_info in optimizer_data.values():
                all_assets.append(set(portfolio_info["assets"]))

        if not all_assets:
            return []

        # Find intersection of all asset sets
        common_assets = set.intersection(*all_assets)
        return sorted(list(common_assets))

    def create_weights_dataframe(self, common_assets_only: bool = True) -> pd.DataFrame:
        """
        Create a DataFrame with portfolio weights for comparison.

        Args:
            common_assets_only: If True, only include assets common to all portfolios

        Returns:
            DataFrame with assets as rows and portfolios as columns
        """
        if common_assets_only:
            assets = self.get_common_assets()
        else:
            # Get all unique assets
            all_assets = set()
            for optimizer_data in self.portfolio_data.values():
                for portfolio_info in optimizer_data.values():
                    all_assets.update(portfolio_info["assets"])
            assets = sorted(list(all_assets))

        # Create DataFrame
        columns = []
        weights_data = []

        for optimizer_name, optimizer_data in self.portfolio_data.items():
            for strategy_key, portfolio_info in optimizer_data.items():
                strategy = portfolio_info["strategy"]
                timestamp = portfolio_info["timestamp"]
                column_name = f"{optimizer_name}_{strategy}_{timestamp}"
                columns.append(column_name)

                # Extract weights for all assets (0 if not present)
                weights = []
                for asset in assets:
                    weight = portfolio_info["weights"].get(asset, 0.0)
                    weights.append(weight)

                weights_data.append(weights)

        # Transpose to have assets as rows and portfolios as columns
        weights_df = pd.DataFrame(
            data=np.array(weights_data).T, index=assets, columns=columns
        )

        logger.info(
            f"Created weights DataFrame with {len(assets)} assets and {len(columns)} portfolios"
        )
        return weights_df

    def calculate_correlation_matrix(
        self, weights_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between portfolio allocations.

        Args:
            weights_df: Portfolio weights DataFrame (if None, creates from loaded data)

        Returns:
            Correlation matrix DataFrame
        """
        if weights_df is None:
            weights_df = self.create_weights_dataframe()

        # Calculate correlation matrix between portfolios (columns)
        correlation_matrix = weights_df.corr()

        logger.info(
            f"Calculated correlation matrix for {len(correlation_matrix)} portfolios"
        )
        return correlation_matrix

    def calculate_difference_metrics(
        self, weights_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Calculate various difference metrics between portfolio allocations.

        Args:
            weights_df: Portfolio weights DataFrame (if None, creates from loaded data)

        Returns:
            Dictionary containing difference metrics
        """
        if weights_df is None:
            weights_df = self.create_weights_dataframe()

        portfolios = weights_df.columns.tolist()
        n_portfolios = len(portfolios)

        # Initialize metrics storage
        metrics = {
            "pairwise_distances": {},
            "manhattan_distances": {},
            "euclidean_distances": {},
            "max_absolute_differences": {},
            "mean_absolute_differences": {},
            "rms_differences": {},
            "summary_statistics": {},
        }

        # Calculate pairwise metrics
        for i in range(n_portfolios):
            for j in range(i + 1, n_portfolios):
                portfolio1 = portfolios[i]
                portfolio2 = portfolios[j]

                weights1 = weights_df[portfolio1].values
                weights2 = weights_df[portfolio2].values

                # Calculate different distance metrics
                manhattan_dist = np.sum(np.abs(weights1 - weights2))
                euclidean_dist = np.sqrt(np.sum((weights1 - weights2) ** 2))
                max_abs_diff = np.max(np.abs(weights1 - weights2))
                mean_abs_diff = np.mean(np.abs(weights1 - weights2))
                rms_diff = np.sqrt(np.mean((weights1 - weights2) ** 2))

                pair_key = f"{portfolio1} vs {portfolio2}"

                metrics["manhattan_distances"][pair_key] = manhattan_dist
                metrics["euclidean_distances"][pair_key] = euclidean_dist
                metrics["max_absolute_differences"][pair_key] = max_abs_diff
                metrics["mean_absolute_differences"][pair_key] = mean_abs_diff
                metrics["rms_differences"][pair_key] = rms_diff

        # Calculate summary statistics
        metrics["summary_statistics"] = {
            "portfolio_concentration": self._calculate_concentration_metrics(
                weights_df
            ),
            "portfolio_diversity": self._calculate_diversity_metrics(weights_df),
            "portfolio_statistics": self._calculate_portfolio_statistics(weights_df),
        }

        logger.info(f"Calculated difference metrics for {n_portfolios} portfolios")
        return metrics

    def _calculate_concentration_metrics(
        self, weights_df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Calculate concentration metrics for each portfolio."""
        concentration_metrics = {}

        for portfolio in weights_df.columns:
            weights = weights_df[portfolio].values

            # Herfindahl-Hirschman Index (HHI)
            hhi = np.sum(weights**2)

            # Effective number of assets
            effective_assets = 1 / hhi if hhi > 0 else 0

            # Concentration ratio (sum of top 3 weights)
            top3_concentration = np.sum(np.sort(weights)[-3:])

            # Gini coefficient (measure of inequality)
            sorted_weights = np.sort(weights)
            n = len(sorted_weights)
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_weights))) / (
                n * np.sum(sorted_weights)
            ) - (n + 1) / n

            concentration_metrics[portfolio] = {
                "herfindahl_index": hhi,
                "effective_assets": effective_assets,
                "top3_concentration": top3_concentration,
                "gini_coefficient": gini,
            }

        return concentration_metrics

    def _calculate_diversity_metrics(
        self, weights_df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Calculate diversity metrics for each portfolio."""
        diversity_metrics = {}

        for portfolio in weights_df.columns:
            weights = weights_df[portfolio].values

            # Shannon entropy
            non_zero_weights = weights[weights > 0]
            if len(non_zero_weights) > 0:
                shannon_entropy = -np.sum(non_zero_weights * np.log(non_zero_weights))
            else:
                shannon_entropy = 0

            # Number of non-zero positions
            num_positions = np.sum(
                weights > 1e-6
            )  # Using small threshold to avoid numerical issues

            # Maximum weight
            max_weight = np.max(weights)

            # Standard deviation of weights
            weight_std = np.std(weights)

            diversity_metrics[portfolio] = {
                "shannon_entropy": shannon_entropy,
                "num_positions": num_positions,
                "max_weight": max_weight,
                "weight_std": weight_std,
            }

        return diversity_metrics

    def _calculate_portfolio_statistics(
        self, weights_df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Calculate basic statistics for each portfolio."""
        stats = {}

        for portfolio in weights_df.columns:
            weights = weights_df[portfolio].values

            stats[portfolio] = {
                "sum_weights": np.sum(weights),
                "mean_weight": np.mean(weights),
                "median_weight": np.median(weights),
                "min_weight": np.min(weights),
                "max_weight": np.max(weights),
                "weight_range": np.max(weights) - np.min(weights),
            }

        return stats

    def get_most_similar_portfolios(
        self, correlation_matrix: Optional[pd.DataFrame] = None, top_n: int = 3
    ) -> List[Tuple[str, str, float]]:
        """
        Find the most similar portfolio pairs based on correlation.

        Args:
            correlation_matrix: Correlation matrix (if None, calculates from loaded data)
            top_n: Number of top similar pairs to return

        Returns:
            List of tuples (portfolio1, portfolio2, correlation)
        """
        if correlation_matrix is None:
            correlation_matrix = self.calculate_correlation_matrix()

        similar_pairs = []

        # Extract upper triangle of correlation matrix (excluding diagonal)
        for i in range(len(correlation_matrix.index)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                portfolio1 = correlation_matrix.index[i]
                portfolio2 = correlation_matrix.columns[j]
                correlation = correlation_matrix.iloc[i, j]

                similar_pairs.append((portfolio1, portfolio2, correlation))

        # Sort by correlation in descending order and return top N
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        return similar_pairs[:top_n]

    def get_most_different_portfolios(
        self, correlation_matrix: Optional[pd.DataFrame] = None, top_n: int = 3
    ) -> List[Tuple[str, str, float]]:
        """
        Find the most different portfolio pairs based on correlation.

        Args:
            correlation_matrix: Correlation matrix (if None, calculates from loaded data)
            top_n: Number of top different pairs to return

        Returns:
            List of tuples (portfolio1, portfolio2, correlation)
        """
        if correlation_matrix is None:
            correlation_matrix = self.calculate_correlation_matrix()

        different_pairs = []

        # Extract upper triangle of correlation matrix (excluding diagonal)
        for i in range(len(correlation_matrix.index)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                portfolio1 = correlation_matrix.index[i]
                portfolio2 = correlation_matrix.columns[j]
                correlation = correlation_matrix.iloc[i, j]

                different_pairs.append((portfolio1, portfolio2, correlation))

                # Sort by correlation in ascending order and return top N
        different_pairs.sort(key=lambda x: x[2])
        return different_pairs[:top_n]

    def create_correlation_heatmap(
        self,
        correlation_matrix: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Create a correlation heatmap visualization.

        Args:
            correlation_matrix: Correlation matrix (if None, calculates from loaded data)
            save_path: Path to save the plot (if None, doesn't save)
            figsize: Figure size tuple

        Returns:
            Matplotlib figure object
        """
        if correlation_matrix is None:
            correlation_matrix = self.calculate_correlation_matrix()

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap with better formatting
        mask = np.triu(
            np.ones_like(correlation_matrix, dtype=bool), k=1
        )  # Mask upper triangle
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="RdYlBu_r",
            center=0,
            square=True,
            mask=mask,
            cbar_kws={"shrink": 0.8},
            fmt=".3f",
            ax=ax,
        )

        ax.set_title(
            "Portfolio Correlation Matrix", fontsize=16, fontweight="bold", pad=20
        )
        ax.set_xlabel("Portfolios", fontsize=12)
        ax.set_ylabel("Portfolios", fontsize=12)

        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved correlation heatmap to {save_path}")

        return fig

    def create_weights_comparison_chart(
        self,
        weights_df: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """
        Create a side-by-side bar chart comparing portfolio weights.

        Args:
            weights_df: Portfolio weights DataFrame (if None, creates from loaded data)
            save_path: Path to save the plot (if None, doesn't save)
            figsize: Figure size tuple

        Returns:
            Matplotlib figure object
        """
        if weights_df is None:
            weights_df = self.create_weights_dataframe()

        # Create figure with subplots
        n_portfolios = len(weights_df.columns)
        n_cols = 3
        n_rows = (n_portfolios + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # Flatten axes for easier iteration
        axes_flat = axes.flatten()

        # Create a bar chart for each portfolio
        for i, portfolio in enumerate(weights_df.columns):
            ax = axes_flat[i]
            weights = weights_df[portfolio].sort_values(ascending=True)

            # Color bars based on weight magnitude
            colors = plt.cm.viridis(weights / weights.max())

            bars = ax.barh(range(len(weights)), weights.values, color=colors)
            ax.set_yticks(range(len(weights)))
            ax.set_yticklabels(weights.index, fontsize=8)
            ax.set_xlabel("Weight", fontsize=10)
            ax.set_title(portfolio, fontsize=10, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)

            # Add value labels on bars
            for j, (bar, value) in enumerate(zip(bars, weights.values)):
                if value > 0.01:  # Only label significant weights
                    ax.text(value + 0.005, j, f"{value:.2f}", va="center", fontsize=8)

        # Hide unused subplots
        for i in range(n_portfolios, len(axes_flat)):
            axes_flat[i].set_visible(False)

        fig.suptitle("Portfolio Weights Comparison", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved weights comparison chart to {save_path}")

        return fig

    def create_concentration_comparison(
        self,
        metrics: Optional[Dict] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        Create charts comparing portfolio concentration metrics.

        Args:
            metrics: Difference metrics dictionary (if None, calculates from loaded data)
            save_path: Path to save the plot (if None, doesn't save)
            figsize: Figure size tuple

        Returns:
            Matplotlib figure object
        """
        if metrics is None:
            metrics = self.calculate_difference_metrics()

        concentration_data = metrics["summary_statistics"]["portfolio_concentration"]

        # Extract data for plotting
        portfolios = list(concentration_data.keys())
        hhi_values = [concentration_data[p]["herfindahl_index"] for p in portfolios]
        effective_assets = [
            concentration_data[p]["effective_assets"] for p in portfolios
        ]
        top3_concentration = [
            concentration_data[p]["top3_concentration"] for p in portfolios
        ]

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. Herfindahl-Hirschman Index
        bars1 = ax1.bar(range(len(portfolios)), hhi_values, color="skyblue", alpha=0.8)
        ax1.set_title("Herfindahl-Hirschman Index (HHI)", fontweight="bold")
        ax1.set_ylabel("HHI Value")
        ax1.set_xticks(range(len(portfolios)))
        ax1.set_xticklabels(
            [p.split("_")[0] for p in portfolios], rotation=45, ha="right"
        )
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, value in zip(bars1, hhi_values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # 2. Effective Number of Assets
        bars2 = ax2.bar(
            range(len(portfolios)), effective_assets, color="lightcoral", alpha=0.8
        )
        ax2.set_title("Effective Number of Assets", fontweight="bold")
        ax2.set_ylabel("Number of Assets")
        ax2.set_xticks(range(len(portfolios)))
        ax2.set_xticklabels(
            [p.split("_")[0] for p in portfolios], rotation=45, ha="right"
        )
        ax2.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, value in zip(bars2, effective_assets):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # 3. Top 3 Concentration
        bars3 = ax3.bar(
            range(len(portfolios)), top3_concentration, color="lightgreen", alpha=0.8
        )
        ax3.set_title("Top 3 Assets Concentration", fontweight="bold")
        ax3.set_ylabel("Concentration Ratio")
        ax3.set_xticks(range(len(portfolios)))
        ax3.set_xticklabels(
            [p.split("_")[0] for p in portfolios], rotation=45, ha="right"
        )
        ax3.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, value in zip(bars3, top3_concentration):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # 4. HHI vs Effective Assets scatter plot
        ax4.scatter(hhi_values, effective_assets, c="purple", alpha=0.7, s=100)
        for i, portfolio in enumerate(portfolios):
            ax4.annotate(
                portfolio.split("_")[0],
                (hhi_values[i], effective_assets[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax4.set_xlabel("Herfindahl-Hirschman Index")
        ax4.set_ylabel("Effective Number of Assets")
        ax4.set_title("HHI vs Effective Assets", fontweight="bold")
        ax4.grid(alpha=0.3)

        fig.suptitle("Portfolio Concentration Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved concentration comparison to {save_path}")

        return fig

    def create_distance_matrix_heatmap(
        self,
        metrics: Optional[Dict] = None,
        metric_type: str = "euclidean_distances",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Create a heatmap of distance metrics between portfolios.

        Args:
            metrics: Difference metrics dictionary (if None, calculates from loaded data)
            metric_type: Type of distance metric to visualize
            save_path: Path to save the plot (if None, doesn't save)
            figsize: Figure size tuple

        Returns:
            Matplotlib figure object
        """
        if metrics is None:
            metrics = self.calculate_difference_metrics()

        # Extract distance data
        distance_data = metrics[metric_type]

        # Get unique portfolio names
        portfolios = set()
        for pair in distance_data.keys():
            p1, p2 = pair.split(" vs ")
            portfolios.add(p1)
            portfolios.add(p2)
        portfolios = sorted(list(portfolios))

        # Create distance matrix
        n = len(portfolios)
        distance_matrix = np.zeros((n, n))

        for i, p1 in enumerate(portfolios):
            for j, p2 in enumerate(portfolios):
                if i == j:
                    distance_matrix[i, j] = 0
                elif i < j:
                    pair_key = f"{p1} vs {p2}"
                    if pair_key in distance_data:
                        distance_matrix[i, j] = distance_data[pair_key]
                        distance_matrix[j, i] = distance_data[pair_key]
                    else:
                        # Try reverse order
                        pair_key = f"{p2} vs {p1}"
                        if pair_key in distance_data:
                            distance_matrix[i, j] = distance_data[pair_key]
                            distance_matrix[j, i] = distance_data[pair_key]

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        # Mask diagonal and upper triangle for cleaner visualization
        mask = np.triu(np.ones_like(distance_matrix, dtype=bool), k=1)

        sns.heatmap(
            distance_matrix,
            annot=True,
            cmap="YlOrRd",
            square=True,
            mask=mask,
            xticklabels=[p.split("_")[0] for p in portfolios],
            yticklabels=[p.split("_")[0] for p in portfolios],
            cbar_kws={"shrink": 0.8},
            fmt=".3f",
            ax=ax,
        )

        metric_name = metric_type.replace("_", " ").title()
        ax.set_title(
            f"Portfolio {metric_name} Matrix", fontsize=16, fontweight="bold", pad=20
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved {metric_type} heatmap to {save_path}")

        return fig

    def generate_all_visualizations(
        self, output_dir: str = "results/plots"
    ) -> Dict[str, str]:
        """
        Generate all visualization charts and save them to the specified directory.

        Args:
            output_dir: Directory to save all plots

        Returns:
            Dictionary with plot types and their file paths
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        saved_plots = {}

        try:
            # 1. Correlation heatmap
            correlation_matrix = self.calculate_correlation_matrix()
            corr_path = output_path / f"correlation_heatmap_{timestamp}.png"
            self.create_correlation_heatmap(correlation_matrix, str(corr_path))
            saved_plots["correlation_heatmap"] = str(corr_path)

            # 2. Weights comparison
            weights_df = self.create_weights_dataframe()
            weights_path = output_path / f"weights_comparison_{timestamp}.png"
            self.create_weights_comparison_chart(weights_df, str(weights_path))
            saved_plots["weights_comparison"] = str(weights_path)

            # 3. Concentration analysis
            metrics = self.calculate_difference_metrics(weights_df)
            concentration_path = output_path / f"concentration_analysis_{timestamp}.png"
            self.create_concentration_comparison(metrics, str(concentration_path))
            saved_plots["concentration_analysis"] = str(concentration_path)

            # 4. Distance matrices
            for metric_type in ["euclidean_distances", "manhattan_distances"]:
                distance_path = output_path / f"{metric_type}_heatmap_{timestamp}.png"
                self.create_distance_matrix_heatmap(
                    metrics, metric_type, str(distance_path)
                )
                saved_plots[f"{metric_type}_heatmap"] = str(distance_path)

            logger.info(
                f"Generated {len(saved_plots)} visualization plots in {output_dir}"
            )

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            raise

        return saved_plots

    def export_metrics_to_json(
        self,
        output_path: str = "results/reports",
        filename: Optional[str] = None,
        include_visualizations: bool = True,
    ) -> str:
        """
        Export all calculated metrics to a comprehensive JSON report.

        Args:
            output_path: Directory to save the JSON report
            filename: Custom filename (if None, generates timestamped name)
            include_visualizations: Whether to include visualization file references

        Returns:
            Path to the saved JSON file
        """
        # Create output directory if it doesn't exist
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_comparison_report_{timestamp}.json"

        file_path = output_dir / filename

        try:
            # Calculate all metrics
            weights_df = self.create_weights_dataframe()
            correlation_matrix = self.calculate_correlation_matrix(weights_df)
            difference_metrics = self.calculate_difference_metrics(weights_df)

            # Get portfolio analysis
            most_similar = self.get_most_similar_portfolios(correlation_matrix, top_n=5)
            most_different = self.get_most_different_portfolios(
                correlation_matrix, top_n=5
            )

            # Build comprehensive report structure
            report = {
                "metadata": {
                    "report_type": "Portfolio Comparison Analysis",
                    "generated_at": datetime.now().isoformat(),
                    "num_portfolios": len(weights_df.columns),
                    "num_assets": len(weights_df.index),
                    "common_assets": self.get_common_assets(),
                    "portfolios_analyzed": list(weights_df.columns.tolist()),
                },
                "portfolio_weights": {
                    "description": "Normalized portfolio weights for all assets",
                    "data": weights_df.to_dict(),
                },
                "correlation_analysis": {
                    "description": "Correlation matrix between portfolio allocations",
                    "correlation_matrix": correlation_matrix.to_dict(),
                    "most_similar_pairs": [
                        {
                            "portfolio_1": p1,
                            "portfolio_2": p2,
                            "correlation": float(corr),
                        }
                        for p1, p2, corr in most_similar
                    ],
                    "most_different_pairs": [
                        {
                            "portfolio_1": p1,
                            "portfolio_2": p2,
                            "correlation": float(corr),
                        }
                        for p1, p2, corr in most_different
                    ],
                },
                "distance_metrics": {
                    "description": "Various distance measures between portfolio pairs",
                    "manhattan_distances": difference_metrics["manhattan_distances"],
                    "euclidean_distances": difference_metrics["euclidean_distances"],
                    "max_absolute_differences": difference_metrics[
                        "max_absolute_differences"
                    ],
                    "mean_absolute_differences": difference_metrics[
                        "mean_absolute_differences"
                    ],
                    "rms_differences": difference_metrics["rms_differences"],
                },
                "concentration_metrics": {
                    "description": "Portfolio concentration and diversity measures",
                    "data": difference_metrics["summary_statistics"][
                        "portfolio_concentration"
                    ],
                },
                "diversity_metrics": {
                    "description": "Portfolio diversity measures",
                    "data": difference_metrics["summary_statistics"][
                        "portfolio_diversity"
                    ],
                },
                "portfolio_statistics": {
                    "description": "Basic statistical measures for each portfolio",
                    "data": difference_metrics["summary_statistics"][
                        "portfolio_statistics"
                    ],
                },
            }

            # Add visualization references if requested
            if include_visualizations:
                try:
                    saved_plots = self.generate_all_visualizations()
                    report["visualizations"] = {
                        "description": "Generated visualization files",
                        "files": saved_plots,
                    }
                except Exception as e:
                    logger.warning(f"Could not include visualizations in report: {e}")
                    report["visualizations"] = {
                        "description": "Visualization generation failed",
                        "error": str(e),
                    }

            # Save to JSON file
            with open(file_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Exported comprehensive metrics report to {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Error exporting metrics to JSON: {e}")
            raise

    def export_correlation_matrix_csv(
        self, output_path: str = "results/reports", filename: Optional[str] = None
    ) -> str:
        """
        Export correlation matrix to CSV format.

        Args:
            output_path: Directory to save the CSV file
            filename: Custom filename (if None, generates timestamped name)

        Returns:
            Path to the saved CSV file
        """
        # Create output directory if it doesn't exist
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"correlation_matrix_{timestamp}.csv"

        file_path = output_dir / filename

        try:
            correlation_matrix = self.calculate_correlation_matrix()
            correlation_matrix.to_csv(file_path, float_format="%.6f")

            logger.info(f"Exported correlation matrix to {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Error exporting correlation matrix to CSV: {e}")
            raise

    def export_portfolio_weights_csv(
        self, output_path: str = "results/reports", filename: Optional[str] = None
    ) -> str:
        """
        Export portfolio weights DataFrame to CSV format.

        Args:
            output_path: Directory to save the CSV file
            filename: Custom filename (if None, generates timestamped name)

        Returns:
            Path to the saved CSV file
        """
        # Create output directory if it doesn't exist
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_weights_{timestamp}.csv"

        file_path = output_dir / filename

        try:
            weights_df = self.create_weights_dataframe()
            weights_df.to_csv(file_path, float_format="%.6f")

            logger.info(f"Exported portfolio weights to {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Error exporting portfolio weights to CSV: {e}")
            raise

    def export_distance_metrics_csv(
        self, output_path: str = "results/reports", filename: Optional[str] = None
    ) -> str:
        """
        Export distance metrics to CSV format with portfolio pairs as rows.

        Args:
            output_path: Directory to save the CSV file
            filename: Custom filename (if None, generates timestamped name)

        Returns:
            Path to the saved CSV file
        """
        # Create output directory if it doesn't exist
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"distance_metrics_{timestamp}.csv"

        file_path = output_dir / filename

        try:
            metrics = self.calculate_difference_metrics()

            # Create a consolidated DataFrame with all distance metrics
            pairs = list(metrics["manhattan_distances"].keys())

            distance_data = {
                "Portfolio_Pair": pairs,
                "Manhattan_Distance": [
                    metrics["manhattan_distances"][pair] for pair in pairs
                ],
                "Euclidean_Distance": [
                    metrics["euclidean_distances"][pair] for pair in pairs
                ],
                "Max_Absolute_Difference": [
                    metrics["max_absolute_differences"][pair] for pair in pairs
                ],
                "Mean_Absolute_Difference": [
                    metrics["mean_absolute_differences"][pair] for pair in pairs
                ],
                "RMS_Difference": [metrics["rms_differences"][pair] for pair in pairs],
            }

            distance_df = pd.DataFrame(distance_data)
            distance_df.to_csv(file_path, index=False, float_format="%.6f")

            logger.info(f"Exported distance metrics to {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Error exporting distance metrics to CSV: {e}")
            raise

    def export_concentration_metrics_csv(
        self, output_path: str = "results/reports", filename: Optional[str] = None
    ) -> str:
        """
        Export portfolio concentration and diversity metrics to CSV format.

        Args:
            output_path: Directory to save the CSV file
            filename: Custom filename (if None, generates timestamped name)

        Returns:
            Path to the saved CSV file
        """
        # Create output directory if it doesn't exist
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"concentration_metrics_{timestamp}.csv"

        file_path = output_dir / filename

        try:
            metrics = self.calculate_difference_metrics()

            # Extract concentration, diversity, and basic statistics
            concentration_data = metrics["summary_statistics"][
                "portfolio_concentration"
            ]
            diversity_data = metrics["summary_statistics"]["portfolio_diversity"]
            stats_data = metrics["summary_statistics"]["portfolio_statistics"]

            # Create consolidated DataFrame
            portfolios = list(concentration_data.keys())

            consolidated_data = {
                "Portfolio": portfolios,
                # Concentration metrics
                "Herfindahl_Index": [
                    concentration_data[p]["herfindahl_index"] for p in portfolios
                ],
                "Effective_Assets": [
                    concentration_data[p]["effective_assets"] for p in portfolios
                ],
                "Top3_Concentration": [
                    concentration_data[p]["top3_concentration"] for p in portfolios
                ],
                "Gini_Coefficient": [
                    concentration_data[p]["gini_coefficient"] for p in portfolios
                ],
                # Diversity metrics
                "Shannon_Entropy": [
                    diversity_data[p]["shannon_entropy"] for p in portfolios
                ],
                "Num_Positions": [
                    diversity_data[p]["num_positions"] for p in portfolios
                ],
                "Max_Weight": [diversity_data[p]["max_weight"] for p in portfolios],
                "Weight_Std": [diversity_data[p]["weight_std"] for p in portfolios],
                # Basic statistics
                "Sum_Weights": [stats_data[p]["sum_weights"] for p in portfolios],
                "Mean_Weight": [stats_data[p]["mean_weight"] for p in portfolios],
                "Median_Weight": [stats_data[p]["median_weight"] for p in portfolios],
                "Min_Weight": [stats_data[p]["min_weight"] for p in portfolios],
                "Weight_Range": [stats_data[p]["weight_range"] for p in portfolios],
            }

            concentration_df = pd.DataFrame(consolidated_data)
            concentration_df.to_csv(file_path, index=False, float_format="%.6f")

            logger.info(f"Exported concentration metrics to {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Error exporting concentration metrics to CSV: {e}")
            raise

    def generate_comprehensive_report(
        self, output_path: str = "results/reports", report_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate a comprehensive analysis report in both JSON and CSV formats.

        Args:
            output_path: Directory to save all report files
            report_name: Base name for report files (if None, uses timestamp)

        Returns:
            Dictionary with report types and their file paths
        """
        # Create output directory if it doesn't exist
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for unique filenames if no report name provided
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"portfolio_analysis_{timestamp}"

        generated_files = {}

        try:
            logger.info(
                f"Generating comprehensive portfolio analysis report: {report_name}"
            )

            # 1. Generate main JSON report (includes visualizations)
            json_filename = f"{report_name}.json"
            json_path = self.export_metrics_to_json(
                output_path=output_path,
                filename=json_filename,
                include_visualizations=True,
            )
            generated_files["json_report"] = json_path

            # 2. Generate individual CSV files
            csv_base = report_name

            # Portfolio weights
            weights_path = self.export_portfolio_weights_csv(
                output_path=output_path, filename=f"{csv_base}_weights.csv"
            )
            generated_files["weights_csv"] = weights_path

            # Correlation matrix
            correlation_path = self.export_correlation_matrix_csv(
                output_path=output_path, filename=f"{csv_base}_correlations.csv"
            )
            generated_files["correlations_csv"] = correlation_path

            # Distance metrics
            distance_path = self.export_distance_metrics_csv(
                output_path=output_path, filename=f"{csv_base}_distances.csv"
            )
            generated_files["distances_csv"] = distance_path

            # Concentration metrics
            concentration_path = self.export_concentration_metrics_csv(
                output_path=output_path, filename=f"{csv_base}_metrics.csv"
            )
            generated_files["metrics_csv"] = concentration_path

            # 3. Create a summary report with file references
            summary_path = output_dir / f"{report_name}_summary.txt"
            self._create_summary_report(generated_files, summary_path)
            generated_files["summary_txt"] = str(summary_path)

            logger.info(f"Successfully generated {len(generated_files)} report files")

            return generated_files

        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            raise

    def _create_summary_report(
        self, generated_files: Dict[str, str], summary_path: Path
    ) -> None:
        """
        Create a text summary report with file references and key findings.

        Args:
            generated_files: Dictionary of generated file types and paths
            summary_path: Path to save the summary report
        """
        try:
            # Calculate key metrics for summary
            weights_df = self.create_weights_dataframe()
            correlation_matrix = self.calculate_correlation_matrix(weights_df)
            metrics = self.calculate_difference_metrics(weights_df)

            # Get top insights
            most_similar = self.get_most_similar_portfolios(correlation_matrix, top_n=3)
            most_different = self.get_most_different_portfolios(
                correlation_matrix, top_n=3
            )

            # Extract concentration insights
            concentration_data = metrics["summary_statistics"][
                "portfolio_concentration"
            ]

            # Find most and least concentrated portfolios
            portfolios_by_hhi = sorted(
                concentration_data.items(), key=lambda x: x[1]["herfindahl_index"]
            )
            least_concentrated = portfolios_by_hhi[0]
            most_concentrated = portfolios_by_hhi[-1]

            with open(summary_path, "w") as f:
                f.write("PORTFOLIO COMPARISON ANALYSIS SUMMARY\n")
                f.write("=" * 50 + "\n\n")

                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Number of Portfolios: {len(weights_df.columns)}\n")
                f.write(f"Number of Assets: {len(weights_df.index)}\n\n")

                f.write("KEY FINDINGS:\n")
                f.write("-" * 20 + "\n\n")

                f.write("Most Similar Portfolios:\n")
                for i, (p1, p2, corr) in enumerate(most_similar, 1):
                    f.write(
                        f"  {i}. {p1.split('_')[0]} vs {p2.split('_')[0]}: {corr:.4f}\n"
                    )
                f.write("\n")

                f.write("Most Different Portfolios:\n")
                for i, (p1, p2, corr) in enumerate(most_different, 1):
                    f.write(
                        f"  {i}. {p1.split('_')[0]} vs {p2.split('_')[0]}: {corr:.4f}\n"
                    )
                f.write("\n")

                f.write("Portfolio Concentration:\n")
                f.write(f"  Least Concentrated: {least_concentrated[0].split('_')[0]} ")
                f.write(f"(HHI: {least_concentrated[1]['herfindahl_index']:.4f}, ")
                f.write(
                    f"Effective Assets: {least_concentrated[1]['effective_assets']:.1f})\n"
                )
                f.write(f"  Most Concentrated: {most_concentrated[0].split('_')[0]} ")
                f.write(f"(HHI: {most_concentrated[1]['herfindahl_index']:.4f}, ")
                f.write(
                    f"Effective Assets: {most_concentrated[1]['effective_assets']:.1f})\n\n"
                )

                f.write("GENERATED FILES:\n")
                f.write("-" * 20 + "\n\n")

                file_descriptions = {
                    "json_report": "Comprehensive analysis in JSON format",
                    "weights_csv": "Portfolio weights matrix",
                    "correlations_csv": "Portfolio correlation matrix",
                    "distances_csv": "Distance metrics between portfolios",
                    "metrics_csv": "Concentration and diversity metrics",
                }

                for file_type, path in generated_files.items():
                    if file_type in file_descriptions:
                        f.write(f"{file_descriptions[file_type]}:\n")
                        f.write(f"  {Path(path).name}\n\n")

                f.write("VISUALIZATIONS:\n")
                f.write("-" * 20 + "\n")
                f.write(
                    "Generated visualization files can be found in the results/plots/ directory.\n"
                )
                f.write("Visualization file paths are referenced in the JSON report.\n")

            logger.info(f"Created summary report: {summary_path}")

        except Exception as e:
            logger.error(f"Error creating summary report: {e}")
            # Don't raise - summary is optional
            pass


def load_portfolios(portfolios_dir: str = "results/portfolios") -> PortfolioComparison:
    """
    Convenience function to load portfolio data.

    Args:
        portfolios_dir: Directory containing portfolio results

    Returns:
        PortfolioComparison object with loaded data
    """
    comparison = PortfolioComparison(portfolios_dir)
    comparison.load_portfolio_data()
    return comparison


if __name__ == "__main__":
    # Test the portfolio loading and comparison functionality
    try:
        comparison = load_portfolios()

        print("Loaded Portfolios:")
        for optimizer, strategies in comparison.portfolio_data.items():
            print(f"  {optimizer}:")
            for strategy_key in strategies.keys():
                print(f"    - {strategy_key}")

        # Create weights DataFrame
        weights_df = comparison.create_weights_dataframe()
        print(f"\nWeights DataFrame shape: {weights_df.shape}")
        print(f"Common assets: {len(comparison.get_common_assets())}")

        # Test correlation matrix calculation
        correlation_matrix = comparison.calculate_correlation_matrix(weights_df)
        print(f"\nCorrelation Matrix shape: {correlation_matrix.shape}")

        # Find most similar and different portfolios
        most_similar = comparison.get_most_similar_portfolios(
            correlation_matrix, top_n=3
        )
        print("\nMost Similar Portfolio Pairs:")
        for p1, p2, corr in most_similar:
            print(f"  {p1} vs {p2}: {corr:.4f}")

        most_different = comparison.get_most_different_portfolios(
            correlation_matrix, top_n=3
        )
        print("\nMost Different Portfolio Pairs:")
        for p1, p2, corr in most_different:
            print(f"  {p1} vs {p2}: {corr:.4f}")

        # Test difference metrics calculation
        metrics = comparison.calculate_difference_metrics(weights_df)
        print(
            f"\nCalculated difference metrics for {len(weights_df.columns)} portfolios"
        )

        # Show a sample of metrics
        print("\nSample Distance Metrics:")
        sample_pairs = list(metrics["euclidean_distances"].items())[:3]
        for pair, distance in sample_pairs:
            print(f"  {pair}: {distance:.4f} (Euclidean)")

            # Show portfolio concentration metrics
        print("\nPortfolio Concentration (HHI):")
        for portfolio, stats in metrics["summary_statistics"][
            "portfolio_concentration"
        ].items():
            hhi = stats["herfindahl_index"]
            effective_assets = stats["effective_assets"]
            print(
                f"  {portfolio}: HHI={hhi:.4f}, Effective Assets={effective_assets:.2f}"
            )

        # Generate visualizations
        print("\nGenerating visualization plots...")
        saved_plots = comparison.generate_all_visualizations()
        print("Saved plots:")
        for plot_type, path in saved_plots.items():
            print(f"  {plot_type}: {path}")

        # Test new CSV/JSON reporting functionality
        print("\nTesting CSV/JSON reporting functionality...")

        # Generate comprehensive report
        generated_reports = comparison.generate_comprehensive_report()
        print(f"\nGenerated {len(generated_reports)} report files:")
        for report_type, path in generated_reports.items():
            file_size = os.path.getsize(path) / 1024  # KB
            print(f"  {report_type}: {path} ({file_size:.1f} KB)")

        print("\nAll tests completed successfully!")
        print("Check results/reports/ directory for generated reports")
        print("Check results/plots/ directory for generated visualizations")

    except Exception as e:
        print(f"Error testing portfolio comparison: {e}")
        import traceback

        traceback.print_exc()
