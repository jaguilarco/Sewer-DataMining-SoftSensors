#!/usr/bin/env python3
"""
statistical_significance_tests.py

- Read ./algorithms_results/<model>/fold_metrics.csv
- Consolidate metrics per fold
- Execute Friedman + Nemenyi tests (statds)
- Save plots in PNG and SVG formats
- Display plots in Jupyter (display) or CLI (optional plt.show)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------
#  GLOBAL CONFIGURATION
# ---------------------------------------------------------
BASE_DIR = "./algorithms_results"
OUT_DIR = "./outputs"
ALPHA = 0.05
METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
SHOW_PLOTS = False  # Change to True to display graphs when running
SUPPRESS_WARNINGS = False
# ---------------------------------------------------------

# Detect if running in a Jupyter/IPython environment
def in_notebook() -> bool:
    try:
        from IPython import get_ipython  # noqa
        ip = get_ipython()
        if ip is None:
            return False
        # ZMQInteractiveShell usually indicates a Jupyter environment
        return ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def sanitize_filename(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def build_metrics_dfs(base_dir: Path) -> dict[str, pd.DataFrame]:
    metrics_dfs: dict[str, pd.DataFrame] = {}

    if not base_dir.exists():
        raise FileNotFoundError(f"base_dir does not exist: {base_dir.resolve()}")

    for model_name in os.listdir(base_dir):
        model_path = base_dir / model_name

        if not model_path.is_dir():
            continue
        if model_name in ["__MACOSX"]:
            continue

        metrics_file_path = model_path / "fold_metrics.csv"
        if not metrics_file_path.exists():
            print(f"Warning: 'fold_metrics.csv' not found in {model_path}")
            continue

        df_metrics = pd.read_csv(metrics_file_path)

        # Metric columns (exclude repeat/fold)
        metric_columns = [c for c in df_metrics.columns if c not in ["repeat", "fold"]]
        if "fold" not in df_metrics.columns:
            raise ValueError(f"File {metrics_file_path} is missing the 'fold' column.")

        # Aggregate by fold (calculating mean)
        df_metrics_agg = df_metrics.groupby("fold")[metric_columns].mean().reset_index()

        for metric_col in metric_columns:
            if metric_col not in metrics_dfs:
                # Initialize DataFrame with folds as index
                metrics_dfs[metric_col] = (
                    pd.DataFrame({"Fold": df_metrics_agg["fold"].unique()})
                    .set_index("Fold")
                )

            # Assign values aligning by fold
            metrics_dfs[metric_col][model_name] = df_metrics_agg.set_index("fold")[metric_col]

    # Sort by Fold index
    for metric_name, df in metrics_dfs.items():
        metrics_dfs[metric_name] = df.sort_index()

    return metrics_dfs


def run_tests_and_plots(
    metrics_dfs: dict[str, pd.DataFrame],
    metrics_to_run: list[str],
    alpha: float,
    out_dir: Path,
    show: bool,
) -> None:
    # Imported here to allow backend selection if necessary
    import matplotlib.pyplot as plt  # noqa: F401
    from statds.no_parametrics import friedman, nemenyi

    out_dir.mkdir(parents=True, exist_ok=True)

    notebook = in_notebook()
    if notebook:
        from IPython.display import display

    for metric_name in metrics_to_run:
        if metric_name not in metrics_dfs:
            print(f"[SKIP] Metric '{metric_name}' not found in metrics_dfs.")
            continue

        df = metrics_dfs[metric_name]
        print(f"\nDataFrame for metric: {metric_name}")
        print(df.head())

        dataset = df.copy()

        rankings, statistic, p_value, critical_value, hypothesis = friedman(
            dataset, alpha, minimize=False
        )
        print(f"Hypothesis Rejected: {hypothesis}")
        print(f"Statistic: {statistic}, Critical Value: {critical_value}, p-value: {p_value}")
        print(f"Rankings:\n{rankings}")

        num_cases = dataset.shape[0]
        ranks_values, critical_distance_nemenyi, fig = nemenyi(
            rankings, num_cases, alpha
        )
        print(f"Ranks Values: {ranks_values}")
        print(f"Critical Distance (Nemenyi): {critical_distance_nemenyi}")

        # ---- Save to files (PNG + SVG) ----
        base = sanitize_filename(f"nemenyi_{metric_name}_alpha_{alpha}")
        png_path = out_dir / f"{base}.png"
        svg_path = out_dir / f"{base}.svg"

        # Using bbox_inches='tight' to crop unnecessary margins
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        print(f"Saved: {png_path}")
        print(f"Saved: {svg_path}")

        # ---- Display Logic ----
        if show:
            if notebook:
                display(fig)  # Jupyter: renders correctly without figure.show()
            else:
                # CLI: plt.show() (opens a window if an interactive backend is used)
                import matplotlib.pyplot as plt
                plt.show()

        # Close the figure to free up memory
        import matplotlib.pyplot as plt
        plt.close(fig)


def main():
    if SUPPRESS_WARNINGS:
        warnings.filterwarnings("ignore", category=UserWarning)

    base_dir = Path(BASE_DIR)
    out_dir = Path(OUT_DIR)

    metrics_dfs = build_metrics_dfs(base_dir)

    print("Consolidation complete. The 'metrics_dfs' dictionary now contains DataFrames for each metric.")
    
    run_tests_and_plots(
        metrics_dfs=metrics_dfs,
        metrics_to_run=METRICS,
        alpha=ALPHA,
        out_dir=out_dir,
        show=SHOW_PLOTS,
    )


if __name__ == "__main__":
    main()