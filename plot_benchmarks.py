"""
Benchmark Plotting Script

This script reads the CPU vs GPU benchmark results from CSV and generates
visualization plots comparing performance across devices, sequence lengths,
and batch sizes.

Plots saved to benchmarks/plots/:
  1) tokens_per_sec vs seq_len (CPU vs GPU comparison)
  2) mean_step_ms vs seq_len (step time comparison)
  3) tokens_per_sec vs batch_size at fixed seq_len
  4) peak_mem_mb vs seq_len (CUDA only)

Usage:
    python plot_benchmarks.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================
CSV_PATH = "benchmarks/results_cpu_gpu.csv"
PLOTS_DIR = "benchmarks/plots"

# Default values for single-variable plots
DEFAULT_BATCH_SIZE = 32  # For seq_len comparison plots
DEFAULT_SEQ_LEN = 256  # For batch_size comparison plots


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess benchmark data."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print("Run bench_cpu_gpu.py first to generate benchmark data.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Check if CSV has headers - if not, add them
    if "timestamp" not in df.columns:
        # Old format without headers - add column names
        column_names = [
            "timestamp",
            "device",
            "precision",
            "seq_len",
            "batch_size",
            "warmup_steps",
            "timed_steps",
            "repeat",
            "mean_step_ms",
            "p50_step_ms",
            "p90_step_ms",
            "tokens_per_sec",
            "peak_mem_mb",
            "final_loss",
            "status",
            "notes",
        ]
        df.columns = column_names[: len(df.columns)]

    # Filter only successful runs (if status column exists)
    if "status" in df.columns:
        df = df[df["status"] == "success"].copy()

    # Ensure numeric columns
    for col in ["seq_len", "batch_size", "mean_step_ms", "tokens_per_sec", "peak_mem_mb"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if len(df) == 0:
        print("Error: No benchmark data found in CSV.")
        sys.exit(1)

    return df


def aggregate_repeats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results across repeats (mean of means)."""
    agg_df = (
        df.groupby(["device", "seq_len", "batch_size"])
        .agg(
            {
                "mean_step_ms": "mean",
                "p50_step_ms": "mean",
                "p90_step_ms": "mean",
                "tokens_per_sec": "mean",
                "peak_mem_mb": "mean",
            }
        )
        .reset_index()
    )

    # Also compute std for error bars
    std_df = (
        df.groupby(["device", "seq_len", "batch_size"])
        .agg(
            {
                "mean_step_ms": "std",
                "tokens_per_sec": "std",
            }
        )
        .reset_index()
    )
    std_df.columns = ["device", "seq_len", "batch_size", "mean_step_ms_std", "tokens_per_sec_std"]

    agg_df = agg_df.merge(std_df, on=["device", "seq_len", "batch_size"])
    agg_df = agg_df.fillna(0)  # Fill NaN std (single repeat) with 0

    return agg_df


def plot_tokens_per_sec_vs_seq_len(df: pd.DataFrame, batch_size: int, output_dir: str):
    """
    Plot 1: tokens/sec vs seq_len for CPU and GPU at a fixed batch size.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    subset = df[df["batch_size"] == batch_size]

    if len(subset) == 0:
        print(f"Warning: No data for batch_size={batch_size}, skipping plot 1")
        plt.close(fig)
        return

    devices = subset["device"].unique()
    colors = {"cpu": "#2196F3", "cuda": "#4CAF50"}
    markers = {"cpu": "o", "cuda": "s"}

    for device in sorted(devices):
        device_data = subset[subset["device"] == device].sort_values("seq_len")
        ax.errorbar(
            device_data["seq_len"],
            device_data["tokens_per_sec"],
            yerr=device_data["tokens_per_sec_std"],
            label=device.upper(),
            color=colors.get(device, "#666"),
            marker=markers.get(device, "o"),
            linewidth=2,
            markersize=8,
            capsize=4,
        )

    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Tokens/sec", fontsize=12)
    ax.set_title(f"Throughput vs Sequence Length (batch_size={batch_size})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("linear")

    # Set x-ticks to actual seq_lens
    ax.set_xticks(subset["seq_len"].unique())

    plt.tight_layout()
    output_path = os.path.join(output_dir, "tokens_per_sec_vs_seq_len.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_step_time_vs_seq_len(df: pd.DataFrame, batch_size: int, output_dir: str):
    """
    Plot 2: mean_step_ms vs seq_len for CPU and GPU at a fixed batch size.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    subset = df[df["batch_size"] == batch_size]

    if len(subset) == 0:
        print(f"Warning: No data for batch_size={batch_size}, skipping plot 2")
        plt.close(fig)
        return

    devices = subset["device"].unique()
    colors = {"cpu": "#2196F3", "cuda": "#4CAF50"}
    markers = {"cpu": "o", "cuda": "s"}

    for device in sorted(devices):
        device_data = subset[subset["device"] == device].sort_values("seq_len")
        ax.errorbar(
            device_data["seq_len"],
            device_data["mean_step_ms"],
            yerr=device_data["mean_step_ms_std"],
            label=device.upper(),
            color=colors.get(device, "#666"),
            marker=markers.get(device, "o"),
            linewidth=2,
            markersize=8,
            capsize=4,
        )

    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Mean Step Time (ms)", fontsize=12)
    ax.set_title(f"Step Time vs Sequence Length (batch_size={batch_size})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Set x-ticks to actual seq_lens
    ax.set_xticks(subset["seq_len"].unique())

    plt.tight_layout()
    output_path = os.path.join(output_dir, "step_time_vs_seq_len.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_tokens_per_sec_vs_batch_size(df: pd.DataFrame, seq_len: int, output_dir: str):
    """
    Plot 3: tokens/sec vs batch_size for CPU and GPU at a fixed seq_len.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    subset = df[df["seq_len"] == seq_len]

    if len(subset) == 0:
        print(f"Warning: No data for seq_len={seq_len}, skipping plot 3")
        plt.close(fig)
        return

    devices = subset["device"].unique()
    colors = {"cpu": "#2196F3", "cuda": "#4CAF50"}
    markers = {"cpu": "o", "cuda": "s"}

    for device in sorted(devices):
        device_data = subset[subset["device"] == device].sort_values("batch_size")
        ax.errorbar(
            device_data["batch_size"],
            device_data["tokens_per_sec"],
            yerr=device_data["tokens_per_sec_std"],
            label=device.upper(),
            color=colors.get(device, "#666"),
            marker=markers.get(device, "o"),
            linewidth=2,
            markersize=8,
            capsize=4,
        )

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Tokens/sec", fontsize=12)
    ax.set_title(f"Throughput vs Batch Size (seq_len={seq_len})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Set x-ticks to actual batch_sizes
    ax.set_xticks(subset["batch_size"].unique())

    plt.tight_layout()
    output_path = os.path.join(output_dir, "tokens_per_sec_vs_batch_size.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_peak_memory_vs_seq_len(df: pd.DataFrame, batch_size: int, output_dir: str):
    """
    Plot 4: peak_mem_mb vs seq_len (CUDA only).
    """
    cuda_data = df[(df["device"] == "cuda") & (df["batch_size"] == batch_size)]

    if len(cuda_data) == 0:
        print("  Skipping memory plot: No CUDA data available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    cuda_data = cuda_data.sort_values("seq_len")

    ax.bar(
        cuda_data["seq_len"].astype(str),
        cuda_data["peak_mem_mb"],
        color="#4CAF50",
        alpha=0.8,
        edgecolor="black",
    )

    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Peak GPU Memory (MB)", fontsize=12)
    ax.set_title(f"GPU Memory Usage vs Sequence Length (batch_size={batch_size})", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (_, row) in enumerate(cuda_data.iterrows()):
        ax.text(i, row["peak_mem_mb"] + 20, f"{row['peak_mem_mb']:.0f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "peak_memory_vs_seq_len.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_heatmap_speedup(df: pd.DataFrame, output_dir: str):
    """
    Bonus plot: Heatmap of CUDA speedup over CPU (ratio of tokens/sec).
    """
    cpu_data = df[df["device"] == "cpu"].set_index(["seq_len", "batch_size"])["tokens_per_sec"]
    cuda_data = df[df["device"] == "cuda"].set_index(["seq_len", "batch_size"])["tokens_per_sec"]

    if len(cpu_data) == 0 or len(cuda_data) == 0:
        print("  Skipping speedup heatmap: Need both CPU and CUDA data")
        return

    # Compute speedup
    speedup = cuda_data / cpu_data
    speedup = speedup.dropna()

    if len(speedup) == 0:
        print("  Skipping speedup heatmap: No matching configurations")
        return

    # Pivot for heatmap
    speedup_df = speedup.reset_index()
    speedup_df.columns = ["seq_len", "batch_size", "speedup"]
    pivot = speedup_df.pivot(index="batch_size", columns="seq_len", values="speedup")

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")

    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Batch Size", fontsize=12)
    ax.set_title("CUDA Speedup over CPU (tokens/sec ratio)", fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Speedup (x)", fontsize=11)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f"{val:.1f}x", ha="center", va="center", color="black", fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "cuda_speedup_heatmap.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    """Main plotting function."""
    print("=" * 70)
    print("Benchmark Plotting")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from: {CSV_PATH}")
    df = load_data(CSV_PATH)
    print(f"  Total rows: {len(df)}")
    print(f"  Devices: {df['device'].unique().tolist()}")
    print(f"  Seq lengths: {sorted(df['seq_len'].unique().tolist())}")
    print(f"  Batch sizes: {sorted(df['batch_size'].unique().tolist())}")

    # Aggregate repeats
    print("\nAggregating repeats...")
    agg_df = aggregate_repeats(df)

    # Create output directory
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"\nGenerating plots in: {PLOTS_DIR}")

    # Determine available batch_sizes and seq_lens
    available_batch_sizes = sorted(agg_df["batch_size"].unique())
    available_seq_lens = sorted(agg_df["seq_len"].unique())

    # Use default or fallback to first available
    batch_size = DEFAULT_BATCH_SIZE if DEFAULT_BATCH_SIZE in available_batch_sizes else available_batch_sizes[0]
    seq_len = DEFAULT_SEQ_LEN if DEFAULT_SEQ_LEN in available_seq_lens else available_seq_lens[0]

    print(f"  Using batch_size={batch_size} for seq_len plots")
    print(f"  Using seq_len={seq_len} for batch_size plots")
    print()

    # Generate plots
    plot_tokens_per_sec_vs_seq_len(agg_df, batch_size, PLOTS_DIR)
    plot_step_time_vs_seq_len(agg_df, batch_size, PLOTS_DIR)
    plot_tokens_per_sec_vs_batch_size(agg_df, seq_len, PLOTS_DIR)
    plot_peak_memory_vs_seq_len(agg_df, batch_size, PLOTS_DIR)
    plot_heatmap_speedup(agg_df, PLOTS_DIR)

    print(f"\n{'=' * 70}")
    print("Plotting complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
