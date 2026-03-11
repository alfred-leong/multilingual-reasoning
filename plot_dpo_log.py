#!/usr/bin/env python3
"""
Plot DPO training rewards/margins from a training log file.

Usage:
    python plot_dpo_log.py <log_file> [--output <output_path>]

Example:
    python plot_dpo_log.py outputs-translate_dpo/logs/JA_JP_dpo_train.log
    python plot_dpo_log.py outputs-translate_dpo/logs/JA_JP_dpo_train.log --output my_plot.png
"""

import argparse
import ast
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def parse_log_file(log_path: str) -> list[dict]:
    """Parse a DPO training log file and extract metric dicts."""
    records = []
    # Match lines that look like Python dicts starting with {'loss': ...}
    dict_pattern = re.compile(r"^\{.*'loss'.*\}$")

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and progress bars
            if not line or line.startswith("\r"):
                continue
            # Handle lines that may have leading/trailing ANSI or progress bar remnants
            # Extract the dict portion
            match = re.search(r"\{[^{}]+\}", line)
            if match:
                candidate = match.group(0)
                if "'loss'" in candidate and "'rewards/margins'" in candidate:
                    try:
                        record = ast.literal_eval(candidate)
                        if isinstance(record, dict):
                            records.append(record)
                    except (ValueError, SyntaxError):
                        continue
    return records


def plot_metrics(records: list[dict], log_path: str, output_path: str | None = None):
    """Create a multi-panel plot of DPO training metrics."""
    if not records:
        print("No training records found in the log file.", file=sys.stderr)
        sys.exit(1)

    # Skip the final summary record (train_runtime) if present
    records = [r for r in records if "train_runtime" not in r]

    steps = list(range(1, len(records) + 1))
    epochs = [r.get("epoch", None) for r in records]
    use_epoch = all(e is not None for e in epochs)
    x = epochs if use_epoch else steps
    x_label = "Epoch" if use_epoch else "Logging Step"

    # Extract metrics
    loss = [r["loss"] for r in records]
    rewards_chosen = [r["rewards/chosen"] for r in records]
    rewards_rejected = [r["rewards/rejected"] for r in records]
    rewards_margins = [r["rewards/margins"] for r in records]
    rewards_accuracies = [r["rewards/accuracies"] for r in records]

    # Optional metrics
    has_logps = "logps/chosen" in records[0]
    has_entropy = "entropy" in records[0]
    has_grad_norm = "grad_norm" in records[0]

    # Determine number of subplots
    n_plots = 4  # loss, rewards, margins, accuracies
    if has_entropy:
        n_plots += 1
    if has_grad_norm:
        n_plots += 1

    # Derive a title from the log filename
    log_name = Path(log_path).stem
    title = log_name.replace("_", " ").title()

    # --- Plotting ---
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.2 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    colors = {
        "loss": "#e74c3c",
        "chosen": "#2ecc71",
        "rejected": "#e67e22",
        "margins": "#3498db",
        "accuracies": "#9b59b6",
        "entropy": "#1abc9c",
        "grad_norm": "#e84393",
    }

    ax_idx = 0

    # 1. Loss
    ax = axes[ax_idx]
    ax.plot(x, loss, color=colors["loss"], linewidth=1.5, alpha=0.85)
    ax.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax.set_title(f"DPO Training Loss", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.fill_between(x, loss, alpha=0.1, color=colors["loss"])
    ax_idx += 1

    # 2. Rewards (chosen vs rejected)
    ax = axes[ax_idx]
    ax.plot(x, rewards_chosen, color=colors["chosen"], linewidth=1.5, alpha=0.85, label="Chosen")
    ax.plot(x, rewards_rejected, color=colors["rejected"], linewidth=1.5, alpha=0.85, label="Rejected")
    ax.fill_between(x, rewards_chosen, rewards_rejected, alpha=0.12, color=colors["margins"])
    ax.set_ylabel("Reward", fontsize=11, fontweight="bold")
    ax.set_title("Rewards: Chosen vs Rejected", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax_idx += 1

    # 3. Margins
    ax = axes[ax_idx]
    ax.plot(x, rewards_margins, color=colors["margins"], linewidth=1.5, alpha=0.85)
    ax.set_ylabel("Margin", fontsize=11, fontweight="bold")
    ax.set_title("Reward Margins (Chosen − Rejected)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.fill_between(x, rewards_margins, alpha=0.1, color=colors["margins"])
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax_idx += 1

    # 4. Accuracies
    ax = axes[ax_idx]
    ax.plot(x, rewards_accuracies, color=colors["accuracies"], linewidth=1.5, alpha=0.85)
    ax.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    ax.set_title("Reward Accuracies", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.fill_between(x, rewards_accuracies, alpha=0.1, color=colors["accuracies"])
    ax_idx += 1

    # 5. Entropy (optional)
    if has_entropy:
        entropy = [r["entropy"] for r in records]
        ax = axes[ax_idx]
        ax.plot(x, entropy, color=colors["entropy"], linewidth=1.5, alpha=0.85)
        ax.set_ylabel("Entropy", fontsize=11, fontweight="bold")
        ax.set_title("Policy Entropy", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.fill_between(x, entropy, alpha=0.1, color=colors["entropy"])
        ax_idx += 1

    # 6. Gradient Norm (optional)
    if has_grad_norm:
        grad_norm = [r["grad_norm"] for r in records]
        ax = axes[ax_idx]
        ax.plot(x, grad_norm, color=colors["grad_norm"], linewidth=1.5, alpha=0.85)
        ax.set_ylabel("Grad Norm", fontsize=11, fontweight="bold")
        ax.set_title("Gradient Norm", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.fill_between(x, grad_norm, alpha=0.1, color=colors["grad_norm"])
        ax_idx += 1

    # Set x label on the last subplot
    axes[-1].set_xlabel(x_label, fontsize=11, fontweight="bold")

    fig.suptitle(f"DPO Training Metrics — {title}", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    # Save
    if output_path is None:
        output_path = str(Path(log_path).with_suffix(".png"))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot DPO training rewards/margins from a log file."
    )
    parser.add_argument("log_file", help="Path to the DPO training log file")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for the plot image (default: same name as log with .png extension)",
    )
    args = parser.parse_args()

    if not Path(args.log_file).exists():
        print(f"Error: Log file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    records = parse_log_file(args.log_file)
    print(f"Found {len(records)} training records in {args.log_file}")
    plot_metrics(records, args.log_file, args.output)


if __name__ == "__main__":
    main()
