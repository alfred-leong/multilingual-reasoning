"""
Plot evaluation results: base vs dpo_mode1 vs dpo_mode2
across reasoning modes (english_only, native_only, natural)
for each language (BN_BD, JA_JP, SW_KE).

Note: BN_BD dpo_mode2 native_only is skipped (still running).
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────── Config ───────────────────────────────────
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

LANGUAGES    = ["BN_BD", "JA_JP", "SW_KE"]
MODEL_TYPES  = ["base", "dpo_mode1", "dpo_mode2"]
THINK_MODES  = ["english_only", "native_only", "natural"]

LANG_LABELS  = {"BN_BD": "Bengali (BN_BD)", "JA_JP": "Japanese (JA_JP)", "SW_KE": "Swahili (SW_KE)"}
MODE_LABELS  = {"english_only": "English Only", "native_only": "Native Only", "natural": "Natural"}
MODEL_LABELS = {"base": "Base", "dpo_mode1": "DPO Mode 1", "dpo_mode2": "DPO Mode 2"}

COLORS = {"base": "#4C72B0", "dpo_mode1": "#55A868", "dpo_mode2": "#C44E52"}

BAR_WIDTH = 0.22
GROUP_GAP = 0.8   # space between thinking-mode groups

# ─────────────────────────────── Load data ────────────────────────────────
def load_results():
    """Return nested dict: data[lang][model_type][think_mode] = accuracy | None"""
    data = {
        lang: {model: {mode: None for mode in THINK_MODES} for model in MODEL_TYPES}
        for lang in LANGUAGES
    }

    for lang in LANGUAGES:
        for model in MODEL_TYPES:
            # map model_type -> filename segment
            model_seg = "base" if model == "base" else model.replace("dpo_", "dpo_")
            for mode in THINK_MODES:
                fname = f"eval_{lang}_{model_seg}_{mode}.json"
                fpath = os.path.join(OUTPUTS_DIR, fname)
                if not os.path.exists(fpath):
                    print(f"  [skip] {fname} not found")
                    continue
                with open(fpath) as f:
                    rec = json.load(f)
                data[lang][model][mode] = rec["accuracy"]

    return data


# ─────────────────────────────── Plot helpers ─────────────────────────────
def plot_accuracy(data: dict, save_path: str | None = None):
    """
    One figure with 3 subplots (one per language).
    Each subplot shows grouped bars by thinking mode, coloured by model type.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle("Accuracy: Base vs DPO Mode 1 vs DPO Mode 2\nacross Reasoning Modes", fontsize=14, fontweight="bold")

    n_models = len(MODEL_TYPES)
    x_base = np.arange(len(THINK_MODES)) * GROUP_GAP

    for ax, lang in zip(axes, LANGUAGES):
        for i, model in enumerate(MODEL_TYPES):
            offsets = (i - (n_models - 1) / 2) * BAR_WIDTH
            heights = []
            hatches = []
            for mode in THINK_MODES:
                val = data[lang][model][mode]
                heights.append(val if val is not None else 0)
                hatches.append("/" if val is None else "")

            bars = ax.bar(
                x_base + offsets,
                heights,
                width=BAR_WIDTH,
                color=COLORS[model],
                label=MODEL_LABELS[model],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )

            # Mark missing bars with a hatched overlay
            for bar, val, hatch in zip(bars, [data[lang][model][m] for m in THINK_MODES], hatches):
                if val is None:
                    bar.set_hatch("////")
                    bar.set_alpha(0.3)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        0.01,
                        "N/A",
                        ha="center", va="bottom",
                        fontsize=7, color="gray", rotation=90,
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.004,
                        f"{val:.2f}",
                        ha="center", va="bottom",
                        fontsize=6.5, rotation=90,
                    )

        ax.set_title(LANG_LABELS[lang], fontsize=11, fontweight="bold")
        ax.set_xticks(x_base)
        ax.set_xticklabels([MODE_LABELS[m] for m in THINK_MODES], fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    # Shared legend
    handles = [mpatches.Patch(color=COLORS[m], label=MODEL_LABELS[m]) for m in MODEL_TYPES]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_language_ratios(data_full: dict, save_path: str | None = None):
    """
    Heatmap-style grid showing mean_target_ratio and mean_english_ratio
    for each (language, model, thinking_mode) combination.
    """
    ratio_keys = ["mean_target_ratio", "mean_english_ratio"]
    ratio_labels = {"mean_target_ratio": "Mean Target-Language Ratio (↑ better)",
                    "mean_english_ratio": "Mean English Ratio"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey="row")
    fig.suptitle("Language Usage Ratios: Base vs DPO Mode 1 vs DPO Mode 2", fontsize=13, fontweight="bold")

    n_models = len(MODEL_TYPES)
    x_base = np.arange(len(THINK_MODES)) * GROUP_GAP

    for row, rkey in enumerate(ratio_keys):
        for col, lang in enumerate(LANGUAGES):
            ax = axes[row][col]
            for i, model in enumerate(MODEL_TYPES):
                offsets = (i - (n_models - 1) / 2) * BAR_WIDTH
                heights = []
                for mode in THINK_MODES:
                    val = data_full[lang][model][mode]
                    heights.append(val.get(rkey, 0) if val is not None else 0)

                bars = ax.bar(
                    x_base + offsets,
                    heights,
                    width=BAR_WIDTH,
                    color=COLORS[model],
                    label=MODEL_LABELS[model],
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.5,
                )

                for bar, h, mode in zip(bars, heights, THINK_MODES):
                    raw = data_full[lang][model][mode]
                    if raw is None:
                        bar.set_hatch("////")
                        bar.set_alpha(0.3)
                    else:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.002,
                            f"{h:.3f}",
                            ha="center", va="bottom", fontsize=6, rotation=90,
                        )

            if row == 0:
                ax.set_title(LANG_LABELS[lang], fontsize=10, fontweight="bold")
            ax.set_xticks(x_base)
            ax.set_xticklabels([MODE_LABELS[m] for m in THINK_MODES], fontsize=8)
            if col == 0:
                ax.set_ylabel(ratio_labels[rkey], fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines[["top", "right"]].set_visible(False)

    handles = [mpatches.Patch(color=COLORS[m], label=MODEL_LABELS[m]) for m in MODEL_TYPES]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_delta_heatmap(data: dict, save_path: str | None = None):
    """
    Show accuracy delta (vs base) for dpo_mode1 and dpo_mode2
    as a heatmap — positive=green (improvement), negative=red (regression).
    """
    compare_models = ["dpo_mode1", "dpo_mode2"]
    n_rows = len(compare_models)
    n_cols = len(LANGUAGES)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6))
    fig.suptitle("Accuracy Delta vs Base (DPO Mode 1 & Mode 2)", fontsize=13, fontweight="bold")

    for row, model in enumerate(compare_models):
        for col, lang in enumerate(LANGUAGES):
            ax = axes[row][col]
            matrix = np.full((1, len(THINK_MODES)), np.nan)
            annotations = []
            for j, mode in enumerate(THINK_MODES):
                base_acc = data[lang]["base"][mode]
                dpo_acc  = data[lang][model][mode]
                if base_acc is not None and dpo_acc is not None:
                    matrix[0, j] = dpo_acc - base_acc
                    annotations.append(f"{dpo_acc - base_acc:+.2f}\n({base_acc:.2f}→{dpo_acc:.2f})")
                else:
                    annotations.append("N/A")

            vmax = 0.12
            im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

            ax.set_xticks(range(len(THINK_MODES)))
            ax.set_xticklabels([MODE_LABELS[m] for m in THINK_MODES], fontsize=8)
            ax.set_yticks([])

            for j, ann in enumerate(annotations):
                color = "black"
                ax.text(j, 0, ann, ha="center", va="center", fontsize=8, color=color,
                        fontweight="bold" if "N/A" not in ann else "normal")

            if row == 0:
                ax.set_title(LANG_LABELS[lang], fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(MODEL_LABELS[model], fontsize=9, fontweight="bold")

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    fig.colorbar(sm, cax=cbar_ax, label="Accuracy Δ vs Base")

    plt.tight_layout(rect=[0, 0, 0.91, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


# ─────────────────────────────── Main ─────────────────────────────────────
def load_full_results():
    """Return nested dict: data_full[lang][model][mode] = record dict | None"""
    data_full = {
        lang: {model: {mode: None for mode in THINK_MODES} for model in MODEL_TYPES}
        for lang in LANGUAGES
    }
    for lang in LANGUAGES:
        for model in MODEL_TYPES:
            model_seg = "base" if model == "base" else model
            for mode in THINK_MODES:
                fname = f"eval_{lang}_{model_seg}_{mode}.json"
                fpath = os.path.join(OUTPUTS_DIR, fname)
                if not os.path.exists(fpath):
                    continue
                with open(fpath) as f:
                    data_full[lang][model][mode] = json.load(f)
    return data_full


def main():
    print("Loading results …")
    data_full = load_full_results()

    # Build accuracy-only dict
    data_acc = {
        lang: {
            model: {
                mode: (data_full[lang][model][mode]["accuracy"]
                       if data_full[lang][model][mode] is not None else None)
                for mode in THINK_MODES
            }
            for model in MODEL_TYPES
        }
        for lang in LANGUAGES
    }

    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print("Generating accuracy bar chart …")
    plot_accuracy(data_acc, save_path=os.path.join(out_dir, "plot_accuracy.png"))

    print("Generating language ratio chart …")
    plot_language_ratios(data_full, save_path=os.path.join(out_dir, "plot_language_ratios.png"))

    print("Generating delta heatmap …")
    plot_delta_heatmap(data_acc, save_path=os.path.join(out_dir, "plot_delta_heatmap.png"))

    print("\nDone. All plots saved to outputs/")


if __name__ == "__main__":
    main()
