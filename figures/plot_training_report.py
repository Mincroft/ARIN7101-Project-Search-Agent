"""Create a report-ready vector figure from the verified W&B run history."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = Path("runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main")
OUT = Path("figures/qwen35_9b_training_report")

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "neutral": "#CFCECE",
    "dark": "#272727",
}


def load_metrics(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(RUN.glob(pattern)):
        with path.open() as handle:
            row = json.load(handle)
        row["update"] = int(path.parent.name.rsplit("-", 1)[1])
        rows.append(row)
    return sorted(rows, key=lambda row: row["update"])


def rolling_mean(values: np.ndarray, window: int = 5) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        result[i] = values[max(0, i - window + 1) : i + 1].mean()
    return result


def main() -> None:
    eval_rows = load_metrics("eval-*/metrics.json")
    train_rows = load_metrics("rollouts-*/metrics.json")
    assert len(eval_rows) == 41, f"expected baseline + 40 eval checkpoints, got {len(eval_rows)}"
    assert len(train_rows) == 80, f"expected 80 training updates, got {len(train_rows)}"

    eval_x = np.array([row["update"] for row in eval_rows])
    f1 = np.array([row["eval/f1"] for row in eval_rows])
    f4 = np.array([row["eval/f_beta"] for row in eval_rows])
    train_x = np.array([row["update"] for row in train_rows])
    reward = np.array([row["train_raw/reward"] for row in train_rows])
    reward_trend = rolling_mean(reward, 5)

    best_f1_i = int(np.argmax(f1))
    best_f4_i = int(np.argmax(f4))
    best_f1_x, best_f1 = eval_x[best_f1_i], f1[best_f1_i]
    best_f4_x, best_f4 = eval_x[best_f4_i], f4[best_f4_i]

    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.linewidth": 1.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.83, wspace=0.27)

    ax = axes[0]
    ax.plot(eval_x, f1, color=PALETTE["blue_main"], lw=2.4, marker="o", ms=4.5, label="F1")
    ax.plot(eval_x, f4, color=PALETTE["teal"], lw=2.4, marker="s", ms=4.0, label="F4")
    ax.scatter([best_f1_x], [best_f1], s=90, color=PALETTE["blue_main"], edgecolor="white", lw=1.3, zorder=5)
    ax.scatter([best_f4_x], [best_f4], s=90, color=PALETTE["teal"], edgecolor="white", lw=1.3, zorder=5)
    ax.annotate(f"best F1\nupdate {best_f1_x}: {best_f1:.3f}", (best_f1_x, best_f1),
                xytext=(-42, 24), textcoords="offset points", ha="center", va="bottom",
                fontsize=9, color=PALETTE["blue_main"],
                arrowprops={"arrowstyle": "-", "color": PALETTE["blue_main"], "lw": 0.9})
    ax.annotate(f"best F4\nupdate {best_f4_x}: {best_f4:.3f}", (best_f4_x, best_f4),
                xytext=(34, -32), textcoords="offset points", ha="center", va="top",
                fontsize=9, color=PALETTE["teal"],
                arrowprops={"arrowstyle": "-", "color": PALETTE["teal"], "lw": 0.9})
    ax.set_title("Development-set quality", loc="left", fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.08, 0.66)
    ax.set_yticks(np.arange(0.1, 0.7, 0.1))
    ax.grid(axis="y", color=PALETTE["neutral"], lw=0.7, alpha=0.55)
    ax.legend(loc="upper left", ncol=2, handlelength=2.4, columnspacing=1.2)

    ax = axes[1]
    ax.scatter(train_x, reward, color=PALETTE["violet"], alpha=0.42, s=19, linewidths=0, label="Per-update reward")
    ax.plot(train_x, reward_trend, color=PALETTE["violet"], lw=2.6, label="5-update trend")
    ax.axvline(best_f4_x, color=PALETTE["teal"], lw=1.2, ls=(0, (4, 3)), alpha=0.8)
    ax.text(best_f4_x + 1.0, 0.02, f"selected\nupdate {best_f4_x}", color=PALETTE["teal"], fontsize=9, va="bottom")
    ax.set_title("Training reward", loc="left", fontweight="bold")
    ax.set_ylabel("Mean reward")
    ax.set_ylim(-0.05, 0.68)
    ax.grid(axis="y", color=PALETTE["neutral"], lw=0.7, alpha=0.55)
    ax.legend(loc="upper left")

    for ax in axes:
        ax.set_xlabel("Optimization update")
        ax.set_xlim(-1, 81)
        ax.set_xticks(np.arange(0, 81, 10))
        ax.tick_params(length=3, width=1.0)

    fig.suptitle("Qwen3.5-9B SEC search training", x=0.075, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.075, 0.885, "Local 8-GPU Dr. GRPO run · 80 updates · fixed 32-question development evaluation",
             ha="left", va="bottom", fontsize=10, color="#4D4D4D")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT.with_suffix(".png"), format="png", dpi=600, bbox_inches="tight", pad_inches=0.08)
    manifest = {
        "source": str(RUN),
        "source_type": "verified W&B run history mirrored in authoritative local metrics",
        "updates": 80,
        "evaluation_checkpoints": len(eval_rows),
        "best_f1": {"update": int(best_f1_x), "value": float(best_f1)},
        "best_f4": {"update": int(best_f4_x), "value": float(best_f4)},
        "selected_checkpoint": f"checkpoint-{best_f4_x:06d}",
        "outputs": [str(OUT.with_suffix(".pdf")), str(OUT.with_suffix(".png"))],
    }
    OUT.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
