"""Build the publication-facing main and appendix figures from reproduced results."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    p = pd.read_csv(RESULTS / "expanded_event_time_paths.csv")
    p = p[p["sample"].eq("expanded34")].copy()

    outcomes = [
        ("ln_licenses", "Log licenses and options", "(a) Licensing"),
        ("filing_margin", "Filing margin", "(b) Filing margin"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for ax, (outcome, ylabel, title) in zip(axes, outcomes):
        d = p[p["outcome"].eq(outcome)].sort_values("event_time")
        ax.axhline(0, linewidth=0.8)
        ax.axvline(-1, linewidth=0.8, linestyle="--")
        ax.errorbar(
            d["event_time"],
            d["gap"],
            yerr=1.96 * d["se"],
            fmt="o-",
            capsize=2.5,
            linewidth=1.2,
        )
        ax.set_title(title)
        ax.set_xlabel("Years from policy revision")
        ax.set_ylabel(ylabel)
        ax.set_xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

    fig.tight_layout()
    fig.savefig(FIGDIR / "figure1_event_study.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "figure1_event_study.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    loo = pd.read_csv(RESULTS / "expanded34_leave_one_out.csv").copy()
    loo = loo.sort_values("gap").reset_index(drop=True)
    labels = (
        loo["dropped_institution"].astype(str)
        + " ("
        + loo["dropped_year"].astype(int).astype(str)
        + ")"
    )

    fig, ax = plt.subplots(figsize=(7.5, 8.2))
    ax.axvline(0.4949823607577816, linewidth=0.9, linestyle="--")
    ax.scatter(loo["gap"], range(len(loo)))
    ax.set_yticks(range(len(loo)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("Licensing gap after dropping one revision")
    ax.set_ylabel("")
    ax.set_title("Leave-one-event-out licensing estimates")
    fig.tight_layout()
    fig.savefig(FIGDIR / "figureA1_leave_one_out.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "figureA1_leave_one_out.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote figures to {FIGDIR}")


if __name__ == "__main__":
    main()
