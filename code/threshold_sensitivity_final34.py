"""Sensitivity of the final 34-event design to stricter PCSI-change cutoffs.

This nested-sample check starts from the final documentary-reviewed 34-event sample and
raises the minimum absolute PCSI change from 0.03 to 0.04 and 0.05. At each cutoff the
revision universe used to exclude contaminated controls is rebuilt.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import revision_event_study as core
import revision_replication as canon
import expanded_manual_revision_analysis as exp

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
THRESHOLDS = [0.03, 0.04, 0.05]
OUTCOMES = ["ln_licenses", "filing_margin"]


def revision_universe_at_threshold(raw: pd.DataFrame, threshold: float) -> pd.DataFrame:
    obs = core.load_p1_observed()
    cw = raw[["Institution_pci", "Institution_std"]].dropna().copy()
    cw["p1_norm"] = core.norm(cw["Institution_pci"])
    cw = (cw.groupby("p1_norm", observed=True)["Institution_std"]
          .agg(lambda z: z.value_counts().index[0]).rename("institution").reset_index())
    obs["p1_norm"] = core.norm(obs["Institution"])
    z = obs.merge(cw, on="p1_norm", how="inner")
    rev = z[z["delta_pcsi"].abs() > threshold].copy()
    rev = rev.rename(columns={"Year": "revision_year"})
    rev["direction"] = np.where(rev["delta_pcsi"] > 0, "up", "down")
    return (rev[["institution", "Institution", "prev_year", "revision_year",
                 "delta_pcsi", "direction"]]
            .sort_values(["institution", "revision_year"])
            .reset_index(drop=True))


def reindex_events(events: pd.DataFrame) -> pd.DataFrame:
    e = events.copy().reset_index(drop=True)
    e["event_id"] = np.arange(len(e))
    e["sign"] = np.where(e["direction"].eq("up"), 1.0, -1.0)
    return e


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(core.RAW, low_memory=False)
    panel = canon.median_size_panel(raw)
    e34, _ = exp.load_expanded_events(raw)

    rows = []
    for j, threshold in enumerate(THRESHOLDS):
        universe = revision_universe_at_threshold(raw, threshold)
        events = reindex_events(e34[e34["delta_pcsi"].abs() > threshold])
        for k, outcome in enumerate(OUTCOMES):
            stacks = core.make_stacks(panel, universe, events)
            point = core.fit_outcome(stacks, outcome)
            perm = exp.monte_carlo_direction_p(
                stacks, events, outcome, draws=50000,
                seed=20261007 + 100 * j + k
            )
            rows.append({
                "threshold": threshold,
                "outcome": outcome,
                "events": len(events),
                "up_events": int((events["sign"] > 0).sum()),
                "down_events": int((events["sign"] < 0).sum()),
                "revision_universe": len(universe),
                "revision_institutions": universe["institution"].nunique(),
                "gap": point["gap"],
                "cluster_se": point["gap_se"],
                "pre_p": point["pre_p"],
                "upward": point["up"],
                "downward": point["down"],
                "permutation_p": perm["permutation_p"],
                "permutation_mc_se": perm["mc_se"],
                "n": point["n"],
            })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "threshold_sensitivity_final34.csv", index=False)
    print("\nSTRICTER PCSI-THRESHOLD SENSITIVITY")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
