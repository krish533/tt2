"""Publication-facing inference for the 34-event and strict 29-event samples.

Uses 100,000 Monte Carlo direction assignments for each outcome to reduce simulation
noise relative to the exploratory 20,000-draw runs. Outputs overwrite the exploratory
expanded-sample result files with the publication-facing estimates.
"""
from __future__ import annotations

import json
import pandas as pd

import expanded_manual_revision_analysis as exp
import revision_event_study as core
import revision_replication as canon

DRAWS = 100_000


def main():
    raw = pd.read_csv(core.RAW, low_memory=False)
    panel = canon.median_size_panel(raw)
    universe = core.revision_universe(raw)
    expanded34, strict29 = exp.load_expanded_events(raw)

    exp.RESULTS.mkdir(parents=True, exist_ok=True)
    expanded34.to_csv(exp.RESULTS / "expanded34_event_list.csv", index=False)
    strict29.to_csv(exp.RESULTS / "strict29_event_list.csv", index=False)

    r34 = exp.run_sample("expanded34", panel, universe, expanded34, draws=DRAWS)
    r29 = exp.run_sample("strict29", panel, universe, strict29, draws=DRAWS)
    out = pd.concat([r34, r29], ignore_index=True)
    out.to_csv(exp.RESULTS / "expanded_manual_sample_results.csv", index=False)

    summary = {
        "permutation_draws": DRAWS,
        "expanded34": {
            "events": 34, "up": 8, "down": 26, "new_events": 9,
            "results": r34.to_dict(orient="records"),
        },
        "strict29": {
            "definition": "uniformly requires >=2 treated core-outcome panel years in t=0..+5",
            "events": 29, "up": 7, "down": 22,
            "results": r29.to_dict(orient="records"),
        },
        "note": "The 34-event sample is the preferred documentary sample; the 29-event sample is the stricter post-support sensitivity."
    }
    (exp.RESULTS / "expanded_manual_sample_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n")
    print("FINAL EXPANDED INFERENCE", json.dumps(summary, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
