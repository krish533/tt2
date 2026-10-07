"""Shared construction and exploratory inference for the expanded documentary samples.

The revised manuscript treats the 34-event hand-reviewed documentary sample as the
preferred specification. It combines the frozen September 25 25-event benchmark with
nine additional revisions that pass the same C/P/N comparability and A/B/C timing
review rules. The 25-event design remains a frozen benchmark/sensitivity for provenance.

Samples:
  * expanded34: existing 25 + all 9 newly reviewed C/P & A/B pairs; preferred sample.
  * strict29: apply a uniform sensitivity requiring at least two observed treated
    institution panel years in event times 0..+5 to the combined hand-reviewed set.
    This drops one existing benchmark event and four of the nine additions, leaving 29.

This module defaults to 20,000 random direction assignments for development runs.
Publication-facing inference is executed by final_expanded_inference.py with 100,000
assignments preserving the number of upward revisions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import revision_event_study as core
import revision_replication as canon

ROOT = Path(__file__).resolve().parents[1]
REREVIEW = ROOT / "data" / "revision_codes_rereview37.csv"
RESULTS = ROOT / "results"
OUTCOMES = ["ln_licenses", "filing_margin", "ln_new_patent_apps",
            "ln_patents_issued", "ln_disclosures"]


def load_expanded_events(raw: pd.DataFrame):
    base = core.load_main_events().copy()
    base["source_sample"] = "sept25_main25"
    base["review_confidence"] = "baseline"

    rr = pd.read_csv(REREVIEW)
    rr["usable_documentary_pair"] = rr["usable_documentary_pair"].astype(str).str.lower().eq("true")
    add = rr[rr["usable_documentary_pair"]].copy()
    add["revision_year"] = pd.to_numeric(add["event_year"], errors="raise").astype(int)
    add["prev_doc_year"] = pd.to_numeric(add["prev_year"], errors="coerce")
    add["source_sample"] = "rereview37_addition"
    add["review_confidence"] = add["confidence"].astype(str)
    add["main_sample"] = 0

    cols = ["institution", "prev_doc_year", "revision_year", "delta_pcsi",
            "comparability", "timing_tier", "direction", "main_sample",
            "source_sample", "review_confidence"]
    base2 = base.copy()
    for c in cols:
        if c not in base2.columns:
            base2[c] = np.nan
    e34 = pd.concat([base2[cols], add[cols]], ignore_index=True)
    if e34.duplicated(["institution", "revision_year"]).any():
        dup = e34[e34.duplicated(["institution", "revision_year"], keep=False)]
        raise AssertionError(f"Duplicate event keys in expanded sample:\n{dup}")
    e34["event_id"] = np.arange(len(e34))
    e34["sign"] = np.where(e34["direction"].eq("up"), 1.0, -1.0)
    if (len(e34), int((e34.sign > 0).sum()), int((e34.sign < 0).sum())) != (34, 8, 26):
        raise AssertionError("Expanded documentary sample must contain 34 events: 8 up / 26 down")

    panel = canon.median_size_panel(raw)
    post_counts = []
    for _, e in e34.iterrows():
        g = panel[(panel["institution"].astype(str) == str(e.institution)) &
                  panel["year"].between(int(e.revision_year), int(e.revision_year) + 5)]
        core_cols = [c for c in OUTCOMES if c in g.columns]
        if core_cols:
            years = g.loc[g[core_cols].notna().any(axis=1), "year"].nunique()
        else:
            years = 0
        post_counts.append(int(years))
    e34["treated_post_core_years"] = post_counts
    strict = e34[e34["treated_post_core_years"] >= 2].copy().reset_index(drop=True)
    strict["event_id"] = np.arange(len(strict))
    strict["sign"] = np.where(strict["direction"].eq("up"), 1.0, -1.0)
    if (len(strict), int((strict.sign > 0).sum()), int((strict.sign < 0).sum())) != (29, 7, 22):
        raise AssertionError(
            "Strict full-post sensitivity expected 29 events: 7 up / 22 down; got "
            f"{len(strict)} / {(strict.sign > 0).sum()} / {(strict.sign < 0).sum()}"
        )
    return e34, strict


def monte_carlo_direction_p(stacks, events, outcome, draws=20000, seed=20260926,
                            batch_size=2000):
    q, dcols, stack_ids, A, arhs, C, Bparts, cparts = canon.sufficient_stats(stacks, outcome)
    n_events = len(stack_ids)
    sign_map = events.set_index("event_id").sign.to_dict()
    observed_signs = np.array([sign_map[int(s)] for s in stack_ids], dtype=float)
    n_up = int((observed_signs > 0).sum())
    m = len(dcols)
    p0 = A.shape[0]
    post = [dcols.index(f"D_p{k}") for k in core.POST_TIMES]
    obs_beta = canon.solve_signs(observed_signs, A, arhs, C, Bparts, cparts)
    obs_gap = float((2 / len(post)) * obs_beta[p0 + np.array(post)].sum())
    point = core.fit_outcome(stacks, outcome)
    if abs(obs_gap - point["gap"]) > 1e-7:
        raise AssertionError(f"Permutation normal-equation gap {obs_gap} != fit gap {point['gap']}")

    rng = np.random.default_rng(seed)
    extreme = 0
    done = 0
    dim = p0 + m
    while done < draws:
        b = min(batch_size, draws - done)
        S = -np.ones((b, n_events), dtype=float)
        for i in range(b):
            ix = rng.choice(n_events, size=n_up, replace=False)
            S[i, ix] = 1.0
        Bb = np.einsum("be,eij->bij", S, Bparts, optimize=True)
        cb = S @ cparts
        mats = np.empty((b, dim, dim), float)
        mats[:, :p0, :p0] = A
        mats[:, :p0, p0:] = Bb
        mats[:, p0:, :p0] = np.swapaxes(Bb, 1, 2)
        mats[:, p0:, p0:] = C
        rhs = np.empty((b, dim), float)
        rhs[:, :p0] = arhs
        rhs[:, p0:] = cb
        try:
            betas = np.linalg.solve(mats, rhs[..., None]).squeeze(-1)
        except np.linalg.LinAlgError:
            betas = np.einsum("bij,bj->bi", np.linalg.pinv(mats), rhs)
        gaps = (2 / len(post)) * betas[:, p0 + np.array(post)].sum(axis=1)
        extreme += int((np.abs(gaps) >= abs(obs_gap) - 1e-12).sum())
        done += b

    p = (extreme + 1) / (draws + 1)
    mcse = float(np.sqrt(p * (1 - p) / (draws + 1)))
    return {"permutation_p": float(p), "mc_se": mcse, "draws": int(draws),
            "extreme": int(extreme), "observed_gap": obs_gap, "n_events": n_events,
            "n_up": n_up, "n_down": n_events - n_up, "n": int(len(q))}


def run_sample(name, panel, universe, events, draws=20000):
    stacks = core.make_stacks(panel, universe, events)
    rows = []
    for j, outcome in enumerate(OUTCOMES):
        point = core.fit_outcome(stacks, outcome)
        perm = monte_carlo_direction_p(stacks, events, outcome, draws=draws,
                                       seed=20260926 + j + (100 if name == "strict29" else 0))
        rows.append({
            "sample": name,
            "outcome": outcome,
            "events": len(events),
            "up_events": int((events.sign > 0).sum()),
            "down_events": int((events.sign < 0).sum()),
            "gap": point["gap"],
            "cluster_se": point["gap_se"],
            "pre_p": point["pre_p"],
            "upward": point["up"],
            "downward": point["down"],
            "permutation_p": perm["permutation_p"],
            "permutation_mc_se": perm["mc_se"],
            "n": point["n"],
            "clusters": point["clusters"],
        })
    return pd.DataFrame(rows)


def main():
    raw = pd.read_csv(core.RAW, low_memory=False)
    panel = canon.median_size_panel(raw)
    universe = core.revision_universe(raw)
    expanded34, strict29 = load_expanded_events(raw)

    RESULTS.mkdir(parents=True, exist_ok=True)
    expanded34.to_csv(RESULTS / "expanded34_event_list.csv", index=False)
    strict29.to_csv(RESULTS / "strict29_event_list.csv", index=False)

    r34 = run_sample("expanded34", panel, universe, expanded34)
    r29 = run_sample("strict29", panel, universe, strict29)
    out = pd.concat([r34, r29], ignore_index=True)
    out.to_csv(RESULTS / "expanded_manual_sample_results.csv", index=False)

    summary = {
        "permutation_draws": 20000,
        "analysis_mode": "development",
        "expanded34": {
            "definition": "preferred revised-manuscript documentary sample; frozen 25-event benchmark plus 9 independently re-reviewed usable additions",
            "events": 34, "up": 8, "down": 26,
            "new_events": 9,
            "results": r34.to_dict(orient="records"),
        },
        "strict29": {
            "definition": "sensitivity uniformly requiring >=2 treated core-outcome panel years in t=0..+5",
            "events": 29, "up": 7, "down": 22,
            "results": r29.to_dict(orient="records"),
        },
        "note": "Development entry point uses 20,000 assignments. Run final_expanded_inference.py for publication-facing 100,000-assignment estimates."
    }
    (RESULTS / "expanded_manual_sample_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
