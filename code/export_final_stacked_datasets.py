"""Export the analysis-ready stacked event-study datasets used by the updated paper.

This makes the regression input explicit instead of reconstructing it only in memory.
It writes one row-level file for the 34-event preferred documentary sample and one
for the 29-event stricter post-support sensitivity sample.

Important: these are stacked university-year-event rows, not independent policy
events. The preferred file contains 34 actual revision events (8 upward, 26
downward); comparison universities can appear in multiple event stacks.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import revision_event_study as core
import revision_replication as canon
import expanded_manual_revision_analysis as expanded

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived"


def annotate_stacks(stacks: pd.DataFrame, events: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    meta = events.copy().rename(columns={
        "institution": "event_institution",
        "revision_year": "event_revision_year_meta",
        "direction": "event_direction_meta",
        "sign": "event_sign_meta",
    })
    keep_meta = [
        "event_id", "event_institution", "event_revision_year_meta",
        "event_direction_meta", "event_sign_meta", "delta_pcsi",
        "prev_doc_year", "comparability", "timing_tier",
        "source_sample", "review_confidence", "treated_post_core_years",
    ]
    keep_meta = [c for c in keep_meta if c in meta.columns]
    x = stacks.merge(meta[keep_meta], left_on="stack", right_on="event_id", how="left", validate="many_to_one")
    x.insert(0, "analysis_sample", sample_name)
    x["control"] = (x["treated"] == 0).astype(int)
    x["post"] = (x["event_time"] >= 0).astype(int)
    x["pre"] = (x["event_time"] < 0).astype(int)

    if "event_revision_year_meta" in x.columns:
        if not (x["event_year"].astype(int) == x["event_revision_year_meta"].astype(int)).all():
            raise AssertionError("Event-year metadata mismatch")
    if "event_direction_meta" in x.columns:
        if not (x["direction"].astype(str) == x["event_direction_meta"].astype(str)).all():
            raise AssertionError("Event-direction metadata mismatch")

    preferred = [
        "analysis_sample", "stack", "event_id", "event_institution",
        "event_year", "event_revision_year_meta", "event_direction_meta",
        "delta_pcsi", "prev_doc_year", "comparability", "timing_tier",
        "source_sample", "review_confidence", "treated_post_core_years",
        "institution", "year", "event_time", "treated", "control",
        "event_sign", "direction", "pre", "post", "raw_id", "private",
        "size_tercile", "research_exp", "ln_research_exp", "licensing_ftes",
        "ln_licensing_ftes", "royalty_share", "disclosures", "ln_disclosures",
        "new_patent_apps", "ln_new_patent_apps", "total_patent_apps",
        "ln_total_patent_apps", "patents_issued", "ln_patents_issued",
        "licenses", "ln_licenses", "filing_margin", "startups", "ln_startups",
    ]
    cols = [c for c in preferred if c in x.columns]
    extra = [c for c in x.columns if c not in cols]
    return x[cols + extra].sort_values(["stack", "institution", "year"]).reset_index(drop=True)


def write_dictionary(path: Path, df34: pd.DataFrame, df29: pd.DataFrame) -> None:
    text = f"""# Derived stacked event-study datasets

These files make the actual row-level inputs to the updated stacked event-study design explicit.

## Files

- `final_stacked_event_dataset_expanded34.csv`: preferred sample, 34 policy-revision events (8 upward, 26 downward).
- `final_stacked_event_dataset_strict29.csv`: stricter sensitivity sample, 29 events (7 upward, 22 downward).

## Row interpretation

A row is a **university-year within a particular event stack**. It is not an independent policy revision. The same comparison university-year can appear in more than one stack when it is a valid clean control for multiple policy revisions.

The preferred stacked file has {len(df34):,} rows, {df34['stack'].nunique()} stacks, and {df34['institution'].nunique()} distinct panel institutions. The strict file has {len(df29):,} rows, {df29['stack'].nunique()} stacks, and {df29['institution'].nunique()} distinct panel institutions.

## Key variables

- `stack` / `event_id`: policy-revision stack identifier.
- `event_institution`: university whose policy revision defines the stack.
- `event_year`: documented revision year used to center event time.
- `event_direction_meta`: upward/supportive or downward/restrictive PCSI revision.
- `delta_pcsi`: change in PCSI from predecessor to successor policy.
- `institution`, `year`: panel university-year.
- `event_time`: calendar year minus event year; main window is -4 through +5.
- `treated`: 1 for the revising university in that stack, 0 for clean controls.
- `control`: 1 for clean controls.
- `size_tercile`, `private`, `ln_research_exp`: main design controls / fixed-effect strata inputs.
- `ln_licenses`, `filing_margin`, `ln_new_patent_apps`, `ln_patents_issued`, `ln_disclosures`: five main outcomes.

Outcome-specific regressions drop rows with missing values required for that outcome and the common regression covariates, so the reported regression N is smaller than the raw number of stacked rows.

## Construction

Generated from `data/merged_autm.csv`, the pinned Paper 1 observed-policy sequence, `data/revision_codes_manual.csv`, and `data/revision_codes_rereview37.csv` using the same panel, revision-universe, clean-control, and event-window code used by the manuscript.
"""
    path.write_text(text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(core.RAW, low_memory=False)
    panel = canon.median_size_panel(raw)
    universe = core.revision_universe(raw)
    events34, events29 = expanded.load_expanded_events(raw)

    stacks34 = core.make_stacks(panel, universe, events34)
    stacks29 = core.make_stacks(panel, universe, events29)
    out34 = annotate_stacks(stacks34, events34, "expanded34")
    out29 = annotate_stacks(stacks29, events29, "strict29")

    if out34["stack"].nunique() != 34:
        raise AssertionError("Preferred stacked dataset must contain 34 stacks")
    if out29["stack"].nunique() != 29:
        raise AssertionError("Strict stacked dataset must contain 29 stacks")
    if set(out34["event_direction_meta"].dropna().unique()) != {"up", "down"}:
        raise AssertionError("Unexpected revision direction labels")

    f34 = OUT / "final_stacked_event_dataset_expanded34.csv"
    f29 = OUT / "final_stacked_event_dataset_strict29.csv"
    out34.to_csv(f34, index=False)
    out29.to_csv(f29, index=False)
    write_dictionary(OUT / "README.md", out34, out29)

    print(f"WROTE {f34}: {len(out34):,} rows, {out34['stack'].nunique()} stacks")
    print(f"WROTE {f29}: {len(out29):,} rows, {out29['stack'].nunique()} stacks")


if __name__ == "__main__":
    main()
