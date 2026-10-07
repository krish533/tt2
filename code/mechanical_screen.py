"""Mechanical non-overlap and outcome-support screen (Table 2, Section 3.3).

Starting from the 126 threshold revisions, a revision passes when
  1. the same institution has no other threshold revision in the event window
     [E-4, E+5], and
  2. the institution has at least one AUTM year with any core outcome before E,
     and at least one in E..E+5.
The screen uses outcome *availability* only, never outcome values. The passing set
must equal the 62 document pairs that were reviewed (25 September-25 pairs plus the 37
re-reviewed pairs); the script stops if it does not.
"""
from __future__ import annotations

import pandas as pd

import revision_event_study as core
import revision_replication as canon
import expanded_manual_revision_analysis as exp

CORE_OUTCOMES = ["ln_licenses", "filing_margin", "ln_new_patent_apps",
                 "ln_patents_issued", "ln_disclosures"]


def mechanical_screen(raw: pd.DataFrame) -> pd.DataFrame:
    panel = canon.median_size_panel(raw)
    universe = core.revision_universe(raw)
    have = panel.loc[panel[CORE_OUTCOMES].notna().any(axis=1), ["institution", "year"]]
    years = have.groupby("institution")["year"].apply(set)
    rows = []
    for _, r in universe.iterrows():
        E = int(r.revision_year)
        others = universe[(universe.institution == r.institution)
                          & (universe.revision_year != E)
                          & universe.revision_year.between(E + core.EVENT_MIN, E + core.EVENT_MAX)]
        ys = years.get(r.institution, set())
        n_pre = sum(y in ys for y in range(E + core.EVENT_MIN, E))
        n_post = sum(y in ys for y in range(E, E + core.EVENT_MAX + 1))
        overlap = len(others) > 0
        rows.append({**r.to_dict(), "other_revision_in_window": overlap,
                     "n_pre_outcome_years": n_pre, "n_post_outcome_years": n_post,
                     "passes_screen": (not overlap) and n_pre >= 1 and n_post >= 1})
    return pd.DataFrame(rows)


def main():
    raw = pd.read_csv(core.RAW, low_memory=False)
    s = mechanical_screen(raw)
    passed = set(zip(s.loc[s.passes_screen, "institution"], s.loc[s.passes_screen, "revision_year"]))
    manual = pd.read_csv(core.EVENTS)
    rr = pd.read_csv(exp.REREVIEW)
    reviewed = (set(zip(manual.institution, manual.revision_year))
                | set(zip(rr.institution, rr.revision_year)))
    if passed != reviewed:
        raise AssertionError(
            f"Screen/review mismatch: {len(passed - reviewed)} passing but not reviewed, "
            f"{len(reviewed - passed)} reviewed but not passing")
    core.RESULTS.mkdir(parents=True, exist_ok=True)
    s.to_csv(core.RESULTS / "mechanical_screen.csv", index=False)
    print(f"MECHANICAL SCREEN {len(s)} revisions -> {int(s.passes_screen.sum())} pass "
          f"(= {len(manual)} September-25 pairs + {len(rr)} re-reviewed pairs)")


if __name__ == "__main__":
    main()
