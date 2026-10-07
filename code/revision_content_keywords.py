"""Table 5: content added in upward and downward revisions (both panels).

For every threshold revision, compare the pooled text of the earlier and later
observed policy documents (Paper 1 sentence file, pooled by institution-year, as in
the PCSI construction). A revision "adds" a provision type when the later text matches
the keyword pattern and the earlier text does not. Two-sided Fisher exact tests compare
upward and downward revisions. Keyword presence is descriptive only.

Panels: "Original 25-event content-coded subset" (the September-25 events, each matched
to its revision by institution and predecessor-document year) and "All revisions" (the
full threshold universe). Appendix Table A12 comes from hand coding and is not
regenerated here (see TABLE_PROVENANCE.md).
"""
from __future__ import annotations

import re

import pandas as pd
from scipy.stats import fisher_exact

import revision_event_study as core

SENTENCES = core.ROOT / "data" / "p1_sentence_scores_canonical.csv"
PATTERNS = {
    "Adds present assignment language": r"hereby assign",
    "Adds equity language": r"\bequity\b|\bstock\b",
    "Adds conflict of interest language": r"conflicts? of interest",
    "Adds startup language": r"start-?up|spin-?off|new venture",
    "Adds copyright language": r"copyright",
}


def compare(pairs: pd.DataFrame, text: pd.Series, panel: str) -> pd.DataFrame:
    flags = []
    for _, r in pairs.iterrows():
        before = text.get((r.Institution, int(r.prev_year)), "")
        after = text.get((r.Institution, int(r.revision_year)), "")
        flags.append({"direction": r.direction,
                      **{k: bool(re.search(p, after)) and not re.search(p, before)
                         for k, p in PATTERNS.items()}})
    f = pd.DataFrame(flags)
    up, down = f[f.direction == "up"], f[f.direction == "down"]
    rows = []
    for k in PATTERNS:
        a, b = int(up[k].sum()), int(down[k].sum())
        p = fisher_exact([[a, len(up) - a], [b, len(down) - b]])[1]
        rows.append({"panel": panel, "content": k, "upward_share": a / len(up),
                     "downward_share": b / len(down), "fisher_p": p,
                     "n_up": len(up), "n_down": len(down)})
    return pd.DataFrame(rows)


def main():
    raw = pd.read_csv(core.RAW, low_memory=False)
    universe = core.revision_universe(raw)
    s = pd.read_csv(SENTENCES, usecols=["Institution", "Year", "Sentence_Cleaned"], low_memory=False)
    text = (s.groupby(["Institution", "Year"])["Sentence_Cleaned"]
            .apply(lambda z: " ".join(map(str, z)).lower()))
    sept25 = core.load_main_events()
    matched = []
    for _, e in sept25.iterrows():
        u = universe[(universe.institution == e.institution) & (universe.prev_year == e.prev_doc_year)]
        if len(u) != 1:
            raise AssertionError(f"Could not match {e.institution} {e.revision_year} to one revision")
        matched.append(u.iloc[0])
    sept25_pairs = pd.DataFrame(matched)
    out = pd.concat([compare(sept25_pairs, text, "Original 25-event content-coded subset"),
                     compare(universe, text, "All revisions")], ignore_index=True)
    core.RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(core.RESULTS / "revision_content_keywords.csv", index=False)
    print("REVISION CONTENT KEYWORDS")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
