"""Descriptive tables: Table 1, Table 2, Table A2, Table A10 and Table A11.

All values are computed from the frozen AUTM input, the frozen Paper 1 policy file and
the two documentary-coding files. Outputs go to results/descriptive_*.csv and
results/descriptive_summary.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import revision_event_study as core
import expanded_manual_revision_analysis as exp
import mechanical_screen as screen

num = core.num


def table1(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    p = core.make_panel(raw)
    pci = num(raw["Mean_Tone_Score"])
    rows = [("PCSI (policy in force)", pci),
            ("Invention disclosures", p.disclosures),
            ("New patent applications", p.new_patent_apps),
            ("U.S. patents issued", p.patents_issued),
            ("Licenses and options executed", p.licenses),
            ("Research expenditure ($ millions)", p.research_exp / 1e6),
            ("Licensing FTEs", p.licensing_ftes),
            ("Startups formed", p.startups)]
    t = pd.DataFrame([{"variable": n, "N": int(s.notna().sum()), "mean": s.mean(), "sd": s.std()}
                      for n, s in rows])
    cf = num(raw["Is_Carried_Forward"])
    d = pd.DataFrame({"i": raw["Institution_std"], "pci": pci}).dropna()
    within = float((d.pci - d.groupby("i").pci.transform("mean")).var(ddof=0) / d.pci.var(ddof=0))
    extra = {"institution_years": int(len(p)),
             "harmonized_institutions": int(p.institution.nunique()),
             "autm_reporting_ids": int(raw["[ID]"].nunique()),
             "institutions_with_multiple_ids": int((raw.groupby("Institution_std")["[ID]"].nunique() > 1).sum()),
             "share_pcsi_carried_forward": float(cf[pci.notna()].mean()),
             "within_institution_share_pcsi_variance": within}
    return t, extra


def table2(raw: pd.DataFrame) -> dict:
    obs = core.load_p1_observed()
    universe = core.revision_universe(raw)
    s = screen.mechanical_screen(raw)
    manual = pd.read_csv(core.EVENTS)
    rr = pd.read_csv(exp.REREVIEW)
    e34, s29 = exp.load_expanded_events(raw)
    return {"observed_policy_records": int(len(obs)),
            "threshold_revisions": int(len(universe)),
            "threshold_revision_institutions": int(universe.institution.nunique()),
            "pass_mechanical_screen": int(s.passes_screen.sum()),
            "previously_reviewed_sept25": int(len(manual)),
            "rereviewed_second_pass": int(len(rr)),
            "usable_second_pass": int(rr.usable_documentary_pair.astype(str).str.lower().eq("true").sum()),
            "preferred_sample": int(len(e34)),
            "preferred_up": int((e34.sign > 0).sum()), "preferred_down": int((e34.sign < 0).sum()),
            "preferred_institutions": int(e34.institution.nunique()),
            "strict_sample": int(len(s29)),
            "strict_up": int((s29.sign > 0).sum()), "strict_down": int((s29.sign < 0).sum())}


def tableA2(raw: pd.DataFrame) -> pd.DataFrame:
    obs = core.load_p1_observed()
    cw = raw[["Institution_pci", "Institution_std"]].dropna().copy()
    cw["k"] = core.norm(cw["Institution_pci"])
    cw = cw.groupby("k")["Institution_std"].agg(lambda z: z.value_counts().index[0])
    z = obs.assign(inst=core.norm(obs["Institution"]).map(cw)).dropna(subset=["inst"])
    out = []
    for t in [0.02, 0.025, 0.03, 0.04, 0.05]:
        r = z[z.delta_pcsi.abs() > t]
        out.append({"threshold": t, "revisions": len(r), "institutions": r.inst.nunique(),
                    "up": int((r.delta_pcsi > 0).sum()), "down": int((r.delta_pcsi < 0).sum())})
    return pd.DataFrame(out)


def tableA10(raw: pd.DataFrame) -> pd.DataFrame:
    yr = num(raw["Year"])
    tot = num(raw["Tot Lic/Opt Exe"])
    comp = num(raw["Lic Iss"]) + num(raw["Opt Iss"])
    out = []
    for y in range(2016, 2024):
        m = yr.eq(y)
        complete = m & tot.notna() & comp.notna()
        out.append({"year": y,
                    # share of all records in which total == licenses + options
                    # (records with a missing field count as not equal)
                    "share_equal_all_records": float((tot[m] == comp[m]).mean()),
                    # same share among records reporting all three fields
                    "share_equal_complete_records": float((tot[complete] == comp[complete]).mean()),
                    "median_total_field": float(tot[m].median()),
                    "median_licenses_plus_options": float(comp[m].median())})
    return pd.DataFrame(out)


def tableA11(raw: pd.DataFrame) -> pd.DataFrame:
    p = core.make_panel(raw)
    out = []
    for y in [1994, 1995, 2000, 2005, 2010, 2015, 2020, 2021, 2022, 2023]:
        g = p[p.year == y]
        rep = g.startups.notna()
        cur = set(g.loc[rep, "institution"])
        prev_g = p[p.year == y - 1]
        prev = set(prev_g.loc[prev_g.startups.notna(), "institution"])
        out.append({"year": y, "reported": float(rep.mean()),
                    "zero_share": float((g.startups[rep] == 0).mean()),
                    "median": float(g.startups.median()),
                    "entering": len(cur - prev) if y > 1994 else np.nan,
                    "exiting": len(prev - cur) if y > 1994 else np.nan})
    return pd.DataFrame(out)


def main():
    raw = pd.read_csv(core.RAW, low_memory=False)
    core.RESULTS.mkdir(parents=True, exist_ok=True)
    t1, t1extra = table1(raw)
    t1.to_csv(core.RESULTS / "descriptive_table1.csv", index=False)
    t2 = table2(raw)
    a2 = tableA2(raw); a2.to_csv(core.RESULTS / "descriptive_tableA2.csv", index=False)
    a10 = tableA10(raw); a10.to_csv(core.RESULTS / "descriptive_tableA10.csv", index=False)
    a11 = tableA11(raw); a11.to_csv(core.RESULTS / "descriptive_tableA11.csv", index=False)
    summary = {"table1_panel": t1extra, "table2": t2}
    (core.RESULTS / "descriptive_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("DESCRIPTIVE TABLES", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
