"""Diagnostics and manuscript-facing outputs for the expanded hand-reviewed revision sample.

The preferred expanded sample contains 34 documentary-reviewed revisions (8 upward,
26 downward). A stricter sensitivity requires at least two treated-institution outcome
panel years in event times 0..+5 and contains 29 revisions (7 upward, 22 downward).

This script produces event-time paths, BH-adjusted outcome results, balance diagnostics,
leave-one-event-out estimates, and stack composition for the revised manuscript.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import expanded_manual_revision_analysis as exp
import revision_event_study as core
import revision_replication as canon

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTCOMES = exp.OUTCOMES


def bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * len(p) / np.arange(1, len(p) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)
    return out


def event_paths(panel, universe, events, sample_name):
    stacks = core.make_stacks(panel, universe, events)
    rows = []
    for outcome in OUTCOMES:
        fit = core.fit_outcome(stacks, outcome)
        for r in fit["path"]:
            rows.append({
                "sample": sample_name,
                "outcome": outcome,
                "event_time": int(r["event_time"]),
                "gap": float(r["gap"]),
                "se": float(r["se"]),
                "lo95": float(r["gap"] - 1.96 * r["se"]),
                "hi95": float(r["gap"] + 1.96 * r["se"]),
            })
    return pd.DataFrame(rows), stacks


def p1_prev_pcsi_map(raw):
    obs = core.load_p1_observed().copy()
    cw = raw[["Institution_pci", "Institution_std"]].dropna().copy()
    cw["p1_norm"] = core.norm(cw["Institution_pci"])
    cw = (cw.groupby("p1_norm", observed=True)["Institution_std"]
          .agg(lambda z: z.value_counts().index[0]).rename("institution").reset_index())
    obs["p1_norm"] = core.norm(obs["Institution"])
    z = obs.merge(cw, on="p1_norm", how="inner")
    z["Year"] = pd.to_numeric(z["Year"], errors="coerce")
    return {(str(r.institution), int(r.Year)): float(r.prev_pcsi)
            for r in z.itertuples() if pd.notna(r.Year) and pd.notna(r.prev_pcsi)}


def ols_slope(year, val):
    y = np.asarray(val, float)
    x = np.asarray(year, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    x = x[ok]; y = y[ok]
    x = x - x.mean()
    den = float(x @ x)
    return float((x @ (y - y.mean())) / den) if den > 0 else np.nan


def event_balance(panel, raw, events, draws=50000, seed=20260926):
    prev_map = p1_prev_pcsi_map(raw)
    rows = []
    for _, e in events.iterrows():
        inst = str(e.institution); E = int(e.revision_year)
        g = panel[panel["institution"].astype(str).eq(inst)].copy()
        pre = g[g["year"].between(E - 3, E - 1)]
        trend = g[g["year"].between(E - 4, E - 1)]
        rows.append({
            "institution": inst,
            "revision_year": E,
            "direction": e.direction,
            "sign": 1 if e.direction == "up" else -1,
            "ln_licenses": pre["ln_licenses"].mean(),
            "ln_disclosures": pre["ln_disclosures"].mean(),
            "ln_new_patent_apps": pre["ln_new_patent_apps"].mean(),
            "filing_margin": pre["filing_margin"].mean(),
            "ln_research_exp": pre["ln_research_exp"].mean(),
            "ln_licensing_ftes": pre["ln_licensing_ftes"].mean(),
            "trend_ln_licenses": ols_slope(trend["year"], trend["ln_licenses"]),
            "trend_filing_margin": ols_slope(trend["year"], trend["filing_margin"]),
            "private": float(g["private"].dropna().iloc[0]) if g["private"].notna().any() else np.nan,
            "size_tercile": float(g["size_tercile"].dropna().iloc[0]) if g["size_tercile"].notna().any() else np.nan,
            "prev_pcsi": prev_map.get((inst, E), np.nan),
            "document_gap_years": float(E - e.prev_doc_year) if pd.notna(e.prev_doc_year) else np.nan,
            "timing_A": 1.0 if str(e.timing_tier) == "A" else 0.0,
            "comparable_C": 1.0 if str(e.comparability) == "C" else 0.0,
        })
    ev = pd.DataFrame(rows)
    labels = {
        "ln_licenses": "Log licenses and options",
        "ln_disclosures": "Log invention disclosures",
        "ln_new_patent_apps": "Log new patent applications",
        "filing_margin": "Filing margin",
        "ln_research_exp": "Log research expenditure",
        "ln_licensing_ftes": "Log licensing FTEs",
        "trend_ln_licenses": "Trend in log licenses (per year)",
        "trend_filing_margin": "Trend in filing margin (per year)",
        "private": "Private institution",
        "size_tercile": "Research-size tercile (0-2)",
        "prev_pcsi": "PCSI of previous document",
        "revision_year": "Revision year",
        "document_gap_years": "Years since previous observed document",
        "timing_A": "Timing tier A",
        "comparable_C": "Comparable document pair",
    }
    rng = np.random.default_rng(seed)
    out = []
    for c, label in labels.items():
        vals = pd.to_numeric(ev[c], errors="coerce").to_numpy(float)
        mask = np.isfinite(vals)
        signs = ev.sign.to_numpy()[mask]
        v = vals[mask]
        up = v[signs > 0]; dn = v[signs < 0]
        mu_u = float(np.mean(up)) if len(up) else np.nan
        mu_d = float(np.mean(dn)) if len(dn) else np.nan
        vu = float(np.var(up, ddof=1)) if len(up) > 1 else np.nan
        vd = float(np.var(dn, ddof=1)) if len(dn) > 1 else np.nan
        denom = np.sqrt(np.nanmean([vu, vd]))
        sd = float((mu_u - mu_d) / denom) if np.isfinite(denom) and denom > 0 else np.nan
        obs = mu_u - mu_d
        m = len(v); m_up = int((signs > 0).sum())
        extreme = 0
        for _ in range(draws):
            ix = rng.choice(m, size=m_up, replace=False)
            lab = np.zeros(m, dtype=bool); lab[ix] = True
            diff = float(v[lab].mean() - v[~lab].mean())
            extreme += abs(diff) >= abs(obs) - 1e-12
        pp = (extreme + 1) / (draws + 1)
        out.append({"characteristic": label, "upward": mu_u, "downward": mu_d,
                    "std_diff": sd, "perm_p": pp, "n": m})
    return ev, pd.DataFrame(out)


def stack_composition(stacks, events):
    rows = []
    em = events.set_index("event_id")
    for sid, s in stacks.groupby("stack", observed=True):
        e = em.loc[int(sid)]
        use = s[s["ln_licenses"].notna() & s["ln_research_exp"].notna()]
        tuse = use[use["treated"].eq(1)]
        cuse = use[use["treated"].eq(0)]
        E = int(e.revision_year)
        rows.append({
            "institution": e.institution,
            "revision_year": E,
            "direction": e.direction,
            "window_start": E + core.EVENT_MIN,
            "window_end": E + core.EVENT_MAX,
            "treated_obs": int(len(tuse)),
            "controls": int(cuse["institution"].nunique()),
            "obs": int(len(use)),
        })
    return pd.DataFrame(rows)


def leave_one_out(panel, universe, events):
    full = core.fit_outcome(core.make_stacks(panel, universe, events), "ln_licenses")["gap"]
    rows = []
    for i in range(len(events)):
        drop = events.iloc[i]
        e = events.drop(events.index[i]).copy().reset_index(drop=True)
        e["event_id"] = np.arange(len(e))
        e["sign"] = np.where(e["direction"].eq("up"), 1.0, -1.0)
        gap = core.fit_outcome(core.make_stacks(panel, universe, e), "ln_licenses")["gap"]
        rows.append({"dropped_institution": drop.institution,
                     "dropped_year": int(drop.revision_year),
                     "dropped_direction": drop.direction,
                     "gap": float(gap)})
    loo = pd.DataFrame(rows)
    summary = {"full_gap": float(full), "min_gap": float(loo.gap.min()),
               "max_gap": float(loo.gap.max())}
    return loo, summary


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(core.RAW, low_memory=False)
    panel = canon.median_size_panel(raw)
    universe = core.revision_universe(raw)
    e34, e29 = exp.load_expanded_events(raw)

    p34, s34 = event_paths(panel, universe, e34, "expanded34")
    p29, _ = event_paths(panel, universe, e29, "strict29")
    pd.concat([p34, p29], ignore_index=True).to_csv(
        RESULTS / "expanded_event_time_paths.csv", index=False)

    res = pd.read_csv(RESULTS / "expanded_manual_sample_results.csv")
    res["bh_q"] = np.nan
    for sample, idx in res.groupby("sample").groups.items():
        res.loc[idx, "bh_q"] = bh_adjust(res.loc[idx, "permutation_p"].to_numpy())
    res.to_csv(RESULTS / "expanded_manual_sample_results.csv", index=False)

    evbal, bal = event_balance(panel, raw, e34)
    evbal.to_csv(RESULTS / "expanded34_event_level_balance_data.csv", index=False)
    bal.to_csv(RESULTS / "expanded34_balance.csv", index=False)

    stacks = stack_composition(s34, e34)
    stacks.to_csv(RESULTS / "expanded34_stack_composition.csv", index=False)

    loo, loo_summary = leave_one_out(panel, universe, e34)
    loo.to_csv(RESULTS / "expanded34_leave_one_out.csv", index=False)

    summary = {
        "expanded34_bh_q": res[res["sample"].eq("expanded34")]
            .set_index("outcome")["bh_q"].to_dict(),
        "strict29_bh_q": res[res["sample"].eq("strict29")]
            .set_index("outcome")["bh_q"].to_dict(),
        "expanded34_leave_one_out": loo_summary,
        "balance_draws": 50000,
    }
    (RESULTS / "expanded34_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print("EXPANDED34 DIAGNOSTICS", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
