"""Reproduce the supporting annual-panel tables in the revised manuscript.

This module uses the same 149-institution harmonization and FY2023 licensing correction
as the documentary event-study code. It produces the continuous-PCSI lag table, the
filing-margin robustness panel, and the revision-predictor diagnostics reported in the
main text/appendix. These are supporting associational analyses, not the preferred design.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, stats

import revision_event_study as core

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 42
CONTROLS = ["ln_research_exp_l1", "ln_licensing_ftes_l1", "royalty_share_l1"]
OUTCOMES = [
    ("ln_disclosures", "Log invention disclosures"),
    ("filing_margin", "Filing margin"),
    ("ln_new_patent_apps", "Log new patent applications"),
    ("ln_total_patent_apps", "Log total patent applications"),
    ("ln_patents_issued", "Log patents issued"),
    ("ln_licenses", "Log licenses/options"),
]


def _numeric(s):
    return pd.to_numeric(s, errors="coerce")


def _percent_numeric(s):
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")


def _calendar_lookup(df: pd.DataFrame, column: str, offset: int) -> np.ndarray:
    lookup = df.set_index(["institution", "year"])[column]
    idx = pd.MultiIndex.from_arrays([df["institution"], df["year"] - offset])
    return lookup.reindex(idx).to_numpy()


def controls_for_lag(lag: int) -> list[str]:
    """Use lag-one controls for contemporaneous/lag-one PCI, then same-lag controls."""
    k = 1 if lag <= 1 else lag
    return [f"ln_research_exp_l{k}", f"ln_licensing_ftes_l{k}", f"royalty_share_l{k}"]


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    p = core.make_panel(raw)
    aux = raw[["Institution_std", "Year", "Mean_Tone_Score", "Carnegie R1", "Royalty Share"]].copy()
    aux = aux.rename(columns={
        "Institution_std": "institution", "Year": "year",
        "Mean_Tone_Score": "pci", "Carnegie R1": "carnegie_r1",
        "Royalty Share": "royalty_share_clean",
    })
    aux["institution"] = aux["institution"].astype("string")
    aux["year"] = _numeric(aux["year"])
    aux["pci"] = _numeric(aux["pci"])
    aux["carnegie_r1"] = _numeric(aux["carnegie_r1"])
    aux["royalty_share_clean"] = _percent_numeric(aux["royalty_share_clean"])
    aux = aux.drop_duplicates(["institution", "year"], keep="last")
    p = p.merge(aux, on=["institution", "year"], how="left", validate="one_to_one")

    # core.make_panel intentionally leaves percentage-formatted royalty shares numeric-missing
    # because the event-study specifications do not use them. The annual supporting models do.
    p["royalty_share"] = p["royalty_share_clean"]

    for col in ["pci", "ln_research_exp", "ln_licensing_ftes", "royalty_share",
                "ln_new_patent_apps", "ln_total_patent_apps", "ln_patents_issued",
                "ln_disclosures", "ln_licenses", "filing_margin"]:
        for k in range(1, 6):
            p[f"{col}_l{k}"] = _calendar_lookup(p, col, k)
    p["ln_research_exp_fwd"] = _calendar_lookup(p, "ln_research_exp", -1)
    return p


def twfe(df: pd.DataFrame, outcome: str, treatment: str,
         controls: list[str] | tuple[str, ...] = CONTROLS):
    cols = [outcome, treatment, "institution", "year", *controls]
    q = df[cols].dropna().copy()
    if q.empty:
        raise ValueError(f"Empty TWFE sample for outcome={outcome}, treatment={treatment}, controls={list(controls)}")
    y = q[outcome].to_numpy(float)
    xcols = [treatment, *controls]
    X = q[xcols].to_numpy(float)
    gi = pd.factorize(q["institution"].astype(str))[0]
    gt = pd.factorize(q["year"].astype(int))[0]
    rz = core._demean(np.column_stack([y, X]), [gi, gt])
    yr, Xr = rz[:, 0], rz[:, 1:]
    beta = np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ yr)
    resid = yr - Xr @ beta
    cov = core._cluster_cov(Xr, resid, q["institution"].astype(str).to_numpy())
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    b, s = float(beta[0]), float(se[0])
    clusters = int(q["institution"].nunique())
    dfree = max(clusters - 1, 1)
    pval = float(2 * stats.t.sf(abs(b / s), dfree)) if s > 0 else np.nan
    return {"beta": b, "se": s, "p": pval, "n": int(len(q)), "clusters": clusters}


def bh_adjust(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * len(p) / np.arange(1, len(p) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)
    return out


def lag_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome, label in OUTCOMES:
        for lag in range(0, 6):
            treatment = "pci" if lag == 0 else f"pci_l{lag}"
            r = twfe(panel, outcome, treatment, controls=controls_for_lag(lag))
            rows.append({"outcome": outcome, "label": label, "lag": lag, **r})
    out = pd.DataFrame(rows)
    lag1 = out[out["lag"].eq(1)].copy()
    lag1["bh_q_lag1"] = bh_adjust(lag1["p"].to_numpy())
    out = out.merge(lag1[["outcome", "bh_q_lag1"]], on="outcome", how="left")
    return out


def winsorized_copy(panel: pd.DataFrame, cols: list[str], q=0.01):
    x = panel.copy()
    for col in cols:
        v = _numeric(x[col])
        lo, hi = v.quantile([q, 1-q])
        x[col] = v.clip(lo, hi)
    return x


def circular_shift_placebo(panel: pd.DataFrame, n_perms=1000, seed=SEED):
    need = ["filing_margin", "pci_l1", "institution", "year", *CONTROLS]
    q = panel[need].dropna().copy().sort_values(["institution", "year"]).reset_index(drop=True)
    nuisance = pd.concat([
        q[CONTROLS].astype(float),
        pd.get_dummies(q["institution"].astype(str), prefix="inst", drop_first=True, dtype=float),
        pd.get_dummies(q["year"].astype(int), prefix="yr", drop_first=True, dtype=float),
    ], axis=1)
    nuisance.insert(0, "const", 1.0)
    z = nuisance.to_numpy(float)
    qq, rr, _ = linalg.qr(z, mode="economic", pivoting=True)
    diag = np.abs(np.diag(rr))
    tol = np.finfo(float).eps * max(z.shape) * (diag.max() if len(diag) else 0)
    rank = int(np.sum(diag > tol))
    qq = qq[:, :rank]

    def resid(v):
        return v - qq @ (qq.T @ v)

    y = resid(q["filing_margin"].to_numpy(float))
    x = q["pci_l1"].to_numpy(float)
    xr = resid(x)
    obs = float(xr @ y / (xr @ xr))
    groups = [np.asarray(idx) for idx in q.groupby("institution", sort=False).indices.values()]
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perms):
        xp = x.copy()
        for idx in groups:
            if len(idx) > 1:
                shift = int(rng.integers(1, len(idx)))
                xp[idx] = np.roll(x[idx], shift)
        xpr = resid(xp)
        b = float(xpr @ y / (xpr @ xpr))
        extreme += abs(b) >= abs(obs) - 1e-12
    p = (extreme + 1) / (n_perms + 1)
    return {"beta": obs, "placebo_p": float(p), "draws": n_perms, "extreme": int(extreme)}


def robustness_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = twfe(panel, "filing_margin", "pci_l1")
    rows.append({"specification": "Baseline, lag 1", **base})

    wcols = ["filing_margin", "pci_l1", *CONTROLS]
    w = winsorized_copy(panel, wcols)
    rows.append({"specification": "Winsorized 1st/99th percentiles",
                 **twfe(w, "filing_margin", "pci_l1")})

    counts = panel.groupby("institution", observed=True)["year"].count()
    keep = counts[counts >= 15].index
    rows.append({"specification": "Institutions observed at least 15 years",
                 **twfe(panel[panel["institution"].isin(keep)], "filing_margin", "pci_l1")})

    rows.append({"specification": "R1 universities only",
                 **twfe(panel[panel["carnegie_r1"].eq(1)], "filing_margin", "pci_l1")})
    rows.append({"specification": "Lag 2 PCSI",
                 **twfe(panel, "filing_margin", "pci_l2", controls=controls_for_lag(2))})
    rows.append({"specification": "Falsification: future ln research expenditure",
                 **twfe(panel, "ln_research_exp_fwd", "pci_l1")})
    return pd.DataFrame(rows)


def revision_predictors(panel: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    universe = core.revision_universe(raw)
    keys = set(zip(universe["institution"].astype(str), universe["revision_year"].astype(int)))
    x = panel.copy()
    x["revision_event"] = [
        1.0 if (str(i), int(y)) in keys else 0.0
        for i, y in zip(x["institution"], x["year"])
    ]
    specs = [
        ("ln_new_patent_apps_l1", "Log new patent applications"),
        ("ln_disclosures_l1", "Log invention disclosures"),
        ("ln_licenses_l1", "Log licenses and options"),
        ("filing_margin_l1", "Filing margin"),
        ("ln_research_exp_l1", "Log research expenditure"),
    ]
    rows = []
    for pred, label in specs:
        r = twfe(x, "revision_event", pred, controls=[])
        rows.append({"predictor": pred, "label": label, **r})
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(core.RAW, low_memory=False)
    panel = build_panel(raw)
    lags = lag_table(panel)
    robust = robustness_table(panel)
    placebo = circular_shift_placebo(panel, n_perms=1000)
    predictors = revision_predictors(panel, raw)

    lags.to_csv(RESULTS / "supporting_annual_lags.csv", index=False)
    robust.to_csv(RESULTS / "supporting_annual_robustness.csv", index=False)
    predictors.to_csv(RESULTS / "revision_predictors.csv", index=False)
    summary = {
        "placebo": placebo,
        "lag1": lags[lags["lag"].eq(1)].to_dict(orient="records"),
        "robustness": robust.to_dict(orient="records"),
        "revision_predictors": predictors.to_dict(orient="records"),
    }
    (RESULTS / "supporting_annual_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUPPORTING ANNUAL ANALYSIS", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
