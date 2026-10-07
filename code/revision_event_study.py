"""Shared event-study utilities for the replication package.

The current manuscript's preferred design is the 34-event documentary sample.
This module contains the common panel construction, revision-universe, stacking,
fixed-effect absorption, and clustered-inference routines. It also retains the
frozen 25-event loader used by the historical benchmark replication.

The code uses Institution_std (149 harmonized AUTM institutions) for panel
fixed effects. Institution_pci is only the sparse link from AUTM to the policy
corpus and must not be used as the panel identifier.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "merged_autm.csv"
EVENTS = ROOT / "data" / "revision_codes_manual.csv"
RESULTS = ROOT / "results"
P1_COMMIT = "25a9472b34334825b6d6c6a334f5b88eb00695b5"
# Frozen copy of Paper 1's policy-level file at P1_COMMIT
# (Tech-transfer-1/P1_replication_package/data/derived/policy_level_indices_institution_year.csv).
P1_FILE = ROOT / "data" / "p1_policy_level_indices_institution_year.csv"
P1_SHA256 = "694c21acc07d2a50ed27199d0e7ec01bb6974f08f843cbce2d7da4318f864198"
# Paper 1 analyses 1944-2025. Its corpus file also carries a one-sentence 1925 Caltech
# record that Paper 1 excludes; dropping it here keeps both papers on the same sample.
P1_FIRST_YEAR = 1944
THRESHOLD = 0.03
EVENT_MIN = -4
EVENT_MAX = 5
OMITTED = -1
EVENT_TIMES = [k for k in range(EVENT_MIN, EVENT_MAX + 1) if k != OMITTED]
POST_TIMES = list(range(0, 6))
PRE_TIMES = [-4, -3, -2]


def norm(s: pd.Series) -> pd.Series:
    return (s.astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip().str.casefold())


def num(s):
    return pd.to_numeric(s, errors="coerce")


def make_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Construct the harmonized AUTM panel used by the event-study code."""
    x = pd.DataFrame(index=raw.index)
    x["institution"] = raw["Institution_std"].astype("string")
    x["year"] = num(raw["Year"])
    x["raw_id"] = raw["[ID]"].astype("string")
    x["private"] = num(raw["Private"]).fillna(0).astype(int)
    x["research_exp"] = num(raw["Tot Res Exp"])
    x["licensing_ftes"] = num(raw["Lic FTEs"])
    x["royalty_share"] = num(raw["Royalty Share"])
    x["disclosures"] = num(raw["Inv Dis Rec"])
    x["new_patent_apps"] = num(raw["New Pat App Fld"])
    x["total_patent_apps"] = num(raw["Tot Pat App Fld"])
    x["patents_issued"] = num(raw["Iss US Pat"])
    x["startups"] = num(raw["St-Ups Formed"])

    # FY2023 changed the total licensing field definition; use components there.
    lic_total = num(raw["Tot Lic/Opt Exe"])
    comp = num(raw["Lic Iss"]) + num(raw["Opt Iss"])
    x["licenses"] = lic_total
    x.loc[x["year"].eq(2023), "licenses"] = comp[x["year"].eq(2023)]

    for c in ["disclosures", "new_patent_apps", "total_patent_apps",
              "patents_issued", "licenses", "startups"]:
        x[f"ln_{c}"] = np.log1p(x[c])
    x["filing_margin"] = x["ln_new_patent_apps"] - x["ln_disclosures"]
    x["ln_research_exp"] = np.where(x["research_exp"] > 0, np.log(x["research_exp"]), np.nan)
    x["ln_licensing_ftes"] = np.log1p(x["licensing_ftes"])

    means = x.groupby("institution", observed=True)["research_exp"].mean()
    ranks = means.rank(method="first", pct=True)
    terc = pd.Series(np.minimum((ranks * 3).apply(np.ceil).astype(int) - 1, 2), index=means.index)
    x["size_tercile"] = x["institution"].map(terc).astype("Int64")

    if x["institution"].nunique() != 149:
        raise AssertionError(f"Expected 149 harmonized institutions, found {x['institution'].nunique()}")
    if x.duplicated(["institution", "year"]).any():
        raise AssertionError("Harmonized institution-year key is not unique")
    return x.sort_values(["institution", "year"]).reset_index(drop=True)


def load_p1_observed() -> pd.DataFrame:
    """Load the frozen Paper 1 observed-policy sequence (1944-2025, as in Paper 1)."""
    import hashlib
    digest = hashlib.sha256(P1_FILE.read_bytes()).hexdigest()
    if digest != P1_SHA256:
        raise RuntimeError(f"Checksum mismatch for {P1_FILE.name}: {digest}")
    p1 = pd.read_csv(P1_FILE, low_memory=False)
    p1["Year"] = num(p1["Year"])
    p1["Mean_Tone_Score"] = num(p1["Mean_Tone_Score"])
    p1["Is_Carried_Forward"] = num(p1["Is_Carried_Forward"])
    p1 = p1[p1["Year"] >= P1_FIRST_YEAR]
    obs = (p1[p1["Is_Carried_Forward"].eq(0)]
           .dropna(subset=["Institution", "Year", "Mean_Tone_Score"])
           .sort_values(["Institution", "Year"])
           .drop_duplicates(["Institution", "Year"], keep="last").copy())
    obs["prev_year"] = obs.groupby("Institution")["Year"].shift(1)
    obs["prev_pcsi"] = obs.groupby("Institution")["Mean_Tone_Score"].shift(1)
    obs["delta_pcsi"] = obs["Mean_Tone_Score"] - obs["prev_pcsi"]
    return obs


def revision_universe(raw: pd.DataFrame) -> pd.DataFrame:
    """Recover all 126 threshold revisions from the P1 observed-document sequence."""
    obs = load_p1_observed()
    cw = raw[["Institution_pci", "Institution_std"]].dropna().copy()
    cw["p1_norm"] = norm(cw["Institution_pci"])
    cw = (cw.groupby("p1_norm", observed=True)["Institution_std"]
          .agg(lambda z: z.value_counts().index[0]).rename("institution").reset_index())
    obs["p1_norm"] = norm(obs["Institution"])
    z = obs.merge(cw, on="p1_norm", how="inner")
    rev = z[z["delta_pcsi"].abs() > THRESHOLD].copy()
    rev = rev.rename(columns={"Year": "revision_year"})
    rev["direction"] = np.where(rev["delta_pcsi"] > 0, "up", "down")
    out = rev[["institution", "Institution", "prev_year", "revision_year",
               "delta_pcsi", "direction"]].sort_values(["institution", "revision_year"])
    if len(out) != 126 or out["institution"].nunique() != 78:
        raise AssertionError(
            f"Expected 126 threshold revisions at 78 linked institutions; got {len(out)} / {out['institution'].nunique()}"
        )
    return out.reset_index(drop=True)


def load_main_events() -> pd.DataFrame:
    """Load the frozen 25-event benchmark sample."""
    e = pd.read_csv(EVENTS)
    e = e[e["main_sample"].eq(1)].copy()
    e["event_id"] = np.arange(len(e))
    e["sign"] = np.where(e["direction"].eq("up"), 1.0, -1.0)
    if (len(e), int((e.sign > 0).sum()), int((e.sign < 0).sum())) != (25, 6, 19):
        raise AssertionError("Frozen benchmark must contain 25 events: 6 upward, 19 downward")
    return e


def make_stacks(panel: pd.DataFrame, universe: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    all_inst = set(panel["institution"].dropna().astype(str))
    rows = []
    for _, e in events.iterrows():
        E = int(e.revision_year)
        treated = str(e.institution)
        if treated not in all_inst:
            raise AssertionError(f"Treated institution not in harmonized AUTM panel: {treated}")
        lo, hi = E + EVENT_MIN, E + EVENT_MAX
        bad = set(universe.loc[universe.revision_year.between(lo, hi), "institution"].astype(str))
        controls = all_inst - bad
        keep = controls | {treated}
        s = panel[panel["institution"].astype(str).isin(keep) & panel["year"].between(lo, hi)].copy()
        s["stack"] = int(e.event_id)
        s["event_year"] = E
        s["event_time"] = s["year"] - E
        s["treated"] = (s["institution"].astype(str) == treated).astype(int)
        s["event_sign"] = float(e.sign)
        s["direction"] = e.direction
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def _demean(arr: np.ndarray, groups: list[np.ndarray], max_iter=500, tol=1e-11) -> np.ndarray:
    """Alternating projection for multiple high-dimensional fixed effects."""
    out = arr.astype(float, copy=True)
    if out.ndim == 1:
        out = out[:, None]
    for _ in range(max_iter):
        old = out.copy()
        for codes in groups:
            ng = int(codes.max()) + 1
            counts = np.bincount(codes, minlength=ng).astype(float)
            for j in range(out.shape[1]):
                sums = np.bincount(codes, weights=out[:, j], minlength=ng)
                means = sums / counts
                out[:, j] -= means[codes]
        if np.nanmax(np.abs(out - old)) < tol:
            break
    return out


def _cluster_cov(X, resid, clusters):
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    _, gcodes = np.unique(clusters, return_inverse=True)
    G = int(gcodes.max()) + 1
    meat = np.zeros((k, k))
    for g in range(G):
        idx = gcodes == g
        score = X[idx].T @ resid[idx]
        meat += np.outer(score, score)
    cov = xtx_inv @ meat @ xtx_inv
    if G > 1 and n > k:
        cov *= (G / (G - 1)) * ((n - 1) / (n - k))
    return cov


def fit_outcome(stacks: pd.DataFrame, outcome: str):
    df = stacks.copy()
    dcols, scols = [], []
    for k in EVENT_TIMES:
        tag = f"m{abs(k)}" if k < 0 else f"p{k}"
        d = f"D_{tag}"
        s = f"DS_{tag}"
        df[d] = ((df.treated == 1) & (df.event_time == k)).astype(float)
        df[s] = df[d] * df.event_sign
        dcols.append(d)
        scols.append(s)
    xcols = dcols + scols + ["ln_research_exp"]
    need = [outcome, "institution", "stack", "year", "size_tercile", "private"] + xcols
    q = df[need].dropna().copy()

    fe1 = pd.factorize(q["stack"].astype(str) + "|" + q["institution"].astype(str))[0]
    fe2 = pd.factorize(q["stack"].astype(str) + "|" + q["year"].astype(str) + "|" +
                       q["size_tercile"].astype(str) + "|" + q["private"].astype(str))[0]
    y = q[outcome].to_numpy(float)
    X = q[xcols].to_numpy(float)
    rz = _demean(np.column_stack([y, X]), [fe1, fe2])
    yr = rz[:, 0]
    Xr = rz[:, 1:]
    keep = np.sqrt((Xr ** 2).sum(axis=0)) > 1e-10
    if not keep.all():
        bad = [c for c, ok in zip(xcols, keep) if not ok]
        raise AssertionError(f"Absorbed regressors have no residual variation: {bad}")
    beta = np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ yr)
    resid = yr - Xr @ beta
    cov = _cluster_cov(Xr, resid, q["institution"].astype(str).to_numpy())
    b = pd.Series(beta, index=xcols)

    def combo(weights):
        w = np.zeros(len(xcols))
        for name, val in weights.items():
            w[xcols.index(name)] = val
        est = float(w @ beta)
        se = float(np.sqrt(max(w @ cov @ w, 0)))
        return est, se

    post_gap_w, post_up_w, post_dn_w = {}, {}, {}
    for k in POST_TIMES:
        tag = f"p{k}"
        post_gap_w[f"DS_{tag}"] = 2 / len(POST_TIMES)
        post_up_w[f"D_{tag}"] = 1 / len(POST_TIMES)
        post_up_w[f"DS_{tag}"] = 1 / len(POST_TIMES)
        post_dn_w[f"D_{tag}"] = 1 / len(POST_TIMES)
        post_dn_w[f"DS_{tag}"] = -1 / len(POST_TIMES)
    gap, gap_se = combo(post_gap_w)
    up, up_se = combo(post_up_w)
    down, down_se = combo(post_dn_w)

    path = []
    for k in EVENT_TIMES:
        tag = f"m{abs(k)}" if k < 0 else f"p{k}"
        est, se = combo({f"DS_{tag}": 2.0})
        path.append({"event_time": k, "gap": est, "se": se})

    idx = [xcols.index(f"DS_m{abs(k)}") for k in PRE_TIMES]
    bb = beta[idx]
    vv = cov[np.ix_(idx, idx)]
    stat = float(bb.T @ np.linalg.pinv(vv) @ bb)
    try:
        from scipy.stats import chi2
        pre_p = float(chi2.sf(stat, len(idx)))
    except Exception:
        pre_p = np.nan

    return {
        "outcome": outcome,
        "n": int(len(q)),
        "clusters": int(q["institution"].nunique()),
        "gap": gap, "gap_se": gap_se,
        "up": up, "up_se": up_se,
        "down": down, "down_se": down_se,
        "pre_p": pre_p,
        "path": path,
        "beta": b.to_dict(),
    }
