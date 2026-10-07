"""Exact replication layer for the September 25 stacked revision design.

Uses institution terciles based on median research expenditure and enumerates
all C(25,6)=177,100 assignments of six upward direction labels.
"""
from __future__ import annotations

import itertools
import json
import numpy as np
import pandas as pd
import revision_event_study as core

TARGETS = {
    "ln_licenses": {"gap": .619, "se": .136, "perm_p": .015, "pre_p": .62, "up": .289, "down": -.330, "n": 20181},
    "filing_margin": {"gap": .374, "se": .168, "perm_p": .155, "pre_p": .63, "up": .166, "down": -.207, "n": 20277},
    "ln_new_patent_apps": {"gap": .262, "se": .182, "perm_p": .243, "pre_p": .50, "up": .151, "down": -.111, "n": 20277},
    "ln_patents_issued": {"gap": .027, "se": .222, "perm_p": .920, "pre_p": .19, "up": .120, "down": .093, "n": 19821},
    "ln_disclosures": {"gap": -.112, "se": .105, "perm_p": .507, "pre_p": .37, "up": -.015, "down": .096, "n": 20277},
}


def median_size_panel(raw):
    p = core.make_panel(raw)
    med = p.groupby("institution", observed=True)["research_exp"].median()
    ranks = med.rank(method="average", pct=True)
    terc = np.minimum(np.ceil(ranks * 3).astype(int) - 1, 2)
    p["size_tercile"] = p["institution"].map(terc).astype("Int64")
    return p


def design_frame(stacks, outcome):
    q = stacks.copy()
    dcols = []
    for k in core.EVENT_TIMES:
        tag = f"m{abs(k)}" if k < 0 else f"p{k}"
        col = f"D_{tag}"
        q[col] = ((q.treated == 1) & (q.event_time == k)).astype(float)
        dcols.append(col)
    need = [outcome, "institution", "stack", "year", "size_tercile", "private",
            "ln_research_exp"] + dcols
    return q[need].dropna().copy(), dcols


def sufficient_stats(stacks, outcome):
    q, dcols = design_frame(stacks, outcome)
    stack_ids = sorted(q["stack"].unique())
    m = len(dcols)
    p0 = m + 1
    A = np.zeros((p0, p0)); arhs = np.zeros(p0); C = np.zeros((m, m))
    Bparts = np.zeros((len(stack_ids), p0, m)); cparts = np.zeros((len(stack_ids), m))
    for pos, sid in enumerate(stack_ids):
        z = q[q["stack"].eq(sid)]
        fe1 = pd.factorize(z["institution"].astype(str))[0]
        fe2 = pd.factorize(z["year"].astype(str) + "|" + z["size_tercile"].astype(str) + "|" + z["private"].astype(str))[0]
        y = z[outcome].to_numpy(float)
        D = z[dcols].to_numpy(float)
        X0 = np.column_stack([D, z["ln_research_exp"].to_numpy(float)])
        rr = core._demean(np.column_stack([y, X0, D]), [fe1, fe2])
        yr = rr[:, 0]; x0 = rr[:, 1:1+p0]; zr = rr[:, 1+p0:]
        A += x0.T @ x0; arhs += x0.T @ yr; C += zr.T @ zr
        Bparts[pos] = x0.T @ zr; cparts[pos] = zr.T @ yr
    return q, dcols, stack_ids, A, arhs, C, Bparts, cparts


def solve_signs(signs, A, arhs, C, Bparts, cparts):
    B = np.einsum("e,eij->ij", signs, Bparts, optimize=True)
    c = signs @ cparts
    mat = np.block([[A, B], [B.T, C]])
    rhs = np.concatenate([arhs, c])
    return np.linalg.pinv(mat) @ rhs


def exact_permutation(stacks, events, outcome, batch_size=4000):
    q, dcols, stack_ids, A, arhs, C, Bparts, cparts = sufficient_stats(stacks, outcome)
    if len(stack_ids) != 25:
        raise AssertionError(f"Expected 25 stacks, got {len(stack_ids)}")
    sign_map = events.set_index("event_id")["sign"].to_dict()
    observed_signs = np.array([sign_map[int(s)] for s in stack_ids], dtype=float)
    obs_beta = solve_signs(observed_signs, A, arhs, C, Bparts, cparts)
    m = len(dcols); p0 = A.shape[0]
    post = [dcols.index(f"D_p{k}") for k in core.POST_TIMES]
    obs_gap = float((2/len(post)) * obs_beta[p0 + np.array(post)].sum())

    point = core.fit_outcome(stacks, outcome)
    if abs(obs_gap - point["gap"]) > 1e-7:
        raise AssertionError(f"Normal-equation gap {obs_gap} != full-fit gap {point['gap']}")

    total = 0; extreme = 0; buf = []
    def run_batch(combos):
        nonlocal total, extreme
        b = len(combos)
        S = -np.ones((b, 25), dtype=float)
        for i, comb in enumerate(combos):
            S[i, list(comb)] = 1.0
        Bb = np.einsum("be,eij->bij", S, Bparts, optimize=True)
        cb = S @ cparts
        dim = p0 + m
        mats = np.empty((b, dim, dim), float)
        mats[:, :p0, :p0] = A
        mats[:, :p0, p0:] = Bb
        mats[:, p0:, :p0] = np.swapaxes(Bb, 1, 2)
        mats[:, p0:, p0:] = C
        rhs = np.empty((b, dim), float)
        rhs[:, :p0] = arhs; rhs[:, p0:] = cb
        try:
            betas = np.linalg.solve(mats, rhs[..., None]).squeeze(-1)
        except np.linalg.LinAlgError:
            betas = np.einsum("bij,bj->bi", np.linalg.pinv(mats), rhs)
        gaps = (2/len(post)) * betas[:, p0 + np.array(post)].sum(axis=1)
        total += b
        extreme += int((np.abs(gaps) >= abs(obs_gap) - 1e-12).sum())

    for comb in itertools.combinations(range(25), 6):
        buf.append(comb)
        if len(buf) == batch_size:
            run_batch(buf); buf = []
    if buf: run_batch(buf)
    if total != 177100: raise AssertionError(total)
    return {"observed_gap": obs_gap, "permutation_p": extreme/total,
            "extreme_assignments": extreme, "total_assignments": total, "n": len(q)}


def bh_adjust(pvals):
    items = sorted(pvals.items(), key=lambda x: x[1]); m = len(items)
    vals = [p*m/(i+1) for i, (_, p) in enumerate(items)]
    for i in range(m-2, -1, -1): vals[i] = min(vals[i], vals[i+1])
    return {name: min(1.0, v) for (name, _), v in zip(items, vals)}


def main():
    raw = pd.read_csv(core.RAW, low_memory=False)
    panel = median_size_panel(raw)
    universe = core.revision_universe(raw)
    events = core.load_main_events()
    stacks = core.make_stacks(panel, universe, events)
    print("CANONICAL SEPT25 REPLICATION", len(panel), panel.institution.nunique(), len(universe), len(events), len(stacks))
    rows = []; pvals = {}
    for outcome, target in TARGETS.items():
        point = core.fit_outcome(stacks, outcome)
        perm = exact_permutation(stacks, events, outcome)
        pvals[outcome] = perm["permutation_p"]
        row = {"outcome": outcome, "gap": point["gap"], "cluster_se": point["gap_se"],
               "perm_p": perm["permutation_p"], "pre_p": point["pre_p"],
               "upward": point["up"], "downward": point["down"], "n": point["n"],
               "extreme_assignments": perm["extreme_assignments"]}
        rows.append(row)
        print("CANONICAL RESULT", outcome, json.dumps(row, sort_keys=True))
        print("PDF TARGET", outcome, json.dumps(target, sort_keys=True))
    qs = bh_adjust(pvals)
    for r in rows: r["bh_q"] = qs[r["outcome"]]
    out = pd.DataFrame(rows)
    core.RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(core.RESULTS / "table3_reproduced.csv", index=False)
    (core.RESULTS / "table3_reproduced.json").write_text(json.dumps(rows, indent=2)+"\n")
    print("BH Q", qs)
    assert len(panel)==3507 and panel.institution.nunique()==149
    assert len(universe)==126 and universe.institution.nunique()==78
    assert len(events)==25 and int((events.sign>0).sum())==6
    for r in rows: assert r["n"] == TARGETS[r["outcome"]]["n"]

if __name__ == "__main__":
    main()
