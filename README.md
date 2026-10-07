# Replication package: Policy Communication and University Licensing

This repository contains the clean replication files for the study of university intellectual property policy revisions, policy communication, and technology transfer.

The repository is intentionally minimal. It contains only:

- `README.md`
- `requirements.txt`
- `code/`
- `data/`

No manuscript files, drafting history, or development materials are included.

## Quick start

Use Python 3.11 from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python code/reproduce.py
```

The runner creates a `results/` directory and generated files under `data/derived/`.

## Main design

The preferred documentary sample contains 34 policy revisions at 32 universities: 8 upward revisions and 26 downward revisions. The main analysis is a stacked event study comparing revising universities with contemporaneous universities that do not experience a threshold revision in the same event window.

The code reproduces the preferred 34 event results, the stricter 29 event sensitivity, balance and influence diagnostics, supporting annual PCSI analyses, and sensitivity to stricter PCSI change thresholds.

The analysis is observational. Universities choose when and how to revise their policies, so the estimates should not be interpreted as the causal effect of wording itself.

## Data

The committed files in `data/` contain the documentary revision coding used to construct the event sample. The larger policy level PCSI file, the AUTM analysis file, and the sentence level policy input are fetched from frozen source snapshots by `code/fetch_inputs.py` and verified by SHA-256 checksum before use.

Users are responsible for complying with applicable AUTM data use terms.

## Reproduction

The single entry point is:

```bash
python code/reproduce.py
```

All reproduced numerical outputs are written to `results/`.


## Tables, figures, and appendix coverage

Running `python code/reproduce.py` creates the numerical inputs for the paper's main tables and appendix tables, together with the main event study figure and the appendix leave one out figure.

Main text coverage:

- Table 1: linked AUTM and PCSI descriptives
- Table 2: revision sample construction
- Table 3: preferred 34 event outcome estimates
- Table 4: preferred, strict, and frozen benchmark comparisons
- Table 5: revision content keyword comparison
- Table 6: supporting annual PCSI models
- Figure 1: licensing and filing margin event study

Appendix coverage:

- preferred revision list and threshold universe counts
- stack composition and event time coefficients
- strict 29 event estimates
- pre revision balance
- leave one event out estimates and figure
- annual robustness, placebo, and revision predictor analyses
- FY2023 licensing field audit
- startup reporting audit
- stricter 0.03, 0.04, and 0.05 PCSI threshold sensitivity

The original provision level coding table is different from the computational outputs above. Its reported aggregate percentages and Fisher test p values are preserved in `data/provision_level_original25_reported.csv`, but the row level hand coding used to construct that descriptive table is not available in this package. It should therefore not be described as independently reproducible from raw row level provision coding.
