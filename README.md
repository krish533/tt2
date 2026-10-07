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

The committed files in `data/` contain the policy level PCSI input and documentary revision coding used to construct the event sample.

The AUTM analysis file is not duplicated in the repository. `code/fetch_inputs.py` retrieves the frozen analysis input from the tested source snapshot and verifies its SHA-256 checksum before use. The same script retrieves the sentence level input used for the descriptive revision content comparison.

Users are responsible for complying with applicable AUTM data use terms.

## Reproduction

The single entry point is:

```bash
python code/reproduce.py
```

All reproduced numerical outputs are written to `results/`.
