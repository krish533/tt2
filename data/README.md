# Data

This directory contains the committed inputs used to construct the policy revision sample.

## Committed files

- `p1_policy_level_indices_institution_year.csv` — policy level PCSI input used to reconstruct the revision universe.
- `revision_codes_manual.csv` — documentary coding for the original benchmark revision set.
- `revision_codes_rereview37.csv` — second pass documentary review used to construct the expanded preferred sample.

## Fetched inputs

The AUTM analysis input is not duplicated here. Run:

```bash
python code/fetch_inputs.py
```

or run the full replication:

```bash
python code/reproduce.py
```

The fetch script downloads the frozen analysis input used by the study and verifies its SHA-256 checksum before use. It also retrieves the sentence level policy file used for the descriptive revision content comparison.

Users are responsible for complying with applicable AUTM data use terms.

## Generated files

The replication scripts create `data/derived/` as needed, including the explicit stacked datasets for the preferred 34 event design and the strict 29 event sensitivity.
