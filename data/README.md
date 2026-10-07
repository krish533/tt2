# Data

This directory contains the documentary coding files used to construct the policy revision sample.

## Committed files

- `revision_codes_manual.csv` — documentary coding for the original benchmark revision set.
- `revision_codes_rereview37.csv` — second pass documentary review used to construct the expanded preferred sample.

## Fetched frozen inputs

Run:

```bash
python code/fetch_inputs.py
```

or run the complete replication:

```bash
python code/reproduce.py
```

The fetch script downloads and checksum verifies three frozen inputs:

- the AUTM analysis file used for the technology transfer outcomes;
- the policy level PCSI file used to reconstruct the revision universe;
- the sentence level policy file used for the descriptive revision content comparison.

The policy level PCSI input is pinned to the Paper 1 source commit and is additionally checked inside the event study code before use.

Users are responsible for complying with applicable AUTM data use terms.

## Generated files

The replication scripts create `data/derived/` as needed, including the explicit stacked datasets for the preferred 34 event design and the strict 29 event sensitivity.
