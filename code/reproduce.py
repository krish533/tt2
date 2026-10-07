"""Single entry point for the clean journal replication package."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    "fetch_inputs.py",
    "mechanical_screen.py",
    "descriptive_tables.py",
    "revision_content_keywords.py",
    "final_expanded_inference.py",
    "expanded34_diagnostics.py",
    "threshold_sensitivity_final34.py",
    "supporting_annual_analysis.py",
    "export_final_stacked_datasets.py",
    "revision_replication.py",
]

EXPECTED = [
    ROOT / "results" / "mechanical_screen.csv",
    ROOT / "results" / "expanded_manual_sample_results.csv",
    ROOT / "results" / "expanded34_balance.csv",
    ROOT / "results" / "expanded34_leave_one_out.csv",
    ROOT / "results" / "threshold_sensitivity_final34.csv",
    ROOT / "results" / "supporting_annual_lags.csv",
    ROOT / "data" / "derived" / "final_stacked_event_dataset_expanded34.csv",
    ROOT / "data" / "derived" / "final_stacked_event_dataset_strict29.csv",
]

def main() -> None:
    for script in STEPS:
        print(f"\n=== Running {script} ===", flush=True)
        subprocess.run([sys.executable, str(ROOT / "code" / script)], cwd=ROOT, check=True)

    missing = [str(p.relative_to(ROOT)) for p in EXPECTED if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Replication completed but expected outputs are missing: " + ", ".join(missing)
        )

    print("\nReplication completed successfully.")
    print("Preferred sample: 34 events (8 upward / 26 downward).")
    print("Strict post-support sample: 29 events (7 upward / 22 downward).")
    print("See results/ for reproduced estimates and diagnostics.")

if __name__ == "__main__":
    main()
