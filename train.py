"""Run the baseline and text model evaluation from a clean environment."""

import json
from pathlib import Path

from triage import fit_and_evaluate, load_issues


ROOT = Path(__file__).parent
DATA = ROOT / "data" / "scikit_learn_issues_date_cohort_labeled.csv"
REPORT = ROOT / "evaluation.json"


if __name__ == "__main__":
    result = fit_and_evaluate(load_issues(DATA))
    REPORT.write_text(json.dumps(result.metrics, indent=2) + "\n")
    print(json.dumps(result.metrics, indent=2))
    print(f"Saved {REPORT}")
