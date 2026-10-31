"""Build the self-contained, executable project notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).parent
DATA = ROOT / "data" / "scikit_learn_issues_date_cohort_labeled.csv"
SOURCE = ROOT / "triage.py"
OUT = ROOT / "issue_triage_full_pipeline_lr103.ipynb"
SHA256 = hashlib.sha256(DATA.read_bytes()).hexdigest()
BASE = "https://raw.githubusercontent.com/LOHITHR29/github-issue-triage-assistant-rcel506/main/data"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md("""
# GitHub Issue Triage Assistant

**Lohith Regalla (lr103) | RCEL 506 | Final project notebook**

## Question and value

Can the title and description of a newly opened scikit-learn issue help a maintainer decide whether it is a **Bug** or a **Feature Request**? The app offers a preliminary suggestion for human review. Its possible value is less first-pass sorting work. This notebook does not measure staff time saved.

The target is based on the repository's labels at collection time. **Bug** maps to Bug. **Feature Request** maps to Enhancement or New Feature, following the repository's [label guide](https://github.com/scikit-learn/scikit-learn/wiki/labels). Issues with neither category or with both categories are outside this binary experiment.

The collection covers issues created from January 2024 through October 2026. It queries every month and checks each page count. Email addresses in issue text were redacted before saving. The saved data makes this notebook reproducible without a GitHub token. This is a repository-specific evaluation, not a promise of reliable automatic labeling.
"""),
    code(f"""
from pathlib import Path
from urllib.request import urlopen
import hashlib
import json
import matplotlib.pyplot as plt
import pandas as pd

DATA_NAME = "scikit_learn_issues_date_cohort_labeled.csv"
DATA_PATH = Path("data") / DATA_NAME
if not DATA_PATH.exists():
    DATA_PATH = Path(DATA_NAME)
    DATA_PATH.write_bytes(urlopen("{BASE}/" + DATA_NAME).read())

EXPECTED_SHA256 = "{SHA256}"
actual_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
assert actual_hash == EXPECTED_SHA256, "The CSV does not match the audited project data"

AUDIT_PATH = Path("data/collection_audit.json")
if AUDIT_PATH.exists():
    audit = json.loads(AUDIT_PATH.read_text())
else:
    audit = json.loads(urlopen("{BASE}/collection_audit.json").read())
print("Data file:", DATA_PATH)
print("SHA-256:", actual_hash)
print("Collected at:", audit["retrieved_at"])
"""),
    md("""
## 1. Read and verify the data

The extraction used GitHub issue search with `is:issue`, so pull requests are excluded by the query. It fetched every page for each creation month and rejected any incomplete month. Labels are a snapshot at retrieval, so a later label change can affect the historical target. The raw cohort and month-level audit are bundled in the repository.
"""),
    code(SOURCE.read_text()),
    code("""
issues = load_issues(DATA_PATH)
print("All issues collected:", audit["raw_count"])
print("Excluded, neither target:", audit["excluded"]["neither_target"])
print("Excluded, both categories:", audit["excluded"]["ambiguous"])
print("Usable binary issues:", len(issues))
print("Unique issue numbers:", issues.issue_number.nunique())
print("Date range:", issues.created_at.min(), "to", issues.created_at.max())
print("Class counts:\\n", issues.issue_type.value_counts())
print("Missing values:\\n", issues[["title", "body", "created_at", "issue_type"]].isna().sum())
assert len(issues) == audit["labeled_count"]
assert len(issues) + sum(audit["excluded"].values()) == audit["raw_count"]
display(issues[["issue_number", "created_at", "title", "issue_type"]].head())
"""),
    code("""
coverage = pd.Series({
    "Bug": audit["class_counts"]["Bug"],
    "Feature Request": audit["class_counts"]["Feature Request"],
    "Neither category": audit["excluded"]["neither_target"],
    "Both categories": audit["excluded"]["ambiguous"],
})
fig, ax = plt.subplots(figsize=(9, 4.5))
colors = ["#2d6a8a", "#e5a23b", "#a9afb4", "#b76565"]
bars = ax.barh(coverage.index[::-1], coverage.values[::-1], color=colors[::-1])
ax.bar_label(bars, padding=5)
ax.set_xlim(0, coverage.max() * 1.15)
ax.set_xlabel("Issues created Jan 2024 to Oct 2026")
ax.set_title("Which collected issues fit the binary target?")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()
"""),
    md("""
More than half of the collected issues have neither target label. The app is therefore a **binary suggestion tool for the defined subset**, and a maintainer must route other issue types.
"""),
    code("""
by_quarter = (issues.assign(quarter=issues.created_at.dt.tz_localize(None).dt.to_period("Q").astype(str))
             .groupby(["quarter", "issue_type"]).size().unstack(fill_value=0)
             .reindex(columns=LABELS, fill_value=0))
ax = by_quarter.plot(kind="bar", stacked=True, figsize=(11, 4.5),
                     color=["#2d6a8a", "#e5a23b"], width=0.78)
ax.set_title("Target-labeled issues by creation quarter")
ax.set_xlabel("")
ax.set_ylabel("Issues")
ax.legend(title="Label", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
display(by_quarter)
"""),
    md("""
## 2. Train and test in time order

The oldest 80% of usable issues are training data and the newest 20% are held out for testing. TF-IDF learns its vocabulary only from training text inside the pipeline. The model uses title and body after removing explicit target words, issue-template headings, the generic “No response” placeholder, and the automatic Versions section. Dates, labels, IDs, and other metadata are never model inputs.

The majority-class classifier uses training labels only. Macro F1 gives equal weight to each class.
"""),
    code("""
result = fit_and_evaluate(issues)
print("Training:", result.metrics["train_count"], result.metrics["train_labels"])
print("Test:", result.metrics["test_count"], result.metrics["test_labels"])
print("Train ends:", result.metrics["train_end"])
print("Test starts:", result.metrics["test_start"])
scores = pd.DataFrame({
    "Majority baseline": result.metrics["baseline"],
    "TF-IDF + Logistic Regression": result.metrics["model"],
}).loc[["accuracy", "macro_f1"]]
display(scores)
display(pd.DataFrame(result.metrics["model"]["per_class"]).T)
"""),
    code("""
matrix = pd.DataFrame(result.metrics["model"]["confusion_matrix"], index=LABELS, columns=LABELS)
fig, ax = plt.subplots(figsize=(6, 5))
image = ax.imshow(matrix.values, cmap="Blues")
for row in range(2):
    for col in range(2):
        ax.text(col, row, str(matrix.iloc[row, col]), ha="center", va="center",
                color="white" if matrix.iloc[row, col] > matrix.values.max() / 2 else "#1d2a35",
                fontsize=16, fontweight="bold")
ax.set_xticks(range(2), LABELS)
ax.set_yticks(range(2), LABELS)
ax.set_xlabel("Suggested")
ax.set_ylabel("Maintainer label snapshot")
ax.set_title("Chronological holdout: 170 issues")
plt.tight_layout()
plt.show()
"""),
    code("""
errors = result.test.loc[[actual != predicted for actual, predicted in
                          zip(result.test.issue_type, result.predictions)],
                         ["issue_number", "created_at", "title", "issue_type", "source_url"]].copy()
errors["suggested"] = [prediction for actual, prediction in
                       zip(result.test.issue_type, result.predictions) if actual != prediction]
print("Errors:", len(errors))
display(errors.head(10))

# Sensitivity check: use only titles, with the same split and model settings.
title_data = issues.copy()
title_data["text"] = [clean_text(title, "") for title in title_data.title]
title_result = fit_and_evaluate(title_data)
print("Title-only macro F1:", title_result.metrics["model"]["macro_f1"])
print("Title and body macro F1:", result.metrics["model"]["macro_f1"])
"""),
    md("""
## 3. Live audit and example

Change `year` below during a live review to inspect any creation year in the saved data. Rerun the cell and the count and rows will update.
"""),
    code("""
year = 2026
year_view = issues.loc[issues.created_at.dt.year == year]
print(f"{year}: {len(year_view)} target-labeled issues")
print(year_view.issue_type.value_counts())
display(year_view[["issue_number", "created_at", "title", "issue_type"]].head(10))
"""),
    code("""
example = predict_issue(
    result,
    "Allow custom threshold in confusion matrix display",
    "I would like an option to specify a threshold when displaying predictions.",
)
print("Example suggestion for maintainer review:", example)
"""),
    md("""
## Interpretation and limits

The text model improves on the majority-class baseline on this saved, chronological holdout. This supports a **proof of concept** for human-reviewed first-pass triage within scikit-learn. It does not prove that maintainer time is saved. It also does not cover unlabeled, documentation, CI, or other issue types.

Labels are the state observed at collection time, not label events reconstructed at creation. The repository's use of labels and issue templates can change. A future operational study should collect another recent cohort, verify label timing, measure reviewer effort, and test a manual-review or abstain route before any automatic labeling.

Live app: https://scikit-learn-issue-triage-lr103.streamlit.app/
Source and data: https://github.com/LOHITHR29/github-issue-triage-assistant-rcel506
"""),
]

notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {
    "display_name": "Python 3", "language": "python", "name": "python3"}})
nbf.validate(notebook)
nbf.write(notebook, OUT)
print(OUT)
