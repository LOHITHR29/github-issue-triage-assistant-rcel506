# GitHub Issue Triage Assistant

**Lohith Regalla (lr103), RCEL 506 final project**

Live app: https://scikit-learn-issue-triage-lr103.streamlit.app/

This app suggests whether a scikit-learn issue is a **Bug** or **Feature Request** from its title and description. A maintainer reviews every suggestion. The project tests whether a simple text model can help with first-pass sorting. It does not measure staff time saved or automatically label GitHub issues.

## Scope and data

The target uses the repository's [label guide](https://github.com/scikit-learn/scikit-learn/wiki/labels): Bug maps to Bug, while Enhancement and New Feature map to Feature Request. Other categories and issues with both target categories are outside the binary model.

`collect_issues.py` queried all public scikit-learn issues created in each month from January 2024 through October 2026, with `is:issue` to exclude pull requests. It checked the reported result count for every month, fetched all pages, and rejected incomplete results. Email addresses in issue text were redacted before saving. The collected labels are a snapshot from 31 October 2026. The source data and counts are in `data/`.

| Collection step | Issues |
| --- | ---: |
| All issues collected by creation date | 1,757 |
| Neither target category | 907 |
| Both target categories | 3 |
| Usable binary issues | 847 |
| Bug | 568 |
| Feature Request | 279 |

The original 908-row, label-capped sample is retained in `data/scikit_learn_issues_bug_enhancement_sample.csv` only to document the earlier project stage. The final model uses `data/scikit_learn_issues_date_cohort_labeled.csv`.

## Model and result

The oldest 80% of usable issues (677) train the model. The newest 20% (170, including 51 Feature Requests) are held out. The model uses TF-IDF on cleaned title and body with Logistic Regression. Preprocessing removes explicit target words, issue-template headings, the generic “No response” placeholder, and automatic Versions sections. Labels, issue numbers, dates, and other metadata are excluded from the features. TF-IDF is fitted on training text only.

| Chronological test | Majority baseline | Text model |
| --- | ---: | ---: |
| Accuracy | 0.7000 | 0.8824 |
| Macro F1 | 0.4118 | 0.8702 |

The text model correctly classified 101 of 119 Bugs and 49 of 51 Feature Requests. A title-only sensitivity check scored 0.8177 macro F1. The exact evaluation and confusion matrix are in `evaluation.json` and the executed notebook.

## Run and inspect

The live app has three tabs: **Try an issue**, **Evaluation**, and **Explore the data**. Enter a title and description to get a suggestion for maintainer review. The score bars are model outputs, not calibrated reliability estimates.

To run locally from this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train.py
.venv/bin/streamlit run app.py
```

On a Mac, `zsh run_demo.command` also starts the local app. The executed `issue_triage_full_pipeline_lr103.ipynb` contains the data checks, coverage and time plots, evaluation, error inspection, and a year filter that can be changed live. In Google Colab, its first code cell downloads the published CSV and audit file if they are not already beside the notebook, and checks the CSV SHA-256 hash.

To refresh the public data with GitHub CLI access, run `python3 collect_issues.py --start 2024-01 --end 2026-10 --refresh`. The saved monthly files are reused if `--refresh` is omitted. After changing the data, rerun `train.py` and rebuild and execute the notebook. Install `requirements-notebook.txt` to run `make_notebook.py` and `execute_notebook.py` locally.

## Limits

Only 847 of the 1,757 collected issues fit this binary target. Current labels may differ from labels when an issue was opened. The evaluation is for this repository and this historical time window. It does not establish performance on every incoming issue, on other repositories, or the amount of maintainer time saved. The app is a public classroom demonstration and does not write labels back to GitHub. Before operational use, collect a new prospective cohort, check label timing, measure reviewer effort, and define a manual route for other issue types.
