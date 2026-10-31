# GitHub Issue Triage Assistant

Working local prototype for RCEL 506. Enter an issue title and description to see a preliminary Bug or Enhancement suggestion. A maintainer must review the suggestion.

## Run the demo

From this folder on a computer with Python 3.9 or newer:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train.py
.venv/bin/streamlit run app.py
```

Open the local address printed by Streamlit, usually `http://localhost:8501`.

The app trains from the bundled CSV at startup. The evaluation report is written to `evaluation.json` by `train.py`. In the app, the Evaluation tab shows the same metrics and the Explore the data tab supports a live year filter.

The executed `issue_triage_full_pipeline_lr103.ipynb` reproduces the data checks, sample coverage chart, chronological evaluation, confusion matrix, and an example suggestion. In Colab, upload the bundled CSV when its data-loading cell asks for it.

## Verified result on the saved sample

The chronological split has 726 training issues and 182 test issues. The TF-IDF plus Logistic Regression model scored 0.8462 accuracy and 0.5886 macro F1, compared with 0.0275 accuracy and 0.0267 macro F1 for the training-majority baseline. The test set has 177 Bugs and only five Enhancements, so these numbers are not a reliable estimate of performance on future issues. All markdown heading lines and explicit target terms are removed from the model text to reduce issue-template leakage.

## What the demo establishes

- The text pipeline runs end to end: clean title and body, fit TF-IDF and Logistic Regression on the oldest 80%, evaluate on the newest 20%, and suggest a label for new text.
- The majority baseline is trained from the training split only.
- Input features use title and body only. Labels, dates, issue numbers, and other metadata are excluded from training.

## Limits

The CSV is a capped historical sample, not a repository census. The newest test split contains only five Enhancements. Saved labels may have changed after issue creation. The result does not establish reliable performance on incoming issues or other repositories. The model is a classroom prototype, not a deployed maintainer tool.

Before claiming operational value, confirm the target label mapping with the CTO, collect a date-based cohort without filtering by label, preserve label timing, and evaluate on a recent holdout containing enough examples of both classes.
