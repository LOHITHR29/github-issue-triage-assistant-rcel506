from pathlib import Path

from collect_issues import target
from triage import clean_text, fit_and_evaluate, load_issues, predict_issue


DATA = Path(__file__).parent / "data" / "scikit_learn_issues_date_cohort_labeled.csv"


def test_cleaning_and_prediction_use_same_text_contract():
    assert clean_text("Bug report: missing values", "### Description\nA bug occurs") == "missing values a occurs"
    assert clean_text("ENH: improve scaler", "### Describe the workflow you want to enable\nAllow clipping") == "improve scaler allow clipping"
    assert clean_text("Broken estimator", "### Describe the bug\nWrong output") == "broken estimator wrong output"
    assert clean_text("Add feature", "### Versions\nNumPy 2.0\n### Additional context\n_No response_") == "add"
    assert clean_text(None, None) == ""


def test_current_label_mapping_excludes_ambiguous_and_other_issues():
    assert target(["Bug"]) == "Bug"
    assert target(["Enhancement"]) == "Feature Request"
    assert target(["New Feature"]) == "Feature Request"
    assert target(["Bug", "New Feature"]) == "Ambiguous"
    assert target(["Documentation"]) is None


def test_data_split_and_prediction_contract():
    result = fit_and_evaluate(load_issues(DATA))
    assert len(result.issues) == 847
    assert len(result.train) == 677
    assert len(result.test) == 170
    assert result.train["created_at"].max() < result.test["created_at"].min()
    assert result.metrics["train_labels"] == {"Bug": 449, "Feature Request": 228}
    assert result.metrics["test_labels"] == {"Bug": 119, "Feature Request": 51}
    assert result.metrics["baseline"]["accuracy"] == 0.7
    prediction = predict_issue(result, "Pipeline fails on missing values", "The estimator raises an error")
    assert prediction["suggested_label"] in {"Bug", "Feature Request"}
    assert abs(sum(prediction["model_probabilities"].values()) - 1) < 0.002
