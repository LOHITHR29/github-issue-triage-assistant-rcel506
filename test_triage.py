from pathlib import Path

from triage import clean_text, fit_and_evaluate, load_issues, predict_issue


DATA = Path(__file__).parent / "data" / "scikit_learn_issues_bug_enhancement_sample.csv"


def test_cleaning_and_prediction_use_same_text_contract():
    assert clean_text("Bug report: missing values", "### Description\nA bug occurs") == "missing values a occurs"
    assert clean_text("ENH: improve scaler", "### Describe the workflow you want to enable\nAllow clipping") == "improve scaler allow clipping"
    assert clean_text("Broken estimator", "### Describe the bug\nWrong output") == "broken estimator wrong output"
    assert clean_text(None, None) == ""


def test_data_split_and_prediction_contract():
    result = fit_and_evaluate(load_issues(DATA))
    assert len(result.issues) == 908
    assert len(result.train) == 726
    assert len(result.test) == 182
    assert result.train["created_at"].max() < result.test["created_at"].min()
    assert result.metrics["train_labels"] == {"Enhancement": 414, "Bug": 312}
    assert result.metrics["test_labels"] == {"Bug": 177, "Enhancement": 5}
    assert result.metrics["baseline"]["accuracy"] == 0.0275
    prediction = predict_issue(result, "Pipeline fails on missing values", "The estimator raises an error")
    assert prediction["suggested_label"] in {"Bug", "Enhancement"}
    assert abs(sum(prediction["model_probabilities"].values()) - 1) < 0.002
