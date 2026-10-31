"""Reproducible, repository-specific Bug versus Feature Request triage."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline


LABELS = ("Bug", "Feature Request")
TARGET_WORDS = re.compile(
    r"\b(?:bug\s+reports?|new\s+features?|feature\s+requests?|bugs?|features?|enhancements?|enh)\b",
    flags=re.IGNORECASE,
)
TEMPLATE_HEADERS = re.compile(r"(?m)^\s*#{1,6}\s+[^\n]*$")
VERSIONS_SECTION = re.compile(
    r"(?ims)^\s*#{1,6}\s+versions?\s*$.*?(?=^\s*#{1,6}\s|\Z)"
)
NO_RESPONSE = re.compile(r"(?im)^\s*_?no response_?\s*$")
SPACE = re.compile(r"\s+")


def clean_text(title: object, body: object) -> str:
    """Use only title and body, with explicit target words and common headers removed."""
    parts = [part if isinstance(part, str) else "" for part in (title, body)]
    text = "\n".join(parts)
    text = VERSIONS_SECTION.sub(" ", text)
    text = NO_RESPONSE.sub(" ", text)
    text = TEMPLATE_HEADERS.sub(" ", text)
    text = TARGET_WORDS.sub(" ", text)
    text = re.sub(r"(?m)^\s*[:;]\s*", " ", text)
    return SPACE.sub(" ", text).strip().lower()


def load_issues(path: Path) -> pd.DataFrame:
    issues = pd.read_csv(path)
    required = {"issue_number", "title", "body", "created_at", "issue_type"}
    missing = required - set(issues.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if issues["issue_number"].isna().any() or issues["issue_number"].duplicated().any():
        raise ValueError("Issue numbers must be present and unique")
    if issues["title"].isna().any():
        raise ValueError("Issue titles must be present")
    if set(issues["issue_type"]) != set(LABELS):
        raise ValueError("Expected exactly Bug and Feature Request labels")
    issues["created_at"] = pd.to_datetime(issues["created_at"], utc=True, errors="raise")
    issues["text"] = [clean_text(t, b) for t, b in zip(issues["title"], issues["body"])]
    if issues["text"].eq("").any():
        raise ValueError("One or more issues have no usable title or body text")
    return issues.sort_values(["created_at", "issue_number"]).reset_index(drop=True)


@dataclass
class Result:
    issues: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    model: Pipeline
    predictions: list[str]
    baseline_predictions: list[str]
    metrics: dict


def fit_and_evaluate(issues: pd.DataFrame) -> Result:
    """Fit text and majority models on the oldest 80% only."""
    if len(issues) < 20:
        raise ValueError("At least 20 issues are required")
    split = int(len(issues) * 0.8)
    train = issues.iloc[:split].copy()
    test = issues.iloc[split:].copy()
    if train["issue_type"].nunique() != 2 or test["issue_type"].nunique() != 2:
        raise ValueError("Both classes must be present in each split")
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                                      max_features=30000, sublinear_tf=True)),
            ("logistic_regression", LogisticRegression(max_iter=1000, class_weight="balanced",
                                                       solver="liblinear")),
        ]
    )
    model.fit(train["text"], train["issue_type"])
    predictions = model.predict(test["text"]).tolist()
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit([[0]] * len(train), train["issue_type"])
    baseline_predictions = baseline.predict([[0]] * len(test)).tolist()
    metrics = {
        "train_count": len(train),
        "test_count": len(test),
        "train_labels": train["issue_type"].value_counts().to_dict(),
        "test_labels": test["issue_type"].value_counts().to_dict(),
        "train_end": train["created_at"].max().isoformat(),
        "test_start": test["created_at"].min().isoformat(),
        "test_end": test["created_at"].max().isoformat(),
        "model": _scores(test["issue_type"], predictions),
        "baseline": _scores(test["issue_type"], baseline_predictions),
    }
    return Result(issues, train, test, model, predictions, baseline_predictions, metrics)


def _scores(actual: pd.Series, predicted: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=LABELS, zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "macro_f1": round(float(f1_score(actual, predicted, labels=LABELS,
                                         average="macro", zero_division=0)), 4),
        "per_class": {
            label: {"precision": round(float(precision[i]), 4),
                    "recall": round(float(recall[i]), 4),
                    "f1": round(float(f1[i]), 4),
                    "support": int(support[i])}
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(),
    }


def predict_issue(result: Result, title: str, body: str) -> dict:
    text = clean_text(title, body)
    if len(text) < 10:
        raise ValueError("Enter a descriptive title or body with at least 10 usable characters")
    probabilities = result.model.predict_proba([text])[0]
    classes = result.model.classes_.tolist()
    return {
        "suggested_label": classes[int(probabilities.argmax())],
        "model_probabilities": {label: round(float(probabilities[classes.index(label)]), 3)
                                for label in LABELS},
    }
