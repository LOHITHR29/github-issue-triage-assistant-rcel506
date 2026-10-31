"""Presentation demo for the scikit-learn GitHub issue triage prototype."""

from pathlib import Path

import pandas as pd
import streamlit as st

from triage import LABELS, fit_and_evaluate, load_issues, predict_issue


DATA = Path(__file__).parent / "data" / "scikit_learn_issues_bug_enhancement_sample.csv"


@st.cache_resource(show_spinner="Training the text model on the historical sample...")
def get_result():
    return fit_and_evaluate(load_issues(DATA))


st.set_page_config(page_title="GitHub Issue Triage Assistant", page_icon="🦉", layout="wide")
st.title("GitHub Issue Triage Assistant")
st.caption("scikit-learn issue classification prototype | Lohith Regalla | RCEL 506")

try:
    result = get_result()
except Exception as exc:
    st.error(f"The demo could not load its data: {exc}")
    st.stop()

st.info(
    "Research prototype only. The historical data was collected with fixed label caps, and the "
    "newest test slice has just five Enhancements. Every suggestion requires maintainer review."
)

tab_demo, tab_evidence, tab_data = st.tabs(["Try an issue", "Evaluation", "Explore the data"])

with tab_demo:
    st.subheader("Suggest an initial label")
    st.write("Paste a new issue title and description. The model uses only that text.")
    with st.form("issue_form"):
        title = st.text_input("Issue title", placeholder="Example: Pipeline fails when input has missing values")
        body = st.text_area("Issue description", height=160,
                            placeholder="Describe the observed behavior, expected behavior, and context.")
        submitted = st.form_submit_button("Suggest label")
    if submitted:
        try:
            suggestion = predict_issue(result, title, body)
            st.metric("Suggested label", suggestion["suggested_label"])
            st.write("Model probabilities for this text:")
            for label, probability in suggestion["model_probabilities"].items():
                st.progress(round(probability * 100), text=f"{label}: {probability:.1%}")
            st.caption("These probabilities are not calibrated reliability estimates. A maintainer makes the final choice.")
        except ValueError as exc:
            st.warning(str(exc))

with tab_evidence:
    st.subheader("Chronological test")
    st.write("Oldest 80% for training, newest 20% for testing. TF-IDF is fit on training text only.")
    a, b, c = st.columns(3)
    a.metric("Training issues", result.metrics["train_count"])
    b.metric("Test issues", result.metrics["test_count"])
    c.metric("Enhancements in test", result.metrics["test_labels"].get("Enhancement", 0))
    scores = pd.DataFrame({
        "Majority baseline": result.metrics["baseline"],
        "TF-IDF + Logistic Regression": result.metrics["model"],
    }).loc[["accuracy", "macro_f1"]]
    st.dataframe(scores, width="stretch")
    st.write("Per-class performance for the text model")
    st.dataframe(pd.DataFrame(result.metrics["model"]["per_class"]).T, width="stretch")
    st.write("Confusion matrix: rows are actual labels, columns are suggested labels")
    st.dataframe(pd.DataFrame(result.metrics["model"]["confusion_matrix"],
                              index=LABELS, columns=LABELS), width="stretch")
    st.caption(f"Training ends {result.metrics['train_end']}; test starts {result.metrics['test_start']}.")

with tab_data:
    st.subheader("What is in the saved sample?")
    st.write("The 908 rows are a selected historical sample, not every issue opened in the repository.")
    years = sorted(result.issues["created_at"].dt.year.unique().tolist())
    year = st.selectbox("Creation year", ["All years"] + years)
    shown = result.issues if year == "All years" else result.issues[result.issues["created_at"].dt.year == year]
    counts = shown["issue_type"].value_counts().reindex(LABELS, fill_value=0)
    largest = max(int(counts.max()), 1)
    for label, count in counts.items():
        st.progress(round(int(count) / largest * 100), text=f"{label}: {count} issues")
    st.write(f"Showing {len(shown)} issues")
    st.dataframe(shown[["issue_number", "created_at", "issue_type", "title"]].head(30),
                 hide_index=True, width="stretch")
    st.caption("Labels are from the saved data, not proven to be the labels present when issues were opened.")
