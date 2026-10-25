"""Streamlit demo.  Run: streamlit run app.py"""
import sys
from pathlib import Path

import joblib
import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))
from common import LABELS, MODELS_DIR, clean_text  # noqa: E402

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.caption("Paste a news article and the model predicts whether it looks real or fake.")


@st.cache_resource
def load_baseline():
    path = MODELS_DIR / "baseline.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_resource
def load_distilbert():
    path = MODELS_DIR / "distilbert"
    if not path.exists():
        return None
    from transformers import pipeline
    return pipeline("text-classification", model=str(path), tokenizer=str(path),
                    truncation=True, max_length=256, top_k=None)


baseline, bert = load_baseline(), load_distilbert()
options = []
if baseline is not None:
    options.append("TF-IDF + Logistic Regression")
if bert is not None:
    options.append("DistilBERT")
if not options:
    st.error("No trained model found. Run `python src/train_baseline.py` first.")
    st.stop()

choice = st.radio("Model", options, horizontal=True)
text = st.text_area("Article text", height=250, placeholder="Paste the article here...")

if st.button("Analyze", type="primary") and text.strip():
    cleaned = clean_text(text)
    if choice == "DistilBERT":
        scores = {int(r["label"].split("_")[1]): r["score"] for r in bert(cleaned)[0]}
        p_real = scores[1]
    else:
        p_real = float(baseline.predict_proba([cleaned])[0][1])

    label = LABELS[int(p_real >= 0.5)]
    confidence = max(p_real, 1 - p_real)
    (st.success if label == "REAL" else st.error)(f"Prediction: **{label}** ({confidence:.1%} confidence)")
    st.progress(p_real, text=f"P(real) = {p_real:.1%}")
    st.caption("Trained on the ISOT dataset (2016-2017, mostly US political news). "
               "It may not generalise to other topics or years. Not a fact-checker.")
