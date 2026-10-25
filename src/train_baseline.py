"""Baseline: TF-IDF + Logistic Regression.  Run: python src/train_baseline.py"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from common import MODELS_DIR, REPORTS_DIR, load_data

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

df = load_data()
print(f"Loaded {len(df)} articles | real: {df.label.sum()} | fake: {(df.label == 0).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                              max_features=100_000, min_df=3, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, C=5.0)),
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred)
print(classification_report(y_test, pred, target_names=["FAKE", "REAL"], digits=4))

joblib.dump(model, MODELS_DIR / "baseline.joblib")
(REPORTS_DIR / "baseline_metrics.json").write_text(
    json.dumps({"accuracy": round(acc, 4), "f1": round(f1, 4), "test_size": len(y_test)}, indent=2)
)

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["FAKE", "REAL"], cmap="Blues")
plt.title("TF-IDF + Logistic Regression")
plt.savefig(REPORTS_DIR / "baseline_confusion_matrix.png", dpi=150, bbox_inches="tight")

# Most influential words (useful for the README and for explaining the model)
vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
names = vec.get_feature_names_out()
order = clf.coef_[0].argsort()
print("\nTop words -> FAKE:", [names[i] for i in order[:15]])
print("Top words -> REAL:", [names[i] for i in order[-15:]])
print(f"\nSaved model to {MODELS_DIR / 'baseline.joblib'}")
