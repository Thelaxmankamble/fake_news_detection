"""Fine-tune DistilBERT.  Run: python src/train_distilbert.py
On a laptop CPU this is slow, so by default it trains on a subset.
Use --train-size 20000 --epochs 2 on a GPU (e.g. free Google Colab).
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from common import MODELS_DIR, REPORTS_DIR, load_data

parser = argparse.ArgumentParser()
parser.add_argument("--train-size", type=int, default=3000)
parser.add_argument("--test-size", type=int, default=1000)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--max-len", type=int, default=256)
args = parser.parse_args()

MODEL_NAME = "distilbert-base-uncased"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

df = load_data()
# Same split and seed as the baseline so the comparison is fair
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
train_df = train_df.sample(min(args.train_size, len(train_df)), random_state=42)
test_df = test_df.sample(min(args.test_size, len(test_df)), random_state=42)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)


class NewsDataset(Dataset):
    def __init__(self, frame):
        self.enc = tok(frame["text"].tolist(), truncation=True,
                       max_length=args.max_len, padding="max_length")
        self.labels = frame["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


train_ds, test_ds = NewsDataset(train_df), NewsDataset(test_df)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


def metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}


trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
    ),
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=metrics,
)
trainer.train()

out = trainer.predict(test_ds)
preds = np.argmax(out.predictions, axis=-1)
print(classification_report(test_ds.labels, preds, target_names=["FAKE", "REAL"], digits=4))

(REPORTS_DIR / "distilbert_metrics.json").write_text(json.dumps({
    "accuracy": round(accuracy_score(test_ds.labels, preds), 4),
    "f1": round(f1_score(test_ds.labels, preds), 4),
    "train_size": len(train_ds), "test_size": len(test_ds), "epochs": args.epochs,
}, indent=2))

model.save_pretrained(MODELS_DIR / "distilbert")
tok.save_pretrained(MODELS_DIR / "distilbert")
print(f"Saved model to {MODELS_DIR / 'distilbert'}")
