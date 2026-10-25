"""Shared data loading and cleaning."""
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# Label convention: 1 = real, 0 = fake
LABELS = {0: "FAKE", 1: "REAL"}

# In this dataset, real articles often start with "CITY (Reuters) - ".
# Leaving it in lets a model "cheat" by spotting the word Reuters,
# which inflates accuracy. We strip it so the metrics are honest.
_REUTERS_PREFIX = re.compile(r"^.{0,60}?\(Reuters\)\s*-?\s*", re.IGNORECASE)
_URL = re.compile(r"http\S+|www\.\S+")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text)
    text = _REUTERS_PREFIX.sub("", text)
    text = _URL.sub(" ", text)
    text = _SPACES.sub(" ", text)
    return text.strip()


def load_data() -> pd.DataFrame:
    fake_path, true_path = DATA_DIR / "Fake.csv", DATA_DIR / "True.csv"
    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            "Put Fake.csv and True.csv in the data/ folder. See data/README.md."
        )
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    fake["label"], true["label"] = 0, 1

    df = pd.concat([fake, true], ignore_index=True)
    df["text"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).map(clean_text)
    df = df[df["text"].str.len() > 20]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df[["text", "label"]]
