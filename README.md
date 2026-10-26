# Fake News Detection

Classifies news articles as **real** or **fake** using two approaches, compared on the same train/test split:

1. **Baseline:** TF-IDF + Logistic Regression
2. **Transformer:** fine-tuned DistilBERT

Includes a Streamlit demo app.

## Results

| Model | Accuracy | F1 | Test samples |
|-------|----------|----|--------------|
| TF-IDF + Logistic Regression | 99.2% | 0.992 | 7819 |
| DistilBERT | _fill in_ | _fill in_ | _fill in_ |

(Numbers are saved in `reports/*.json` after training. Paste your real results here.)

## Dataset

[ISOT Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (~44k articles).
Not included in this repo; see `data/README.md`.

**Data cleaning notes:** duplicates are removed, and the "CITY (Reuters) -" prefix is stripped from
articles. Without this, models can reach near-perfect accuracy just by spotting the word "Reuters",
which would not generalise.

## Project structure

```
fake-news-detection/
├── app.py                   # Streamlit demo
├── requirements.txt
├── data/                    # put Fake.csv and True.csv here
├── src/
│   ├── common.py            # loading + cleaning
│   ├── train_baseline.py    # TF-IDF + Logistic Regression
│   └── train_distilbert.py  # DistilBERT fine-tuning
└── reports/                 # metrics and confusion matrix
```

## How to run

```bash
pip install -r requirements.txt
python src/train_baseline.py
python src/train_distilbert.py     # slow on CPU; use --train-size 20000 on a GPU
streamlit run app.py
```

## Limitations

- The model reaches 99.2% on this dataset, but confidence drops on articles from other topics and years, because it learns the dataset’s writing style rather than factual truth.
- The data is from 2016-2017 and mostly US political news, so the model may not generalise to other topics or years.
- The model learns writing style and vocabulary, not factual truth. It is not a fact-checker.

## Tech stack

Python, pandas, scikit-learn, PyTorch, Hugging Face Transformers, Streamlit
