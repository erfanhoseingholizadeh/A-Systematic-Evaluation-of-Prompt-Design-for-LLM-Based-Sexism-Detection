"""Classical (non-LLM) baseline: TF-IDF + logistic regression.

Cheap, CPU-only, torch-free. Unlike the encoder/QLoRA baselines in
``finetune.py``, this needs nothing beyond the base ``scikit-learn``
dependency already in ``pyproject.toml``, so it's fully testable in this
dev sandbox. The cheapest non-LLM comparison point this project can add.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .metrics import compute_auc, compute_metrics_from_predictions

DEFAULT_TFIDF_KWARGS: Dict = dict(
    ngram_range=(1, 2),
    min_df=2,
    max_features=50_000,
    sublinear_tf=True,
)
DEFAULT_LOGREG_KWARGS: Dict = dict(
    max_iter=1000,
    class_weight="balanced",  # EDOS's real ~24/76 split would otherwise bias toward "not sexist"
)


def train_classical_baseline(
    train_df: pd.DataFrame,
    seed: int = 42,
    tfidf_kwargs: Optional[Dict] = None,
    logreg_kwargs: Optional[Dict] = None,
) -> Pipeline:
    """Fit a TF-IDF + logistic-regression pipeline on ``train_df``'s text/label columns."""
    tfidf_kwargs = {**DEFAULT_TFIDF_KWARGS, **(tfidf_kwargs or {})}
    logreg_kwargs = {**DEFAULT_LOGREG_KWARGS, **(logreg_kwargs or {}), "random_state": seed}

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
            ("logreg", LogisticRegression(**logreg_kwargs)),
        ]
    )
    pipeline.fit(train_df["text"].tolist(), train_df["label"].tolist())
    return pipeline


def evaluate_classical_baseline(pipeline: Pipeline, test_df: pd.DataFrame) -> Dict[str, float]:
    """Accuracy/precision/recall/F1/AUC on ``test_df``, same metric shape as every other baseline.

    AUC comes from ``predict_proba``'s P(sexist) column, the same
    ``compute_auc`` helper the LLM confidence-capture path uses — this is
    the only baseline in the study whose class-probability estimate can be
    computed for free from an already-fitted classifier, no extra run needed.
    """
    texts = test_df["text"].tolist()
    y_true = test_df["label"].tolist()

    preds = pipeline.predict(texts)
    metrics = compute_metrics_from_predictions(preds, y_true)

    logreg = pipeline.named_steps["logreg"]
    sexist_col = list(logreg.classes_).index(1)
    proba_sexist = pipeline.predict_proba(texts)[:, sexist_col]
    metrics["auc"] = compute_auc(proba_sexist, y_true)

    return metrics
