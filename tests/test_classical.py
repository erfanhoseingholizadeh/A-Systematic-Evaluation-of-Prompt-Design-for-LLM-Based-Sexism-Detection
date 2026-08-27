import pandas as pd

from sexism_prompting.classical import evaluate_classical_baseline, train_classical_baseline

# A separable synthetic signal ("terrible" vs "lovely") so the pipeline has
# something real to learn, with enough repeated tokens to survive min_df=2.
_SEXIST = [
    "women are terrible at this terrible job",
    "she is terrible and should stay home terrible",
    "girls are always terrible drivers terrible",
    "that terrible woman ruined the terrible project",
    "terrible females cannot do terrible math",
    "terrible girl terrible behavior terrible today",
]
_NOT_SEXIST = [
    "the weather is lovely today lovely",
    "lovely dinner with lovely friends lovely",
    "lovely sunset over the lovely lake",
    "what a lovely lovely morning walk",
    "lovely music playing at the lovely cafe",
    "lovely book about lovely gardens lovely",
]


def _toy_df():
    texts = _SEXIST + _NOT_SEXIST
    labels = [1] * len(_SEXIST) + [0] * len(_NOT_SEXIST)
    return pd.DataFrame({"text": texts, "label": labels})


def test_train_classical_baseline_returns_fitted_pipeline():
    pipeline = train_classical_baseline(_toy_df())
    preds = pipeline.predict(["terrible terrible terrible", "lovely lovely lovely"])
    assert list(preds) == [1, 0]


def test_evaluate_classical_baseline_returns_standard_metric_keys():
    train_df = _toy_df()
    pipeline = train_classical_baseline(train_df)
    metrics = evaluate_classical_baseline(pipeline, train_df)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "auc"}
    assert metrics["accuracy"] == 1.0  # trivially separable signal, evaluated on its own training data


def test_evaluate_classical_baseline_auc_reflects_separable_signal():
    train_df = _toy_df()
    pipeline = train_classical_baseline(train_df)
    metrics = evaluate_classical_baseline(pipeline, train_df)
    assert metrics["auc"] == 1.0  # trivially separable signal, evaluated on its own training data


def test_train_classical_baseline_is_deterministic_given_seed():
    train_df = _toy_df()
    p1 = train_classical_baseline(train_df, seed=7)
    p2 = train_classical_baseline(train_df, seed=7)
    preds1 = list(p1.predict(train_df["text"]))
    preds2 = list(p2.predict(train_df["text"]))
    assert preds1 == preds2


def test_evaluate_classical_baseline_handles_unseen_vocabulary():
    train_df = _toy_df()
    pipeline = train_classical_baseline(train_df)
    unseen_df = pd.DataFrame({"text": ["completely novel wording never seen before"], "label": [0]})
    metrics = evaluate_classical_baseline(pipeline, unseen_df)  # must not raise on OOV tokens
    assert metrics["accuracy"] in (0.0, 1.0)
