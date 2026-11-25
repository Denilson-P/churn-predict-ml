import numpy as np
import pandas as pd
import pytest

from churn_predict_ml.experiment import evaluate_model


def test_evaluate_model_perfect_predictions() -> None:
    """Verifica as métricas quando todas as previsões estão corretas."""
    y_true = pd.Series([0, 1, 0, 1])
    predictions = np.array([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8])

    metrics = evaluate_model(y_true, predictions, probabilities)

    assert all(value == pytest.approx(1.0) for value in metrics.values())


def test_evaluate_model_partial_predictions() -> None:
    """Verifica as métricas quando existem erros de classificação."""
    y_true = pd.Series([0, 0, 1, 1])
    predictions = np.array([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.8, 0.4, 0.9])

    metrics = evaluate_model(y_true, predictions, probabilities)

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1_score"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)


def test_evaluate_model_no_positive_predictions() -> None:
    """Verifica o tratamento de previsões sem classes positivas."""
    y_true = pd.Series([0, 0, 1, 1])
    predictions = np.array([0, 0, 0, 0])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])

    metrics = evaluate_model(y_true, predictions, probabilities)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_evaluate_model_returns_expected_metrics() -> None:
    """Verifica se todas as métricas esperadas são retornadas."""
    y_true = pd.Series([0, 1])
    predictions = np.array([0, 1])
    probabilities = np.array([0.2, 0.8])

    metrics = evaluate_model(y_true, predictions, probabilities)

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    }
