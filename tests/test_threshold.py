import numpy as np
import pandas as pd
import pytest

from churn_predict_ml.threshold import (
    compare_thresholds,
    predict_with_threshold,
    select_best_threshold,
)


def test_predict_with_default_threshold() -> None:
    """Verifica a classificação com threshold padrão de 0.5."""
    probabilities = np.array([0.2, 0.5, 0.8])

    predictions = predict_with_threshold(probabilities)

    np.testing.assert_array_equal(predictions, [0, 1, 1])


def test_predict_with_custom_threshold() -> None:
    """Verifica a classificação com um threshold personalizado."""
    probabilities = np.array([0.2, 0.4, 0.7])

    predictions = predict_with_threshold(probabilities, threshold=0.3)

    np.testing.assert_array_equal(predictions, [0, 1, 1])


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_predict_rejects_invalid_threshold(threshold: float) -> None:
    """Verifica a rejeição de thresholds fora do intervalo permitido."""
    probabilities = np.array([0.2, 0.8])

    with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
        predict_with_threshold(probabilities, threshold)


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([np.nan, 0.8]),
        np.array([np.inf, 0.8]),
        np.array([-0.1, 0.8]),
        np.array([0.2, 1.1]),
        np.array([[0.2, 0.8]]),
    ],
)
def test_predict_rejects_invalid_probabilities(
    probabilities: np.ndarray,
) -> None:
    """Verifica a rejeição de probabilidades inválidas."""
    with pytest.raises(ValueError):
        predict_with_threshold(probabilities)


def test_compare_thresholds_returns_expected_metrics() -> None:
    """Verifica as métricas calculadas para diferentes thresholds."""
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    results = compare_thresholds(
        y_true,
        probabilities,
        thresholds=[0.3, 0.5],
    )

    assert len(results) == 2

    assert results.loc[0, "precision"] == pytest.approx(2 / 3)
    assert results.loc[0, "recall"] == pytest.approx(1.0)
    assert results.loc[0, "f1_score"] == pytest.approx(0.8)

    assert results.loc[1, "precision"] == pytest.approx(1.0)
    assert results.loc[1, "recall"] == pytest.approx(1.0)
    assert results.loc[1, "f1_score"] == pytest.approx(1.0)


def test_compare_thresholds_rejects_empty_list() -> None:
    """Verifica a rejeição de uma lista vazia de thresholds."""
    y_true = pd.Series([0, 1])
    probabilities = np.array([0.2, 0.8])

    with pytest.raises(ValueError, match="Thresholds list cannot be empty"):
        compare_thresholds(y_true, probabilities, thresholds=[])


def test_compare_thresholds_rejects_different_lengths() -> None:
    """Verifica a rejeição de entradas com tamanhos incompatíveis."""
    y_true = pd.Series([0, 1, 0])
    probabilities = np.array([0.2, 0.8])

    with pytest.raises(ValueError, match="same length"):
        compare_thresholds(y_true, probabilities, thresholds=[0.5])


def test_select_best_threshold() -> None:
    """Verifica a seleção do threshold com maior F1-score."""
    results = pd.DataFrame(
        {
            "threshold": [0.3, 0.4, 0.5],
            "f1_score": [0.65, 0.72, 0.60],
        }
    )

    best_threshold = select_best_threshold(results)

    assert best_threshold == pytest.approx(0.4)


def test_select_best_threshold_keeps_first_on_tie() -> None:
    """Verifica o desempate pela ordem original dos resultados."""
    results = pd.DataFrame(
        {
            "threshold": [0.3, 0.4],
            "f1_score": [0.7, 0.7],
        }
    )

    assert select_best_threshold(results) == pytest.approx(0.3)


def test_select_best_threshold_rejects_empty_results() -> None:
    """Verifica a rejeição de resultados vazios."""
    results = pd.DataFrame(columns=["threshold", "f1_score"])

    with pytest.raises(ValueError, match="cannot be empty"):
        select_best_threshold(results)


def test_select_best_threshold_rejects_missing_columns() -> None:
    """Verifica a validação das colunas obrigatórias."""
    results = pd.DataFrame({"threshold": [0.3, 0.4]})

    with pytest.raises(ValueError, match="Missing required"):
        select_best_threshold(results)
