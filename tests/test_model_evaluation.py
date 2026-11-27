import numpy as np
import pandas as pd
import pytest

from churn_predict_ml.ml.model_evaluation import evaluate_saved_model
from churn_predict_ml.ml.training import build_logistic_regression_pipeline


@pytest.fixture
def evaluation_data() -> tuple[dict, pd.DataFrame, pd.Series]:
    """Prepara dados sintéticos e um modelo para avaliação."""
    rng = np.random.default_rng(42)
    rows = 60

    X = pd.DataFrame(
        {
            "tenure": rng.integers(1, 73, size=rows),
            "MonthlyCharges": rng.uniform(20, 120, size=rows),
            "Contract": np.tile(
                ["Month-to-month", "One year", "Two year"],
                rows // 3,
            ),
        }
    )

    y = pd.Series(np.tile([0, 1], rows // 2))

    pipeline = build_logistic_regression_pipeline(X)
    pipeline.fit(X, y)

    artifact = {
        "pipeline": pipeline,
        "threshold": 0.40,
        "feature_names": X.columns.tolist(),
    }

    return artifact, X, y


def test_evaluation_returns_expected_keys(evaluation_data) -> None:
    """Verifica a estrutura do resultado da avaliação."""
    artifact, X, y = evaluation_data

    results = evaluate_saved_model(artifact, X, y)

    assert set(results) == {
        "metrics",
        "confusion_matrix",
        "classification_report",
        "threshold",
        "test_samples",
    }


def test_evaluation_returns_valid_metrics(evaluation_data) -> None:
    """Verifica se as métricas estão nos intervalos esperados."""
    artifact, X, y = evaluation_data

    results = evaluate_saved_model(artifact, X, y)

    assert set(results["metrics"]) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    }

    assert all(0 <= value <= 1 for value in results["metrics"].values())


def test_confusion_matrix_has_correct_shape(evaluation_data) -> None:
    """Verifica a estrutura e a contagem da matriz de confusão."""
    artifact, X, y = evaluation_data

    results = evaluate_saved_model(artifact, X, y)
    matrix = results["confusion_matrix"]

    assert matrix.shape == (2, 2)
    assert matrix.sum() == len(y)


def test_evaluation_uses_saved_threshold(evaluation_data) -> None:
    """Verifica se a avaliação utiliza o threshold persistido."""
    artifact, X, y = evaluation_data

    results = evaluate_saved_model(artifact, X, y)

    assert results["threshold"] == 0.40
    assert results["test_samples"] == len(y)


def test_evaluation_rejects_mismatched_lengths(evaluation_data) -> None:
    """Verifica o tratamento de quantidades incompatíveis."""
    artifact, X, y = evaluation_data

    with pytest.raises(ValueError, match="same length"):
        evaluate_saved_model(artifact, X, y.iloc[:-1])


def test_evaluation_rejects_empty_dataset(evaluation_data) -> None:
    """Verifica se um conjunto de teste vazio é rejeitado."""
    artifact, X, y = evaluation_data

    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_saved_model(artifact, X.iloc[:0], y.iloc[:0])


def test_evaluation_preserves_input_data(evaluation_data) -> None:
    """Verifica se a avaliação não modifica os dados recebidos."""
    artifact, X, y = evaluation_data

    original_X = X.copy(deep=True)
    original_y = y.copy(deep=True)

    evaluate_saved_model(artifact, X, y)

    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)
