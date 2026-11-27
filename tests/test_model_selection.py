import numpy as np
import pandas as pd
import pytest

from churn_predict_ml.ml.model_selection import compare_models_with_thresholds


@pytest.fixture
def sample_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Cria dados sintéticos para testar a seleção de modelos."""
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

    return X, y


def test_model_selection_returns_both_models(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica se os dois classificadores são comparados."""
    X, y = sample_dataset

    results = compare_models_with_thresholds(
        X,
        y,
        n_iter=2,
        n_splits=2,
    )

    assert set(results["model"]) == {
        "Logistic Regression",
        "Random Forest Tuned",
    }


def test_model_selection_returns_expected_columns(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica a estrutura dos resultados de validação."""
    X, y = sample_dataset

    results = compare_models_with_thresholds(
        X,
        y,
        n_iter=2,
        n_splits=2,
    )

    expected_columns = {
        "model",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    }

    assert set(results.columns) == expected_columns
    assert len(results) == 2


def test_model_selection_returns_valid_scores(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica se os resultados estão nos intervalos válidos."""
    X, y = sample_dataset

    results = compare_models_with_thresholds(
        X,
        y,
        n_iter=2,
        n_splits=2,
    )

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]

    assert results[metric_columns].notna().all().all()
    assert results[metric_columns].ge(0).all().all()
    assert results[metric_columns].le(1).all().all()
    assert results["threshold"].between(0, 1).all()


def test_model_selection_uses_supplied_thresholds(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica se a seleção respeita os thresholds informados."""
    X, y = sample_dataset

    thresholds = [0.35, 0.50]

    results = compare_models_with_thresholds(
        X,
        y,
        thresholds=thresholds,
        n_iter=2,
        n_splits=2,
    )

    assert results["threshold"].isin(thresholds).all()


def test_model_selection_preserves_original_data(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica se os dados de entrada permanecem inalterados."""
    X, y = sample_dataset

    original_X = X.copy(deep=True)
    original_y = y.copy(deep=True)

    compare_models_with_thresholds(
        X,
        y,
        n_iter=2,
        n_splits=2,
    )

    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)
