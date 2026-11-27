import numpy as np
import pandas as pd
import pytest

from churn_predict_ml.ml.nested_model_comparison import compare_models_nested


@pytest.fixture
def sample_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Cria um dataset sintético para testar a comparação aninhada."""
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


def test_nested_comparison_returns_expected_models(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica se os dois modelos aparecem nos resultados."""
    X, y = sample_dataset

    results = compare_models_nested(
        X,
        y,
        outer_splits=2,
        inner_splits=2,
        n_iter=1,
    )

    assert set(results["model"]) == {
        "Logistic Regression",
        "Random Forest Tuned",
    }


def test_nested_comparison_returns_expected_metrics(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica a presença das métricas e seus desvios padrão."""
    X, y = sample_dataset

    results = compare_models_nested(
        X,
        y,
        outer_splits=2,
        inner_splits=2,
        n_iter=1,
    )

    expected_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]

    for metric in expected_metrics:
        assert f"{metric}_mean" in results.columns
        assert f"{metric}_std" in results.columns


def test_nested_comparison_returns_valid_scores(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica se as métricas estão em intervalos válidos."""
    X, y = sample_dataset

    results = compare_models_nested(
        X,
        y,
        outer_splits=2,
        inner_splits=2,
        n_iter=1,
    )

    metric_columns = [
        column for column in results.columns if column.endswith(("_mean", "_std"))
    ]

    assert results[metric_columns].notna().all().all()
    assert results[metric_columns].ge(0).all().all()
    assert results[metric_columns].le(1).all().all()


def test_nested_comparison_is_reproducible(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica a reprodutibilidade com a mesma semente."""
    X, y = sample_dataset

    first_results = compare_models_nested(
        X,
        y,
        outer_splits=2,
        inner_splits=2,
        n_iter=1,
        random_state=42,
    )

    second_results = compare_models_nested(
        X,
        y,
        outer_splits=2,
        inner_splits=2,
        n_iter=1,
        random_state=42,
    )

    pd.testing.assert_frame_equal(
        first_results,
        second_results,
    )


def test_nested_comparison_preserves_original_data(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica se o processo não modifica os dados originais."""
    X, y = sample_dataset

    original_X = X.copy(deep=True)
    original_y = y.copy(deep=True)

    compare_models_nested(
        X,
        y,
        outer_splits=2,
        inner_splits=2,
        n_iter=1,
    )

    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)
