import numpy as np
import pandas as pd
import pytest

from churn_predict_ml.ml.model_comparison import compare_models
from churn_predict_ml.ml.training import (
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
)


@pytest.fixture
def sample_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Cria um dataset sintético para comparação dos modelos."""
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


def test_compare_models_returns_expected_columns(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica a estrutura dos resultados da comparação."""
    X, y = sample_dataset

    models = {
        "Logistic Regression": build_logistic_regression_pipeline(X),
        "Random Forest": build_random_forest_pipeline(
            X,
            n_estimators=10,
        ),
    }

    results = compare_models(
        X,
        y,
        models,
        n_splits=3,
    )

    expected_columns = {
        "model",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_score_mean",
        "f1_score_std",
        "roc_auc_mean",
        "roc_auc_std",
    }

    assert set(results.columns) == expected_columns
    assert len(results) == 2
    assert set(results["model"]) == set(models)


def test_compare_models_returns_valid_metrics(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica os intervalos das métricas retornadas."""
    X, y = sample_dataset

    models = {
        "Random Forest": build_random_forest_pipeline(
            X,
            n_estimators=10,
        ),
    }

    results = compare_models(
        X,
        y,
        models,
        n_splits=3,
    )

    mean_columns = [column for column in results.columns if column.endswith("_mean")]

    std_columns = [column for column in results.columns if column.endswith("_std")]

    assert results[mean_columns].ge(0).all().all()
    assert results[mean_columns].le(1).all().all()
    assert results[std_columns].ge(0).all().all()


def test_compare_models_is_reproducible(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica resultados idênticos usando a mesma semente."""
    X, y = sample_dataset

    models = {
        "Random Forest": build_random_forest_pipeline(
            X,
            random_state=42,
            n_estimators=10,
        ),
    }

    first_results = compare_models(
        X,
        y,
        models,
        n_splits=3,
        random_state=42,
    )

    second_results = compare_models(
        X,
        y,
        models,
        n_splits=3,
        random_state=42,
    )

    pd.testing.assert_frame_equal(
        first_results,
        second_results,
    )


def test_compare_models_does_not_fit_original_pipeline(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica que a validação cruzada não ajusta o pipeline original."""
    X, y = sample_dataset

    pipeline = build_random_forest_pipeline(
        X,
        n_estimators=10,
    )

    compare_models(
        X,
        y,
        {"Random Forest": pipeline},
        n_splits=3,
    )

    assert not hasattr(
        pipeline.named_steps["classifier"],
        "estimators_",
    )
