import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import RandomizedSearchCV

from churn_predict_ml.hyperparameter_tuning import tune_random_forest
from churn_predict_ml.training import build_random_forest_pipeline


@pytest.fixture
def sample_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Cria um dataset sintético para testar a otimização."""
    rng = np.random.default_rng(42)

    rows = 40

    X = pd.DataFrame(
        {
            "tenure": rng.integers(1, 73, size=rows),
            "MonthlyCharges": rng.uniform(20, 120, size=rows),
            "Contract": np.tile(
                ["Month-to-month", "One year"],
                rows // 2,
            ),
        }
    )

    y = pd.Series(np.tile([0, 1], rows // 2))

    return X, y


def test_tune_random_forest_returns_fitted_search(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica se a busca retorna um objeto ajustado."""
    X, y = sample_dataset

    pipeline = build_random_forest_pipeline(
        X,
        n_estimators=10,
    )

    search = tune_random_forest(
        X,
        y,
        pipeline,
        n_iter=2,
        n_splits=2,
    )

    assert isinstance(search, RandomizedSearchCV)
    assert hasattr(search, "best_estimator_")
    assert hasattr(search, "best_params_")
    assert hasattr(search, "best_score_")


def test_tune_random_forest_uses_f1_score(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica se a seleção utiliza o F1-score."""
    X, y = sample_dataset

    pipeline = build_random_forest_pipeline(
        X,
        n_estimators=10,
    )

    search = tune_random_forest(
        X,
        y,
        pipeline,
        n_iter=2,
        n_splits=2,
    )

    assert search.scoring == "f1"
    assert 0.0 <= search.best_score_ <= 1.0


def test_tune_random_forest_is_reproducible(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica a reprodutibilidade da busca."""
    X, y = sample_dataset

    pipeline = build_random_forest_pipeline(
        X,
        n_estimators=10,
    )

    first_search = tune_random_forest(
        X,
        y,
        pipeline,
        n_iter=2,
        n_splits=2,
        random_state=42,
    )

    second_search = tune_random_forest(
        X,
        y,
        pipeline,
        n_iter=2,
        n_splits=2,
        random_state=42,
    )

    assert first_search.best_params_ == second_search.best_params_
    assert first_search.best_score_ == pytest.approx(second_search.best_score_)


def test_tune_random_forest_preserves_original_pipeline(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica que a busca não ajusta o pipeline original."""
    X, y = sample_dataset

    pipeline = build_random_forest_pipeline(
        X,
        n_estimators=10,
    )

    tune_random_forest(
        X,
        y,
        pipeline,
        n_iter=2,
        n_splits=2,
    )

    assert not hasattr(
        pipeline.named_steps["classifier"],
        "estimators_",
    )
