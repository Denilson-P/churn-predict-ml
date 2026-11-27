import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from churn_predict_ml.ml.training import build_random_forest_pipeline


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Cria um conjunto de features para testar o Random Forest."""
    return pd.DataFrame(
        {
            "tenure": [1, 2, 12, 24, 36, 48, 60, 72],
            "MonthlyCharges": [
                90.0,
                85.0,
                75.0,
                65.0,
                55.0,
                45.0,
                35.0,
                25.0,
            ],
            "Contract": [
                "Month-to-month",
                "Month-to-month",
                "Month-to-month",
                "One year",
                "One year",
                "Two year",
                "Two year",
                "Two year",
            ],
        }
    )


@pytest.fixture
def sample_target() -> pd.Series:
    """Cria os rótulos binários utilizados nos testes."""
    return pd.Series([1, 1, 1, 0, 0, 0, 0, 0])


def test_build_random_forest_pipeline_contains_expected_steps(
    sample_features: pd.DataFrame,
) -> None:
    """Verifica os componentes do pipeline."""
    pipeline = build_random_forest_pipeline(sample_features)

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocessor", "classifier"]
    assert isinstance(
        pipeline.named_steps["classifier"],
        RandomForestClassifier,
    )


def test_random_forest_preserves_numerical_features(
    sample_features: pd.DataFrame,
) -> None:
    """Verifica que variáveis numéricas não são padronizadas."""
    pipeline = build_random_forest_pipeline(sample_features)

    preprocessor = pipeline.named_steps["preprocessor"]
    numerical_transformer = preprocessor.transformers[1][1]

    assert numerical_transformer == "passthrough"


def test_random_forest_uses_configured_estimators(
    sample_features: pd.DataFrame,
) -> None:
    """Verifica a configuração da quantidade de árvores."""
    pipeline = build_random_forest_pipeline(
        sample_features,
        n_estimators=50,
    )

    classifier = pipeline.named_steps["classifier"]

    assert classifier.n_estimators == 50


def test_random_forest_trains_and_predicts(
    sample_features: pd.DataFrame,
    sample_target: pd.Series,
) -> None:
    """Verifica treinamento e geração de previsões binárias."""
    pipeline = build_random_forest_pipeline(
        sample_features,
        n_estimators=10,
    )

    pipeline.fit(sample_features, sample_target)

    predictions = pipeline.predict(sample_features)

    assert len(predictions) == len(sample_features)
    assert set(predictions).issubset({0, 1})


def test_random_forest_predicts_probabilities(
    sample_features: pd.DataFrame,
    sample_target: pd.Series,
) -> None:
    """Verifica a geração de probabilidades para cada classe."""
    pipeline = build_random_forest_pipeline(
        sample_features,
        n_estimators=10,
    )

    pipeline.fit(sample_features, sample_target)

    probabilities = pipeline.predict_proba(sample_features)

    assert probabilities.shape == (len(sample_features), 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(len(sample_features)),
    )


def test_random_forest_is_reproducible(
    sample_features: pd.DataFrame,
    sample_target: pd.Series,
) -> None:
    """Verifica a reprodutibilidade com a mesma semente."""
    first_pipeline = build_random_forest_pipeline(
        sample_features,
        random_state=42,
        n_estimators=10,
    )

    second_pipeline = build_random_forest_pipeline(
        sample_features,
        random_state=42,
        n_estimators=10,
    )

    first_pipeline.fit(sample_features, sample_target)
    second_pipeline.fit(sample_features, sample_target)

    first_probabilities = first_pipeline.predict_proba(sample_features)
    second_probabilities = second_pipeline.predict_proba(sample_features)

    np.testing.assert_allclose(
        first_probabilities,
        second_probabilities,
    )


def test_random_forest_rejects_empty_features() -> None:
    """Verifica a rejeição de um DataFrame sem colunas."""
    empty_features = pd.DataFrame(index=range(3))

    with pytest.raises(ValueError, match="must contain columns"):
        build_random_forest_pipeline(empty_features)


def test_random_forest_rejects_unsupported_features() -> None:
    """Verifica a rejeição de tipos de dados incompatíveis."""
    unsupported_features = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        }
    )

    with pytest.raises(ValueError, match="Unsupported feature columns"):
        build_random_forest_pipeline(unsupported_features)
