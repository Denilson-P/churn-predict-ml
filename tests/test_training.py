import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn_predict_ml.training import build_logistic_regression_pipeline


@pytest.fixture
def sample_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Cria dados de exemplo para testar o pipeline de treinamento."""
    X = pd.DataFrame(
        {
            "Contract": ["Month-to-month", "One year", "Two year"] * 10,
            "tenure": list(range(1, 31)),
            "MonthlyCharges": [50.0, 75.0, 100.0] * 10,
        }
    )
    y = pd.Series([0, 1, 0] * 10)

    return X, y


def test_build_pipeline_contains_expected_steps(
    sample_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica os componentes do pipeline."""
    X, _ = sample_training_data

    pipeline = build_logistic_regression_pipeline(X)

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocessor", "classifier"]
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)


def test_build_pipeline_scales_numerical_features(
    sample_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica a padronização das variáveis numéricas."""
    X, _ = sample_training_data

    pipeline = build_logistic_regression_pipeline(X)
    preprocessor = pipeline.named_steps["preprocessor"]

    transformed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    numerical_indices = [
        index
        for index, name in enumerate(feature_names)
        if name.startswith("numerical__")
    ]

    numerical_values = transformed[:, numerical_indices]

    assert isinstance(preprocessor.named_transformers_["numerical"], StandardScaler)
    np.testing.assert_allclose(numerical_values.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(numerical_values.std(axis=0), 1.0, atol=1e-10)


def test_build_pipeline_trains_and_predicts(
    sample_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verifica se o modelo consegue treinar e gerar previsões."""
    X, y = sample_training_data

    pipeline = build_logistic_regression_pipeline(X)
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})
    assert probabilities.shape == (len(X), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_build_pipeline_rejects_empty_features() -> None:
    """Verifica a rejeição de DataFrames sem colunas."""
    X = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="Features DataFrame must contain columns",
    ):
        build_logistic_regression_pipeline(X)
