import numpy as np
import pandas as pd
import pytest

from churn_predict_ml.inference.model_persistence import (
    load_model,
    predict_churn,
    save_model,
)
from churn_predict_ml.ml.training import build_logistic_regression_pipeline


@pytest.fixture
def trained_model() -> tuple:
    """Cria um modelo treinado com dados sintéticos."""
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

    return pipeline, X


def test_save_and_load_model(trained_model, tmp_path) -> None:
    """Verifica se o modelo é salvo e carregado corretamente."""
    pipeline, X = trained_model
    artifact_path = tmp_path / "model.joblib"

    save_model(
        pipeline,
        X.columns.tolist(),
        artifact_path,
    )

    artifact = load_model(artifact_path)

    assert artifact_path.is_file()
    assert artifact["threshold"] == 0.40
    assert artifact["feature_names"] == X.columns.tolist()


def test_predictions_remain_consistent_after_loading(
    trained_model,
    tmp_path,
) -> None:
    """Verifica se as previsões permanecem iguais após carregar."""
    pipeline, X = trained_model
    artifact_path = tmp_path / "model.joblib"

    save_model(
        pipeline,
        X.columns.tolist(),
        artifact_path,
    )

    original_probabilities = pipeline.predict_proba(X)[:, 1]

    artifact = load_model(artifact_path)
    predictions = predict_churn(artifact, X)

    np.testing.assert_allclose(
        predictions["churn_probability"].to_numpy(),
        original_probabilities,
    )


def test_predictions_have_valid_values(trained_model, tmp_path) -> None:
    """Verifica os valores e a estrutura das previsões."""
    pipeline, X = trained_model
    artifact_path = tmp_path / "model.joblib"

    save_model(pipeline, X.columns.tolist(), artifact_path)

    artifact = load_model(artifact_path)
    predictions = predict_churn(artifact, X)

    assert len(predictions) == len(X)
    assert set(predictions.columns) == {
        "churn_probability",
        "churn_prediction",
    }
    assert predictions["churn_probability"].between(0, 1).all()
    assert predictions["churn_prediction"].isin([0, 1]).all()


def test_predict_churn_rejects_missing_features(
    trained_model,
    tmp_path,
) -> None:
    """Verifica se features ausentes produzem erro."""
    pipeline, X = trained_model
    artifact_path = tmp_path / "model.joblib"

    save_model(pipeline, X.columns.tolist(), artifact_path)

    artifact = load_model(artifact_path)

    with pytest.raises(ValueError, match="Missing required features"):
        predict_churn(artifact, X.drop(columns=["tenure"]))


def test_load_model_rejects_missing_file(tmp_path) -> None:
    """Verifica o tratamento de artefatos inexistentes."""
    artifact_path = tmp_path / "missing.joblib"

    with pytest.raises(FileNotFoundError):
        load_model(artifact_path)


def test_save_model_rejects_invalid_threshold(
    trained_model,
    tmp_path,
) -> None:
    """Verifica se thresholds inválidos são rejeitados."""
    pipeline, X = trained_model
    artifact_path = tmp_path / "model.joblib"

    with pytest.raises(ValueError, match="Threshold"):
        save_model(
            pipeline,
            X.columns.tolist(),
            artifact_path,
            threshold=1.5,
        )

    assert not artifact_path.exists()
