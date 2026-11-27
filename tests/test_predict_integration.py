from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from churn_predict_ml.api.main import app
from churn_predict_ml.api.routes.predict.route import get_workflow
from churn_predict_ml.config.settings import get_model_path
from churn_predict_ml.inference.model_persistence import load_model


@pytest.fixture
def valid_payload() -> dict:
    """Retorna os dados válidos de um cliente para previsão real.

    Returns:
        Dicionário com as 19 features exigidas pelo modelo.
    """
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 1026.00,
    }


@pytest.fixture
def model_path() -> Path:
    """Disponibiliza o caminho configurado do modelo treinado.

    Returns:
        Caminho do artefato Joblib.

    Raises:
        pytest.skip: Se o modelo não estiver disponível.
    """
    path = get_model_path()

    if not path.is_file():
        pytest.skip("Trained model artifact is not available.")

    return path


@pytest.fixture
def client(model_path: Path):
    """Cria um cliente HTTP para testar a inferência real.

    Args:
        model_path: Caminho validado do modelo treinado.

    Yields:
        Cliente HTTP configurado.
    """
    app.dependency_overrides.clear()
    get_workflow.cache_clear()

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        get_workflow.cache_clear()


def test_predict_real_model_returns_valid_response(
    client: TestClient,
    valid_payload: dict,
    model_path: Path,
) -> None:
    """Verifica uma previsão real utilizando o modelo persistido."""
    artifact = load_model(model_path)
    threshold = artifact["threshold"]

    response = client.post("/api/v1/predict", json=valid_payload)

    assert response.status_code == 200

    result = response.json()

    assert set(result) == {
        "churn_probability",
        "churn_prediction",
    }

    probability = result["churn_probability"]
    prediction = result["churn_prediction"]

    assert 0 <= probability <= 1
    assert prediction in (0, 1)
    assert prediction == int(probability >= threshold)


def test_predict_real_model_is_reused(
    client: TestClient,
    valid_payload: dict,
) -> None:
    """Verifica se o modelo é carregado uma única vez."""
    with patch(
        "churn_predict_ml.api.routes.predict.route.load_model",
        wraps=load_model,
    ) as mock_load:
        first_response = client.post(
            "/api/v1/predict",
            json=valid_payload,
        )

        second_response = client.post(
            "/api/v1/predict",
            json=valid_payload,
        )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    assert first_response.json() == second_response.json()

    mock_load.assert_called_once()
