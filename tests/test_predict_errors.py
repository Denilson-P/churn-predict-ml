from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from churn_predict_ml.api.main import app
from churn_predict_ml.api.routes.predict.route import (
    get_controller,
    get_workflow,
)


@pytest.fixture
def valid_payload() -> dict:
    """Retorna os dados válidos de um cliente.

    Returns:
        Dicionário contendo as 19 features do modelo.
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
def client():
    """Cria um cliente HTTP isolado para os testes.

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


def test_predict_returns_503_when_model_is_missing(
    client: TestClient,
    valid_payload: dict,
) -> None:
    """Verifica o retorno HTTP 503 quando o modelo não existe."""
    with patch(
        "churn_predict_ml.api.routes.predict.route.load_model",
        side_effect=FileNotFoundError("Model not found"),
    ):
        response = client.post("/api/v1/predict", json=valid_payload)

    assert response.status_code == 503
    assert response.json() == {"detail": "Prediction model is unavailable."}


def test_predict_returns_503_when_artifact_is_invalid(
    client: TestClient,
    valid_payload: dict,
) -> None:
    """Verifica o retorno HTTP 503 para um artefato inválido."""
    with patch(
        "churn_predict_ml.api.routes.predict.route.load_model",
        side_effect=ValueError("Invalid model artifact"),
    ):
        response = client.post("/api/v1/predict", json=valid_payload)

    assert response.status_code == 503
    assert response.json() == {"detail": "Prediction model is unavailable."}


def test_predict_returns_500_when_inference_fails(
    client: TestClient,
    valid_payload: dict,
) -> None:
    """Verifica o tratamento de falhas inesperadas na inferência."""
    controller = Mock()
    controller.predict.side_effect = RuntimeError("Internal ML error")

    app.dependency_overrides[get_controller] = lambda: controller

    response = client.post("/api/v1/predict", json=valid_payload)

    assert response.status_code == 500
    assert response.json() == {"detail": "An internal prediction error occurred."}

    controller.predict.assert_called_once()


def test_predict_returns_422_for_invalid_request(
    client: TestClient,
    valid_payload: dict,
) -> None:
    """Verifica que erros de validação continuam retornando 422."""
    valid_payload["tenure"] = -1

    response = client.post("/api/v1/predict", json=valid_payload)

    assert response.status_code == 422
