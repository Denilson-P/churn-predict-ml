from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from churn_predict_ml.api.main import app
from churn_predict_ml.api.routes.predict.dtos.dto import PredictResponse
from churn_predict_ml.api.routes.predict.route import get_controller


@pytest.fixture
def valid_payload() -> dict:
    """Retorna um payload válido para previsão."""
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
    """Cria um cliente HTTP isolado para os testes."""
    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_predict_endpoint_returns_prediction(
    client: TestClient,
    valid_payload: dict,
) -> None:
    """Verifica a resposta HTTP de uma previsão válida."""
    controller = Mock()
    controller.predict.return_value = PredictResponse(
        churn_probability=0.73,
        churn_prediction=1,
    )

    app.dependency_overrides[get_controller] = lambda: controller

    response = client.post("/api/v1/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json() == {
        "churn_probability": 0.73,
        "churn_prediction": 1,
    }

    controller.predict.assert_called_once()


def test_predict_endpoint_rejects_invalid_payload(
    client: TestClient,
    valid_payload: dict,
) -> None:
    """Verifica a rejeição de dados inválidos."""
    valid_payload["tenure"] = -1

    response = client.post("/api/v1/predict", json=valid_payload)

    assert response.status_code == 422


def test_predict_endpoint_rejects_missing_feature(
    client: TestClient,
    valid_payload: dict,
) -> None:
    """Verifica a rejeição de uma feature obrigatória ausente."""
    valid_payload.pop("Contract")

    response = client.post("/api/v1/predict", json=valid_payload)

    assert response.status_code == 422
