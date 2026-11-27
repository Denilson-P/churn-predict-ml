from unittest.mock import Mock

import pytest

from churn_predict_ml.api.routes.predict.dtos.dto import (
    PredictRequest,
    PredictResponse,
)
from churn_predict_ml.api.routes.predict.services.service import PredictService


@pytest.fixture
def valid_request() -> PredictRequest:
    """Retorna uma requisição válida para previsão."""
    return PredictRequest(
        gender="Female",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="No",
        tenure=12,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="Yes",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="Yes",
        StreamingMovies="Yes",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=85.50,
        TotalCharges=1026.00,
    )


def test_service_returns_prediction(valid_request: PredictRequest) -> None:
    """Verifica se o Service retorna uma previsão válida."""
    workflow = Mock()
    workflow.execute.return_value = {
        "churn_probability": 0.73,
        "churn_prediction": 1,
    }

    service = PredictService(workflow=workflow)

    response = service.predict(valid_request)

    assert isinstance(response, PredictResponse)
    assert response.churn_probability == 0.73
    assert response.churn_prediction == 1

    workflow.execute.assert_called_once_with(valid_request.model_dump())


def test_service_returns_negative_prediction(valid_request: PredictRequest) -> None:
    """Verifica se o Service aceita uma previsão negativa."""
    workflow = Mock()
    workflow.execute.return_value = {
        "churn_probability": 0.15,
        "churn_prediction": 0,
    }

    service = PredictService(workflow=workflow)

    response = service.predict(valid_request)

    assert response.churn_probability == 0.15
    assert response.churn_prediction == 0


def test_service_rejects_invalid_prediction(valid_request: PredictRequest) -> None:
    """Verifica se o Service rejeita resultados inválidos."""
    from pydantic import ValidationError

    workflow = Mock()
    workflow.execute.return_value = {
        "churn_probability": 1.5,
        "churn_prediction": 1,
    }

    service = PredictService(workflow=workflow)

    with pytest.raises(ValidationError):
        service.predict(valid_request)


def test_service_propagates_workflow_error(valid_request: PredictRequest) -> None:
    """Verifica se erros do Workflow são propagados."""
    workflow = Mock()
    workflow.execute.side_effect = ValueError("Missing required features")

    service = PredictService(workflow=workflow)

    with pytest.raises(ValueError, match="Missing required features"):
        service.predict(valid_request)
