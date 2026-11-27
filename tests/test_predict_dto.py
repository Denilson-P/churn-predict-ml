import pytest
from pydantic import ValidationError

from churn_predict_ml.api.routes.predict.dtos.dto import (
    PredictRequest,
    PredictResponse,
)


@pytest.fixture
def valid_customer() -> dict:
    """Retorna um cliente com dados válidos para previsão."""
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


def test_predict_request_accepts_valid_customer(valid_customer) -> None:
    """Verifica a aceitação de um cliente com dados válidos."""
    request = PredictRequest(**valid_customer)

    assert request.gender == "Female"
    assert request.tenure == 12
    assert len(request.model_dump()) == 19


def test_predict_request_rejects_missing_field(valid_customer) -> None:
    """Verifica a rejeição de uma feature obrigatória ausente."""
    valid_customer.pop("Contract")

    with pytest.raises(ValidationError):
        PredictRequest(**valid_customer)


def test_predict_request_rejects_invalid_category(valid_customer) -> None:
    """Verifica a rejeição de categorias desconhecidas."""
    valid_customer["Contract"] = "Weekly"

    with pytest.raises(ValidationError):
        PredictRequest(**valid_customer)


def test_predict_request_rejects_negative_tenure(valid_customer) -> None:
    """Verifica a rejeição de tempo de permanência negativo."""
    valid_customer["tenure"] = -1

    with pytest.raises(ValidationError):
        PredictRequest(**valid_customer)


def test_predict_request_rejects_negative_charges(valid_customer) -> None:
    """Verifica a rejeição de cobranças negativas."""
    valid_customer["MonthlyCharges"] = -10

    with pytest.raises(ValidationError):
        PredictRequest(**valid_customer)


def test_predict_request_rejects_extra_fields(valid_customer) -> None:
    """Verifica a rejeição de campos não utilizados pelo modelo."""
    valid_customer["customerID"] = "12345"

    with pytest.raises(ValidationError):
        PredictRequest(**valid_customer)


def test_predict_request_rejects_invalid_phone_service(valid_customer) -> None:
    """Verifica inconsistências entre telefone e múltiplas linhas."""
    valid_customer["PhoneService"] = "No"
    valid_customer["MultipleLines"] = "Yes"

    with pytest.raises(ValidationError):
        PredictRequest(**valid_customer)


def test_predict_request_rejects_invalid_internet_service(valid_customer) -> None:
    """Verifica inconsistências entre internet e serviços associados."""
    valid_customer["InternetService"] = "No"

    with pytest.raises(ValidationError):
        PredictRequest(**valid_customer)


def test_predict_response_accepts_valid_prediction() -> None:
    """Verifica a estrutura válida da resposta."""
    response = PredictResponse(
        churn_probability=0.73,
        churn_prediction=1,
    )

    assert response.churn_probability == 0.73
    assert response.churn_prediction == 1


def test_predict_response_rejects_invalid_probability() -> None:
    """Verifica a rejeição de probabilidades fora do intervalo."""
    with pytest.raises(ValidationError):
        PredictResponse(
            churn_probability=1.5,
            churn_prediction=1,
        )
