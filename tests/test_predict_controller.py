from unittest.mock import Mock

import pytest

from churn_predict_ml.api.routes.predict.controllers.controller import (
    PredictController,
)
from churn_predict_ml.api.routes.predict.dtos.dto import (
    PredictRequest,
    PredictResponse,
)


@pytest.fixture
def valid_request() -> PredictRequest:
    """Retorna uma requisição válida de previsão."""
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


def test_controller_returns_service_prediction(
    valid_request: PredictRequest,
) -> None:
    """Verifica se o Controller retorna a previsão do Service."""
    expected_response = PredictResponse(
        churn_probability=0.73,
        churn_prediction=1,
    )

    service = Mock()
    service.predict.return_value = expected_response

    controller = PredictController(service=service)

    response = controller.predict(valid_request)

    assert response == expected_response
    service.predict.assert_called_once_with(valid_request)


def test_controller_propagates_service_error(
    valid_request: PredictRequest,
) -> None:
    """Verifica se o Controller propaga erros do Service."""
    service = Mock()
    service.predict.side_effect = ValueError("Prediction failed")

    controller = PredictController(service=service)

    with pytest.raises(ValueError, match="Prediction failed"):
        controller.predict(valid_request)

    service.predict.assert_called_once_with(valid_request)
