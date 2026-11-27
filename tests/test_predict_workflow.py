from unittest.mock import patch

import pandas as pd
import pytest

from churn_predict_ml.api.routes.predict.dtos.dto import PredictRequest
from churn_predict_ml.core.predict_workflow import PredictWorkflow


@pytest.fixture
def valid_customer_data() -> dict:
    """Retorna os dados válidos de um cliente."""
    request = PredictRequest(
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

    return request.model_dump()


def test_workflow_returns_prediction(valid_customer_data: dict) -> None:
    """Verifica o retorno de uma previsão positiva."""
    artifact = {
        "pipeline": object(),
        "threshold": 0.40,
        "feature_names": list(valid_customer_data),
    }

    expected_predictions = pd.DataFrame(
        {
            "churn_probability": [0.73],
            "churn_prediction": [1],
        }
    )

    with patch(
        "churn_predict_ml.core.predict_workflow.predict_churn",
        return_value=expected_predictions,
    ) as mock_predict:
        workflow = PredictWorkflow(artifact=artifact)
        response = workflow.execute(valid_customer_data)

    assert response["churn_probability"] == 0.73
    assert response["churn_prediction"] == 1

    mock_predict.assert_called_once()

    called_artifact = mock_predict.call_args.kwargs["artifact"]
    called_features = mock_predict.call_args.kwargs["X"]

    assert called_artifact == artifact
    assert isinstance(called_features, pd.DataFrame)
    assert called_features.shape == (1, 19)
    assert called_features.columns.tolist() == list(valid_customer_data)


def test_workflow_returns_negative_prediction(valid_customer_data: dict) -> None:
    """Verifica o retorno de uma previsão negativa."""
    artifact = {"threshold": 0.40}

    expected_predictions = pd.DataFrame(
        {
            "churn_probability": [0.15],
            "churn_prediction": [0],
        }
    )

    with patch(
        "churn_predict_ml.core.predict_workflow.predict_churn",
        return_value=expected_predictions,
    ):
        workflow = PredictWorkflow(artifact=artifact)
        response = workflow.execute(valid_customer_data)

    assert response["churn_probability"] == 0.15
    assert response["churn_prediction"] == 0


def test_workflow_propagates_inference_error(valid_customer_data: dict) -> None:
    """Verifica a propagação de erros da inferência."""
    artifact = {"threshold": 0.40}

    with patch(
        "churn_predict_ml.core.predict_workflow.predict_churn",
        side_effect=ValueError("Missing required features"),
    ):
        workflow = PredictWorkflow(artifact=artifact)

        with pytest.raises(ValueError, match="Missing required features"):
            workflow.execute(valid_customer_data)
