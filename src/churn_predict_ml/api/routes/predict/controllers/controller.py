from typing import Protocol

from churn_predict_ml.api.routes.predict.dtos.dto import (
    PredictRequest,
    PredictResponse,
)


class PredictionServiceProtocol(Protocol):
    """Define o contrato do serviço de previsão."""

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Realiza uma previsão de churn."""
        ...


class PredictController:
    """Coordena as solicitações de previsão de churn.

    Delega a operação ao serviço injetado, sem conhecer
    os detalhes da inferência ou do modelo treinado.
    """

    def __init__(self, service: PredictionServiceProtocol) -> None:
        """Inicializa o Controller.

        Args:
            service: Serviço responsável pelas previsões.
        """
        self._service = service

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Solicita uma previsão ao Service.

        Args:
            request: Dados validados do cliente.

        Returns:
            Resultado da previsão de churn.
        """
        return self._service.predict(request)
