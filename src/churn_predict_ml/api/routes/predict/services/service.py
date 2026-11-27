from typing import Protocol

from churn_predict_ml.api.routes.predict.dtos.dto import (
    PredictRequest,
    PredictResponse,
)


class PredictionWorkflowProtocol(Protocol):
    """Define o contrato esperado de um Workflow de previsão."""

    def execute(self, customer_data: dict) -> dict[str, float | int]:
        """Executa uma previsão e retorna seus resultados."""
        ...


class PredictService:
    """Executa operações de previsão de churn.

    Depende de uma abstração do Workflow, permitindo
    substituir sua implementação durante os testes.
    """

    def __init__(self, workflow: PredictionWorkflowProtocol) -> None:
        """Inicializa o Service.

        Args:
            workflow: Implementação do contrato de previsão.
        """
        self._workflow = workflow

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Realiza a previsão de churn de um cliente.

        Args:
            request: Dados validados do cliente.

        Returns:
            Resposta contendo a probabilidade e a classificação.
        """
        result = self._workflow.execute(request.model_dump())

        return PredictResponse(**result)
