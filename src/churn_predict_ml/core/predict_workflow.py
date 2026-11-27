from collections.abc import Mapping
from typing import Any

import pandas as pd

from churn_predict_ml.inference.model_persistence import predict_churn


class PredictWorkflow:
    """Orquestra a execução de previsões de churn.

    Recebe o artefato de Machine Learning por injeção
    e executa a inferência sem depender da camada HTTP.
    """

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        """Inicializa o Workflow com o modelo carregado.

        Args:
            artifact: Artefato contendo pipeline, threshold
                e nomes das features utilizadas pelo modelo.
        """
        self._artifact = dict(artifact)

    def execute(self, customer_data: dict[str, Any]) -> dict[str, float | int]:
        """Executa a previsão de churn de um cliente.

        Args:
            customer_data: Dicionário contendo as features do cliente.

        Returns:
            Dicionário com probabilidade e classificação de churn.

        Raises:
            ValueError: Se as features não forem compatíveis com o modelo.
        """
        features = pd.DataFrame([customer_data])

        predictions = predict_churn(
            artifact=self._artifact,
            X=features,
        )

        prediction = predictions.iloc[0]

        return {
            "churn_probability": float(prediction["churn_probability"]),
            "churn_prediction": int(prediction["churn_prediction"]),
        }
