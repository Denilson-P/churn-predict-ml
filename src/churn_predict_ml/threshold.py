import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def predict_with_threshold(
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Converte probabilidades em previsões binárias.

    Args:
        probabilities: Probabilidades previstas para a classe positiva.
        threshold: Limiar de classificação entre zero e um.

    Returns:
        Array contendo as classes previstas, sendo 0 ou 1.

    Raises:
        ValueError: Se o threshold for inválido ou se as
            probabilidades não forem unidimensionais, finitas
            ou estiverem fora do intervalo de zero a um.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 1:
        raise ValueError("Probabilities must be one-dimensional.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must contain only finite values.")

    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    return (probabilities >= threshold).astype(int)


def compare_thresholds(
    y_true: pd.Series,
    probabilities: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    """Compara métricas de classificação para diferentes limiares.

    Args:
        y_true: Valores reais da variável-alvo.
        probabilities: Probabilidades previstas para a classe positiva.
        thresholds: Lista de limiares a serem avaliados.

    Returns:
        DataFrame contendo threshold, precision, recall e F1-score.

    Raises:
        ValueError: Se a lista de thresholds estiver vazia
            ou os dados possuírem tamanhos incompatíveis.
    """
    if not thresholds:
        raise ValueError("Thresholds list cannot be empty.")

    if len(y_true) != len(probabilities):
        raise ValueError("Target and probabilities must have the same length.")

    results = []

    for threshold in thresholds:
        predictions = predict_with_threshold(probabilities, threshold)

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "f1_score": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(results)


def select_best_threshold(results: pd.DataFrame) -> float:
    """Seleciona o threshold com maior F1-score.

    Em caso de empate, mantém o primeiro threshold
    encontrado na ordem original dos resultados.

    Args:
        results: DataFrame com thresholds e métricas calculadas
            sobre dados de validação.

    Returns:
        Threshold correspondente ao maior F1-score.

    Raises:
        ValueError: Se o DataFrame estiver vazio ou não
            contiver as colunas obrigatórias.
    """
    required_columns = {"threshold", "f1_score"}

    if results.empty:
        raise ValueError("Threshold results cannot be empty.")

    if not required_columns.issubset(results.columns):
        raise ValueError("Missing required threshold result columns.")

    best_index = results["f1_score"].idxmax()

    return float(results.loc[best_index, "threshold"])
