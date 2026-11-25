from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn_predict_ml.preprocessing import (
    clean_total_charges,
    split_features_target,
    split_train_test,
)
from churn_predict_ml.training import build_logistic_regression_pipeline


def evaluate_model(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calcula as métricas de avaliação de um classificador binário.

    Args:
        y_true: Valores reais da variável-alvo.
        predictions: Classes previstas pelo modelo.
        probabilities: Probabilidades previstas para a classe positiva.

    Returns:
        Dicionário contendo accuracy, precision, recall,
        F1-score e ROC-AUC.
    """
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1_score": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def run_experiment(dataset_path: Path) -> None:
    """Executa o experimento de classificação de churn.

    Carrega o dataset, realiza a limpeza dos dados, separa
    as variáveis preditoras e a variável-alvo, divide os
    registros em treinamento e teste e avalia o modelo.

    Args:
        dataset_path: Caminho do arquivo CSV contendo os dados.
    """
    df = pd.read_csv(dataset_path)

    cleaned_df = clean_total_charges(df)

    X, y = split_features_target(cleaned_df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    pipeline = build_logistic_regression_pipeline(X_train)

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, predictions, probabilities)

    print(f"Total de registros: {len(df)}")
    print(f"Registros de treinamento: {len(X_train)}")
    print(f"Registros de teste: {len(X_test)}")

    print("\nModel Evaluation:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    dataset_path = Path("data/raw/telco_customer_churn.csv")
    run_experiment(dataset_path)
