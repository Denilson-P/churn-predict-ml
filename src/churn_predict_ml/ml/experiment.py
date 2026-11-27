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

from churn_predict_ml.ml.preprocessing import (
    clean_total_charges,
    split_features_target,
    split_train_test,
)
from churn_predict_ml.ml.threshold import (
    compare_thresholds,
    predict_with_threshold,
    select_best_threshold,
)
from churn_predict_ml.ml.training import build_logistic_regression_pipeline


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

    Carrega e limpa o dataset, separa os dados de desenvolvimento
    e teste, utiliza validação interna para selecionar o threshold,
    treina o modelo final e compara os resultados no teste.

    Args:
        dataset_path: Caminho do arquivo CSV contendo os dados.
    """
    df = pd.read_csv(dataset_path)

    cleaned_df = clean_total_charges(df)

    X, y = split_features_target(cleaned_df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    X_train_inner, X_validation, y_train_inner, y_validation = split_train_test(
        X_train,
        y_train,
    )

    validation_pipeline = build_logistic_regression_pipeline(X_train_inner)
    validation_pipeline.fit(X_train_inner, y_train_inner)

    validation_probabilities = validation_pipeline.predict_proba(X_validation)[:, 1]

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]

    threshold_results = compare_thresholds(
        y_validation,
        validation_probabilities,
        thresholds,
    )

    best_threshold = select_best_threshold(threshold_results)

    pipeline = build_logistic_regression_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    default_predictions = predict_with_threshold(probabilities, threshold=0.50)
    optimized_predictions = predict_with_threshold(
        probabilities,
        threshold=best_threshold,
    )

    default_metrics = evaluate_model(
        y_test,
        default_predictions,
        probabilities,
    )

    optimized_metrics = evaluate_model(
        y_test,
        optimized_predictions,
        probabilities,
    )

    print(f"Total de registros: {len(df)}")
    print(f"Registros de desenvolvimento: {len(X_train)}")
    print(f"Registros de treinamento interno: {len(X_train_inner)}")
    print(f"Registros de validação: {len(X_validation)}")
    print(f"Registros de teste: {len(X_test)}")

    print("\nValidation Threshold Comparison:")
    print(threshold_results.to_string(index=False, float_format="%.4f"))

    print(f"\nSelected Threshold: {best_threshold:.2f}")

    print("\nModel Evaluation:")
    print(f"{'Metric':<15} {'Default':>10} {'Optimized':>12}")

    for metric_name in default_metrics:
        print(
            f"{metric_name:<15} "
            f"{default_metrics[metric_name]:>10.4f} "
            f"{optimized_metrics[metric_name]:>12.4f}"
        )

    print("\nClassification Report — Optimized Threshold:")
    print(classification_report(y_test, optimized_predictions, zero_division=0))

    print("\nConfusion Matrix — Optimized Threshold:")
    print(confusion_matrix(y_test, optimized_predictions))


if __name__ == "__main__":
    dataset_path = Path("data/raw/telco_customer_churn.csv")
    run_experiment(dataset_path)
