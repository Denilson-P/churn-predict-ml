from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from churn_predict_ml.inference.model_persistence import load_model, predict_churn
from churn_predict_ml.ml.experiment import evaluate_model
from churn_predict_ml.ml.preprocessing import (
    clean_total_charges,
    split_features_target,
    split_train_test,
)


def evaluate_saved_model(
    artifact: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Avalia um modelo persistido em dados de teste.

    Utiliza o pipeline e o threshold armazenados no artefato,
    sem realizar novo treinamento ou ajustar parâmetros.

    Args:
        artifact: Artefato contendo pipeline e metadados.
        X_test: Features reservadas para avaliação.
        y_test: Classes reais dos registros de teste.

    Returns:
        Dicionário com métricas, matriz de confusão,
        relatório de classificação e quantidade de registros.

    Raises:
        ValueError: Se features e rótulos forem incompatíveis.
    """
    if len(X_test) != len(y_test):
        raise ValueError("Features and targets must have the same length.")

    if X_test.empty:
        raise ValueError("Test dataset must not be empty.")

    if not X_test.index.equals(y_test.index):
        raise ValueError("Features and targets must have matching indexes.")

    predictions_df = predict_churn(artifact, X_test)

    predictions = predictions_df["churn_prediction"].to_numpy()
    probabilities = predictions_df["churn_probability"].to_numpy()

    metrics = evaluate_model(
        y_true=y_test,
        predictions=predictions,
        probabilities=probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No Churn", "Churn"],
        zero_division=0,
        output_dict=True,
    )

    return {
        "metrics": metrics,
        "confusion_matrix": matrix,
        "classification_report": report,
        "threshold": artifact["threshold"],
        "test_samples": len(X_test),
    }


def run_model_evaluation(
    dataset_path: Path,
    artifact_path: Path,
) -> dict:
    """Executa a avaliação do modelo salvo no conjunto de teste.

    Reconstrói a divisão original dos dados para recuperar
    os registros reservados, sem treinar novamente o modelo.

    Args:
        dataset_path: Caminho do dataset original.
        artifact_path: Caminho do artefato Joblib.

    Returns:
        Dicionário contendo os resultados da avaliação.
    """
    df = pd.read_csv(dataset_path)

    cleaned_df = clean_total_charges(df)
    X, y = split_features_target(cleaned_df)

    _, X_test, _, y_test = split_train_test(
        X,
        y,
        random_state=42,
    )

    artifact = load_model(artifact_path)

    results = evaluate_saved_model(
        artifact=artifact,
        X_test=X_test,
        y_test=y_test,
    )

    print("\nFinal Model Evaluation")
    print(f"Test samples: {results['test_samples']}")
    print(f"Classification threshold: {results['threshold']:.2f}")

    print("\nClassification Metrics:")

    for metric_name, value in results["metrics"].items():
        print(f"{metric_name:<15} {value:.4f}")

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    print("\nClassification Report:")
    report = results["classification_report"]

    for class_name in ["No Churn", "Churn"]:
        class_metrics = report[class_name]

        print(
            f"{class_name:<12} "
            f"precision={class_metrics['precision']:.4f} "
            f"recall={class_metrics['recall']:.4f} "
            f"f1={class_metrics['f1-score']:.4f} "
            f"support={class_metrics['support']:.0f}"
        )

    return results


if __name__ == "__main__":
    run_model_evaluation(
        dataset_path=Path("data/raw/telco_customer_churn.csv"),
        artifact_path=Path("models/churn_logistic_regression.joblib"),
    )
