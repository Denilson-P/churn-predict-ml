from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from churn_predict_ml.ml.preprocessing import (
    clean_total_charges,
    split_features_target,
    split_train_test,
)
from churn_predict_ml.ml.training import (
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
)


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, Pipeline],
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compara modelos de classificação usando validação cruzada.

    Args:
        X: DataFrame contendo as variáveis preditoras.
        y: Série contendo os rótulos binários.
        models: Dicionário com os nomes e pipelines dos modelos.
        n_splits: Número de divisões da validação cruzada.
        random_state: Semente utilizada na divisão dos dados.

    Returns:
        DataFrame com a média e o desvio padrão das métricas
        de cada modelo nos folds de validação.
    """
    cross_validation = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1_score": "f1",
        "roc_auc": "roc_auc",
    }

    results = []

    for model_name, pipeline in models.items():
        scores = cross_validate(
            estimator=pipeline,
            X=X,
            y=y,
            cv=cross_validation,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        model_results = {"model": model_name}

        for metric_name in scoring:
            values = scores[f"test_{metric_name}"]

            model_results[f"{metric_name}_mean"] = values.mean()
            model_results[f"{metric_name}_std"] = values.std()

        results.append(model_results)

    return pd.DataFrame(results)


def run_model_comparison(dataset_path: Path) -> pd.DataFrame:
    """Executa a comparação de modelos no dataset de churn.

    Carrega e limpa os dados, reserva o conjunto de teste,
    constrói os pipelines e executa a validação cruzada
    exclusivamente nos registros de desenvolvimento.

    Args:
        dataset_path: Caminho do dataset CSV.

    Returns:
        DataFrame contendo as métricas comparativas dos modelos.
    """
    df = pd.read_csv(dataset_path)

    cleaned_df = clean_total_charges(df)

    X, y = split_features_target(cleaned_df)

    X_train, X_test, y_train, _ = split_train_test(X, y)

    models = {
        "Logistic Regression": build_logistic_regression_pipeline(X_train),
        "Random Forest": build_random_forest_pipeline(X_train),
    }

    results = compare_models(
        X=X_train,
        y=y_train,
        models=models,
    )

    print(f"Total de registros: {len(df)}")
    print(f"Registros de desenvolvimento: {len(X_train)}")
    print(f"Registros de teste reservados: {len(X_test)}")

    print("\nCross-Validation Model Comparison:")

    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    return results


if __name__ == "__main__":
    dataset_path = Path("data/raw/telco_customer_churn.csv")
    run_model_comparison(dataset_path)
