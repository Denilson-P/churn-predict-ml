from pathlib import Path

import pandas as pd

from churn_predict_ml.experiment import evaluate_model
from churn_predict_ml.hyperparameter_tuning import tune_random_forest
from churn_predict_ml.preprocessing import (
    clean_total_charges,
    split_features_target,
    split_train_test,
)
from churn_predict_ml.threshold import (
    compare_thresholds,
    predict_with_threshold,
    select_best_threshold,
)
from churn_predict_ml.training import (
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
)


def compare_models_with_thresholds(
    X: pd.DataFrame,
    y: pd.Series,
    thresholds: list[float] | None = None,
    n_iter: int = 10,
    n_splits: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compara modelos e thresholds usando validação independente.

    Separa os dados de desenvolvimento em treinamento e
    validação. Otimiza a Random Forest somente no treinamento,
    seleciona o melhor threshold de cada modelo na validação
    e calcula suas métricas.

    Args:
        X: DataFrame contendo as variáveis preditoras.
        y: Série contendo os rótulos binários.
        thresholds: Limiares de classificação a comparar.
        n_iter: Quantidade de combinações da busca aleatória.
        n_splits: Número de folds da otimização.
        random_state: Semente utilizada nos experimentos.

    Returns:
        DataFrame com os thresholds selecionados e as
        métricas de validação dos dois modelos.
    """
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]

    X_train, X_validation, y_train, y_validation = split_train_test(
        X,
        y,
        random_state=random_state,
    )

    logistic_pipeline = build_logistic_regression_pipeline(
        X_train,
        random_state=random_state,
    )
    logistic_pipeline.fit(X_train, y_train)

    forest_pipeline = build_random_forest_pipeline(
        X_train,
        random_state=random_state,
    )

    search = tune_random_forest(
        X=X_train,
        y=y_train,
        pipeline=forest_pipeline,
        n_iter=n_iter,
        n_splits=n_splits,
        random_state=random_state,
    )

    models = {
        "Logistic Regression": logistic_pipeline,
        "Random Forest Tuned": search.best_estimator_,
    }

    results = []

    for model_name, pipeline in models.items():
        probabilities = pipeline.predict_proba(X_validation)[:, 1]

        threshold_results = compare_thresholds(
            y_validation,
            probabilities,
            thresholds,
        )

        best_threshold = select_best_threshold(threshold_results)

        predictions = predict_with_threshold(
            probabilities,
            threshold=best_threshold,
        )

        metrics = evaluate_model(
            y_validation,
            predictions,
            probabilities,
        )

        results.append(
            {
                "model": model_name,
                "threshold": best_threshold,
                **metrics,
            }
        )

    return pd.DataFrame(results)


def run_model_selection(dataset_path: Path) -> pd.DataFrame:
    """Executa a seleção dos modelos no dataset de churn.

    Reserva o conjunto de teste e utiliza exclusivamente
    os registros de desenvolvimento para comparar os modelos.

    Args:
        dataset_path: Caminho do arquivo CSV.

    Returns:
        DataFrame com os resultados de validação.
    """
    df = pd.read_csv(dataset_path)

    cleaned_df = clean_total_charges(df)
    X, y = split_features_target(cleaned_df)

    X_development, X_test, y_development, _ = split_train_test(
        X,
        y,
    )

    results = compare_models_with_thresholds(
        X=X_development,
        y=y_development,
    )

    print(f"Total de registros: {len(df)}")
    print(f"Registros de desenvolvimento: {len(X_development)}")
    print(f"Registros de teste reservados: {len(X_test)}")

    print("\nModel Selection — Validation Results:")
    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    return results


if __name__ == "__main__":
    dataset_path = Path("data/raw/telco_customer_churn.csv")
    run_model_selection(dataset_path)
