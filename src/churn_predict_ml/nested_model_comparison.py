from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from churn_predict_ml.experiment import evaluate_model
from churn_predict_ml.hyperparameter_tuning import tune_random_forest
from churn_predict_ml.preprocessing import (
    clean_total_charges,
    split_features_target,
    split_train_test,
)
from churn_predict_ml.training import (
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
)


def compare_models_nested(
    X: pd.DataFrame,
    y: pd.Series,
    outer_splits: int = 3,
    inner_splits: int = 3,
    n_iter: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compara modelos utilizando validação cruzada aninhada.

    Em cada fold externo, treina a Regressão Logística
    e otimiza a Random Forest exclusivamente nos dados
    de treinamento daquele fold.

    Args:
        X: DataFrame contendo as variáveis preditoras.
        y: Série contendo os rótulos binários.
        outer_splits: Número de folds externos.
        inner_splits: Número de folds internos da busca.
        n_iter: Quantidade de combinações da busca aleatória.
        random_state: Semente utilizada nas divisões.

    Returns:
        DataFrame contendo as médias e os desvios padrão
        das métricas obtidas nos folds externos.
    """
    outer_cv = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=random_state,
    )

    results = {
        "Logistic Regression": [],
        "Random Forest Tuned": [],
    }

    for train_indices, validation_indices in outer_cv.split(X, y):
        X_train = X.iloc[train_indices]
        X_validation = X.iloc[validation_indices]

        y_train = y.iloc[train_indices]
        y_validation = y.iloc[validation_indices]

        logistic_pipeline = build_logistic_regression_pipeline(
            X_train,
            random_state=random_state,
        )

        logistic_pipeline = clone(logistic_pipeline)
        logistic_pipeline.fit(X_train, y_train)

        random_forest_pipeline = build_random_forest_pipeline(
            X_train,
            random_state=random_state,
        )

        search = tune_random_forest(
            X=X_train,
            y=y_train,
            pipeline=random_forest_pipeline,
            n_iter=n_iter,
            n_splits=inner_splits,
            random_state=random_state,
        )

        models = {
            "Logistic Regression": logistic_pipeline,
            "Random Forest Tuned": search.best_estimator_,
        }

        for model_name, pipeline in models.items():
            predictions = pipeline.predict(X_validation)
            probabilities = pipeline.predict_proba(X_validation)[:, 1]

            metrics = evaluate_model(
                y_validation,
                predictions,
                probabilities,
            )

            results[model_name].append(metrics)

    summary = []

    for model_name, fold_results in results.items():
        metrics_df = pd.DataFrame(fold_results)

        model_summary = {"model": model_name}

        for metric_name in metrics_df.columns:
            model_summary[f"{metric_name}_mean"] = metrics_df[metric_name].mean()

            model_summary[f"{metric_name}_std"] = metrics_df[metric_name].std(ddof=0)

        summary.append(model_summary)

    return pd.DataFrame(summary)


def run_nested_model_comparison(dataset_path: Path) -> pd.DataFrame:
    """Executa a comparação aninhada no dataset de churn.

    Carrega e limpa os dados, reserva o conjunto de teste
    e executa a validação cruzada aninhada exclusivamente
    nos registros de desenvolvimento.

    Args:
        dataset_path: Caminho do arquivo CSV.

    Returns:
        DataFrame com os resultados comparativos.
    """
    df = pd.read_csv(dataset_path)

    cleaned_df = clean_total_charges(df)

    X, y = split_features_target(cleaned_df)

    X_train, X_test, y_train, _ = split_train_test(X, y)

    results = compare_models_nested(
        X=X_train,
        y=y_train,
    )

    print(f"Total de registros: {len(df)}")
    print(f"Registros de desenvolvimento: {len(X_train)}")
    print(f"Registros de teste reservados: {len(X_test)}")

    print("\nNested Cross-Validation Model Comparison:")
    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    return results


if __name__ == "__main__":
    dataset_path = Path("data/raw/telco_customer_churn.csv")
    run_nested_model_comparison(dataset_path)
