from pathlib import Path

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from churn_predict_ml.ml.preprocessing import (
    clean_total_charges,
    split_features_target,
    split_train_test,
)
from churn_predict_ml.ml.training import build_random_forest_pipeline


def tune_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
    n_iter: int = 20,
    n_splits: int = 5,
    random_state: int = 42,
) -> RandomizedSearchCV:
    """Otimiza os hiperparâmetros de um pipeline Random Forest.

    Executa uma busca aleatória com validação cruzada
    estratificada, utilizando o F1-score como critério
    de seleção.

    Args:
        X: DataFrame contendo as variáveis preditoras.
        y: Série contendo os rótulos binários.
        pipeline: Pipeline de Random Forest a ser otimizado.
        n_iter: Quantidade de combinações avaliadas.
        n_splits: Número de folds da validação cruzada.
        random_state: Semente para reprodutibilidade.

    Returns:
        Objeto RandomizedSearchCV ajustado com a melhor
        configuração encontrada.
    """
    parameter_distributions = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 5, 10, 15, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
        "classifier__class_weight": [None, "balanced"],
    }

    cross_validation = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=n_iter,
        scoring="f1",
        cv=cross_validation,
        random_state=random_state,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )

    search.fit(X, y)

    return search


def run_hyperparameter_tuning(dataset_path: Path) -> RandomizedSearchCV:
    """Executa a otimização da Random Forest no dataset de churn.

    Carrega os dados, reserva o conjunto de teste e executa
    a busca de hiperparâmetros exclusivamente sobre os
    registros de desenvolvimento.

    Args:
        dataset_path: Caminho do dataset CSV.

    Returns:
        Busca ajustada com a melhor configuração encontrada.
    """
    df = pd.read_csv(dataset_path)

    cleaned_df = clean_total_charges(df)

    X, y = split_features_target(cleaned_df)

    X_train, X_test, y_train, _ = split_train_test(X, y)

    pipeline = build_random_forest_pipeline(X_train)

    search = tune_random_forest(
        X=X_train,
        y=y_train,
        pipeline=pipeline,
    )

    print(f"Total de registros: {len(df)}")
    print(f"Registros de desenvolvimento: {len(X_train)}")
    print(f"Registros de teste reservados: {len(X_test)}")

    print("\nBest Random Forest Parameters:")

    for parameter, value in search.best_params_.items():
        print(f"{parameter}: {value}")

    print(f"\nBest Cross-Validation F1-score: {search.best_score_:.4f}")

    return search


if __name__ == "__main__":
    dataset_path = Path("data/raw/telco_customer_churn.csv")
    run_hyperparameter_tuning(dataset_path)
