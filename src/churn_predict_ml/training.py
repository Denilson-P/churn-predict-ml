import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn_predict_ml.transformers import build_preprocessor


def build_logistic_regression_pipeline(
    X: pd.DataFrame,
    random_state: int = 42,
) -> Pipeline:
    """Constrói um pipeline de classificação com Regressão Logística.

    Reutiliza o pré-processador compartilhado para codificar
    variáveis categóricas e padronizar variáveis numéricas,
    integrando essas transformações ao classificador.

    Args:
        X: DataFrame contendo as variáveis preditoras.
        random_state: Semente utilizada pelo classificador.

    Returns:
        Pipeline configurado com pré-processamento e modelo.

    Raises:
        ValueError: Se não houver features ou existirem tipos
            de dados incompatíveis.
    """
    preprocessor = build_preprocessor(X, scale_numeric=True)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return pipeline


def build_random_forest_pipeline(
    X: pd.DataFrame,
    random_state: int = 42,
    n_estimators: int = 100,
) -> Pipeline:
    """Constrói um pipeline de classificação com Random Forest.

    Reutiliza o pré-processador compartilhado para codificar
    variáveis categóricas e preservar a escala original das
    variáveis numéricas.

    Args:
        X: DataFrame contendo as variáveis preditoras.
        random_state: Semente utilizada pelo classificador.
        n_estimators: Quantidade de árvores de decisão.

    Returns:
        Pipeline configurado com pré-processamento e Random Forest.

    Raises:
        ValueError: Se não houver features ou existirem tipos
            de dados incompatíveis.
    """
    preprocessor = build_preprocessor(X, scale_numeric=False)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return pipeline
