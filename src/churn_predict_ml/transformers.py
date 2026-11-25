import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Constrói o pré-processador das variáveis preditoras.

    Identifica automaticamente as colunas categóricas e numéricas,
    aplica OneHotEncoder às categóricas e preserva as numéricas
    sem alterações.

    Args:
        X: DataFrame contendo as variáveis preditoras, sem
            customerID e Churn.

    Returns:
        ColumnTransformer configurado para transformar as
        variáveis categóricas e preservar as numéricas.

    Raises:
        ValueError: Se o DataFrame não possuir colunas.
    """
    if X.shape[1] == 0:
        raise ValueError("Features DataFrame must contain columns.")

    categorical_columns = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()

    unsupported_columns = X.columns.difference(categorical_columns + numerical_columns)

    if not unsupported_columns.empty:
        raise ValueError(f"Unsupported feature columns: {unsupported_columns.tolist()}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_columns,
            ),
            (
                "numerical",
                "passthrough",
                numerical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor
