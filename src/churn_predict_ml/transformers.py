import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    X: pd.DataFrame,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Constrói o pré-processador das variáveis preditoras.

    Identifica automaticamente as colunas categóricas e numéricas,
    aplica OneHotEncoder às categóricas e permite padronizar
    opcionalmente as variáveis numéricas.

    Args:
        X: DataFrame contendo as variáveis preditoras, sem
            customerID e Churn.
        scale_numeric: Define se as variáveis numéricas serão
            padronizadas com StandardScaler. Por padrão,
            seus valores são preservados.

    Returns:
        ColumnTransformer configurado para transformar as
        variáveis categóricas e processar as numéricas.

    Raises:
        ValueError: Se o DataFrame não possuir colunas ou
            contiver tipos de dados incompatíveis.
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

    numerical_transformer = StandardScaler() if scale_numeric else "passthrough"

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
                numerical_transformer,
                numerical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor
