
import pandas as pd


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Converte TotalCharges para numérico e trata valores ausentes.

    Valores vazios são preenchidos com zero somente quando tenure
    também é zero, conforme o padrão identificado na análise
    exploratória.

    Args:
        df: DataFrame contendo as colunas TotalCharges e tenure.

    Returns:
        Uma cópia do DataFrame com TotalCharges convertida para float.

    Raises:
        ValueError: Se houver valores inválidos de TotalCharges
            associados a clientes com tenure diferente de zero.
        KeyError: Se alguma coluna obrigatória estiver ausente.
    """
    cleaned_df = df.copy()

    cleaned_df["TotalCharges"] = pd.to_numeric(
        cleaned_df["TotalCharges"],
        errors="coerce",
    )

    missing_total_charges = cleaned_df["TotalCharges"].isna()
    invalid_records = missing_total_charges & cleaned_df["tenure"].ne(0)

    if invalid_records.any():
        raise ValueError(
            "TotalCharges contains invalid values for customers "
            "with tenure different from zero."
        )

    cleaned_df.loc[
        missing_total_charges & cleaned_df["tenure"].eq(0),
        "TotalCharges",
    ] = 0.0

    return cleaned_df


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa as variáveis preditoras da variável-alvo Churn.

    Remove o identificador customerID das features e converte
    os valores de Churn para uma representação binária.

    Args:
        df: DataFrame contendo as informações dos clientes,
            incluindo customerID e Churn.

    Returns:
        Uma tupla contendo:
            - X: DataFrame com as variáveis preditoras.
            - y: Series com a variável-alvo binária,
              sendo No = 0 e Yes = 1.

    Raises:
        KeyError: Se customerID ou Churn não estiver presente.
        ValueError: Se Churn contiver valores diferentes de
            Yes e No, incluindo valores ausentes.
    """
    required_columns = {"customerID", "Churn"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    valid_target = df["Churn"].isin(["Yes", "No"])

    if not valid_target.all():
        raise ValueError(
            "Churn must contain only 'Yes' or 'No' values."
        )

    X = df.drop(columns=["customerID", "Churn"]).copy()
    y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)

    return X, y
