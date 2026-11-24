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
