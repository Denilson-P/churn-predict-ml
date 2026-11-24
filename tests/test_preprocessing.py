import pandas as pd
import pytest

from churn_predict_ml.preprocessing import clean_total_charges


def test_clean_total_charges_converts_numeric_values() -> None:
    """Verifica a conversão de valores textuais para números."""
    df = pd.DataFrame(
        {
            "tenure": [1, 12],
            "TotalCharges": ["29.85", "1889.5"],
        }
    )

    result = clean_total_charges(df)

    assert result["TotalCharges"].tolist() == [29.85, 1889.5]


def test_clean_total_charges_fills_zero_tenure() -> None:
    """Verifica o preenchimento de valores vazios para novos clientes."""
    df = pd.DataFrame(
        {
            "tenure": [0, 2],
            "TotalCharges": [" ", "108.15"],
        }
    )

    result = clean_total_charges(df)

    assert result["TotalCharges"].tolist() == [0.0, 108.15]


def test_clean_total_charges_rejects_invalid_records() -> None:
    """Verifica a rejeição de valores inválidos fora da regra definida."""
    df = pd.DataFrame(
        {
            "tenure": [5],
            "TotalCharges": ["invalid"],
        }
    )

    with pytest.raises(ValueError, match="TotalCharges contains invalid"):
        clean_total_charges(df)


def test_clean_total_charges_preserves_original_dataframe() -> None:
    """Verifica que a transformação não modifica o DataFrame original."""
    df = pd.DataFrame(
        {
            "tenure": [0],
            "TotalCharges": [" "],
        }
    )

    clean_total_charges(df)

    assert df.loc[0, "TotalCharges"] == " "
