
import pandas as pd
import pytest

from churn_predict_ml.preprocessing import (
    clean_total_charges,
    split_features_target,
)


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


def test_split_features_target_separates_features_and_target() -> None:
    """Verifica a separação das features e a conversão de Churn."""
    df = pd.DataFrame(
        {
            "customerID": ["001", "002"],
            "tenure": [12, 3],
            "MonthlyCharges": [50.0, 95.0],
            "Churn": ["No", "Yes"],
        }
    )

    X, y = split_features_target(df)

    assert X.columns.tolist() == ["tenure", "MonthlyCharges"]
    assert y.tolist() == [0, 1]


def test_split_features_target_rejects_invalid_churn() -> None:
    """Verifica a rejeição de categorias inválidas em Churn."""
    df = pd.DataFrame(
        {
            "customerID": ["001", "002"],
            "tenure": [12, 3],
            "Churn": ["No", "Unknown"],
        }
    )

    with pytest.raises(ValueError, match="Churn must contain"):
        split_features_target(df)


def test_split_features_target_rejects_missing_columns() -> None:
    """Verifica a validação das colunas obrigatórias."""
    df = pd.DataFrame(
        {
            "tenure": [12],
            "Churn": ["No"],
        }
    )

    with pytest.raises(KeyError, match="customerID"):
        split_features_target(df)


def test_split_features_target_preserves_original_dataframe() -> None:
    """Verifica que a separação não modifica os dados originais."""
    df = pd.DataFrame(
        {
            "customerID": ["001"],
            "tenure": [12],
            "Churn": ["Yes"],
        }
    )

    original_df = df.copy(deep=True)

    split_features_target(df)

    pd.testing.assert_frame_equal(df, original_df)
