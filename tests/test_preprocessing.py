import pandas as pd
import pytest

from churn_predict_ml.ml.preprocessing import (
    clean_total_charges,
    split_features_target,
    split_train_test,
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


def test_split_train_test_preserves_dataset_size() -> None:
    """Verifica a divisão 80/20 sem perda de registros."""
    X = pd.DataFrame({"tenure": range(100)})
    y = pd.Series([0] * 75 + [1] * 25)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20

    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_train.index).union(X_test.index) == set(X.index)


def test_split_train_test_preserves_class_distribution() -> None:
    """Verifica a preservação da proporção das classes."""
    X = pd.DataFrame({"tenure": range(100)})
    y = pd.Series([0] * 75 + [1] * 25)

    _, _, y_train, y_test = split_train_test(X, y)

    assert y_train.mean() == pytest.approx(0.25)
    assert y_test.mean() == pytest.approx(0.25)


def test_split_train_test_is_reproducible() -> None:
    """Verifica resultados idênticos com a mesma semente."""
    X = pd.DataFrame({"tenure": range(100)})
    y = pd.Series([0] * 75 + [1] * 25)

    first_split = split_train_test(X, y)
    second_split = split_train_test(X, y)

    for first, second in zip(first_split, second_split, strict=True):
        assert first.index.tolist() == second.index.tolist()
