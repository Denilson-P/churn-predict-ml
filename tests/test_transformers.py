import pandas as pd
import pytest

from churn_predict_ml.ml.transformers import build_preprocessor


def test_build_preprocessor_encodes_categorical_columns() -> None:
    """Verifica a conversão de variáveis categóricas em colunas numéricas."""
    X = pd.DataFrame(
        {
            "Contract": ["Month-to-month", "One year", "Two year"],
            "tenure": [3, 24, 60],
        }
    )

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()

    assert transformed.shape == (3, 4)
    assert "categorical__Contract_Month-to-month" in feature_names
    assert "categorical__Contract_One year" in feature_names
    assert "categorical__Contract_Two year" in feature_names


def test_build_preprocessor_preserves_numerical_columns() -> None:
    """Verifica que as variáveis numéricas mantêm seus valores originais."""
    X = pd.DataFrame(
        {
            "Contract": ["Month-to-month", "One year"],
            "tenure": [3, 24],
            "MonthlyCharges": [95.5, 50.0],
        }
    )

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()

    tenure_index = list(feature_names).index("numerical__tenure")
    charges_index = list(feature_names).index("numerical__MonthlyCharges")

    assert transformed[:, tenure_index].tolist() == [3, 24]
    assert transformed[:, charges_index].tolist() == [95.5, 50.0]


def test_build_preprocessor_handles_unknown_categories() -> None:
    """Verifica o tratamento de categorias não vistas no treinamento."""
    X_train = pd.DataFrame(
        {
            "Contract": ["Month-to-month", "One year"],
            "tenure": [3, 24],
        }
    )

    X_test = pd.DataFrame(
        {
            "Contract": ["Two year"],
            "tenure": [60],
        }
    )

    preprocessor = build_preprocessor(X_train)

    preprocessor.fit(X_train)
    transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    categorical_indices = [
        index
        for index, name in enumerate(feature_names)
        if name.startswith("categorical__")
    ]

    assert transformed.shape == (1, 3)
    assert transformed[0, categorical_indices].tolist() == [0.0, 0.0]


def test_build_preprocessor_rejects_empty_dataframe() -> None:
    """Verifica a rejeição de DataFrames sem colunas."""
    X = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="Features DataFrame must contain columns",
    ):
        build_preprocessor(X)


def test_build_preprocessor_rejects_unsupported_columns() -> None:
    """Verifica a rejeição de tipos de dados não suportados."""
    X = pd.DataFrame(
        {
            "tenure": [3, 24],
            "registration_date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
        }
    )

    with pytest.raises(
        ValueError,
        match="Unsupported feature columns",
    ):
        build_preprocessor(X)
