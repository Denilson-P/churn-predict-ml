from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from churn_predict_ml.ml.preprocessing import (
    clean_total_charges,
    split_features_target,
    split_train_test,
)
from churn_predict_ml.ml.threshold import predict_with_threshold
from churn_predict_ml.ml.training import build_logistic_regression_pipeline


def save_model(
    pipeline: Pipeline,
    feature_names: list[str],
    artifact_path: Path,
    threshold: float = 0.40,
) -> None:
    """Salva o pipeline treinado e os metadados de previsão.

    Args:
        pipeline: Pipeline ajustado com os dados de treinamento.
        feature_names: Nomes das features esperadas pelo modelo.
        artifact_path: Caminho de destino do artefato.
        threshold: Limiar utilizado na classificação.

    Raises:
        ValueError: Se o threshold ou as features forem inválidos.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("Feature names must be non-empty and unique.")

    artifact = {
        "pipeline": pipeline,
        "threshold": threshold,
        "feature_names": feature_names,
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)


def load_model(artifact_path: Path) -> dict:
    """Carrega um artefato de modelo previamente salvo.

    Args:
        artifact_path: Caminho do artefato Joblib.

    Returns:
        Dicionário com pipeline, threshold e nomes das features.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se a estrutura do artefato for inválida.
    """
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

    artifact = joblib.load(artifact_path)

    required_keys = {"pipeline", "threshold", "feature_names"}

    if not isinstance(artifact, dict) or not required_keys.issubset(artifact):
        raise ValueError("Invalid model artifact structure.")

    if not isinstance(artifact["pipeline"], Pipeline):
        raise ValueError("Invalid pipeline in model artifact.")

    threshold = artifact["threshold"]
    feature_names = artifact["feature_names"]

    if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
        raise ValueError("Invalid threshold in model artifact.")

    if (
        not isinstance(feature_names, list)
        or not feature_names
        or any(not isinstance(name, str) for name in feature_names)
        or len(feature_names) != len(set(feature_names))
    ):
        raise ValueError("Invalid feature names in model artifact.")

    return artifact


def predict_churn(
    artifact: dict,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Realiza previsões utilizando um modelo carregado.

    Args:
        artifact: Dicionário contendo o modelo e seus metadados.
        X: DataFrame com as informações dos clientes.

    Returns:
        DataFrame com probabilidade e classificação de churn.

    Raises:
        ValueError: Se alguma feature obrigatória estiver ausente.
    """
    feature_names = artifact["feature_names"]

    missing_columns = set(feature_names).difference(X.columns)

    if missing_columns:
        raise ValueError(f"Missing required features: {sorted(missing_columns)}")

    features = X.loc[:, feature_names]
    probabilities = artifact["pipeline"].predict_proba(features)[:, 1]

    predictions = predict_with_threshold(
        probabilities,
        threshold=artifact["threshold"],
    )

    return pd.DataFrame(
        {
            "churn_probability": probabilities,
            "churn_prediction": predictions,
        },
        index=X.index,
    )


def train_and_save_model(
    dataset_path: Path,
    artifact_path: Path,
    threshold: float = 0.40,
    random_state: int = 42,
) -> Path:
    """Treina o modelo selecionado e salva o artefato.

    Utiliza somente o conjunto de desenvolvimento, mantendo
    o conjunto de teste separado do treinamento.

    Args:
        dataset_path: Caminho do dataset original.
        artifact_path: Caminho de destino do modelo.
        threshold: Limiar selecionado na validação.
        random_state: Semente utilizada na divisão dos dados.

    Returns:
        Caminho do artefato salvo.
    """
    df = pd.read_csv(dataset_path)

    cleaned_df = clean_total_charges(df)
    X, y = split_features_target(cleaned_df)

    X_development, X_test, y_development, _ = split_train_test(
        X,
        y,
        random_state=random_state,
    )

    pipeline = build_logistic_regression_pipeline(
        X_development,
        random_state=random_state,
    )

    pipeline.fit(X_development, y_development)

    save_model(
        pipeline=pipeline,
        feature_names=X_development.columns.tolist(),
        artifact_path=artifact_path,
        threshold=threshold,
    )

    print(f"Training records: {len(X_development)}")
    print(f"Reserved test records: {len(X_test)}")
    print(f"Classification threshold: {threshold:.2f}")
    print(f"Model saved to: {artifact_path}")

    return artifact_path


if __name__ == "__main__":
    train_and_save_model(
        dataset_path=Path("data/raw/telco_customer_churn.csv"),
        artifact_path=Path("models/churn_logistic_regression.joblib"),
    )
