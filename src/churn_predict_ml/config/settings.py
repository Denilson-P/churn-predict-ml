import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "churn_logistic_regression.joblib"


def get_model_path() -> Path:
    """Obtém o caminho configurado para o modelo treinado.

    Utiliza a variável de ambiente CHURN_MODEL_PATH quando
    definida. Caso contrário, utiliza o caminho padrão
    do projeto.

    Returns:
        Caminho do artefato de Machine Learning.
    """
    configured_path = os.getenv("CHURN_MODEL_PATH")

    if configured_path:
        return Path(configured_path).expanduser().resolve()

    return DEFAULT_MODEL_PATH
