from pathlib import Path

from churn_predict_ml.config.settings import (
    DEFAULT_MODEL_PATH,
    get_model_path,
)


def test_get_model_path_returns_default_when_env_is_missing(
    monkeypatch,
) -> None:
    """Verifica o caminho padrão sem variável de ambiente."""
    monkeypatch.delenv("CHURN_MODEL_PATH", raising=False)

    result = get_model_path()

    assert result == DEFAULT_MODEL_PATH


def test_get_model_path_uses_environment_variable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Verifica a utilização de um caminho personalizado."""
    custom_path = tmp_path / "custom_model.joblib"

    monkeypatch.setenv("CHURN_MODEL_PATH", str(custom_path))

    result = get_model_path()

    assert result == custom_path.resolve()


def test_get_model_path_expands_user_directory(
    monkeypatch,
) -> None:
    """Verifica a expansão do diretório pessoal."""
    monkeypatch.setenv(
        "CHURN_MODEL_PATH",
        "~/models/churn_model.joblib",
    )

    result = get_model_path()

    expected = (Path.home() / "models" / "churn_model.joblib").resolve()

    assert result == expected
