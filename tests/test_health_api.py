from fastapi.testclient import TestClient

from churn_predict_ml.api.main import app

client = TestClient(app)


def test_health_check_returns_success() -> None:
    """Verifica se o endpoint de saúde retorna HTTP 200."""
    response = client.get("/health")

    assert response.status_code == 200


def test_health_check_returns_expected_body() -> None:
    """Verifica se o endpoint retorna o estado esperado."""
    response = client.get("/health")

    assert response.json() == {"status": "ok"}
