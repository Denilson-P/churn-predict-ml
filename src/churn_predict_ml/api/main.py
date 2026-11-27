from fastapi import FastAPI

from churn_predict_ml.api.routes.health import router as health_router


def create_app() -> FastAPI:
    """Cria e configura a aplicação FastAPI.

    Returns:
        Instância configurada da aplicação.
    """
    app = FastAPI(
        title="Churn Prediction API",
        description="REST API for customer churn prediction using Machine Learning.",
        version="0.1.0",
    )

    app.include_router(health_router)

    return app


app = create_app()
