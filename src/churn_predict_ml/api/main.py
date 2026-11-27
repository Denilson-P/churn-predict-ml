from fastapi import FastAPI

from churn_predict_ml.api.routes.health.health import router as health_router
from churn_predict_ml.api.routes.predict.route import router as predict_router


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
    app.include_router(predict_router)

    return app


app = create_app()
