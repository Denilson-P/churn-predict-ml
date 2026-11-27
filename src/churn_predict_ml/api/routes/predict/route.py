import logging
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from churn_predict_ml.api.routes.predict.controllers.controller import (
    PredictController,
)
from churn_predict_ml.api.routes.predict.dtos.dto import (
    PredictRequest,
    PredictResponse,
)
from churn_predict_ml.api.routes.predict.services.service import PredictService
from churn_predict_ml.config.settings import get_model_path
from churn_predict_ml.core.predict_workflow import PredictWorkflow
from churn_predict_ml.inference.model_persistence import load_model

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["Prediction"],
)


@lru_cache(maxsize=1)
def get_workflow() -> PredictWorkflow:
    """Cria e reutiliza o Workflow com o modelo carregado.

    Returns:
        Workflow configurado com o artefato de ML.

    Raises:
        HTTPException: Se o modelo não puder ser carregado.
    """
    try:
        artifact = load_model(get_model_path())
    except Exception as exc:
        logger.exception("Failed to load prediction model.")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is unavailable.",
        ) from exc

    return PredictWorkflow(artifact=artifact)


def get_service(
    workflow: Annotated[PredictWorkflow, Depends(get_workflow)],
) -> PredictService:
    """Cria o serviço com o Workflow injetado.

    Args:
        workflow: Workflow responsável pela inferência.

    Returns:
        Serviço de previsão configurado.
    """
    return PredictService(workflow=workflow)


def get_controller(
    service: Annotated[PredictService, Depends(get_service)],
) -> PredictController:
    """Cria o Controller com o Service injetado.

    Args:
        service: Serviço responsável pela previsão.

    Returns:
        Controller configurado.
    """
    return PredictController(service=service)


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"description": "Invalid customer data."},
        503: {"description": "Prediction model is unavailable."},
        500: {"description": "Internal prediction error."},
    },
)
def predict(
    request: PredictRequest,
    controller: Annotated[PredictController, Depends(get_controller)],
) -> PredictResponse:
    """Realiza uma previsão de cancelamento de cliente.

    Args:
        request: Informações do cliente para inferência.
        controller: Controller injetado pelo FastAPI.

    Returns:
        Probabilidade e classificação de churn.

    Raises:
        HTTPException: Se ocorrer uma falha durante a inferência.
    """
    try:
        return controller.predict(request)
    except Exception as exc:
        logger.exception("Failed to execute churn prediction.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal prediction error occurred.",
        ) from exc
