from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Verifica se a aplicação está respondendo às requisições.

    Returns:
        Dicionário contendo o estado operacional da API.
    """
    return {"status": "ok"}
