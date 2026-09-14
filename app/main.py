from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.routes import router
from app.pipeline import ExplainPipeline
from app.retrieval.knowledge_base import load_knowledge_base

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Load everything heavy once at startup, never per request.
    app.state.pipeline = ExplainPipeline(load_knowledge_base())
    yield


app = FastAPI(
    title="Medical Miscommunication Detector",
    description=(
        "Explains documented findings in medical report text. "
        "Does not diagnose, does not prescribe, does not replace clinician judgment."
    ),
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/")
def demo_page() -> FileResponse:
    """The patient-facing demo page (roadmap step 7). Not a production frontend — see
    README section 3 for the deliberate gap between this and a real patient product."""
    return FileResponse(STATIC_DIR / "index.html")
