from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import ExplainRequest, ExplainResponse
from app.pipeline import ExplainPipeline, FeatureUnavailableError, PipelineStageError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/explain", response_model=ExplainResponse)
def explain(body: ExplainRequest, request: Request) -> ExplainResponse:
    pipeline: ExplainPipeline = request.app.state.pipeline
    try:
        return pipeline.run(body.report_text, body.target_language, body.include_audio)
    except FeatureUnavailableError as exc:
        raise HTTPException(
            status_code=501, detail={"stage": exc.stage, "message": exc.message}
        ) from exc
    except PipelineStageError as exc:
        logger.exception("Pipeline stage %s failed", exc.stage)
        raise HTTPException(
            status_code=500, detail={"stage": exc.stage, "message": exc.message}
        ) from exc
