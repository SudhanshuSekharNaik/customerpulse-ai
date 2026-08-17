"""AI Analyst Agent interaction endpoints."""

from fastapi import APIRouter
from backend.app.schemas import AgentQueryRequest, AgentResponse
from backend.app.services.agent_service import AgentService

router = APIRouter(prefix="/api/agent", tags=["AI Agent"])


@router.post("/query", response_model=AgentResponse)
def query_agent(req: AgentQueryRequest):
    return AgentService.process_query(prompt=req.prompt, session_id=req.session_id or "default")
