"""Agent service orchestrator for CustomerPulse AI."""

from typing import Dict, Any, List
from ai.agent import CustomerPulseAnalystAgent


class AgentService:
    @staticmethod
    def process_query(prompt: str, session_id: str = "default") -> Dict[str, Any]:
        agent = CustomerPulseAnalystAgent()
        return agent.answer_query(prompt=prompt, session_id=session_id)
