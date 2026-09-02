"""Compatibility exports; canonical configs belong to their agents."""
from app.agents.central.config import CentralAgentConfig
from app.agents.research.config import AgentConfig

__all__ = ["AgentConfig", "CentralAgentConfig"]
