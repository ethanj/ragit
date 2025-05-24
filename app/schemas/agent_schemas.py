"""
Pydantic Schemas for Agent operations.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from prisma.models import (
    Agent as PrismaAgent,  # For from_prisma_agent, if we keep it here
)

logger = logging.getLogger(__name__)  # In case logger is needed in methods


# Pydantic Models for Agent
class AgentBase(BaseModel):
    name: str = Field(
        ..., min_length=3, max_length=100, description="Name of the agent"
    )
    modelConfig: Optional[Dict[str, Any]] = Field(
        None, description="JSON configuration for the agent's LLM and embedding models"
    )


class AgentCreate(AgentBase):
    pass


class AgentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=3, max_length=100)
    modelConfig: Optional[Dict[str, Any]] = None


class AgentResponse(AgentBase):
    id: str
    # modelConfig is already in AgentBase and can be Optional[Dict]
    createdAt: datetime
    updatedAt: datetime

    class Config:
        from_attributes = True  # For Pydantic v2

    @classmethod
    def from_prisma_agent(cls, agent: PrismaAgent) -> "AgentResponse":
        config_dict = None
        if agent.modelConfig:
            try:
                config_dict = json.loads(agent.modelConfig)
            except json.JSONDecodeError:
                # Consider logging this from the service layer if it fails there during creation/update
                # or ensure modelConfig is always valid JSON if stored.
                # For response model, providing an error or None is reasonable.
                logger.warning(
                    f"Could not parse modelConfig JSON for agent {agent.id} in schema: {agent.modelConfig}"
                )
                config_dict = {"error": "Could not parse modelConfig from database"}
        return cls(
            id=agent.id,
            name=agent.name,
            modelConfig=config_dict,
            createdAt=agent.createdAt,
            updatedAt=agent.updatedAt,
        )
