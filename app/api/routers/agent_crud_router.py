"""
API Router for Agent CRUD operations.
"""

import logging
from typing import Any, List, Optional  # Keep for types used in endpoints

from fastapi import APIRouter, Depends, HTTPException, status

from app.db import get_db
from app.schemas.agent_schemas import AgentCreate, AgentResponse, AgentUpdate
from app.services import agent_service  # Only agent_service should be needed here
from prisma import Prisma

logger = logging.getLogger(__name__)
router = APIRouter()


# API Endpoints for Agent CRUD
@router.post(
    "/agents", response_model=AgentResponse, status_code=status.HTTP_201_CREATED
)
async def create_new_agent(agent_data: AgentCreate, db: Prisma = Depends(get_db)):
    try:
        existing_agent = await agent_service.get_agent_by_name(db, agent_data.name)
        if existing_agent:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Agent with name '{agent_data.name}' already exists.",
            )
        created_agent = await agent_service.create_agent(
            db, name=agent_data.name, model_config=agent_data.modelConfig
        )
        return AgentResponse.from_prisma_agent(created_agent)
    except HTTPException:  # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        logger.error(f"Error creating agent '{agent_data.name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create agent",
        )


@router.get("/agents", response_model=List[AgentResponse])
async def list_all_agents(
    skip: int = 0, limit: int = 100, db: Prisma = Depends(get_db)
):
    try:
        agents_prisma = await db.agent.find_many(
            skip=skip, take=limit, order={"createdAt": "desc"}
        )
        return [AgentResponse.from_prisma_agent(agent) for agent in agents_prisma]
    except Exception as e:
        logger.error(f"Error listing agents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not list agents",
        )


@router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent_details(agent_id: str, db: Prisma = Depends(get_db)):
    try:
        agent = await db.agent.find_unique(where={"id": agent_id})
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{agent_id}' not found",
            )
        return AgentResponse.from_prisma_agent(agent)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent '{agent_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve agent",
        )


@router.put("/agents/{agent_id}", response_model=AgentResponse)
async def update_existing_agent(
    agent_id: str, agent_data: AgentUpdate, db: Prisma = Depends(get_db)
):
    try:
        # Prepare data for the service call
        update_payload = agent_data.model_dump(exclude_unset=True)

        name_to_pass: Optional[str] = update_payload.get("name")
        model_config_to_pass: Any = update_payload.get(
            "modelConfig", Ellipsis
        )  # Pass Ellipsis if not in payload

        # Call the service function
        updated_agent = await agent_service.update_agent(
            db, agent_id=agent_id, name=name_to_pass, model_config=model_config_to_pass
        )
        return AgentResponse.from_prisma_agent(updated_agent)

    except ValueError as ve:
        # Check for the specific "no fields to update" error
        if "No fields to update were provided for the agent." in str(ve):
            logger.warning(
                f"Update for agent '{agent_id}' failed due to no update data: {ve}"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(ve),
            )
        # If it's another ValueError (like "not found" from the service), let it fall through
        # to the more general Exception block or a more specific ValueError check below.
        # For now, we expect other ValueErrors to be caught by the generic Exception handler
        # which will then check for "not found" patterns.
        # A more robust solution would be custom exceptions from the service.
        # Re-raising to be caught by the next block if not the "no fields" error.
        # This means the order of except blocks below matters.
        # The most specific `isinstance(e, ValueError) and "not found"` should come first in the Exception block.
        # Or better, we handle specific ValueErrors from service here and let others be 500.

        # Let's explicitly check for the "not found" ValueErrors from the service here.
        if (
            "not found during update attempt" in str(ve)
            or "not found during prisma update operation" in str(ve)
            or "could not be updated (possibly not found by Prisma update)" in str(ve)
        ):
            logger.warning(
                f"Update for agent '{agent_id}' failed as agent was not found: {ve}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{agent_id}' not found for update.",
            )
        # If it's a ValueError but not one of the above, it's unexpected for this path.
        logger.error(
            f"Unexpected ValueError during agent update '{agent_id}': {ve}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected value error occurred: {ve}",
        )

    except Exception as e:
        # This block will now primarily catch non-ValueError exceptions, or ValueErrors not handled above.
        # The specific "isinstance(e, ValueError) and 'not found'" check is now effectively handled above.
        # We can simplify this block or keep it for other Prisma/DB errors.

        # Example: Catch Prisma's specific RecordNotFound if it somehow bypasses the service's ValueError
        if "RecordNotFoundError" in str(type(e)):
            logger.warning(
                f"Agent '{agent_id}' not found directly by Prisma during update: {e}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{agent_id}' not found (Prisma).",
            )

        logger.error(f"Error updating agent '{agent_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not update agent",
        )


@router.delete("/agents/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_existing_agent(agent_id: str, db: Prisma = Depends(get_db)):
    """Deletes an agent and all its associated data comprehensively."""
    logger.info(f"Received request to delete agent '{agent_id}' comprehensively.")
    try:
        success = await agent_service.delete_agent_comprehensively(db, agent_id)

        if not success:
            logger.error(
                f"Comprehensive deletion of agent '{agent_id}' reported a failure from the service layer."
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to comprehensively delete agent '{agent_id}'. Check logs for details.",
            )
        logger.info(
            f"Comprehensive deletion process for agent '{agent_id}' completed via service call."
        )

    except ValueError as ve:
        if "not found, cannot delete" in str(ve):
            logger.warning(
                f"Deletion failed for agent '{agent_id}' as it was not found: {ve}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{agent_id}' not found.",
            )
        # Other ValueErrors from the service during deletion might indicate partial failure
        logger.error(
            f"ValueError during comprehensive deletion of agent '{agent_id}': {ve}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"A value error occurred during deletion: {ve}",
        )
    except HTTPException:  # Re-raise HTTPExceptions from service or this level
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error during comprehensive deletion of agent '{agent_id}': {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while trying to delete agent '{agent_id}'.",
        )
