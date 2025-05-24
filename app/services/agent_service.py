"""
Service layer for agent-related operations.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import settings
from app.services import file_service, vector_db_service
from prisma import Prisma
from prisma.models import Agent, ChatMessage

logger = logging.getLogger(__name__)

UPLOAD_DIR_BASE = Path(settings.CHROMA_PERSIST_DIR).parent / "uploads"


def model_config_to_json_str(model_config: Dict) -> str:
    """Converts a model config dictionary to a JSON string."""
    return json.dumps(model_config)


async def create_agent(
    db: Prisma, name: str, model_config: Optional[Dict] = None
) -> Agent:
    """
    Creates a new agent.

    Args:
        db: Prisma client instance.
        name: Name of the agent.
        model_config: Optional model configuration for the agent.

    Returns:
        The created Agent object.

    Raises:
        Exception: If agent creation fails.
    """
    try:
        agent = await db.agent.create(
            data={
                "name": name,
                "modelConfig": model_config_to_json_str(model_config)
                if model_config
                else None,
            }
        )
        logger.info(f"Agent '{name}' created successfully with ID: {agent.id}")
        return agent
    except Exception as e:
        logger.error(f"Failed to create agent '{name}': {e}", exc_info=True)
        raise


async def get_agent_by_name(db: Prisma, name: str) -> Optional[Agent]:
    """
    Retrieves an agent by its name.

    Args:
        db: Prisma client instance.
        name: Name of the agent to retrieve.

    Returns:
        The Agent object if found, otherwise None.

    Raises:
        Exception: If retrieval fails.
    """
    try:
        agent = await db.agent.find_unique(where={"name": name})
        if agent:
            logger.info(f"Retrieved agent '{name}' with ID: {agent.id}")
        else:
            logger.info(f"Agent with name '{name}' not found.")
        return agent
    except Exception as e:
        logger.error(f"Failed to retrieve agent '{name}': {e}", exc_info=True)
        raise


async def get_agent_by_id(db: Prisma, agent_id: str) -> Optional[Agent]:
    """
    Retrieves an agent by its ID.

    Args:
        db: Prisma client instance.
        agent_id: ID of the agent to retrieve.

    Returns:
        The Agent object if found, otherwise None.

    Raises:
        Exception: If retrieval fails.
    """
    try:
        agent = await db.agent.find_unique(where={"id": agent_id})
        if agent:
            logger.info(f"Retrieved agent with ID '{agent_id}'.")
        else:
            logger.info(f"Agent with ID '{agent_id}' not found.")
        return agent
    except Exception as e:
        logger.error(
            f"Failed to retrieve agent with ID '{agent_id}': {e}", exc_info=True
        )
        raise


async def get_chat_message(db: Prisma, message_id: str) -> Optional[ChatMessage]:
    """Retrieves a specific chat message by its ID."""
    try:
        message = await db.chatmessage.find_unique(where={"id": message_id})
        if message:
            logger.info(f"Retrieved chat message with ID '{message_id}'.")
        else:
            logger.info(f"Chat message with ID '{message_id}' not found.")
        return message
    except Exception as e:
        logger.error(
            f"Failed to retrieve chat message with ID '{message_id}': {e}",
            exc_info=True,
        )
        raise  # Re-raise for the caller to handle


async def delete_chat_history_for_agent(db: Prisma, agent_id: str) -> int:
    """Deletes all chat messages for a specific agent and returns the count of deleted messages."""
    logger.info(f"Attempting to delete all chat messages for agent_id: {agent_id}")
    try:
        delete_result = await db.chatmessage.delete_many(where={"agentId": agent_id})
        logger.info(
            f"Successfully deleted {delete_result} chat messages for agent_id: {agent_id}"
        )
        return delete_result
    except Exception as e:
        logger.error(
            f"Error deleting chat history for agent_id {agent_id}: {e}", exc_info=True
        )
        # Depending on desired error handling, could raise an exception here
        # or return a specific value like -1 or re-raise a custom service exception.
        raise  # Re-raise the original exception for the router to handle as 500 or specific error


async def delete_agent_comprehensively(db: Prisma, agent_id: str) -> bool:
    """
    Comprehensively deletes an agent and all its associated data.
    This includes: chat messages, uploaded files (physical and DB records),
    vector data from ChromaDB, and finally the agent's own DB record.

    Args:
        db: Prisma client instance.
        agent_id: The ID of the agent to delete.

    Returns:
        True if deletion was successful (or agent wasn't found initially).
        False if a critical part of the deletion failed (e.g., agent record not deleted).
        Raises ValueError if agent is not found for deletion.
    """
    logger.info(f"Starting comprehensive deletion for agent_id: {agent_id}")

    agent = await db.agent.find_unique(where={"id": agent_id})
    if not agent:
        logger.warning(f"Agent '{agent_id}' not found. Cannot perform deletion.")
        raise ValueError(f"Agent with ID '{agent_id}' not found, cannot delete.")

    # 1. Delete Chat Messages for the agent
    try:
        deleted_chat_count = await delete_chat_history_for_agent(db, agent_id)
        logger.info(
            f"Deleted {deleted_chat_count} chat messages for agent '{agent_id}'."
        )
    except Exception as e:
        logger.error(
            f"Error deleting chat history for agent '{agent_id}': {e}. Aborting comprehensive delete.",
            exc_info=True,
        )
        # This could be considered a critical failure for data integrity
        return False

    # 2. Delete associated files and their vector embeddings
    try:
        uploaded_files = await file_service.get_files_by_agent_id(db, agent_id)
        logger.info(
            f"Found {len(uploaded_files)} files associated with agent '{agent_id}' for deletion."
        )
        for file_record in uploaded_files:
            file_id = file_record.id
            file_path_in_db = (
                file_record.filePath
            )  # e.g., "uploads/agent_id/filename_uuid.csv"
            logger.info(
                f"Processing deletion for file ID '{file_id}' (Path: {file_path_in_db})."
            )

            # 2a. Delete vector embeddings for the file
            try:
                await vector_db_service.clear_vectors_for_file(
                    agent_id=agent_id, file_id=file_id
                )
                logger.info(
                    f"Successfully cleared vector embeddings for file '{file_id}'."
                )
            except Exception as e_vec:
                logger.error(
                    f"Error clearing vector embeddings for file '{file_id}' of agent '{agent_id}': {e_vec}. Continuing with file deletion.",
                    exc_info=True,
                )
                # Non-critical for overall agent deletion, but log it.

            # 2b. Delete physical file
            # Path is relative to UPLOAD_DIR_BASE.parent (data/)
            physical_file_path = UPLOAD_DIR_BASE.parent / file_path_in_db
            if physical_file_path.exists():
                try:
                    os.remove(physical_file_path)
                    logger.info(f"Deleted physical file: {physical_file_path}")
                except OSError as e_os:
                    logger.error(
                        f"Error deleting physical file {physical_file_path}: {e_os}. Continuing.",
                        exc_info=True,
                    )
            else:
                logger.warning(
                    f"Physical file not found at {physical_file_path} for file ID '{file_id}'."
                )

            # 2c. Delete UploadedFile DB record
            try:
                await file_service.delete_file_by_id(db, file_id=file_id)
                logger.info(f"Deleted UploadedFile DB record for file ID '{file_id}'.")
            except Exception as e_db_file:
                logger.error(
                    f"Error deleting UploadedFile DB record for '{file_id}': {e_db_file}. Continuing.",
                    exc_info=True,
                )

    except Exception as e_files:
        logger.error(
            f"Error during batch file deletion process for agent '{agent_id}': {e_files}. Aborting comprehensive delete.",
            exc_info=True,
        )
        return False

    # 3. Clear the agent's entire vector collection (as a safeguard)
    try:
        await vector_db_service.clear_agent_vector_collection(agent_id)
        logger.info(
            f"Successfully cleared entire vector collection for agent '{agent_id}'."
        )
    except Exception as e_collection:
        logger.error(
            f"Error clearing entire vector collection for agent '{agent_id}': {e_collection}. This might leave orphaned collections.",
            exc_info=True,
        )
        # Depending on strictness, could return False here.
        # For now, consider it non-critical if individual file vectors were attempted.

    # 4. Delete the agent record itself
    try:
        await db.agent.delete(where={"id": agent_id})
        logger.info(f"Successfully deleted agent record for ID '{agent_id}'.")
    except Exception as e_agent_delete:
        logger.error(
            f"CRITICAL: Failed to delete agent record for ID '{agent_id}': {e_agent_delete}. Manual cleanup required.",
            exc_info=True,
        )
        return False  # This is a critical failure

    logger.info(f"Comprehensive deletion for agent ID '{agent_id}' completed.")
    return True


async def update_agent(
    db: Prisma,
    agent_id: str,
    name: Optional[str] = None,
    model_config: Optional[Dict[str, Any]] = Ellipsis,
) -> Agent:
    """
    Updates an existing agent.

    Args:
        db: Prisma client instance.
        agent_id: ID of the agent to update.
        name: Optional new name for the agent.
        model_config: Optional new model configuration.
                      If provided as a dict, it's updated.
                      If provided as None, modelConfig is set to null in DB.
                      If not provided (Ellipsis), modelConfig is not changed.

    Returns:
        The updated Agent object.

    Raises:
        ValueError: If no update data is actually provided (all args are None/Ellipsis).
        Exception: If Prisma update fails (e.g., agent not found RecordNotFoundError, DB error).
    """
    update_data: Dict[str, Any] = {}
    if name is not None:
        update_data["name"] = name

    if model_config is not Ellipsis:  # model_config was explicitly passed
        if model_config is None:
            update_data["modelConfig"] = json.dumps(None)  # Set to JSON null
        else:
            # Ensure model_config is a dict before dumping, though type hint suggests it.
            if isinstance(model_config, dict):
                update_data["modelConfig"] = model_config_to_json_str(model_config)
            else:
                # This case should ideally be caught by Pydantic validation in the router
                logger.error(
                    f"model_config for agent '{agent_id}' was not a dict, but {type(model_config)}. Skipping update of modelConfig."
                )

    if not update_data:
        logger.warning(
            f"No actual update data provided for agent '{agent_id}'. Agent not modified."
        )
        # To prevent an error with empty data for Prisma update, either raise or return current agent.
        # Raising error is better to indicate no operation was performed as requested.
        raise ValueError("No fields to update were provided for the agent.")

    try:
        logger.info(f"Attempting to update agent '{agent_id}' with data: {update_data}")
        updated_agent_prisma = await db.agent.update(
            where={"id": agent_id}, data=update_data
        )
        if not updated_agent_prisma:
            # This case might occur if the record is deleted between the find_unique and update,
            # or if Prisma's update behaves this way for a non-existent record.
            logger.error(
                f"Prisma update call returned None for agent '{agent_id}', likely meaning it was not found or an issue occurred."
            )
            raise ValueError(
                f"Agent with ID '{agent_id}' not found during prisma update operation or no update performed."
            )
        logger.info(
            f"Agent '{agent_id}' updated successfully. Name: {updated_agent_prisma.name}"
        )
        return updated_agent_prisma
    except Exception as e:  # Catch other potential Prisma errors
        logger.error(f"Database error updating agent '{agent_id}': {e}", exc_info=True)
        # Re-raise a more generic error or a specific custom DB error
        raise ValueError(f"Failed to update agent '{agent_id}' in database. Error: {e}")
