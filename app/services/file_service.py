"""
Service layer for managing UploadedFile records in the database.
"""

import logging
from typing import List, Optional

from prisma import Prisma
from prisma.models import UploadedFile as PrismaUploadedFile
from prisma.types import UploadedFileCreateInput, UploadedFileUpdateInput

logger = logging.getLogger(__name__)


async def create_uploaded_file(
    db: Prisma,
    file_name: str,
    file_path: str,
    file_size: int,
    mime_type: Optional[str],
    agent_id: str,
) -> PrismaUploadedFile:
    """
    Creates a new UploadedFile record in the database.

    Args:
        db: The Prisma client instance.
        file_name: Original name of the file.
        file_path: Storage path of the file.
        file_size: Size of the file in bytes.
        mime_type: Mime type of the file.
        agent_id: ID of the agent this file belongs to.

    Returns:
        The created UploadedFile record.

    Raises:
        Exception: If database operation fails.
    """
    try:
        file_create_input = UploadedFileCreateInput(
            fileName=file_name,
            filePath=file_path,
            fileSize=file_size,
            mimeType=mime_type,
            agentId=agent_id,
            status="pending",  # Initial status
        )
        uploaded_file = await db.uploadedfile.create(data=file_create_input)
        logger.info(
            f"Created UploadedFile record with ID: {uploaded_file.id} for agent {agent_id}"
        )
        return uploaded_file
    except Exception as e:
        logger.error(f"Error creating UploadedFile record for {file_name}: {e}")
        raise  # Re-raise to be handled by the caller


async def update_uploaded_file_status(
    db: Prisma,
    file_id: str,
    status: str,
    error_message: Optional[str] = None,
) -> Optional[PrismaUploadedFile]:
    """
    Updates the status and optionally an error message for an UploadedFile record.

    Args:
        db: The Prisma client instance.
        file_id: The ID of the UploadedFile record to update.
        status: The new status (e.g., 'processing', 'completed', 'failed').
        error_message: An error message if the status is 'failed'.

    Returns:
        The updated UploadedFile record, or None if the file was not found.

    Raises:
        Exception: If database operation fails.
    """
    try:
        update_data = UploadedFileUpdateInput(status=status)
        if error_message is not None:
            update_data["error"] = error_message

        updated_file = await db.uploadedfile.update(
            where={"id": file_id},
            data=update_data,
        )
        if updated_file:
            logger.info(f"Updated status of UploadedFile ID {file_id} to {status}")
        else:
            logger.warning(f"UploadedFile ID {file_id} not found for status update.")
        return updated_file
    except Exception as e:
        logger.error(f"Error updating status for UploadedFile ID {file_id}: {e}")
        raise  # Re-raise to be handled by the caller


async def get_uploaded_file_by_name(
    db: Prisma, file_name: str, agent_id: str
) -> Optional[PrismaUploadedFile]:
    """
    Retrieves an UploadedFile record by its name and agent ID.

    Args:
        db: The Prisma client.
        file_name: The name of the file.
        agent_id: The ID of the agent.

    Returns:
        The UploadedFile record if found, else None.
    """
    try:
        file_record = await db.uploadedfile.find_first(
            where={"fileName": file_name, "agentId": agent_id}
        )
        return file_record
    except Exception as e:
        logger.error(
            f"Error fetching uploaded file by name '{file_name}' for agent '{agent_id}': {e}"
        )
        return None


async def get_files_by_agent_id(db: Prisma, agent_id: str) -> List[PrismaUploadedFile]:
    """Retrieves all files associated with a given agent_id."""
    try:
        files = await db.uploadedfile.find_many(
            where={"agentId": agent_id}, order={"createdAt": "desc"}
        )
        logger.info(f"Found {len(files)} files for agent_id '{agent_id}'.")
        return files
    except Exception as e:
        logger.error(
            f"Error retrieving files for agent_id '{agent_id}': {e}", exc_info=True
        )
        raise


async def get_file_by_id_and_agent_id(
    db: Prisma, file_id: str, agent_id: str
) -> Optional[PrismaUploadedFile]:
    """Retrieves a specific file by its ID, ensuring it belongs to the agent."""
    try:
        file_record = await db.uploadedfile.find_first(
            where={"id": file_id, "agentId": agent_id}
        )
        if file_record:
            logger.info(f"Found file '{file_id}' for agent '{agent_id}'.")
        else:
            logger.info(
                f"File '{file_id}' not found or not owned by agent '{agent_id}'."
            )
        return file_record
    except Exception as e:
        logger.error(
            f"Error retrieving file '{file_id}' for agent '{agent_id}': {e}",
            exc_info=True,
        )
        raise


async def get_uploaded_file_by_path_and_agent_id(
    db: Prisma, file_path: str, agent_id: str
) -> Optional[PrismaUploadedFile]:
    """
    Retrieves an UploadedFile record by its full path and agent ID.

    Args:
        db: The Prisma client.
        file_path: The storage path of the file.
        agent_id: The ID of the agent.

    Returns:
        The UploadedFile record if found, else None.
    """
    try:
        file_record = await db.uploadedfile.find_first(
            where={"filePath": file_path, "agentId": agent_id}
        )
        if file_record:
            logger.info(f"Found file with path '{file_path}' for agent '{agent_id}'.")
        else:
            logger.info(
                f"File with path '{file_path}' not found for agent '{agent_id}'."
            )
        return file_record
    except Exception as e:
        logger.error(
            f"Error fetching uploaded file by path '{file_path}' for agent '{agent_id}': {e}"
        )
        return None


async def delete_file_by_id(db: Prisma, file_id: str) -> Optional[PrismaUploadedFile]:
    """Deletes a file record by its ID.

    Returns:
        The deleted file record if successful.
    Raises:
        Exception: If deletion fails (e.g., Prisma's RecordNotFoundError if not found).
    """
    try:
        # Prisma's delete will raise an exception if the record is not found.
        # This is often a `prisma.errors.RecordNotFoundError`.
        deleted_file = await db.uploadedfile.delete(where={"id": file_id})
        logger.info(f"Deleted UploadedFile record with ID '{file_id}'.")
        return deleted_file
    except Exception as e:
        logger.error(
            f"Error deleting UploadedFile record '{file_id}': {e}", exc_info=True
        )
        raise
