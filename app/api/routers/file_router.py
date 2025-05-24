"""
API Router for File Management under an Agent.
"""

import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from pydantic import BaseModel

from app.db import get_db
from app.services import file_service, vector_db_service
from app.services import ingestion_orchestrator as ingestion_service
from prisma import Prisma
from prisma.models import UploadedFile as PrismaUploadedFile

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agents/{agent_id}/files", tags=["Files"])

UPLOAD_DIR_BASE = Path("data/uploads")

# Ensure base upload directory exists
UPLOAD_DIR_BASE.mkdir(parents=True, exist_ok=True)


# Pydantic Models for File API
class UploadedFileResponse(BaseModel):
    id: str
    fileName: str
    filePath: str
    fileSize: int
    mimeType: Optional[str]
    status: str
    error: Optional[str]
    createdAt: datetime
    updatedAt: datetime
    agentId: str

    class Config:
        from_attributes = True


# Helper function to save uploaded file to disk
async def _save_upload_file_to_disk(file: UploadFile, destination_path: Path) -> int:
    """Saves an UploadFile to the specified destination_path. Returns file size."""
    try:
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File '{file.filename}' saved to path: {destination_path}")
        return destination_path.stat().st_size
    except Exception as e:
        logger.error(
            f"Failed to save uploaded file '{file.filename}' to {destination_path}: {e}",
            exc_info=True,
        )
        # If saving fails, attempt to remove partial file if it exists
        if destination_path.exists():
            try:
                os.remove(destination_path)
            except OSError as e_os:
                logger.error(
                    f"Failed to remove partially saved file {destination_path}: {e_os}"
                )
        raise  # Re-raise the original exception to be caught by the endpoint
    finally:
        await file.close()  # Ensure the UploadFile is closed


async def run_ingestion_pipeline(
    db: Prisma, file_id: str, agent_id: str, temp_file_path: Path
):
    """The actual ingestion pipeline logic to be run in the background.
    This now delegates to ingestion_service.process_ingestion_from_router.
    """
    try:
        logger.info(
            f"Background task: Starting ingestion for file ID {file_id}, agent ID {agent_id} using new service orchestrator."
        )
        # No need to update status here, the service function will do it.
        # await file_service.update_uploaded_file_status(db, file_id, "processing") # Done by service

        await ingestion_service.process_ingestion_from_router(
            db=db, file_id=file_id, agent_id=agent_id, temp_file_path=temp_file_path
        )

        # Status updates (COMPLETED/FAILED) are handled within process_ingestion_from_router
        logger.info(
            f"Background task: Ingestion process completed (or failed and handled) by service for file ID {file_id}."
        )

    except (
        Exception
    ) as e:  # General catch-all, though service should handle most specifics
        logger.error(
            f"Background task: Unexpected error directly in run_ingestion_pipeline for file {file_id}: {e}",
            exc_info=True,
        )
        # Attempt to mark as failed if not already, as a last resort.
        # The service function should have already tried to do this.
        try:
            await file_service.update_uploaded_file_status(
                db, file_id, "FAILED", f"Router-level error: {str(e)}"
            )
        except Exception as db_update_err:
            logger.error(
                f"Background task: Failed to update status to FAILED after router-level error for {file_id}: {db_update_err}"
            )

    finally:
        # Clean up the temporary file
        if temp_file_path.exists():
            try:
                os.remove(temp_file_path)
                logger.info(
                    f"Background task: Cleaned up temporary file: {temp_file_path}"
                )
            except OSError as e_os:
                logger.error(
                    f"Background task: Error deleting temporary file {temp_file_path}: {e_os}"
                )


@router.post(
    "/", response_model=UploadedFileResponse, status_code=status.HTTP_202_ACCEPTED
)
async def upload_file_for_agent(
    agent_id: str,
    background_tasks: BackgroundTasks,
    db: Prisma = Depends(get_db),
    file: UploadFile = File(...),
) -> UploadedFileResponse:
    """
    Upload a file for an agent. The file will be processed in the background.
    Currently only accepts CSV files as per ingestion service.
    """
    agent = await db.agent.find_unique(where={"id": agent_id})
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID '{agent_id}' not found.",
        )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File name cannot be empty.",
        )

    # Basic validation (file extension)
    # TODO: More robust content type validation if possible beyond filename
    allowed_extensions = (
        ".csv",
        ".txt",
    )  # Extend as more types are supported by ingestion
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Accepted types: {', '.join(allowed_extensions)} based on filename.",
        )

    safe_original_filename = Path(file.filename).name

    # --- BEGIN MODIFICATION: Check for and delete existing file with the same name ---
    try:
        existing_file_record = await file_service.get_uploaded_file_by_name(
            db, safe_original_filename, agent_id
        )
        if existing_file_record:
            logger.info(
                f"File '{safe_original_filename}' already exists for agent '{agent_id}' (ID: {existing_file_record.id}). Replacing it."
            )

            # 1. Clear vector embeddings
            try:
                await vector_db_service.clear_vectors_for_file(
                    agent_id=agent_id, file_id=existing_file_record.id
                )
                logger.info(
                    f"Successfully cleared vector embeddings for old file ID {existing_file_record.id}."
                )
            except Exception as e_vec:
                logger.error(
                    f"Error clearing vector embeddings for old file ID {existing_file_record.id}: {e_vec}. Proceeding with replacement.",
                    exc_info=True,
                )

            # 2. Delete physical file
            # filePath is stored relative to UPLOAD_DIR_BASE.parent (i.e., 'data/')
            # e.g., 'uploads/agent_id_xyz/file_abc.csv'
            existing_physical_file_path = (
                UPLOAD_DIR_BASE.parent / existing_file_record.filePath
            )
            if existing_physical_file_path.exists():
                try:
                    os.remove(existing_physical_file_path)
                    logger.info(
                        f"Successfully deleted old physical file: {existing_physical_file_path}"
                    )
                except OSError as e_os:
                    logger.error(
                        f"Error deleting old physical file {existing_physical_file_path}: {e_os}. Proceeding with replacement.",
                        exc_info=True,
                    )
            else:
                logger.warning(
                    f"Old physical file not found at {existing_physical_file_path}, might have been already deleted."
                )

            # 3. Delete old database record
            try:
                await file_service.delete_file_by_id(db, existing_file_record.id)
                logger.info(
                    f"Successfully deleted old database record for file ID {existing_file_record.id}."
                )
            except Exception as e_db_del:
                logger.error(
                    f"Error deleting old database record for file ID {existing_file_record.id}: {e_db_del}. Proceeding with replacement.",
                    exc_info=True,
                )
    except Exception as e_check:
        logger.error(
            f"Error during pre-check for existing file '{safe_original_filename}' for agent '{agent_id}': {e_check}. Proceeding with new upload attempt.",
            exc_info=True,
        )
    # --- END MODIFICATION ---

    file_uuid = uuid.uuid4()
    agent_upload_dir = UPLOAD_DIR_BASE / agent_id
    agent_upload_dir.mkdir(parents=True, exist_ok=True)
    stored_filename = f"{file_uuid}_{safe_original_filename}"
    disk_save_path = agent_upload_dir / stored_filename

    try:
        file_size = await _save_upload_file_to_disk(file, disk_save_path)
    except Exception as e:  # Catch exceptions from _save_upload_file_to_disk
        # The helper already logged the specific saving error.
        # disk_save_path cleanup is attempted in the helper.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save uploaded file: {str(e)}",  # Provide some context from the original error
        )

    # Path for DB: relative to project root (parent of UPLOAD_DIR_BASE)
    # UPLOAD_DIR_BASE = data/uploads. UPLOAD_DIR_BASE.parent = data.
    # We want to store a path like "data/uploads/agent_id/file_name" to be independent of project root location if DB is moved.
    # So, relative to the parent of UPLOAD_DIR_BASE (which is 'data/' in this case) seems incorrect.
    # Path(disk_save_path).relative_to(UPLOAD_DIR_BASE.parent.parent) would be relative to project root.
    # Or, more simply, store the path from UPLOAD_DIR_BASE onwards: f"{agent_id}/{stored_filename}"
    # and prepend UPLOAD_DIR_BASE when reconstructing full path.
    # For now, let's simplify the stored path to be relative to UPLOAD_DIR_BASE itself for clarity.
    # db_file_path_str = str(disk_save_path.relative_to(UPLOAD_DIR_BASE))
    # However, the original code stored it relative to UPLOAD_DIR_BASE.parent which is data/
    # Let's keep the original logic for path storage to minimize unintended changes for now.
    # Original: temp_file_path.relative_to(UPLOAD_DIR_BASE.parent)
    # This becomes: disk_save_path.relative_to(UPLOAD_DIR_BASE.parent)
    # If UPLOAD_DIR_BASE = /abs/path/to/ragit/data/uploads
    # UPLOAD_DIR_BASE.parent = /abs/path/to/ragit/data
    # disk_save_path = /abs/path/to/ragit/data/uploads/agent_id/filename
    # relative_to(UPLOAD_DIR_BASE.parent) -> uploads/agent_id/filename
    # This seems reasonable for DB storage. The file_service then needs to reconstruct it with UPLOAD_DIR_BASE.parent

    db_relative_path = disk_save_path.relative_to(UPLOAD_DIR_BASE.parent)

    try:
        uploaded_file_db_record = await file_service.create_uploaded_file(
            db=db,
            agent_id=agent_id,
            file_name=safe_original_filename,
            file_path=str(db_relative_path),  # Store relative path
            file_size=file_size,
            mime_type=file.content_type,
        )
        logger.info(
            f"Created UploadedFile DB record ID: {uploaded_file_db_record.id} for file '{file.filename}'"
        )
    except Exception as e:
        logger.error(
            f"Failed to create DB record for file '{file.filename}': {e}", exc_info=True
        )
        if disk_save_path.exists():  # Clean up saved file if DB record fails
            os.remove(disk_save_path)
            logger.info(
                f"Cleaned up disk file {disk_save_path} after DB record creation failure."
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create file record in database.",
        )

    background_tasks.add_task(
        run_ingestion_pipeline, db, uploaded_file_db_record.id, agent_id, disk_save_path
    )

    return UploadedFileResponse.from_orm(uploaded_file_db_record)


@router.get("/", response_model=List[UploadedFileResponse])
async def list_files_for_agent(
    agent_id: str, db: Prisma = Depends(get_db)
) -> List[UploadedFileResponse]:
    """
    List all files uploaded for a specific agent.
    """
    # Check if agent exists first (optional, but good practice)
    agent = await db.agent.find_unique(where={"id": agent_id})
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID '{agent_id}' not found.",
        )

    try:
        files = await db.uploadedfile.find_many(
            where={"agentId": agent_id},
            order={"createdAt": "desc"},  # Show newest first
        )
        logger.info(f"Retrieved {len(files)} files for agent ID {agent_id}")
        return [UploadedFileResponse.model_validate(f) for f in files]
    except Exception as e:
        logger.error(
            f"Error listing files for agent ID '{agent_id}': {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list files for agent: {str(e)}",
        )


@router.get("/{file_id}", response_model=UploadedFileResponse)
async def get_file_details_for_agent(
    agent_id: str, file_id: str, db: Prisma = Depends(get_db)
) -> UploadedFileResponse:
    """
    Get details (including status) for a specific file belonging to an agent.
    """
    try:
        file_record = await db.uploadedfile.find_first(
            where={"id": file_id, "agentId": agent_id}
        )
        if not file_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File with ID '{file_id}' not found for agent '{agent_id}'.",
            )
        logger.info(f"Retrieved details for file ID {file_id} for agent ID {agent_id}")
        return UploadedFileResponse.model_validate(file_record)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error retrieving file ID '{file_id}' for agent '{agent_id}': {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve file details: {str(e)}",
        )


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file_for_agent(
    agent_id: str, file_id: str, db: Prisma = Depends(get_db)
) -> None:
    """Deletes a specific file for an agent, including its DB record, physical file, and associated vector embeddings."""
    # 1. Fetch the file record to ensure it exists and belongs to the agent
    file_record: Optional[
        PrismaUploadedFile
    ] = await file_service.get_file_by_id_and_agent_id(
        db, file_id=file_id, agent_id=agent_id
    )
    if not file_record:
        logger.warning(
            f"File with ID '{file_id}' not found for agent '{agent_id}'. Cannot delete."
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File with ID '{file_id}' not found for agent '{agent_id}'.",
        )

    # 2. Attempt to delete associated vector embeddings
    try:
        logger.info(
            f"Attempting to delete vector embeddings for file ID '{file_id}' of agent '{agent_id}'."
        )
        await vector_db_service.clear_vectors_for_file(
            agent_id=agent_id,
            file_id=file_id,  # file_id is the original_document_id
        )
        logger.info(
            f"Successfully initiated deletion of vector embeddings for file ID '{file_id}'."
        )
    except vector_db_service.VectorDBError as e:
        # Log the error but proceed with deleting the file record and physical file
        # This is a critical operation; if vectors fail to delete, we should still allow file deletion
        # but ensure this is logged prominently.
        logger.error(
            f"VectorDBError while deleting embeddings for file '{file_id}', agent '{agent_id}': {e}. Proceeding with file deletion.",
            exc_info=True,
        )
    except ValueError as e:  # From clear_vectors_for_file if IDs are empty
        logger.error(
            f"ValueError (likely empty agent_id/file_id) calling clear_vectors_for_file for file '{file_id}': {e}. Proceeding.",
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            f"Unexpected error deleting vector embeddings for file '{file_id}', agent '{agent_id}': {e}. Proceeding with file deletion.",
            exc_info=True,
        )

    # 3. Delete the physical file
    # Construct the full path from the stored relative path
    # Path stored in DB is relative to UPLOAD_DIR_BASE.parent (e.g. 'data/')
    # Example: db_file_path = "uploads/agent_id_xyz/uuid_filename.csv"
    # UPLOAD_DIR_BASE.parent = Path("data")
    # physical_file_path = Path("data") / "uploads/agent_id_xyz/uuid_filename.csv"
    physical_file_path = UPLOAD_DIR_BASE.parent / file_record.filePath
    if physical_file_path.exists():
        try:
            os.remove(physical_file_path)
            logger.info(f"Successfully deleted physical file: {physical_file_path}")
        except OSError as e:
            # Log error but proceed to delete DB record. Orphaned file is less critical than orphaned DB record.
            logger.error(
                f"Error deleting physical file {physical_file_path}: {e}. Proceeding with DB record deletion.",
                exc_info=True,
            )
    else:
        logger.warning(
            f"Physical file not found at {physical_file_path} during deletion for file record '{file_id}'."
        )

    # 4. Delete the UploadedFile record from the database
    try:
        await file_service.delete_file_by_id(db, file_id=file_id)
        logger.info(f"Successfully deleted UploadedFile DB record for ID '{file_id}'.")
    except Exception as e:
        # This is more critical. If DB deletion fails, we have an issue.
        logger.error(
            f"Failed to delete UploadedFile DB record for '{file_id}' after physical/vector deletion attempts: {e}",
            exc_info=True,
        )
        # Raise an HTTP exception because the primary record deletion failed.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file record from database for ID '{file_id}'. Manual cleanup may be required.",
        )

    return  # For 204 No Content
