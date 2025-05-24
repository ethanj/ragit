"""
This script seeds the database with an initial agent and ingests data from a sample CSV file.

It's designed to be run as a standalone script to quickly populate the system with
data for development and testing purposes.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to sys.path to allow importing app modules
import sys

project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from app.core.config import settings  # For OPENAI_API_KEY check, logging
from app.db import Prisma, close_db, connect_db
from app.services.agent_service import create_agent
from app.services.file_service import (
    create_uploaded_file,
)
from app.services.ingestion_orchestrator import (
    process_ingestion_from_router,
)
from app.services.vector_db_service import (
    clear_all_agent_vector_data_from_chroma,
)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_AGENT_NAME = "LangChain AI Assistant"
SEED_CSV_FILE_PATH = "data/langchain_com_pages_4.csv"

# Define the base path for uploads, consistent with file_router.py
# UPLOAD_DIR_BASE = Path(project_root_path) / "data" / "uploads" # This might be too generic for seed.py's needs now

# Define the structure for seed data from JSON
SEED_DATA_PATH = Path(__file__).parent.parent / "data" / "seed_data.json"
# Base directory for uploads, relative to the project root (ragit/)
# UPLOAD_DIR_BASE should align with how file_router stores them, e.g., data/uploads/
# The actual file_router stores them under data/uploads/<agent_id>/<filename_uuid_ext>
# For seeding, we place initial files directly under a seed_uploads folder within data/
SEED_UPLOAD_DIR = Path(__file__).parent.parent / "data" / "seed_uploads"


async def seed_database(db: Prisma):
    """Seeds the database with predefined agents and files using the provided db client."""
    logger.info("Starting database seeding with FULL CLEANUP...")

    if (
        not settings.OPENAI_API_KEY
        or settings.OPENAI_API_KEY == "your_openai_api_key_here"
    ):
        logger.warning(
            "OPENAI_API_KEY is not set or is set to the default placeholder."
            " File ingestion requiring embeddings will likely fail or use dummy embeddings."
            " Please set a valid OPENAI_API_KEY in your .env file for full functionality."
        )
        # Depending on strictness, could raise an error or exit here.
        # For now, we'll allow seeding to proceed, issues will arise during ingestion.

    # --- 1. Global ChromaDB Cleanup ---
    logger.info("Attempting to clear ALL vector data from ChromaDB globally...")
    try:
        await clear_all_agent_vector_data_from_chroma()
        logger.info(
            "Successfully cleared all agent-specific vector collections from ChromaDB."
        )
    except Exception as e:
        logger.error(
            f"Error during global ChromaDB data clearing: {e}. Some vector data may persist."
        )

    # --- 2. Comprehensive Database Cleanup ---
    logger.info("Attempting to clear all agent-related data from the database...")

    # 2a. Delete physical files associated with UploadedFile records
    logger.info(
        "Fetching all UploadedFile records to delete associated physical files..."
    )
    try:
        all_uploaded_files = await db.uploadedfile.find_many()
        if all_uploaded_files:
            logger.info(
                f"Found {len(all_uploaded_files)} uploaded files to process for physical deletion."
            )
            for file_record in all_uploaded_files:
                physical_file_path_str = file_record.filePath
                full_physical_path = (
                    Path(project_root_path) / "data" / physical_file_path_str
                )
                if full_physical_path.exists():
                    try:
                        os.remove(full_physical_path)
                        logger.info(f"Deleted physical file: {full_physical_path}")
                    except Exception as e_remove:
                        logger.error(
                            f"Error deleting physical file {full_physical_path}: {e_remove}"
                        )
                else:
                    logger.warning(
                        f"Physical file not found (already deleted or moved?): {full_physical_path}"
                    )
        else:
            logger.info(
                "No UploadedFile records found in the database to process for physical deletion."
            )
    except Exception as e:
        logger.error(
            f"Error fetching or processing UploadedFile records for physical file deletion: {e}"
        )

    # 2b. Delete database records in order: ChatMessage -> UploadedFile -> Agent
    try:
        logger.info("Deleting all ChatMessage records...")
        deleted_chat_count = await db.chatmessage.delete_many()
        logger.info(f"Deleted {deleted_chat_count} ChatMessage records.")

        logger.info("Deleting all UploadedFile records...")
        deleted_files_count = await db.uploadedfile.delete_many()
        logger.info(f"Deleted {deleted_files_count} UploadedFile records.")

        logger.info("Deleting all Agent records...")
        deleted_agents_count = await db.agent.delete_many()
        logger.info(f"Deleted {deleted_agents_count} Agent records.")

        logger.info("Successfully cleared all agent-related tables from the database.")
    except Exception as e:
        logger.error(
            f"Error during database table clearing: {e}. Some database records may persist."
        )

    # --- 3. Proceed with Seeding New Data ---
    logger.info("Proceeding to seed new data...")
    SEED_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open(SEED_DATA_PATH, "r") as f:
            seed_content = json.load(f)
    except FileNotFoundError:
        logger.error(f"Seed data file not found at {SEED_DATA_PATH}. Aborting seeding.")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {SEED_DATA_PATH}. Aborting seeding.")
        return

    agents_data: List[Dict] = seed_content.get("agents", [])

    for agent_info in agents_data:
        agent_name = agent_info.get("name")
        model_config = agent_info.get("modelConfig")
        files_to_seed = agent_info.get(
            "files", []
        )  # Renamed from 'files' to avoid conflict

        if not agent_name:
            logger.warning("Skipping agent with no name in seed data.")
            continue

        logger.info(f"Processing agent: {agent_name}")

        # Since we've cleared all agents, we always create new ones.
        # No need to check for existing_agent anymore for cleanup purposes here.
        try:
            agent_to_process = await create_agent(
                db, name=agent_name, model_config=model_config
            )
            logger.info(
                f"Agent '{agent_name}' created successfully with ID: {agent_to_process.id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to create agent '{agent_name}': {e}. Skipping files for this agent."
            )
            continue

        agent_id = agent_to_process.id

        # Prepare files for ingestion
        for file_detail in files_to_seed:  # Use renamed variable
            original_filename = file_detail.get("filename")
            # file_description = file_detail.get("description", "") # Description not used in create_uploaded_file

            if not original_filename:
                logger.warning(
                    f"Skipping file with no filename for agent '{agent_name}'."
                )
                continue

            source_file_path = SEED_UPLOAD_DIR / original_filename
            if not source_file_path.exists():
                logger.warning(
                    f"Seed file '{original_filename}' not found at {source_file_path} for agent '{agent_name}'. Skipping."
                )
                continue

            agent_upload_dir = (
                Path(settings.CHROMA_PERSIST_DIR).parent / "uploads" / agent_id
            )
            agent_upload_dir.mkdir(parents=True, exist_ok=True)
            target_disk_path = agent_upload_dir / original_filename

            try:
                with (
                    open(source_file_path, "rb") as src_f,
                    open(target_disk_path, "wb") as dest_f,
                ):
                    dest_f.write(src_f.read())
                logger.info(
                    f"Copied seed file '{original_filename}' to '{target_disk_path}' for agent '{agent_name}'."
                )
            except Exception as e:
                logger.error(
                    f"Failed to copy seed file '{original_filename}' to '{target_disk_path}': {e}. Skipping this file."
                )
                continue

            db_file_path = str(
                target_disk_path.relative_to(Path(settings.CHROMA_PERSIST_DIR).parent)
            )

            # No need to check for existing_file_record as we've cleared UploadedFile table
            file_id_to_ingest: Optional[str] = None
            try:
                new_file_record = await create_uploaded_file(
                    db=db,
                    file_name=original_filename,
                    file_path=db_file_path,
                    file_size=source_file_path.stat().st_size,
                    mime_type="text/csv",  # Assuming CSV, might need to make dynamic if seed files vary
                    agent_id=agent_id,
                )
                file_id_to_ingest = new_file_record.id
                logger.info(
                    f"Created UploadedFile record for '{original_filename}' (ID: {file_id_to_ingest}) for agent '{agent_id}'."
                )
            except Exception as e:
                logger.error(
                    f"Failed to create UploadedFile record for '{original_filename}': {e}. Skipping ingestion for this file."
                )
                if target_disk_path.exists():
                    try:
                        os.remove(target_disk_path)  # Clean up copied file
                    except Exception as e_remove_copy:
                        logger.error(
                            f"Error cleaning up copied file {target_disk_path} after DB error: {e_remove_copy}"
                        )
                continue

            if file_id_to_ingest:
                logger.info(
                    f"Starting ingestion for file: {original_filename} (ID: {file_id_to_ingest}) for agent {agent_name} (ID: {agent_id})"
                )
                try:
                    await process_ingestion_from_router(
                        db=db,
                        file_id=file_id_to_ingest,
                        agent_id=agent_id,
                        temp_file_path=target_disk_path,
                    )
                    logger.info(
                        f"Successfully processed and ingested file '{original_filename}' for agent '{agent_name}'."
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing file '{original_filename}' for agent '{agent_name}': {e}",
                        exc_info=True,
                    )
                    # If ingestion fails, we might want to clean up the created UploadedFile record and physical file.
                    # For now, leaving them as is, which indicates a failed ingestion attempt.

    logger.info("Database seeding completed.")


async def main():
    logger.info("Initializing database client for seeding...")
    db_client: Optional[Prisma] = None
    try:
        db_client = await connect_db()
        if db_client:
            await seed_database(db_client)
        else:
            logger.error(
                "Failed to get database client from connect_db. Seeding aborted."
            )
    except Exception as e:
        logger.error(
            f"An error occurred during the seeding process: {e}", exc_info=True
        )
    finally:
        if db_client:
            await close_db(db_client)
        else:
            await (
                close_db()
            )  # Attempt to close global client if specific one wasn't obtained
        logger.info("Seeding script finished and database client disconnected/checked.")


if __name__ == "__main__":
    # Ensure parent directory for script exists if running this directly and scripts/ doesn't exist
    # Though typically Makefile handles this context from project root.
    # Path("scripts").mkdir(exist_ok=True) # Not strictly needed if scripts/ already exists from git clone
    asyncio.run(main())
