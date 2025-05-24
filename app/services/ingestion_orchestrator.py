"""
Service layer for data ingestion pipeline.

This module will contain functions for:
- Loading and validating CSV files.
- Cleaning and preprocessing text data.
- Chunking text.
- Generating embeddings.
- Storing vectors and metadata.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from app.services.embedding_service import (
    EmbeddingError,
    embed_chunks,
)
from app.services.file_service import (
    create_uploaded_file,
    get_file_by_id_and_agent_id,
    update_uploaded_file_status,
)

# Import from new validation service
from app.services.file_validation_service import (
    CSVValidationError,
    load_and_validate_csv_from_path,
)

# Import from new text_processing_service
from app.services.text_processing_service import (
    chunk_data,
    clean_and_preprocess_data,
)

# Import from new vector_db_service
from app.services.vector_db_service import (
    VectorDBError,
    store_chunks_in_vector_db,
)
from prisma import Prisma
from prisma.models import UploadedFile as PrismaUploadedFile

# MAX_FILE_SIZE_MB = 50 # Moved to file_validation_service
# MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024 # Moved

# Common column names that might contain primary textual content
# POTENTIAL_CONTENT_COLUMNS = [...] # Moved

# Common column names for HTML content
# POTENTIAL_HTML_COLUMNS = [...] # Moved

logger = logging.getLogger(__name__)

# CSVValidationError class, _validate_csv_content_and_structure,
# load_and_validate_csv, and load_and_validate_csv_from_path functions were moved to file_validation_service.py


# Assuming tiktoken is used by SentenceSplitter by default for token counting with OpenAI models.
# If not, specific model name might be needed for tokenizer init.
# DEFAULT_CHUNK_SIZE = 1024 # Moved to settings
# DEFAULT_CHUNK_OVERLAP = 100 # Moved to settings


class UnsupportedFileTypeError(Exception):
    """Custom error for unsupported file types during ingestion."""

    pass


async def _handle_empty_or_unprocessed_data(
    db: Prisma,
    uploaded_file_id: str,
    data_identifier: str,
    reason_message: str,
    original_file_name: str,
):
    """Logs a warning and updates file status to COMPLETED with a note when data is empty."""
    logger.warning(
        f"{data_identifier} for '{original_file_name}' (ID: {uploaded_file_id}) was empty or resulted in no data. {reason_message}"
    )
    await update_uploaded_file_status(
        db,
        uploaded_file_id,
        "COMPLETED",
        notes=f"{data_identifier.capitalize()} empty. {reason_message}",
    )


async def _load_and_validate_data_by_type(
    file_path_str: str, file_content_type: str, original_file_name: str
) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """Loads and validates data based on file type.

    Returns:
        Tuple of (DataFrame, content_column_name, metadata_columns_dict)

    Raises:
        CSVValidationError: If CSV validation fails.
        UnsupportedFileTypeError: If the file type is not supported.
        IOError: If file reading fails for plain text.
    """
    if file_content_type == "text/csv":
        df, content_col, meta_cols = load_and_validate_csv_from_path(file_path_str)
        logger.info(
            f"CSV '{original_file_name}' validated. Content: '{content_col}', Metadata: {meta_cols}"
        )
        return df, content_col, meta_cols
    elif file_content_type == "text/plain":
        try:
            with open(file_path_str, "r", encoding="utf-8") as f_txt:
                text_content = f_txt.read()
            df = pd.DataFrame([{"text": text_content}])
            content_col = "text"
            meta_cols = {}
            logger.info(
                f"Read plain text file '{original_file_name}'. Length: {len(text_content)}"
            )
            return df, content_col, meta_cols
        except IOError as e:
            logger.error(f"IOError reading plain text file '{original_file_name}': {e}")
            raise
    else:
        logger.error(
            f"Unsupported file type: {file_content_type} for '{original_file_name}'."
        )
        raise UnsupportedFileTypeError(f"Unsupported file type: {file_content_type}")


async def process_file_for_agent(
    db: Prisma,
    agent_id: str,
    file_path: str,  # Local path to the already uploaded file
    original_file_name: str,
    file_content_type: str,
    file_size_bytes: int,
) -> None:
    """
    Full pipeline for processing a single file for an agent:
    1. Create UploadedFile record in DB.
    2. Load and validate (currently CSV specific, extend for others).
    3. Clean and preprocess data.
    4. Chunk data.
    5. Embed chunks.
    6. Store chunks in vector DB.
    7. Update UploadedFile record status.
    """
    uploaded_file_record: PrismaUploadedFile | None = None
    try:
        logger.info(
            f"Starting processing for file '{original_file_name}' for agent {agent_id}."
        )
        # 1. Create initial UploadedFile record
        uploaded_file_record = await create_uploaded_file(
            db=db,
            agent_id=agent_id,
            file_name=original_file_name,
            file_path=file_path,  # Store relative path from perspective of upload dir
            content_type=file_content_type,
            size_bytes=file_size_bytes,
            status="PENDING",
        )
        logger.info(f"Created UploadedFile record ID: {uploaded_file_record.id}")

        df, content_col, meta_cols = await _load_and_validate_data_by_type(
            file_path, file_content_type, original_file_name
        )

        # 3. Clean and preprocess
        cleaned_df = clean_and_preprocess_data(df, content_col, meta_cols)
        if cleaned_df.empty:
            await _handle_empty_or_unprocessed_data(
                db,
                uploaded_file_record.id,
                "Cleaned data",
                "No processable content found after cleaning.",
                original_file_name,
            )
            return
        logger.info(
            f"Data cleaned for '{original_file_name}'. Shape: {cleaned_df.shape}"
        )

        # 4. Chunk data
        chunks = chunk_data(cleaned_df, uploaded_file_record.id, meta_cols)
        if not chunks:
            await _handle_empty_or_unprocessed_data(
                db,
                uploaded_file_record.id,
                "Chunks",
                "No chunks generated from content.",
                original_file_name,
            )
            return
        logger.info(f"Created {len(chunks)} chunks for '{original_file_name}'.")

        # 5. Embed chunks
        chunks_with_embeddings = embed_chunks(chunks)
        logger.info(
            f"Embeddings generated for {len(chunks_with_embeddings)} chunks for '{original_file_name}'."
        )

        # 6. Store chunks in vector DB
        store_chunks_in_vector_db(agent_id, chunks_with_embeddings)
        logger.info(
            f"Chunks stored in vector DB for agent {agent_id} from file '{original_file_name}'."
        )

        # 7. Update UploadedFile record status to COMPLETED
        await update_uploaded_file_status(db, uploaded_file_record.id, "COMPLETED")
        logger.info(
            f"Successfully processed file '{original_file_name}' for agent {agent_id}. Status: COMPLETED."
        )

    except (
        CSVValidationError,
        UnsupportedFileTypeError,
        EmbeddingError,
        VectorDBError,
        IOError,
    ) as e:
        error_type_name = type(e).__name__
        error_detail = str(e.detail) if hasattr(e, "detail") else str(e)
        logger.error(
            f"{error_type_name} for '{original_file_name}' (ID: {uploaded_file_record.id if uploaded_file_record else 'N/A'}) for agent {agent_id}: {error_detail}",
            exc_info=True,
        )
        if uploaded_file_record:
            await update_uploaded_file_status(
                db,
                uploaded_file_record.id,
                "FAILED",
                error_message=f"{error_type_name}: {error_detail}",
            )
    except Exception as e:
        logger.error(
            f"Unexpected error processing file '{original_file_name}' (ID: {uploaded_file_record.id if uploaded_file_record else 'N/A'}) for agent {agent_id}: {e}",
            exc_info=True,
        )
        if uploaded_file_record:
            await update_uploaded_file_status(
                db,
                uploaded_file_record.id,
                "FAILED",
                error_message=f"Unexpected error: {str(e)}",
            )


async def process_ingestion_from_router(
    db: Prisma,
    file_id: str,  # ID of the existing UploadedFile record
    agent_id: str,  # Agent ID passed for context
    temp_file_path: Path,  # Path to the physically stored temporary file
):
    """
    Processes an already uploaded and recorded file for ingestion.
    Fetches the UploadedFile record, then proceeds with validation, cleaning, chunking, embedding, and storage.
    """
    uploaded_file_record: PrismaUploadedFile | None = None
    try:
        logger.info(
            f"Starting ingestion from router for file ID '{file_id}' for agent '{agent_id}'."
        )

        # Get the existing UploadedFile record, ensuring it belongs to the agent
        uploaded_file_record = await get_file_by_id_and_agent_id(
            db, file_id=file_id, agent_id=agent_id
        )
        if not uploaded_file_record:
            logger.error(
                f"UploadedFile record ID '{file_id}' not found for agent '{agent_id}'. Cannot process ingestion."
            )
            # No status update here as the record might not exist or belong to this agent.
            # The router initiated this, so it should handle the 404 if agent/file is invalid.
            return

        # Update status to 'processing'
        await update_uploaded_file_status(db, file_id, "PROCESSING")

        original_file_name = uploaded_file_record.fileName
        file_content_type = uploaded_file_record.mimeType

        df, content_col, meta_cols = await _load_and_validate_data_by_type(
            str(temp_file_path), file_content_type, original_file_name
        )

        # 3. Clean and preprocess
        cleaned_df = clean_and_preprocess_data(df, content_col, meta_cols)
        if cleaned_df.empty:
            await _handle_empty_or_unprocessed_data(
                db,
                file_id,
                "Cleaned data",
                "No processable content found after cleaning.",
                original_file_name,
            )
            return
        logger.info(
            f"Ingestion from router: Data cleaned for '{original_file_name}' (ID: {file_id}). Shape: {cleaned_df.shape}"
        )

        # 4. Chunk data
        chunks = chunk_data(cleaned_df, file_id, meta_cols)
        if not chunks:
            await _handle_empty_or_unprocessed_data(
                db,
                file_id,
                "Chunks",
                "No chunks generated from content.",
                original_file_name,
            )
            return
        logger.info(
            f"Ingestion from router: Created {len(chunks)} chunks for '{original_file_name}' (ID: {file_id})."
        )

        # 5. Embed chunks
        chunks_with_embeddings = embed_chunks(chunks)
        logger.info(
            f"Ingestion from router: Embeddings generated for {len(chunks_with_embeddings)} chunks for '{original_file_name}' (ID: {file_id})."
        )

        # 6. Store chunks in vector DB
        store_chunks_in_vector_db(
            agent_id, chunks_with_embeddings
        )  # agent_id is passed correctly
        logger.info(
            f"Ingestion from router: Chunks stored in vector DB for agent {agent_id} from file '{original_file_name}' (ID: {file_id})."
        )

        # 7. Update UploadedFile record status to COMPLETED
        await update_uploaded_file_status(db, file_id, "COMPLETED")
        logger.info(
            f"Ingestion from router: Successfully processed file '{original_file_name}' (ID: {file_id}) for agent {agent_id}. Status: COMPLETED."
        )

    except (
        CSVValidationError,
        UnsupportedFileTypeError,
        EmbeddingError,
        VectorDBError,
        IOError,
    ) as e:
        error_type_name = type(e).__name__
        error_detail = str(e.detail) if hasattr(e, "detail") else str(e)
        logger.error(
            f"Ingestion from router: {error_type_name} for '{original_file_name}' (ID: {file_id}): {error_detail}",
            exc_info=True,
        )
        await update_uploaded_file_status(
            db, file_id, "FAILED", error_message=f"{error_type_name}: {error_detail}"
        )
    except Exception as e:
        logger.error(
            f"Ingestion from router: Unexpected error for '{original_file_name}' (ID: {file_id}): {e}",
            exc_info=True,
        )
        await update_uploaded_file_status(
            db, file_id, "FAILED", error_message=f"Unexpected error: {str(e)}"
        )


# End of process_file_for_agent, the functions below are removed.
