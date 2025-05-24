"""
Service for interacting with the Vector Database (ChromaDB).
"""

import functools
import logging
import os

import chromadb

from app.core.config import settings

# Embedding related imports are not directly needed here unless we re-introduce
# OpenAIEmbeddingFunction for ChromaDB, which we are not currently doing as embeddings are pre-computed.

logger = logging.getLogger(__name__)


class VectorDBError(Exception):
    """Custom error for vector DB operations."""

    pass


@functools.lru_cache(maxsize=None)
def get_chroma_client():
    """Initializes and returns a ChromaDB persistent client."""
    try:
        if not os.path.exists(settings.CHROMA_PERSIST_DIR):
            os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
            logger.info(
                f"Created ChromaDB persistence directory: {settings.CHROMA_PERSIST_DIR}"
            )

        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        logger.info(
            f"ChromaDB PersistentClient initialized from path: {settings.CHROMA_PERSIST_DIR}"
        )
        return client
    except ImportError:
        logger.error(
            "ChromaDB library not installed. Please install it with `pip install chromadb`."
        )
        raise VectorDBError("ChromaDB library not installed.")
    except chromadb.errors.ChromaError as e:  # Catch ChromaDB specific errors
        logger.error(
            f"Failed to initialize ChromaDB client (ChromaError): {e}", exc_info=True
        )
        raise VectorDBError(f"ChromaDB client initialization failed: {e}")
    except Exception as e:  # General fallback for other unexpected errors
        logger.error(
            f"Failed to initialize ChromaDB client (Unexpected Error): {e}",
            exc_info=True,
        )
        raise VectorDBError(
            f"ChromaDB client initialization failed with unexpected error: {e}"
        )


def store_chunks_in_vector_db(agent_id: str, chunks_with_embeddings: list[dict]):
    """
    Stores chunks with their embeddings in ChromaDB, associated with an agent_id.

    Args:
        agent_id: The ID of the agent to associate the chunks with.
        chunks_with_embeddings: A list of chunk dictionaries, each must have
                                'chunk_id', 'text', 'embedding', and 'metadata'.

    Raises:
        VectorDBError: If storing chunks fails.
    """
    if not chunks_with_embeddings:
        logger.info(f"No chunks provided to store for agent {agent_id}.")
        return

    chroma_client = get_chroma_client()
    collection_name = f"agent_{agent_id}_vectors"

    try:
        collection = chroma_client.get_or_create_collection(name=collection_name)
        logger.info(f"Accessed/Created ChromaDB collection: {collection_name}")
    except chromadb.errors.ChromaError as e:  # Catch ChromaDB specific errors
        logger.error(
            f"Failed to get or create ChromaDB collection '{collection_name}' (ChromaError): {e}",
            exc_info=True,
        )
        raise VectorDBError(f"ChromaDB collection operation failed: {e}")
    except Exception as e:  # General fallback
        logger.error(
            f"Failed to get or create ChromaDB collection '{collection_name}' (Unexpected Error): {e}",
            exc_info=True,
        )
        raise VectorDBError(
            f"ChromaDB collection operation failed with unexpected error: {e}"
        )

    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for chunk in chunks_with_embeddings:
        if not all(k in chunk for k in ["chunk_id", "text", "embedding", "metadata"]):
            logger.warning(
                f"Skipping chunk due to missing required fields: {chunk.get('chunk_id', 'UNKNOWN_ID')}"
            )
            continue

        ids.append(chunk["chunk_id"])
        embeddings.append(chunk["embedding"])
        current_metadata = chunk["metadata"].copy()
        metadatas.append(current_metadata)
        documents.append(chunk["text"])

    if not ids:
        logger.info(f"No valid chunks to store after filtering for agent {agent_id}.")
        return

    try:
        collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )
        logger.info(
            f"Successfully upserted {len(ids)} chunks into '{collection_name}'."
        )
    except chromadb.errors.ChromaError as e:  # Catch ChromaDB specific errors
        logger.error(
            f"Error upserting chunks to ChromaDB collection '{collection_name}' (ChromaError): {e}",
            exc_info=True,
        )
        raise VectorDBError(f"ChromaDB upsert failed: {e}")
    except Exception as e:  # General fallback
        logger.error(
            f"Error upserting chunks to ChromaDB collection '{collection_name}' (Unexpected Error): {e}",
            exc_info=True,
        )
        raise VectorDBError(f"ChromaDB upsert failed with unexpected error: {e}")


async def clear_vectors_for_file(agent_id: str, file_id: str):
    """Clears vectors associated with a specific file_id for an agent from ChromaDB.

    Args:
        agent_id: The ID of the agent.
        file_id: The ID of the file (corresponds to 'original_document_id' in chunk metadata).

    Raises:
        VectorDBError: If deleting vectors fails.
        ValueError: If agent_id or file_id is empty.
    """
    if not agent_id:
        logger.error("Agent ID is required to clear vectors for a file.")
        raise ValueError("Agent ID cannot be empty.")
    if not file_id:
        logger.error("File ID is required to clear vectors for a file.")
        raise ValueError("File ID cannot be empty.")

    chroma_client = get_chroma_client()
    collection_name = f"agent_{agent_id}_vectors"

    try:
        existing_collection_names = chroma_client.list_collections()
        if collection_name in existing_collection_names:
            collection = chroma_client.get_collection(name=collection_name)
            # Delete vectors where metadata field 'original_document_id' matches file_id
            collection.delete(where={"original_document_id": file_id})
            logger.info(
                f"Attempted to delete vectors for file_id '{file_id}' from collection '{collection_name}'. "
                f"ChromaDB's delete operation does not return count of deleted items directly through this API."
            )
        else:
            logger.warning(
                f"Collection '{collection_name}' not found when trying to delete vectors for file_id '{file_id}'. No action taken."
            )
    except chromadb.errors.ChromaError as e:  # Catch other ChromaDB specific errors
        logger.error(
            f"Error deleting vectors for file_id '{file_id}' from '{collection_name}' (ChromaError): {e}",
            exc_info=True,
        )
        raise VectorDBError(
            f"Failed to delete vectors for file_id '{file_id}' from '{collection_name}': {e}"
        )
    except Exception as e:  # General fallback
        logger.error(
            f"Error deleting vectors for file_id '{file_id}' from '{collection_name}' (Unexpected Error): {e}",
            exc_info=True,
        )
        raise VectorDBError(
            f"Failed to delete vectors for file_id '{file_id}' from '{collection_name}' with unexpected error: {e}"
        )


async def clear_agent_vector_collection(agent_id: str):
    """Clears all vectors for a specific agent from ChromaDB."""
    if not agent_id:
        logger.error("Agent ID is required to clear vector collection.")
        raise ValueError("Agent ID cannot be empty.")

    try:
        chroma_client = get_chroma_client()
        collection_name = f"agent_{agent_id}_vectors"

        existing_collection_names = (
            chroma_client.list_collections()
        )  # Returns List[str] in Chroma 0.6.0+

        if collection_name in existing_collection_names:  # Correct check
            chroma_client.delete_collection(name=collection_name)
            logger.info(f"Successfully deleted ChromaDB collection: {collection_name}")
            return True
        else:
            logger.info(
                f"ChromaDB collection '{collection_name}' not found. No action taken."
            )
            return False
    except chromadb.errors.ChromaError as ce:
        logger.error(
            f"Error deleting ChromaDB collection '{collection_name}' (ChromaError): {ce}",
            exc_info=True,
        )
        raise VectorDBError(
            f"Failed to delete ChromaDB collection '{collection_name}': {ce}"
        )
    except Exception as e:  # General fallback
        logger.error(
            f"Error deleting ChromaDB collection '{collection_name}' (Unexpected Error): {e}",
            exc_info=True,
        )
        raise VectorDBError(
            f"Failed to delete ChromaDB collection '{collection_name}' with unexpected error: {e}"
        )


async def clear_all_agent_vector_data_from_chroma():
    """Clears ALL agent-specific vector collections from ChromaDB."""
    # TODO: Consider safety features if this function were to be exposed via an API
    # or critical script. For example, requiring a specific confirmation parameter (e.g., a specific string or boolean flag)
    # passed to the function, or an environment variable to be set, to prevent accidental mass deletion.
    # Currently, this function is a utility and not directly exposed via an API endpoint.
    chroma_client = get_chroma_client()
    try:
        collections = chroma_client.list_collections()
        deleted_count = 0
        for collection_name_str in collections:
            if collection_name_str.startswith(
                "agent_"
            ) and collection_name_str.endswith("_vectors"):
                chroma_client.delete_collection(name=collection_name_str)
                logger.info(f"Deleted ChromaDB collection: {collection_name_str}")
                deleted_count += 1
        logger.info(
            f"Cleared {deleted_count} agent-specific collections from ChromaDB."
        )
        return deleted_count
    except chromadb.errors.ChromaError as e:  # Catch ChromaDB specific errors
        logger.error(
            f"Error clearing all agent vector data from ChromaDB (ChromaError): {e}",
            exc_info=True,
        )
        raise VectorDBError(f"Failed to clear all agent vector data from ChromaDB: {e}")
    except Exception as e:  # General fallback
        logger.error(
            f"Error clearing all agent vector data from ChromaDB (Unexpected Error): {e}",
            exc_info=True,
        )
        raise VectorDBError(
            f"Failed to clear all agent vector data from ChromaDB with unexpected error: {e}"
        )
