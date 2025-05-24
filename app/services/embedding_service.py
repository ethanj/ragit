"""
Service for handling text embeddings.
"""

import functools
import logging
from typing import Dict, List

from llama_index.embeddings.openai import OpenAIEmbedding

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Custom error for embedding failures."""

    pass


@functools.lru_cache(maxsize=None)
def get_embedding_model():
    """Initializes and returns the OpenAI embedding model based on app settings."""
    if (
        not settings.OPENAI_API_KEY
        or settings.OPENAI_API_KEY == "your_openai_api_key_here"
    ):  # Added check for placeholder
        logger.error(
            "OPENAI_API_KEY is not set or is a placeholder. Please set it in your environment variables or .env file."
        )
        raise ValueError(
            "OPENAI_API_KEY is not set or is a placeholder. Please set it in your environment variables or .env file."
        )
    try:
        embed_model = OpenAIEmbedding(
            model=settings.EMBEDDING_MODEL_NAME, api_key=settings.OPENAI_API_KEY
        )
        logger.info(
            f"Successfully initialized OpenAIEmbedding model: {settings.EMBEDDING_MODEL_NAME}"
        )
        return embed_model
    except Exception as e:
        logger.error(
            f"Failed to initialize OpenAIEmbedding model '{settings.EMBEDDING_MODEL_NAME}': {e}",
            exc_info=True,
        )
        raise ValueError(
            f"Failed to initialize embedding model '{settings.EMBEDDING_MODEL_NAME}'. Check API key and model name."
        )


def embed_chunks(chunks: List[Dict[str, any]]) -> List[Dict[str, any]]:
    """
    Generates embeddings for a list of text chunks.
    Each chunk dictionary is expected to have a "text" field.
    Adds an "embedding" field to each chunk dictionary.

    Args:
        chunks: A list of chunk dictionaries.

    Returns:
        The list of chunk dictionaries, with embeddings added.

    Raises:
        EmbeddingError: If embedding generation fails for any chunk after retries (if any).
        ValueError: If the embedding model cannot be initialized.
    """
    if not chunks:
        return []

    embed_model = (
        get_embedding_model()
    )  # This will raise ValueError if model can't be init

    texts_to_embed = [chunk["text"] for chunk in chunks if "text" in chunk]
    if not texts_to_embed:
        logger.warning("No text found in chunks to embed.")
        return chunks

    try:
        embeddings = embed_model.get_text_embedding_batch(
            texts_to_embed,
            show_progress=True,  # Consider making show_progress configurable
        )
    except Exception as e:
        logger.error(f"Error during batch embedding generation: {e}", exc_info=True)
        raise EmbeddingError(f"Batch embedding generation failed: {e}")

    if len(embeddings) != len(texts_to_embed):
        logger.error(
            f"Mismatch in number of embeddings ({len(embeddings)}) and texts embedded ({len(texts_to_embed)})."
        )
        raise EmbeddingError(
            "Embedding count mismatch. Cannot reliably assign embeddings to chunks."
        )

    embedding_idx = 0
    for chunk in chunks:
        if "text" in chunk:
            if embedding_idx < len(embeddings):
                chunk["embedding"] = embeddings[embedding_idx]
                embedding_idx += 1
            else:
                logger.error(
                    "Ran out of embeddings to assign. Logic error or prior mismatch."
                )
                break
    return chunks
