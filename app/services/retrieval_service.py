"""
Service layer for retrieving relevant document chunks.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import chromadb  # Added for specific exception handling

from app.core.config import settings

# from app.services.ingestion import get_chroma_client, get_embedding_model # Old import
from app.services.embedding_service import get_embedding_model  # New import
from app.services.ranking_service import (  # New import for ranking
    reciprocal_rank_fusion,
    rerank_documents,
)
from app.services.vector_db_service import get_chroma_client  # New import

logger = logging.getLogger(__name__)

# DEFAULT_SEMANTIC_TOP_K = 10 # Removed, will use settings.DEFAULT_SEMANTIC_TOP_K


class RetrievalError(Exception):
    """Custom exception for retrieval errors."""

    pass


def _get_chroma_collection_for_agent(agent_id: str, chroma_client: Any) -> Any | None:
    """Gets the ChromaDB collection for a given agent.

    Args:
        agent_id: The ID of the agent.
        chroma_client: The ChromaDB client instance.

    Returns:
        The ChromaDB collection object if found, otherwise None.
    """
    collection_name = f"agent_{agent_id}_vectors"
    try:
        collection = chroma_client.get_collection(name=collection_name)
        logger.info(
            f"Accessed ChromaDB collection for semantic search: {collection_name}"
        )
        return collection
    except chromadb.errors.ChromaError as ce:  # Catch more general Chroma errors
        logger.error(f"ChromaDB error for {collection_name}: {ce}")
    except Exception as e:  # Catch any other unexpected errors
        logger.error(f"Unexpected error accessing collection {collection_name}: {e}")
    return None


def _get_query_embedding(query_text: str) -> List[float]:
    """Generates embedding for a given query text.

    Args:
        query_text: The user's query string.

    Returns:
        The query embedding as a list of floats.

    Raises:
        RetrievalError: If API key is not configured or embedding generation fails.
    """
    if (
        not settings.OPENAI_API_KEY
        or settings.OPENAI_API_KEY == "your_openai_api_key_here"
    ):
        logger.error(
            "OpenAI API Key is not configured. Cannot generate query embedding."
        )
        raise RetrievalError("OpenAI API Key not configured for query embedding.")

    try:
        embed_model = (
            get_embedding_model()
        )  # Uses the cached model from embedding_service
        query_embedding = embed_model.get_text_embedding(query_text)
        return query_embedding
    except ValueError as e:  # Catch specific error from get_embedding_model if API key issue persists in it
        logger.error(
            f"Cannot generate query embedding due to embedding model configuration: {e}"
        )
        raise RetrievalError(f"Embedding model configuration error: {e}")
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}", exc_info=True)
        raise RetrievalError(f"Query embedding generation failed: {e}")


def _get_all_documents_for_bm25(
    agent_id: str, chroma_client: Any
) -> Tuple[Optional[List[str]], Optional[List[Dict[str, Any]]], Optional[List[str]]]:
    """Fetches all document texts, metadatas, and ids for an agent from ChromaDB.

    Args:
        agent_id: The ID of the agent.
        chroma_client: The ChromaDB client instance.

    Returns:
        A tuple containing (corpus_texts, corpus_metadatas, corpus_ids).
        Returns (None, None, None) if collection not found or other error.
    """
    collection_name = f"agent_{agent_id}_vectors"
    try:
        collection = chroma_client.get_collection(name=collection_name)
        # Fetch all documents and metadatas from this collection for BM25
        # This could be inefficient for very large collections.
        collection_data = collection.get(include=["documents", "metadatas", "ids"])

        corpus_texts = collection_data.get("documents")
        corpus_metadatas = collection_data.get("metadatas")
        corpus_ids = collection_data.get("ids")

        if not corpus_texts:
            logger.info(
                f"No documents found in collection '{collection_name}' for BM25 indexing."
            )
            return None, None, None  # Explicitly return None for all if no texts

        return corpus_texts, corpus_metadatas, corpus_ids

    except Exception as e:
        logger.warning(
            f"Failed to get ChromaDB collection '{collection_name}' or its data for BM25: {e}. Agent may have no documents."
        )
        return None, None, None


from rank_bm25 import BM25Okapi


def _calculate_bm25_scores(
    query_text: str, corpus_texts: List[str], bm25_class: Any = BM25Okapi
) -> List[float]:
    """Calculates BM25 scores for a query against a corpus.

    Args:
        query_text: The user's query string.
        corpus_texts: A list of document texts forming the corpus.
        bm25_class: The BM25 implementation class (e.g., BM25Okapi).

    Returns:
        A list of BM25 scores for each document in the corpus.

    Raises:
        ImportError: If rank_bm25 is not installed.
        Exception: For other errors during BM25 processing.
    """
    if not corpus_texts:
        logger.info(
            "Empty corpus_texts provided to _calculate_bm25_scores. Returning empty list."
        )
        return []

    # Tokenize corpus (simple space split for now)
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
    bm25_model = bm25_class(tokenized_corpus)

    # Tokenize query
    tokenized_query = query_text.lower().split()

    # Get BM25 scores
    doc_scores = bm25_model.get_scores(tokenized_query)
    return doc_scores


async def semantic_search(
    agent_id: str, query_text: str, top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Performs semantic search for a given agent and query using ChromaDB.

    Args:
        agent_id: The ID of the agent whose data to search.
        query_text: The user's query string.
        top_k: The number of top results to retrieve.

    Returns:
        A list of dictionaries, where each dictionary represents a retrieved chunk
        and includes its text, metadata, and potentially a relevance score.
        Returns an empty list if no results or an error occurs.

    Raises:
        RetrievalError: If there's an issue with ChromaDB operations or configuration.
    """
    if not agent_id:
        logger.error("Agent ID is required for semantic search.")
        raise RetrievalError("Agent ID cannot be empty for semantic search.")
    if not query_text:
        logger.info("Query text is empty, returning no results for semantic search.")
        return []

    effective_top_k = top_k if top_k is not None else settings.DEFAULT_SEMANTIC_TOP_K

    collection_name = f"agent_{agent_id}_vectors"

    try:
        chroma_client = get_chroma_client()
    except Exception as e:
        logger.error(f"Could not connect to ChromaDB for semantic search. Error: {e}")
        raise RetrievalError(f"ChromaDB connection failed for semantic search: {e}")

    collection = _get_chroma_collection_for_agent(agent_id, chroma_client)
    if collection is None:
        return []  # Collection not found or error accessing it, return empty list

    # OpenAI API Key check and embedding generation moved to _get_query_embedding helper
    try:
        query_embedding = _get_query_embedding(query_text)
    except (
        RetrievalError
    ):  # Catch and re-raise to ensure semantic_search's contract is met
        # The logger.error is already handled in _get_query_embedding
        raise

    raw_chroma_results = None  # Initialize
    try:
        raw_chroma_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_top_k,
            include=[
                "metadatas",
                "documents",
                "distances",
            ],  # documents are the text, distances are scores
        )
    except Exception as e:
        logger.error(
            f"Error during ChromaDB query for collection '{collection_name}': {e}"
        )
        raise RetrievalError(
            f"ChromaDB query failed for collection '{collection_name}': {e}"
        )

    # Handle ChromaDB results (which might be None or a dict)
    if raw_chroma_results is None:
        logger.info(
            f"Semantic search from '{collection_name}' returned None (no results or query issue handled by client returning None)."
        )
        return []  # _format_semantic_results would also return [] for None input
    else:
        # This log is safe now as raw_chroma_results is confirmed to be a dict
        logger.info(
            f"Semantic search returned {len(raw_chroma_results.get('ids', [[]])[0])} results from '{collection_name}'."
        )
        return _format_semantic_results(raw_chroma_results)


def _format_semantic_results(chroma_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Formats the raw results from ChromaDB query into a list of dictionaries.

    Args:
        chroma_results: The dictionary returned by ChromaDB's query method.
                        Expected to contain 'ids', 'documents', 'metadatas', 'distances' as lists of lists.

    Returns:
        A list of formatted result dictionaries.
    """
    formatted_results = []
    if not chroma_results:
        return formatted_results

    # ChromaDB query results for a single query embedding are lists within lists.
    # e.g., ids = [['id1', 'id2']], documents = [['doc1', 'doc2']], etc.
    # We need to access the first element of these lists.

    ids_list = chroma_results.get("ids", [[]])[0]
    documents_list = chroma_results.get("documents", [[]])[0]
    metadatas_list = chroma_results.get("metadatas", [[]])[0]
    distances_list = chroma_results.get("distances", [[]])[0]

    # Ensure all lists have the same length to avoid IndexError
    min_len = min(
        len(ids_list) if ids_list else 0,
        len(documents_list) if documents_list else 0,
        len(metadatas_list) if metadatas_list else 0,
        len(distances_list) if distances_list else 0,
    )

    if not min_len and (ids_list or documents_list or metadatas_list or distances_list):
        logger.warning(
            f"Mismatched lengths in ChromaDB results or some lists are None/empty. "
            f"IDs: {len(ids_list) if ids_list else 'N/A'}, "
            f"Docs: {len(documents_list) if documents_list else 'N/A'}, "
            f"Metadatas: {len(metadatas_list) if metadatas_list else 'N/A'}, "
            f"Distances: {len(distances_list) if distances_list else 'N/A'}. "
            "Returning empty formatted results."
        )
        return []

    for i in range(min_len):
        # Ensure metadata is a dict, default to empty dict if None
        metadata = metadatas_list[i] if metadatas_list[i] is not None else {}
        formatted_results.append(
            {
                "id": ids_list[i],
                "text": documents_list[i],
                "metadata": metadata,
                "score": 1.0
                - distances_list[i],  # Convert distance to similarity score
                "retrieval_strategy": "semantic",
            }
        )
    return formatted_results


DEFAULT_KEYWORD_TOP_K = 10


async def keyword_search_bm25(
    agent_id: str, query_text: str, top_k: int = settings.DEFAULT_KEYWORD_TOP_K
) -> List[Dict[str, Any]]:
    """
    Performs keyword search for a given agent and query using BM25.
    Fetches all documents from the agent's ChromaDB collection to build an in-memory BM25 index.

    Args:
        agent_id: The ID of the agent whose data to search.
        query_text: The user's query string.
        top_k: The number of top results to retrieve.

    Returns:
        A list of dictionaries, where each dictionary represents a retrieved chunk
        and includes its text, metadata, and BM25 score.
        Returns an empty list if no results or an error occurs.
    """
    # PERFORMANCE WARNING:
    # This implementation fetches ALL documents for the agent from ChromaDB
    # and builds an in-memory BM25 index on-the-fly for each query.
    # This can be very inefficient and slow for agents with a large number of documents.
    # The tokenizer used (simple space split) is also very basic.
    # Consider:
    # 1. Pre-calculating or caching BM25 indexes per agent/corpus.
    # 2. Using a more scalable BM25 library or a search system with built-in BM25.
    # 3. Limiting the size of the corpus fetched for BM25 if full-corpus search is too slow.

    if not agent_id:
        logger.error("Agent ID cannot be empty for keyword search.")
        raise RetrievalError("Agent ID cannot be empty for keyword search.")
    if not query_text:
        logger.info(
            f"Empty query for keyword search for agent '{agent_id}'. Returning no results."
        )
        return []

    effective_top_k = top_k if top_k is not None else settings.DEFAULT_KEYWORD_TOP_K

    try:
        chroma_client = get_chroma_client()
    except Exception as e:
        logger.error(f"Could not connect to ChromaDB for keyword search. Error: {e}")
        raise RetrievalError(f"ChromaDB connection failed for keyword search: {e}")

    corpus_texts, corpus_metadatas, corpus_ids = _get_all_documents_for_bm25(
        agent_id, chroma_client
    )
    if not corpus_texts or not corpus_ids or not corpus_metadatas:
        logger.info(
            f"No documents found for agent '{agent_id}' to build BM25 index. Returning empty list."
        )
        return []

    try:
        doc_scores = _calculate_bm25_scores(query_text, corpus_texts)
        final_results = _format_bm25_results(
            doc_scores, corpus_ids, corpus_texts, corpus_metadatas, effective_top_k
        )

        logger.info(
            f"Keyword search (BM25) returned {len(final_results)} results from '{agent_id}'."
        )
        return final_results

    except ImportError:
        logger.error(
            "rank_bm25 library not installed. Please install it for keyword search: `pip install rank-bm25`"
        )
        raise RetrievalError(
            "rank_bm25 library not installed, required for keyword search."
        )
    except Exception as e:
        logger.error(
            f"Error during keyword search for agent {agent_id}: {e}", exc_info=True
        )
        return []  # Fallback to empty list


def _format_bm25_results(
    doc_scores: List[float],
    corpus_ids: List[str],
    corpus_texts: List[str],
    corpus_metadatas: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Helper function to format BM25 scoring results."""
    scored_documents = []
    for i, score in enumerate(doc_scores):
        # Ensure we don't go out of bounds if lists aren't perfectly aligned (should be though)
        if i < len(corpus_ids) and i < len(corpus_texts) and i < len(corpus_metadatas):
            scored_documents.append(
                {
                    "id": corpus_ids[i],
                    "text": corpus_texts[i],
                    "metadata": corpus_metadatas[i],
                    "score": score,
                }
            )
        else:
            logger.warning(
                f"Index {i} out of bounds when formatting BM25 results. Scores len: {len(doc_scores)}"
            )
            break  # Stop if data is misaligned

    # Sort by score in descending order and take top_k
    sorted_documents = sorted(scored_documents, key=lambda x: x["score"], reverse=True)
    top_results = sorted_documents[:top_k]

    # Filter out results with score 0 or less for BM25Okapi (scores are usually >= 0)
    final_results = [doc for doc in top_results if doc["score"] > 0]
    return final_results


DEFAULT_HYBRID_TOP_K = settings.DEFAULT_HYBRID_TOP_K  # Corrected to use settings


async def hybrid_retrieve(
    agent_id: str,
    query_text: str,
    semantic_top_k: Optional[int] = None,
    keyword_top_k: Optional[int] = None,
    final_top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Performs hybrid retrieval using both semantic and keyword search,
    then fuses the results using Reciprocal Rank Fusion (RRF),
    and finally reranks the top N results.
    """
    # Use settings for defaults if parameters are None
    effective_semantic_top_k = (
        semantic_top_k
        if semantic_top_k is not None
        else settings.DEFAULT_SEMANTIC_TOP_K
    )
    effective_keyword_top_k = (
        keyword_top_k if keyword_top_k is not None else settings.DEFAULT_KEYWORD_TOP_K
    )
    # DEFAULT_HYBRID_TOP_K from settings refers to the number of results after RRF / initial retrieval for reranker.
    # Let's rename final_top_k parameter to initial_rerank_top_n for clarity,
    # and use settings.RERANKER_TOP_N for the *final* number of results after reranking.
    # The `final_top_k` parameter here seems to correspond to how many results to feed into the reranker.
    effective_initial_rerank_top_n = (
        final_top_k if final_top_k is not None else settings.DEFAULT_HYBRID_TOP_K
    )

    if not agent_id or not query_text:
        logger.warning("Agent ID or query text is empty for hybrid retrieval.")
        return []

    try:
        semantic_results = await semantic_search(
            agent_id, query_text, top_k=effective_semantic_top_k
        )
        logger.info(
            f"Hybrid retrieve: Semantic search found {len(semantic_results)} results for agent {agent_id}."
        )
    except RetrievalError as e:
        logger.error(
            f"Hybrid retrieve: Semantic search failed for agent {agent_id}: {e}"
        )
        semantic_results = []
    except Exception as e:
        logger.error(
            f"Hybrid retrieve: Unexpected error in semantic search for agent {agent_id}: {e}",
            exc_info=True,
        )
        semantic_results = []

    try:
        keyword_results = await keyword_search_bm25(
            agent_id, query_text, top_k=effective_keyword_top_k
        )
        logger.info(
            f"Hybrid retrieve: Keyword search found {len(keyword_results)} results for agent {agent_id}."
        )
    except RetrievalError as e:  # Assuming keyword_search_bm25 might raise this
        logger.error(
            f"Hybrid retrieve: Keyword search failed for agent {agent_id}: {e}"
        )
        keyword_results = []
    except Exception as e:
        logger.error(
            f"Hybrid retrieve: Unexpected error in keyword search for agent {agent_id}: {e}",
            exc_info=True,
        )
        keyword_results = []

    if not semantic_results and not keyword_results:
        logger.info("Both semantic and keyword search returned no results.")
        return []

    # Call from ranking_service
    fused_results = reciprocal_rank_fusion(
        [semantic_results, keyword_results], k=settings.RRF_K_CONSTANT
    )

    candidate_documents_for_reranking = fused_results[:effective_initial_rerank_top_n]

    if settings.RERANKER_MODEL_NAME and candidate_documents_for_reranking:
        try:
            # Call from ranking_service
            reranked_documents = await rerank_documents(
                query_text,
                candidate_documents_for_reranking,
                top_n=settings.RERANKER_TOP_N,
            )
            logger.info(
                f"Reranked {len(candidate_documents_for_reranking)} documents down to {len(reranked_documents)}."
            )
            # The rerank_documents function already applies top_n, so no slicing needed here.
            return reranked_documents
        except Exception as e:
            logger.error(
                f"Error during reranking: {e}. Returning non-reranked (fused and truncated) results."
            )
            return candidate_documents_for_reranking[
                : settings.RERANKER_TOP_N
            ]  # Fallback to RRF results truncated by RERANKER_TOP_N
    else:
        logger.info(
            "Reranker not configured or no documents to rerank. Returning fused and truncated results."
        )
        return candidate_documents_for_reranking[
            : settings.DEFAULT_HYBRID_TOP_K
        ]  # Truncate by DEFAULT_HYBRID_TOP_K if reranker is off
