"""
Service layer for ranking and fusing search results.
"""

import functools
import logging
from typing import Any, Dict, List, Tuple

from app.core.config import settings

# Attempt to import FlagEmbedding, log if not found
try:
    from FlagEmbedding import FlagReranker  # type: ignore
except ImportError:
    FlagReranker = None  # Allows functions to check for its presence
    logging.getLogger(__name__).warning(
        "FlagEmbedding library not found. Reranking functionality will be disabled. "
        "Please install with: pip install FlagEmbedding"
    )


logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def _get_reranker_model(model_name: str, use_fp16: bool):
    """Loads and caches the FlagReranker model."""
    if FlagReranker is None:
        logger.error("FlagEmbedding library not installed. Cannot load reranker model.")
        return None
    try:
        logger.info(
            f"Initializing FlagReranker model: {model_name} with use_fp16={use_fp16}"
        )
        return FlagReranker(model_name, use_fp16=use_fp16)
    except Exception as e:  # More specific exceptions could be caught if known
        logger.error(
            f"Error initializing FlagReranker model {model_name}: {e}", exc_info=True
        )
        return None


def _validate_rrf_inputs(results_lists: List[List[Dict[str, Any]]]) -> bool:
    """Validates the structure of inputs for RRF.

    Args:
        results_lists: A list of lists of result dictionaries.

    Returns:
        True if inputs are valid, False otherwise.
    """
    if not results_lists or not any(results_lists):
        logger.info("Empty results_lists provided to RRF.")
        return False  # Invalid, leads to empty list return in caller

    is_valid = True
    for res_list_idx, res_list in enumerate(results_lists):
        if not isinstance(res_list, list):
            logger.warning(
                f"Item at index {res_list_idx} in results_lists is not a list. Invalid input for RRF."
            )
            is_valid = False  # Mark as invalid, but continue checking other lists for full diagnostics
            continue  # Skip to next list

        if not res_list:  # If an inner list is empty, it's fine, just skip its items
            continue

        for item_idx, item in enumerate(res_list):
            if not isinstance(item, dict) or "id" not in item or "score" not in item:
                logger.warning(
                    f"Item {item_idx} in list {res_list_idx} is malformed: {item}. Required 'id' and 'score'. Invalid input for RRF."
                )
                is_valid = False  # Mark as invalid
                # Optionally, could break here if one bad item invalidates the whole RRF run
    return is_valid


def reciprocal_rank_fusion(
    results_lists: List[List[Dict[str, Any]]], k: int = settings.RRF_K_CONSTANT
) -> List[Dict[str, Any]]:
    """
    Performs Reciprocal Rank Fusion on a list of ranked result lists.

    Args:
        results_lists: A list of lists, where each inner list contains dictionaries
                       representing search results. Each dictionary must have an 'id'
                       (unique identifier for the item) and a 'score'.
        k: A constant used in RRF, typically 60 or 61. Controls the influence
           of lower-ranked items.

    Returns:
        A single list of items, fused and re-ranked according to RRF,
        with a new 'score'. Sorted by score descending.
    """
    if not results_lists or not any(results_lists):
        logger.info("Empty results_lists provided to RRF. Returning empty list.")
        return []

    if not _validate_rrf_inputs(results_lists):
        logger.warning("RRF input validation failed. Returning empty list.")
        return []

    # Validate structure of results # This loop is now in _validate_rrf_inputs
    # for res_list_idx, res_list in enumerate(results_lists):
    #     if not isinstance(res_list, list):
    #         logger.warning(
    #             f"Item at index {res_list_idx} in results_lists is not a list. Skipping."
    #         )
    #         continue
    #     for item_idx, item in enumerate(res_list):
    #         if not isinstance(item, dict) or "id" not in item or "score" not in item:
    #             logger.warning(
    #                 f"Item {item_idx} in list {res_list_idx} is malformed: {item}. Skipping."
    #             )

    rrf_scores: Dict[str, float] = {}
    doc_content_map: Dict[str, Dict[str, Any]] = {}

    for results in results_lists:
        if not results:
            continue
        for rank, doc in enumerate(results):
            if not isinstance(doc, dict) or "id" not in doc or "score" not in doc:
                continue

            doc_id = doc["id"]
            score = 1.0 / (k + rank)

            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                doc_content_map[doc_id] = {
                    key: val for key, val in doc.items() if key != "score"
                }

            rrf_scores[doc_id] += score

    fused_results = []
    for doc_id, total_score in rrf_scores.items():
        doc_data = doc_content_map.get(doc_id)
        if doc_data:
            fused_doc = doc_data.copy()
            fused_doc["score"] = total_score
            fused_results.append(fused_doc)
        else:
            logger.warning(
                f"Document content for ID {doc_id} not found in map. This should not happen."
            )

    fused_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    logger.info(f"RRF resulted in {len(fused_results)} fused documents.")
    return fused_results


def _prepare_sentence_pairs_for_reranking(
    query: str, documents: List[Dict[str, Any]]
) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
    """Prepares sentence pairs for reranking and filters invalid documents.

    Args:
        query: The search query.
        documents: A list of document dictionaries.

    Returns:
        A tuple containing (sentence_pairs, valid_documents_for_reranking).
    """
    sentence_pairs = []
    valid_documents_for_reranking = []
    if not documents:
        return sentence_pairs, valid_documents_for_reranking

    for doc in documents:
        if isinstance(doc.get("text"), str) and doc["text"]:
            sentence_pairs.append([query, doc["text"]])
            valid_documents_for_reranking.append(doc)
        else:
            logger.warning(
                f"Document with id {doc.get('id', 'UNKNOWN_ID')} missing 'text' field, not a string, or empty. Skipping for reranking."
            )
    return sentence_pairs, valid_documents_for_reranking


async def rerank_documents(
    query: str, documents: List[Dict[str, Any]], top_n: int
) -> List[Dict[str, Any]]:
    """
    Reranks a list of documents based on a query using a cross-encoder model.

    Args:
        query: The search query.
        documents: A list of document dictionaries, each with a "text" field.
        top_n: The number of top documents to return after reranking.

    Returns:
        A sorted list of the top_n documents, with an added "rerank_score".
        Returns original documents up to top_n if model fails or not configured.
    """
    if not settings.RERANKER_MODEL_NAME:
        logger.warning("Reranker model name not configured. Skipping reranking.")
        return documents[:top_n]

    if not documents:
        return []

    reranker = _get_reranker_model(
        settings.RERANKER_MODEL_NAME,
        use_fp16=True,  # Default from original code
    )

    if reranker is None:
        logger.error("Reranker model could not be loaded. Skipping reranking.")
        return documents[:top_n]

    sentence_pairs, valid_documents_for_reranking = (
        _prepare_sentence_pairs_for_reranking(query, documents)
    )

    if not sentence_pairs:
        logger.warning(
            "No valid documents with text found for reranking. Returning original documents up to top_n."
        )
        return documents[:top_n]

    try:
        scores = reranker.compute_score(sentence_pairs, normalize=True)
    except Exception as e:
        logger.error(
            f"Error computing rerank scores: {e}. Propagating error.", exc_info=True
        )
        # In the original code, this re-raises, which seems correct for hybrid_retrieve to catch.
        raise

    for i, doc in enumerate(valid_documents_for_reranking):
        if i < len(scores):
            doc["rerank_score"] = scores[i]
        else:
            doc["rerank_score"] = -1.0

    reranked_docs = sorted(
        [
            doc
            for doc in valid_documents_for_reranking
            if "rerank_score" in doc and doc["rerank_score"] != -1.0
        ],
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    return reranked_docs[:top_n]
