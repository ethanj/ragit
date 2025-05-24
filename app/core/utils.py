"""
Core utility functions for the application.
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def parse_text_for_citations(
    text: str, chunk_references: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Parses text for [N] style citation markers and resolves them against chunk_references.

    Args:
        text: The text content (e.g., from LLM response) to parse.
        chunk_references: A list of dictionaries, where each dictionary represents a
                          source chunk and must contain at least a 'number' key
                          (1-indexed) and other details like 'id', 'text_snippet', 'title'.

    Returns:
        A list of citation dictionaries, each corresponding to a successfully parsed
        and resolved marker. Each dictionary includes the original marker, and details
        from the resolved chunk_reference.
    """
    parsed_citations = []
    # Find all occurrences of [N] where N is one or more digits
    citation_markers_found = set(re.findall(r"\[(\d+)\]", text))

    if citation_markers_found:
        logger.info(f"Found citation markers in text: {citation_markers_found}")
        # Create a lookup for chunk_references by their number
        references_lookup = {
            ref["number"]: ref
            for ref in chunk_references
            if isinstance(ref, dict) and "number" in ref
        }

        for marker_num_str in citation_markers_found:
            try:
                marker_num = int(marker_num_str)
                if marker_num in references_lookup:
                    chunk_ref = references_lookup[marker_num]
                    parsed_citations.append(
                        {
                            "marker": f"[{marker_num}]",  # Store the marker string e.g. "[1]"
                            "id": chunk_ref.get("id"),
                            "text_snippet": chunk_ref.get("text_snippet"),
                            "title": chunk_ref.get("title"),
                            "original_document_id": chunk_ref.get(
                                "original_document_id"
                            ),
                            "url": chunk_ref.get("url"),
                            # Add any other relevant fields from chunk_ref
                        }
                    )
                else:
                    logger.warning(
                        f"Cited chunk number {marker_num}, but it was not found in provided context references or references were malformed."
                    )
            except ValueError:
                logger.warning(
                    f"Could not parse citation marker number: {marker_num_str}"
                )
        # Sort citations by their numerical order for consistency
        parsed_citations.sort(
            key=lambda c: int(re.search(r"(\d+)", c["marker"]).group(1))
        )

    return parsed_citations
