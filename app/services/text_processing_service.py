"""
Service layer for cleaning, preprocessing, and chunking text data.
"""

import logging
from typing import Dict

import pandas as pd
from bs4 import BeautifulSoup
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from app.core.config import settings

# Import constants needed from file_validation_service
from app.services.file_validation_service import (
    POTENTIAL_HTML_COLUMNS,
)

logger = logging.getLogger(__name__)

# Constants for metadata processing in chunk_data
ALLOWED_ADDITIONAL_METADATA_PREFIXES = ["metadata/", "crawl/"]
EXACT_ALLOWED_ADDITIONAL_METADATA = [
    "markdown",
    "screenshotUrl",
    "category",
]
# MAX_METADATA_CHAR_LENGTH_BUDGET is calculated dynamically based on chunk_size


def _handle_html_and_select_content_helper(row, content_column_name: str) -> str:
    """
    Helper function to process a single row to extract and clean its primary content,
    handling potential HTML.
    """
    cleaned_content = ""

    original_content = row[content_column_name]
    if pd.isna(original_content):
        return ""  # Return empty string for NaN content

    content_to_process = str(original_content)

    # Check if content is likely HTML by checking for tags or common HTML column names
    is_html_content = False
    if content_column_name.lower() in POTENTIAL_HTML_COLUMNS:
        is_html_content = True
    elif (
        "<html" in content_to_process.lower()
        or "<body" in content_to_process.lower()
        or "<div" in content_to_process.lower()
        or "<p" in content_to_process.lower()
    ):
        is_html_content = True

    if is_html_content:
        try:
            soup = BeautifulSoup(content_to_process, "html.parser")
            # Attempt to extract meaningful text, e.g., from body or specific tags
            body = soup.find("body")
            if body:
                cleaned_content = body.get_text(separator=" ", strip=True)
            else:  # Fallback to full soup if no body tag
                cleaned_content = soup.get_text(separator=" ", strip=True)
            logger.debug(
                f"Stripped HTML from content. Original length: {len(content_to_process)}, Cleaned length: {len(cleaned_content)}"
            )
        except Exception as e:
            logger.warning(
                f"Error stripping HTML, using original content: {e}. Content starts: '{content_to_process[:100]}'"
            )
            cleaned_content = (
                content_to_process.strip()  # Fallback to original (stripped) if parsing fails
            )
    else:
        cleaned_content = (
            content_to_process.strip()
        )  # Ensure non-HTML content is also stripped

    return cleaned_content


def clean_and_preprocess_data(
    df: pd.DataFrame,
    content_column_name: str,
    identified_metadata_columns: Dict[str, str],
) -> pd.DataFrame:
    """
    Cleans and preprocesses the DataFrame content.
    - Strips HTML from the identified content column if it seems like HTML.
    - Deduplicates based on the content.
    - Renames the cleaned content column to 'processed_text'.
    - Ensures key metadata columns (if identified) are carried forward.

    Args:
        df: The input DataFrame.
        content_column_name: The name of the column containing the primary text content.
        identified_metadata_columns: A map of identified metadata types to their column names.

    Returns:
        A cleaned and deduplicated pandas DataFrame ready for chunking.
    """
    if df.empty:
        logger.warning(
            "Input DataFrame to clean_and_preprocess_data is empty. Returning empty DataFrame."
        )
        return df

    if content_column_name not in df.columns:
        logger.error(
            f"Identified content column '{content_column_name}' not found in DataFrame columns: {df.columns.tolist()}. "
            "This should not happen if load_and_validate_csv was successful."
        )
        raise ValueError(
            f"Critical error: content column '{content_column_name}' missing."
        )

    df["processed_text"] = df.apply(
        lambda r: _handle_html_and_select_content_helper(r, content_column_name), axis=1
    )

    # Fallback logic for rows where processed_text is empty and an HTML alternative might exist
    # This primarily handles cases where content_column_name was, e.g., 'content_text',
    # it was NaN/None, and 'content_html' (or similar) has usable content.
    if content_column_name.lower() not in POTENTIAL_HTML_COLUMNS:
        # Attempt fallback only if the primary designated column was not already an HTML one.
        for html_col_candidate in POTENTIAL_HTML_COLUMNS:
            if html_col_candidate in df.columns:
                # Rows where initial processed_text is empty AND the candidate HTML col has content
                condition = (df["processed_text"].str.strip() == "") & (
                    df[html_col_candidate].notna()
                    & (df[html_col_candidate].str.strip() != "")
                )
                if condition.any():
                    logger.info(
                        f"Attempting HTML fallback for empty processed_text using column: '{html_col_candidate}'"
                    )
                    df.loc[condition, "processed_text"] = df[condition].apply(
                        lambda r: _handle_html_and_select_content_helper(
                            r, html_col_candidate
                        ),
                        axis=1,
                    )
                break  # Only try one HTML fallback column

    # Remove rows where processed_text is empty or only whitespace AFTER potential fallback
    original_row_count = len(df)
    df = df[df["processed_text"].str.strip() != ""]
    if len(df) < original_row_count:
        logger.info(
            f"Removed {original_row_count - len(df)} rows with empty 'processed_text' after cleaning and fallback."
        )

    # Deduplicate based on the 'processed_text' column
    df_deduplicated = df.drop_duplicates(subset=["processed_text"], keep="first")
    if len(df_deduplicated) < len(df):
        logger.info(
            f"Deduplicated rows: {len(df) - len(df_deduplicated)} rows removed based on 'processed_text'."
        )
    df = df_deduplicated

    columns_to_keep = ["processed_text"]
    # Add all identified metadata columns (using their original names)
    for original_col_name in identified_metadata_columns.values():
        if original_col_name in df.columns and original_col_name not in columns_to_keep:
            columns_to_keep.append(original_col_name)
            # The commented-out logic regarding df[meta_type] was correctly removed previously.

    # Preserve any other columns from the DataFrame, excluding the original content column
    # (if it's not already selected as metadata or is not 'processed_text').
    for (
        col
    ) in df.columns:  # Iterate current df's columns (already potentially de-duplicated)
        if col not in columns_to_keep and col != content_column_name:
            columns_to_keep.append(col)
            logger.info(f"Preserving additional unspecified column from CSV: '{col}'")

    # Ensure all columns in columns_to_keep actually exist in the df at this stage
    # This check is mostly a safeguard; df.columns should be the source of truth.
    final_columns_to_keep = [col for col in columns_to_keep if col in df.columns]

    if len(final_columns_to_keep) != len(columns_to_keep):
        missing_cols = set(columns_to_keep) - set(final_columns_to_keep)
        logger.warning(
            f"Some columns intended to be kept were not found in the DataFrame. Missing: {missing_cols}. "
        )

    if not final_columns_to_keep:
        logger.error(
            "No columns selected to keep after processing. Returning empty DataFrame."
        )
        return pd.DataFrame()

    final_df = df[final_columns_to_keep].copy()

    logger.info(
        f"Data cleaning and preprocessing complete. Final shape: {final_df.shape}. Columns: {final_df.columns.tolist()}"
    )
    return final_df


def _prepare_and_prune_chunk_metadata(
    row: pd.Series,
    file_id: str,
    item_index: int,  # Using item_index to avoid conflict with pandas' row.index
    identified_metadata_columns: Dict[str, str],
    max_metadata_char_length_budget: int,
    df_columns: list[
        str
    ],  # Pass all columns of the DataFrame for iterating additional metadata
) -> Dict[str, any]:
    """
    Prepares and prunes metadata for a single document/row to be chunked.
    """
    doc_metadata = {
        "file_id": file_id,
    }

    # Determine original_document_id
    original_doc_id_from_meta = identified_metadata_columns.get("id")
    if (
        original_doc_id_from_meta
        and original_doc_id_from_meta in row
        and pd.notna(row[original_doc_id_from_meta])
    ):
        doc_metadata["original_document_id"] = str(row[original_doc_id_from_meta])
    elif "id" in row and pd.notna(row["id"]):
        doc_metadata["original_document_id"] = str(row["id"])
    else:
        doc_metadata["original_document_id"] = f"file_{file_id}_row_{item_index}"

    # Add identified metadata types (e.g., title, url)
    for meta_type, original_col_name in identified_metadata_columns.items():
        if (
            meta_type != "id"  # Already handled as original_document_id
            and original_col_name in row
            and pd.notna(row[original_col_name])
        ):
            doc_metadata[meta_type] = str(row[original_col_name])

    # Add other allowed additional metadata from the row
    for col_name in df_columns:  # Iterate over passed df_columns list
        if (
            col_name != "processed_text"
            and col_name
            not in identified_metadata_columns.values()  # Avoid reprocessing identified keys
            and col_name
            not in doc_metadata  # Avoid overwriting file_id, original_document_id, or already set meta_types
        ):
            if pd.notna(
                row.get(col_name)
            ):  # Check if col_name exists in Series and is not NaN
                is_allowed = False
                if col_name in EXACT_ALLOWED_ADDITIONAL_METADATA:
                    is_allowed = True
                else:
                    for prefix in ALLOWED_ADDITIONAL_METADATA_PREFIXES:
                        if col_name.startswith(prefix):
                            # Further filter down very verbose metadata like jsonLd and openGraph structures
                            if "jsonLd/" in col_name and col_name not in [
                                "metadata/jsonLd/0/name",
                                "metadata/jsonLd/1/name",
                            ]:
                                continue
                            if "openGraph/" in col_name and col_name not in [
                                "metadata/openGraph/0/property",
                                "metadata/openGraph/0/content",
                                "metadata/openGraph/1/property",
                                "metadata/openGraph/1/content",
                            ]:
                                continue
                            is_allowed = True
                            break
                if is_allowed:
                    doc_metadata[col_name] = str(row[col_name])
                    # logger.debug(f"Adding allowed additional metadata '{col_name}' to chunks from item {item_index}.")

    # --- METADATA PRUNING LOGIC --- (copied and adapted from chunk_data)
    current_metadata_char_length = sum(
        len(str(v)) for v in doc_metadata.values() if pd.notna(v)
    )

    if current_metadata_char_length > max_metadata_char_length_budget:
        logger.warning(
            f"Item {item_index} (orig_id: {doc_metadata.get('original_document_id')}): Initial metadata char length ({current_metadata_char_length}) exceeds budget ({max_metadata_char_length_budget}). Pruning..."
        )
        pruned_metadata = {}
        temp_char_count = 0

        # Essential keys to prioritize (includes identified_metadata_columns keys + fixed ones)
        # Order of processing: file_id, original_document_id, then keys from identified_metadata_columns, then others.
        priority_keys = ["file_id", "original_document_id"]
        for meta_type in identified_metadata_columns.keys():
            if (
                meta_type not in priority_keys
            ):  # meta_type is e.g. 'title', 'url', not the raw column name
                priority_keys.append(meta_type)

        # Add remaining keys from doc_metadata, not in priority, to process them last
        all_keys_in_order = priority_keys + [
            k for k in doc_metadata.keys() if k not in priority_keys
        ]

        for key in all_keys_in_order:
            if key in doc_metadata and pd.notna(doc_metadata[key]):
                value_str = str(doc_metadata[key])
                # Apply field-level length limits for non-essential additional metadata before budget check
                # For essential/identified metadata, we only truncate if over budget.
                # For *additional* allowed ones, we apply MAX_METADATA_FIELD_LENGTH first.
                if key not in priority_keys:  # It's an 'additional' metadata field
                    if len(value_str) >= settings.MIN_METADATA_FIELD_LENGTH:
                        if len(value_str) > settings.MAX_METADATA_FIELD_LENGTH:
                            value_str = (
                                value_str[
                                    : settings.MAX_METADATA_FIELD_LENGTH
                                    - len(settings.METADATA_TRUNCATION_SUFFIX)
                                ]
                                + settings.METADATA_TRUNCATION_SUFFIX
                            )
                            # logger.debug(f"Truncated additional metadata '{key}' to max field length before budget check.")
                    else:
                        # logger.debug(f"Skipping short/empty additional metadata '{key}' (value: '{doc_metadata[key]}') before budget check.")
                        continue  # Skip this short additional field

                if (
                    temp_char_count + len(value_str)
                ) <= max_metadata_char_length_budget:
                    pruned_metadata[key] = (
                        value_str  # Use the potentially field-level-truncated value_str
                    )
                    temp_char_count += len(value_str)
                else:  # Truncate due to overall budget
                    remaining_budget_for_key = (
                        max_metadata_char_length_budget - temp_char_count
                    )
                    if remaining_budget_for_key > len(
                        settings.METADATA_TRUNCATION_SUFFIX
                    ):
                        pruned_metadata[key] = (
                            value_str[
                                : remaining_budget_for_key
                                - len(settings.METADATA_TRUNCATION_SUFFIX)
                            ]
                            + settings.METADATA_TRUNCATION_SUFFIX
                        )
                        temp_char_count += len(pruned_metadata[key])
                        logger.warning(
                            f"Truncated metadata key '{key}' due to overall budget limit."
                        )
                    # else: cannot fit even truncated, omitted for this key.
        doc_metadata = pruned_metadata
        logger.info(
            f"Item {item_index} (orig_id: {doc_metadata.get('original_document_id')}): Metadata after pruning: {doc_metadata}"
        )
    return doc_metadata


def chunk_data(
    df: pd.DataFrame,
    file_id: str,
    identified_metadata_columns: Dict[str, str],
    chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Chunks the processed text in the DataFrame.
    Assumes 'processed_text' column exists from clean_and_preprocess_data.
    Dynamically includes other columns from the DataFrame as metadata.

    Args:
        df: DataFrame with a 'processed_text' column and other metadata.
        file_id: The unique ID of the source UploadedFile record.
        identified_metadata_columns: Map of semantic types to original column names.
        chunk_size: Target size of chunks in tokens.
        chunk_overlap: Overlap between chunks in tokens.

    Returns:
        A list of dictionaries, where each dictionary represents a chunk.
    """
    if "processed_text" not in df.columns:
        logger.error("DataFrame for chunking must contain 'processed_text' column.")
        return []

    if df.empty:
        logger.info("Received empty DataFrame for chunking.")
        return []

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks = []
    # Moved ALLOWED_ADDITIONAL_METADATA_PREFIXES and EXACT_ALLOWED_ADDITIONAL_METADATA to module level
    max_metadata_char_length_budget = (
        chunk_size * 4
    )  # Estimate 4 chars/token for budget
    df_column_names = list(df.columns)  # Get column names once for the helper

    for item_idx, row in df.iterrows():  # item_idx to pass to helper
        text_to_chunk = row["processed_text"]
        if (
            not text_to_chunk
            or not isinstance(text_to_chunk, str)
            or text_to_chunk.strip() == ""
        ):
            logger.warning(
                f"Skipping row {item_idx} due to empty or invalid 'processed_text'."
            )
            continue

        doc_metadata = _prepare_and_prune_chunk_metadata(
            row=row,
            file_id=file_id,
            item_index=item_idx,  # Pass the iterator index
            identified_metadata_columns=identified_metadata_columns,
            max_metadata_char_length_budget=max_metadata_char_length_budget,
            df_columns=df_column_names,
        )

        llama_document = Document(text=text_to_chunk, metadata=doc_metadata)

        try:
            nodes = splitter.get_nodes_from_documents([llama_document])
        except Exception as e:
            logger.error(
                f"Error splitting document from row {item_idx} (orig_id: {doc_metadata.get('original_document_id')}): {e}"
            )
            continue

        for i, node in enumerate(nodes):
            chunk_id_base = doc_metadata.get("original_document_id", f"row_{item_idx}")
            chunk_dict = {
                "chunk_id": f"{file_id}_{chunk_id_base}_chunk_{i}",
                "file_id": file_id,
                "original_document_id": doc_metadata.get("original_document_id"),
                "text": node.get_content(),
                "metadata": node.metadata.copy(),
            }
            chunk_dict["metadata"]["chunk_sequence"] = i
            all_chunks.append(chunk_dict)

    logger.info(
        f"Chunked {len(df)} documents from file_id '{file_id}' into {len(all_chunks)} chunks."
    )
    if not all_chunks and not df.empty:
        logger.warning(
            f"No chunks were generated from {len(df)} non-empty documents for file_id '{file_id}'. Review content and chunking parameters."
        )
    return all_chunks
