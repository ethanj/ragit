"""
Service layer for CSV file validation and initial content identification.
"""

import logging
from io import StringIO
from typing import Dict, Tuple

import pandas as pd
from fastapi import HTTPException, UploadFile

# from app.core.config import settings # Not directly used by these functions yet, but good to keep if settings influence validation

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 50  # Consider moving to settings if configurable
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Common column names that might contain primary textual content
POTENTIAL_CONTENT_COLUMNS = [
    "content",
    "text",
    "body",
    "description",
    "summary",
    "article",
    "comment",
    "content_text",
    "content_html",
]

# Common column names for metadata
POTENTIAL_METADATA_COLUMNS = {
    "title": ["title", "name", "headline"],
    "url": ["url", "link", "href"],
    "id": ["id", "identifier", "uuid", "guid"],
}

# Common column names for HTML content
POTENTIAL_HTML_COLUMNS = [  # This seems unused by the moved functions, but was grouped with them
    "html",
    "content_html",
]


class CSVValidationError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=400, detail=detail)


def _validate_csv_content_and_structure(
    contents: bytes, file_name_for_logging: str
) -> Tuple[pd.DataFrame, str, Dict[str, str]]:
    """
    Validates CSV content, identifies primary content and metadata columns.

    Args:
        contents: The byte content of the CSV file.
        file_name_for_logging: The name of the file for logging purposes.

    Returns:
        A tuple containing:
            - pandas DataFrame with the CSV content.
            - The name of the identified primary content column.
            - A dictionary mapping identified metadata types (e.g., 'title', 'url') to their column names.

    Raises:
        CSVValidationError: If validation fails.
    """
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise CSVValidationError(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit.")

    if not contents:
        raise CSVValidationError("File is empty.")

    try:
        csv_string = contents.decode("utf-8")
    except UnicodeDecodeError:
        raise CSVValidationError("File encoding is not UTF-8.")

    try:
        df = pd.read_csv(StringIO(csv_string))
    except pd.errors.EmptyDataError:
        raise CSVValidationError("CSV file is empty or contains no data.")
    except Exception as e:
        logger.error(f"Error parsing CSV '{file_name_for_logging}': {e}")
        raise CSVValidationError(
            f"Could not parse CSV file '{file_name_for_logging}': {e}"
        )

    if df.empty:
        raise CSVValidationError(
            f"CSV file '{file_name_for_logging}' resulted in an empty DataFrame."
        )

    identified_content_column = None
    df_columns_lower = {col.lower(): col for col in df.columns}

    for potential_col_name in POTENTIAL_CONTENT_COLUMNS:
        if potential_col_name.lower() in df_columns_lower:
            identified_content_column = df_columns_lower[potential_col_name.lower()]
            logger.info(
                f"Identified '{identified_content_column}' as the primary content column in '{file_name_for_logging}'."
            )
            break

    if not identified_content_column:
        raise CSVValidationError(
            f"Could not identify a primary content column in '{file_name_for_logging}'. "
            f"Expected one of: {', '.join(POTENTIAL_CONTENT_COLUMNS)}. "
            f"Found columns: {', '.join(df.columns)}"
        )

    identified_metadata_columns = {}
    for meta_type, potential_names in POTENTIAL_METADATA_COLUMNS.items():
        # TODO: Handle cases where multiple metadata columns for the SAME meta_type might be present
        # (e.g., if POTENTIAL_METADATA_COLUMNS["title"] = ["header", "subject"] and CSV has both).
        # Current logic takes the first one found based on the order in `potential_names`.
        # Future enhancements could include: concatenating values, prioritizing based on non-null counts,
        # or allowing user configuration for tie-breaking.
        for potential_name in potential_names:
            if potential_name.lower() in df_columns_lower:
                column_name = df_columns_lower[potential_name.lower()]
                identified_metadata_columns[meta_type] = column_name
                logger.info(
                    f"Identified '{column_name}' as the '{meta_type}' column in '{file_name_for_logging}'."
                )
                break  # Found for this meta_type, move to next

    if not identified_metadata_columns:
        logger.warning(
            f"Could not identify any standard metadata columns (e.g., title, url, id) in '{file_name_for_logging}'. "
            f"Found columns: {', '.join(df.columns)}"
        )

    logger.info(
        f"Successfully validated CSV: {file_name_for_logging}, shape: {df.shape}. "
        f"Content column: '{identified_content_column}'. "
        f"Metadata columns: {identified_metadata_columns}"
    )
    return df, identified_content_column, identified_metadata_columns


def load_and_validate_csv(file: UploadFile) -> Tuple[pd.DataFrame, str, Dict[str, str]]:
    """
    Loads an uploaded CSV file, validates it, and identifies content/metadata columns.

    Args:
        file: The uploaded CSV file (FastAPI UploadFile).

    Returns:
        A tuple: (pandas DataFrame, identified content column name, identified metadata columns map).
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise CSVValidationError(
            f"Invalid file type: {file.filename}. Only CSV files are accepted."
        )

    contents = file.file.read()
    file.file.seek(0)  # Reset file pointer for potential re-reads

    return _validate_csv_content_and_structure(contents, file.filename)


def load_and_validate_csv_from_path(
    file_path: str,
) -> Tuple[pd.DataFrame, str, Dict[str, str]]:
    """
    Loads a CSV file from a given path, validates it, and identifies content/metadata columns.

    Args:
        file_path: The local path to the CSV file.

    Returns:
        A tuple: (pandas DataFrame, identified content column name, identified metadata columns map).
    """
    file_name = file_path.split("/")[-1]
    if not file_name.endswith(".csv"):
        raise CSVValidationError(
            f"Invalid file type: {file_name} (from path {file_path}). Only CSV files are accepted."
        )

    try:
        with open(file_path, "rb") as f:
            contents = f.read()
    except FileNotFoundError:
        raise CSVValidationError(f"File not found at path: {file_path}")
    except Exception as e:
        raise CSVValidationError(f"Error reading file at path {file_path}: {e}")

    return _validate_csv_content_and_structure(contents, file_name)
