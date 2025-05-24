"""
Tests for the ingestion service.
"""

from io import BytesIO

import pandas as pd
import pytest
from fastapi import UploadFile

from app.core.config import settings as app_settings
from app.services.embedding_service import get_embedding_model
from app.services.file_validation_service import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    CSVValidationError,
    load_and_validate_csv,
)
from app.services.ingestion_orchestrator import (
    EmbeddingError,
    VectorDBError,
    chunk_data,
    clean_and_preprocess_data,
    embed_chunks,
    store_chunks_in_vector_db,
)


# Helper to create a mock UploadFile
def create_mock_upload_file(
    filename: str, content: bytes, content_type: str = "text/csv"
) -> UploadFile:
    mock_file = BytesIO(content)
    upload_file = UploadFile(filename=filename, file=mock_file)
    # FastAPI's UploadFile sets content_type from headers, mock it if necessary
    # For current load_and_validate_csv, filename extension is primary check,
    # but good to be aware if content_type was used directly.
    # setattr(upload_file, 'content_type', content_type) # if needed
    return upload_file


@pytest.fixture
def valid_csv_content() -> str:
    return "id,title,url,content_text,content_html\n1,Test Title,http://example.com,Hello world,<h1>Hello</h1>"


@pytest.fixture
def csv_missing_content_cols_content() -> str:
    return "id,title,url\n1,Test Title,http://example.com"


def test_load_and_validate_csv_valid(valid_csv_content: str):
    """Test loading a valid CSV file."""
    mock_file = create_mock_upload_file("valid.csv", valid_csv_content.encode("utf-8"))
    result = load_and_validate_csv(mock_file)
    assert isinstance(result, tuple)
    df, content_col, meta_cols = result
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert content_col is not None  # or a specific expected column name
    assert isinstance(meta_cols, dict)


def test_load_and_validate_csv_invalid_extension():
    """Test with an invalid file extension."""
    mock_file = create_mock_upload_file("invalid.txt", b"some content")
    with pytest.raises(CSVValidationError, match="Invalid file type: invalid.txt"):
        load_and_validate_csv(mock_file)


def test_load_and_validate_csv_file_too_large():
    """Test with a file that exceeds the size limit."""
    # Use the imported constants from app.services.ingestion
    large_content = b"a" * (MAX_FILE_SIZE_BYTES + 1)
    mock_file = create_mock_upload_file("large.csv", large_content)
    with pytest.raises(
        CSVValidationError, match=f"File size exceeds {MAX_FILE_SIZE_MB}MB limit."
    ):
        load_and_validate_csv(mock_file)


def test_load_and_validate_csv_empty_file_content():
    """Test with an empty file content."""
    mock_file = create_mock_upload_file("empty.csv", b"")
    with pytest.raises(CSVValidationError, match="File is empty."):
        load_and_validate_csv(mock_file)


def test_load_and_validate_csv_non_utf8_encoding():
    """Test with non-UTF-8 encoded content."""
    non_utf8_content = "id,text\n1,résumé".encode(
        "latin-1"
    )  # latin-1 is not utf-8 for résumé
    mock_file = create_mock_upload_file("encoding.csv", non_utf8_content)
    with pytest.raises(CSVValidationError, match="File encoding is not UTF-8."):
        load_and_validate_csv(mock_file)


def test_load_and_validate_csv_empty_dataframe_after_read():
    """Test CSV that results in an empty DataFrame (e.g., only headers)."""
    csv_content_only_headers = "id,content_text,content_html\n".encode("utf-8")
    mock_file = create_mock_upload_file("headers.csv", csv_content_only_headers)
    with pytest.raises(
        CSVValidationError,
        match=r"400: CSV file 'headers.csv' resulted in an empty DataFrame.",
    ):
        load_and_validate_csv(mock_file)


def test_load_and_validate_csv_missing_required_content_columns(
    csv_missing_content_cols_content: str,
):
    """Test CSV missing a potential primary content column."""
    mock_file = create_mock_upload_file(
        "missing_cols.csv", csv_missing_content_cols_content.encode("utf-8")
    )
    expected_message_regex = (
        r"400: Could not identify a primary content column in 'missing_cols.csv'. "
        r"Expected one of: .* Found columns: .*"
    )
    with pytest.raises(CSVValidationError, match=expected_message_regex):
        load_and_validate_csv(mock_file)


def test_load_and_validate_csv_valid_with_content_html_only():
    """Test valid CSV with only content_html (content_text can be missing)."""
    csv_content = "id,title,content_html\n1,Test HTML,<b>Hello</b>".encode("utf-8")
    mock_file = create_mock_upload_file("html_only.csv", csv_content)
    result = load_and_validate_csv(mock_file)
    assert isinstance(result, tuple)
    df, content_col, _ = result  # Don't care about meta_cols for this specific check
    assert isinstance(df, pd.DataFrame)
    assert content_col == "content_html"


def test_load_and_validate_csv_valid_with_content_text_only():
    """Test valid CSV with only content_text (content_html can be missing)."""
    csv_content = "id,title,content_text\n1,Test Text,Just text".encode("utf-8")
    mock_file = create_mock_upload_file("text_only.csv", csv_content)
    result = load_and_validate_csv(mock_file)
    assert isinstance(result, tuple)
    df, content_col, _ = result
    assert isinstance(df, pd.DataFrame)
    assert content_col == "content_text"


def test_load_and_validate_csv_no_utf8_encoding():
    """Test CSV with non-UTF-8 encoding."""
    # ISO-8859-1 (Latin-1) content with a character not in ASCII
    non_utf8_content = "id,text\n1,café".encode("iso-8859-1")
    mock_file = create_mock_upload_file("latin1.csv", non_utf8_content)
    with pytest.raises(CSVValidationError, match=r"400: File encoding is not UTF-8."):
        load_and_validate_csv(mock_file)


def test_load_and_validate_csv_malformed_csv_content():
    """Test with content that cannot be parsed as CSV."""
    malformed_text_content = 'header1,header2\nvalue1,"unclosed quote field'.encode(
        "utf-8"
    )
    mock_file_text = create_mock_upload_file(
        "malformed_text.csv", malformed_text_content
    )
    # This should trigger the generic "Could not parse CSV file" which wraps the pandas error.
    # The exact pandas error message can vary, so we match part of our wrapper message.
    with pytest.raises(
        CSVValidationError,
        match=r"400: Could not parse CSV file 'malformed_text.csv': Error tokenizing data.",
    ):
        load_and_validate_csv(mock_file_text)


# --- Tests for clean_and_preprocess_data ---


@pytest.fixture
def df_for_cleaning() -> pd.DataFrame:
    data = {
        "id": ["ID1", "ID2", "ID3", "ID4", "ID5", "ID6", "ID7"],
        "title": [
            "T1",
            "T2_HTML_Only",
            "T3_Override",
            "T4_Text_Only",
            "T5_Duplicate",
            "T6_Whitespace",
            "T7_Empty_HTML",
        ],
        "content_text": [
            "Some text for ID1.",  # ID1
            None,  # ID2
            "This text should be ignored.",  # ID3
            "  Text only with spaces.  ",  # ID4
            "Duplicate content.",  # ID5
            "  \n\t  Whitespace galore  \n  ",  # ID6
            None,  # ID7
        ],
        "content_html": [
            "<p>Some HTML for <b>ID1</b>.</p>",  # ID1 (HTML preferred)
            "<p>HTML Only With details.</p>",  # ID2
            "<p>Override HTML <b>This should be used.</b></p>",  # ID3
            None,  # ID4
            "<p>Duplicate content.</p>",  # ID5 (same as text, but HTML derived)
            "<p>  Leading/Trailing space in HTML  </p>",  # ID6
            "<p> </p><br /><i></i>",  # ID7 (becomes empty after stripping)
        ],
        "other_col": ["A", "B", "C", "D", "E", "F", "G"],
        "url": ["url1", "url2", "url3", "url4", "url5", "url6", "url7"],
    }
    return pd.DataFrame(data)


def test_clean_and_preprocess_data_html_stripping(df_for_cleaning: pd.DataFrame):
    """Test HTML stripping and population of 'processed_text'."""
    df_copy = df_for_cleaning.copy()
    # Assuming 'content_html' was identified as primary, and 'id', 'title', 'url' as metadata
    # For rows where content_html is primary:
    cleaned_df = clean_and_preprocess_data(
        df_copy,
        content_column_name="content_html",
        identified_metadata_columns={"id": "id", "title": "title", "url": "url"},
    )
    # ID1: HTML preferred and processed. BeautifulSoup with separator=' ' might leave a space before punctuation after tags.
    assert (
        cleaned_df[cleaned_df["id"] == "ID1"]["processed_text"].iloc[0]
        == "Some HTML for ID1 ."  # Adjusted to expect the space from BS4
    )
    # ID2: HTML only, processed
    assert (
        cleaned_df[cleaned_df["id"] == "ID2"]["processed_text"].iloc[0]
        == "HTML Only With details."
    )
    # ID3: HTML overrides text, processed
    assert (
        cleaned_df[cleaned_df["id"] == "ID3"]["processed_text"].iloc[0]
        == "Override HTML This should be used."
    )

    # Test with 'content_text' as primary when HTML is None
    df_text_primary_copy = df_for_cleaning[df_for_cleaning["id"] == "ID4"].copy()
    cleaned_df_text_primary = clean_and_preprocess_data(
        df_text_primary_copy,
        content_column_name="content_text",
        identified_metadata_columns={"id": "id", "title": "title", "url": "url"},
    )
    assert (
        cleaned_df_text_primary[cleaned_df_text_primary["id"] == "ID4"][
            "processed_text"
        ].iloc[0]
        == "Text only with spaces."
    )


def test_clean_and_preprocess_data_whitespace_handling(df_for_cleaning: pd.DataFrame):
    """Test whitespace handling from both text and HTML."""
    df_copy = df_for_cleaning.copy()
    # For ID4 (Text only with spaces), assuming content_text is primary
    df_id4 = df_copy[df_copy["id"] == "ID4"].copy()
    cleaned_id4 = clean_and_preprocess_data(
        df_id4,
        content_column_name="content_text",
        identified_metadata_columns={"id": "id"},
    )
    assert cleaned_id4["processed_text"].iloc[0] == "Text only with spaces."

    # For ID6 (Whitespace galore in text, clean HTML), assuming content_text is primary
    df_id6_text_primary = df_copy[df_copy["id"] == "ID6"].copy()
    cleaned_id6_text_primary = clean_and_preprocess_data(
        df_id6_text_primary,
        content_column_name="content_text",
        identified_metadata_columns={"id": "id"},
    )
    assert cleaned_id6_text_primary["processed_text"].iloc[0] == "Whitespace galore"

    # For ID6 (Whitespace galore in text, clean HTML), assuming content_html is primary
    df_id6_html_primary = df_copy[df_copy["id"] == "ID6"].copy()
    cleaned_id6_html_primary = clean_and_preprocess_data(
        df_id6_html_primary,
        content_column_name="content_html",
        identified_metadata_columns={"id": "id"},
    )
    assert (
        cleaned_id6_html_primary["processed_text"].iloc[0]
        == "Leading/Trailing space in HTML"
    )


def test_clean_and_preprocess_data_deduplication(df_for_cleaning: pd.DataFrame):
    """Test deduplication based on processed_text content."""
    # ID5 has "Duplicate content." in text and "<p>Duplicate content.</p>" in HTML.
    # If content_html is primary, processed_text becomes "Duplicate content."
    # If content_text is primary (and HTML is None or different), processed_text is "Duplicate content."
    # Let's create a scenario for deduplication based on the output of processed_text
    data_for_dedup = {
        "id": ["D1", "D2", "D3"],
        "content_text": ["Unique A", "Duplicate Target", "Unique B"],
        "content_html": [
            None,
            "<p>Duplicate Target</p>",
            None,
        ],  # D2 will be primary source
    }
    df_dedup_test = pd.DataFrame(data_for_dedup)

    # First pass where content_html is primary for D2, content_text for D1, D3
    # This setup is a bit complex for clean_and_preprocess which takes ONE content_column_name.
    # Let's simplify: assume load_and_validate determined one content source, then clean it.
    # Create a df that *would be the input* to clean_and_preprocess if content column was already selected
    # and potentially pre-processed by `_handle_html_and_select_content` (which is internal to clean_and_preprocess)
    # The current `clean_and_preprocess_data` takes a `content_column_name` and processes based on that.
    # So, if content_column_name is 'content_html', it uses that. If 'content_text', it uses that.

    # Scenario: content_column_name = 'content_html' (so D1 and D3 will have None processed_text initially before text fallback)
    # This won't work as clean_and_preprocess expects the content_column_name to exist and be string.
    # We need to test the state *after* initial content selection and HTML stripping has populated a base 'cleaned_content'.
    # The deduplication happens on 'cleaned_content'.

    # Let's construct a DataFrame as if it's *just before* deduplication step inside clean_and_preprocess_data.
    # This means 'cleaned_content' column exists.
    # The public API is clean_and_preprocess_data(df, content_column_name, identified_metadata_columns)
    # So we must call it with that.

    df_to_dedup = pd.DataFrame(
        {
            "id": ["dup1", "dup2", "uniq1", "dup3"],
            "title": ["t_d1", "t_d2", "t_u1", "t_d3"],
            "some_content_col": [
                "<p>Text A</p>",  # Becomes "Text A"
                "Text A",  # Stays "Text A"
                "Text B",  # Stays "Text B"
                "<p>Text A</p>",  # Becomes "Text A"
            ],
        }
    )

    cleaned_df = clean_and_preprocess_data(
        df_to_dedup,
        content_column_name="some_content_col",
        identified_metadata_columns={"id": "id", "title": "title"},
    )

    assert len(cleaned_df) == 2  # "Text A" and "Text B"
    assert "Text A" in cleaned_df["processed_text"].values
    assert "Text B" in cleaned_df["processed_text"].values
    # Check that we kept the first occurrences (dup1, uniq1)
    assert "dup1" in cleaned_df["id"].values
    assert "uniq1" in cleaned_df["id"].values


def test_clean_and_preprocess_data_empty_after_strip(df_for_cleaning: pd.DataFrame):
    """Test rows that become empty after HTML stripping are dropped."""
    # ID7: content_html is "<p> </p><br /><i></i>", content_text is None
    # BeautifulSoup("<p> </p><br /><i></i>", "html.parser").get_text(separator=" ", strip=True) == ""
    # So this row should be dropped.
    df_id7 = df_for_cleaning[df_for_cleaning["id"] == "ID7"].copy()
    cleaned_df = clean_and_preprocess_data(
        df_id7,
        content_column_name="content_html",
        identified_metadata_columns={"id": "id"},
    )
    assert len(cleaned_df) == 0


def test_clean_and_preprocess_data_no_content_column_provided():
    """Test behavior when DataFrame is missing the specified content_column_name."""
    df_no_content = pd.DataFrame({"id": [1], "title": ["T1"], "other": ["data"]})
    with pytest.raises(
        ValueError, match="Critical error: content column 'missing_content' missing."
    ):
        clean_and_preprocess_data(
            df_no_content.copy(),
            content_column_name="missing_content",  # A column that doesn't exist
            identified_metadata_columns={"id": "id"},
        )


def test_clean_and_preprocess_data_preserves_other_columns(
    df_for_cleaning: pd.DataFrame,
):
    """Test that other columns are preserved through cleaning and deduplication."""
    # Use ID1, ID2, ID4 (HTML primary, HTML primary, Text primary)
    # ID1 (orig index 0): "Some HTML for ID1.", other_col A, url url1
    # ID2 (orig index 1): "HTML Only With details.", other_col B, url url2
    # ID4 (orig index 3): "Text only with spaces.", other_col D, url url4
    df_subset = df_for_cleaning[
        df_for_cleaning["id"].isin(["ID1", "ID2", "ID4"])
    ].copy()

    # Simulate content_column_name and identified_metadata_columns as if from load_and_validate_csv
    # This is tricky as load_and_validate would pick one content col for the whole df.
    # For this test, let's assume 'content_html' was picked, and 'content_text' is a fallback within clean_and_preprocess
    # if content_html is empty for a row (which it is for ID4).
    # The current clean_and_preprocess logic takes ONE content_column_name.
    # If content_column_name is 'content_html', it will try to use it. If row['content_html'] is NaN/None,
    # it will then try row[original_text_col_name_if_any].
    # This internal fallback is based on content_column_name not being one of the hardcoded html names.

    # Let's simplify: test with a DataFrame where a single content_column_name makes sense.
    df_simple_preserve = pd.DataFrame(
        {
            "my_id": ["id_A", "id_B"],
            "main_text": ["<p>Hello A</p>", "Text for B"],
            "extra_meta1": ["metaA1", "metaB1"],
            "another_url_col": ["urlA", "urlB"],
        }
    )

    cleaned_df = clean_and_preprocess_data(
        df_simple_preserve,
        content_column_name="main_text",
        identified_metadata_columns={"id": "my_id", "url": "another_url_col"},
    )

    assert "processed_text" in cleaned_df.columns
    assert "my_id" in cleaned_df.columns  # Explicitly identified
    assert "another_url_col" in cleaned_df.columns  # Explicitly identified
    assert "extra_meta1" in cleaned_df.columns  # Auto-preserved
    assert len(cleaned_df) == 2
    assert (
        cleaned_df[cleaned_df["my_id"] == "id_A"]["processed_text"].iloc[0] == "Hello A"
    )
    assert cleaned_df[cleaned_df["my_id"] == "id_A"]["extra_meta1"].iloc[0] == "metaA1"


def test_clean_and_preprocess_data_handles_all_none_content(
    df_for_cleaning: pd.DataFrame,
):
    """Test rows where the designated content_column_name is None/NaN."""
    # ID2 has content_text=None, content_html="<p>HTML Only With details.</p>"
    # If content_column_name="content_text" is passed, it should then try to use content_html if content_text is null.
    # This specific internal fallback needs content_column_name NOT to be 'content_html' etc.

    # Test case: content_column_name is 'content_text'. Row has None for 'content_text', but valid 'content_html'
    df_id2 = df_for_cleaning[
        df_for_cleaning["id"] == "ID2"
    ].copy()  # content_text is None, content_html is not
    cleaned_df_id2 = clean_and_preprocess_data(
        df_id2,
        content_column_name="content_text",  # This row has None here
        identified_metadata_columns={"id": "id", "title": "title"},
    )
    # The logic is: use content_column_name. If it's an HTML col, strip. If not, use as is.
    # Then, if resulting cleaned_content is empty, AND original content_column_name was NOT an HTML col,
    # AND an HTML col *exists*, THEN try the HTML col.
    # For ID2, if content_column_name="content_text" (None), initial cleaned_content is None/empty.
    # Then it will check for content_html, find it, parse it.
    assert len(cleaned_df_id2) == 1
    assert cleaned_df_id2["processed_text"].iloc[0] == "HTML Only With details."

    # Test case: content_column_name is 'some_other_text'. Both it and content_html are None.
    df_all_none_direct = pd.DataFrame(
        {
            "id": ["N1"],
            "some_other_text": [None],
            "content_html": [None],  # Explicitly no HTML fallback either
        }
    )
    cleaned_all_none = clean_and_preprocess_data(
        df_all_none_direct,
        content_column_name="some_other_text",
        identified_metadata_columns={"id": "id"},
    )
    assert (
        len(cleaned_all_none) == 0
    )  # Should be dropped as it results in empty processed_text


# --- Tests for chunk_data ---


@pytest.fixture
def df_for_chunking() -> pd.DataFrame:
    data = {
        "id": ["doc1", "doc2"],  # Original doc IDs from CSV
        "title": ["First Document", "Second Document With Longer Text"],
        "url": ["http://example.com/doc1", "http://example.com/doc2"],
        "processed_text": [
            "This is the first document. It is short and sweet.",  # ~10 words
            (
                "This is the second document. It has a significantly longer text content designed to test "
                "the chunking mechanism. We want to see if SentenceSplitter correctly breaks this down "
                "into multiple pieces based on the specified chunk size and overlap parameters. "
                "Let us add even more sentences to ensure it spans across multiple chunks. "
                "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
                "This should be enough text to generate at least two, if not three, chunks. "
                "Final sentence for good measure."
            ),  # ~80 words
        ],
        "other_meta_col": [
            "meta_val1",
            "meta_val2",
        ],  # Renamed to avoid collision with 'metadata' key in chunks
        "another_col_to_keep": ["keep1", "keep2"],
    }
    return pd.DataFrame(data)


def test_chunk_data_output_structure_and_metadata(df_for_chunking: pd.DataFrame):
    """Test the basic output structure and metadata preservation of chunk_data."""
    file_id = "test_file_123"
    # These would typically be derived from load_and_validate_csv output
    identified_meta = {
        "id": "id",
        "title": "title",
        "url": "url",
        "other_meta_col": "other_meta_col",
        "another_col_to_keep": "another_col_to_keep",
    }
    chunks = chunk_data(
        df_for_chunking, file_id=file_id, identified_metadata_columns=identified_meta
    )

    assert isinstance(chunks, list)
    assert len(chunks) > 0

    # Check first chunk (from doc1)
    chunk1 = next(c for c in chunks if c["original_document_id"] == "doc1")
    assert chunk1["chunk_id"].startswith(f"{file_id}_doc1_chunk_")
    assert chunk1["file_id"] == file_id
    assert chunk1["text"] == "This is the first document. It is short and sweet."
    assert chunk1["metadata"]["original_document_id"] == "doc1"
    assert chunk1["metadata"]["file_id"] == file_id
    assert chunk1["metadata"]["title"] == "First Document"
    assert chunk1["metadata"]["url"] == "http://example.com/doc1"
    assert chunk1["metadata"]["other_meta_col"] == "meta_val1"
    assert chunk1["metadata"]["another_col_to_keep"] == "keep1"
    assert "chunk_sequence" in chunk1["metadata"]
    assert chunk1["metadata"]["chunk_sequence"] == 0


def test_chunk_data_short_text_single_chunk(df_for_chunking: pd.DataFrame):
    """Test that short text results in a single chunk."""
    file_id = "test_file_single_chunk"
    df_short_doc = df_for_chunking[df_for_chunking["id"] == "doc1"].copy()
    identified_meta = {
        "id": "id",
        "title": "title",
        "url": "url",
        "other_meta_col": "other_meta_col",
        "another_col_to_keep": "another_col_to_keep",
    }
    chunks = chunk_data(
        df_short_doc, file_id=file_id, identified_metadata_columns=identified_meta
    )

    assert len(chunks) == 1
    assert chunks[0]["text"] == "This is the first document. It is short and sweet."
    assert chunks[0]["metadata"]["chunk_sequence"] == 0


def test_chunk_data_long_text_multiple_chunks(df_for_chunking: pd.DataFrame):
    """Test that long text results in multiple chunks with correct sequencing."""
    file_id = "test_file_multi_chunk"
    df_long_doc = df_for_chunking[df_for_chunking["id"] == "doc2"].copy()
    identified_meta = {
        "id": "id",
        "title": "title",
        "url": "url",
        "other_meta_col": "other_meta_col",
        "another_col_to_keep": "another_col_to_keep",
    }

    small_chunk_size = 60
    small_chunk_overlap = 5
    chunks = chunk_data(
        df_long_doc,
        file_id=file_id,
        identified_metadata_columns=identified_meta,
        chunk_size=small_chunk_size,
        chunk_overlap=small_chunk_overlap,
    )

    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_id"] == f"{file_id}_doc2_chunk_{i}"
        assert chunk["metadata"]["original_document_id"] == "doc2"
        assert chunk["metadata"]["file_id"] == file_id
        assert chunk["metadata"]["title"] == "Second Document With Longer Text"
        assert chunk["metadata"]["url"] == "http://example.com/doc2"
        assert chunk["metadata"]["other_meta_col"] == "meta_val2"
        assert chunk["metadata"]["another_col_to_keep"] == "keep2"
        assert chunk["metadata"]["chunk_sequence"] == i
        assert len(chunk["text"]) > 0


def test_chunk_data_empty_dataframe():
    """Test chunk_data with an empty DataFrame."""
    empty_df = pd.DataFrame(columns=["id", "title", "processed_text", "other_col"])
    identified_meta = {"id": "id", "title": "title", "other_col": "other_col"}
    chunks = chunk_data(
        empty_df,
        file_id="test_file_empty_df",
        identified_metadata_columns=identified_meta,
    )
    assert chunks == []


def test_chunk_data_missing_processed_text_column():
    """Test chunk_data if DataFrame is missing the 'processed_text' column."""
    df_no_text = pd.DataFrame({"id": ["doc1"], "title": ["Title Only"]})
    identified_meta = {"id": "id", "title": "title"}
    chunks = chunk_data(
        df_no_text,
        file_id="test_file_no_text_col",
        identified_metadata_columns=identified_meta,
    )
    assert chunks == []


# --- Tests for embed_chunks and get_embedding_model ---

from unittest.mock import MagicMock, patch

from openai import OpenAIError  # For testing exceptions


@pytest.fixture
def sample_chunks_for_embedding() -> list[dict]:
    return [
        {
            "chunk_id": "file1_doc1_chunk_0",
            "file_id": "file1",
            "original_document_id": "doc1",
            "text": "This is the first chunk of text.",
            "metadata": {"title": "Doc 1", "chunk_sequence": 0},
        },
        {
            "chunk_id": "file1_doc1_chunk_1",
            "file_id": "file1",
            "original_document_id": "doc1",
            "text": "This is the second chunk of text from the same document.",
            "metadata": {"title": "Doc 1", "chunk_sequence": 1},
        },
        {
            "chunk_id": "file1_doc2_chunk_0",
            "file_id": "file1",
            "original_document_id": "doc2",
            "text": "Text from a different document.",
            "metadata": {"title": "Doc 2", "chunk_sequence": 0},
        },
    ]


@patch("app.services.embedding_service.get_embedding_model")
def test_embed_chunks_successful(
    mock_get_embedding_model, sample_chunks_for_embedding: list[dict]
):
    """Test successful embedding of chunks."""
    mock_embed_model = MagicMock()
    # Simulate get_text_embedding_batch returning a list of embeddings
    mock_embeddings = [[0.1] * 10, [0.2] * 10, [0.3] * 10]  # Example: 10-dim embeddings
    mock_embed_model.get_text_embedding_batch.return_value = mock_embeddings
    mock_get_embedding_model.return_value = mock_embed_model

    # Ensure OPENAI_API_KEY is mocked in settings for get_embedding_model internal check if any
    with patch.object(app_settings, "OPENAI_API_KEY", "test_api_key_value"):
        embedded_chunks = embed_chunks(sample_chunks_for_embedding.copy())

    assert len(embedded_chunks) == len(sample_chunks_for_embedding)
    for i, chunk in enumerate(embedded_chunks):
        assert "embedding" in chunk
        assert chunk["embedding"] == mock_embeddings[i]

    texts_to_embed = [c["text"] for c in sample_chunks_for_embedding]
    mock_embed_model.get_text_embedding_batch.assert_called_once_with(
        texts_to_embed, show_progress=True
    )


@patch("app.services.embedding_service.get_embedding_model")
def test_embed_chunks_openai_error(
    mock_get_embedding_model, sample_chunks_for_embedding: list[dict]
):
    """Test EmbeddingError is raised when OpenAIError occurs."""
    mock_embed_model = MagicMock()
    mock_embed_model.get_text_embedding_batch.side_effect = OpenAIError(
        "Simulated OpenAI API Error"
    )
    mock_get_embedding_model.return_value = mock_embed_model

    with patch.object(app_settings, "OPENAI_API_KEY", "test_api_key_value"):
        with pytest.raises(
            EmbeddingError,
            match="Batch embedding generation failed: Simulated OpenAI API Error",
        ):
            embed_chunks(sample_chunks_for_embedding.copy())


@patch("app.services.embedding_service.get_embedding_model")
def test_embed_chunks_value_error(
    mock_get_embedding_model, sample_chunks_for_embedding: list[dict]
):
    """Test EmbeddingError is raised when a generic ValueError occurs during embedding."""
    mock_embed_model = MagicMock()
    mock_embed_model.get_text_embedding_batch.side_effect = ValueError(
        "Simulated generic embedding error"
    )
    mock_get_embedding_model.return_value = mock_embed_model

    with patch.object(app_settings, "OPENAI_API_KEY", "test_api_key_value"):
        with pytest.raises(
            EmbeddingError,
            match="Batch embedding generation failed: Simulated generic embedding error",
        ):
            embed_chunks(sample_chunks_for_embedding.copy())


def test_embed_chunks_empty_list():
    """Test embed_chunks with an empty list of chunks."""
    # No mocking needed as get_embedding_model won't be called if chunks_data is empty
    # and embed_chunks should handle this gracefully.
    result = embed_chunks([])
    assert result == []


# Test get_embedding_model directly (now from embedding_service)
@pytest.mark.skipif(
    not app_settings.OPENAI_API_KEY
    or app_settings.OPENAI_API_KEY == "your_openai_api_key_here",
    reason="OPENAI_API_KEY not set or is placeholder in environment",
)
def test_get_embedding_model_integration():
    """Test that get_embedding_model returns a valid OpenAIEmbedding model if API key is present."""
    model = get_embedding_model()
    assert model is not None
    # Further checks could involve model type or a simple embedding call


@patch("app.services.embedding_service.OpenAIEmbedding")
def test_get_embedding_model_no_api_key(mock_openai_embedding):
    get_embedding_model.cache_clear()  # Clear cache before test
    original_api_key = app_settings.OPENAI_API_KEY
    try:
        app_settings.OPENAI_API_KEY = ""  # Simulate missing API key
        with pytest.raises(
            ValueError,  # Expect ValueError
            match="OPENAI_API_KEY is not set or is a placeholder. Please set it in your environment variables or .env file.",
        ):
            get_embedding_model()
    finally:
        app_settings.OPENAI_API_KEY = original_api_key


# --- Tests for store_chunks_in_vector_db ---


@pytest.fixture
def sample_embedded_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "file1_doc1_chunk_0",
            "text": "Text for chunk 0",
            "embedding": [0.1] * 10,
            "metadata": {"title": "Doc 1", "original_document_id": "doc1"},
        },
        {
            "chunk_id": "file1_doc1_chunk_1",
            "text": "Text for chunk 1",
            "embedding": [0.2] * 10,
            "metadata": {"title": "Doc 1", "original_document_id": "doc1"},
        },
    ]


@patch("app.services.vector_db_service.get_chroma_client")
def test_store_chunks_in_vector_db_successful(
    mock_get_chroma_client, sample_embedded_chunks: list[dict]
):
    """Test successful storage of chunks in ChromaDB."""
    mock_chroma_client = MagicMock()
    mock_collection = MagicMock()
    mock_get_chroma_client.return_value = mock_chroma_client
    mock_chroma_client.get_or_create_collection.return_value = mock_collection

    agent_id = "test_agent_001"
    store_chunks_in_vector_db(agent_id, sample_embedded_chunks)

    expected_collection_name = f"agent_{agent_id}_vectors"
    mock_chroma_client.get_or_create_collection.assert_called_once_with(
        name=expected_collection_name
    )

    expected_ids = [c["chunk_id"] for c in sample_embedded_chunks]
    expected_embeddings = [c["embedding"] for c in sample_embedded_chunks]
    expected_metadatas = [c["metadata"] for c in sample_embedded_chunks]
    expected_documents = [c["text"] for c in sample_embedded_chunks]

    mock_collection.upsert.assert_called_once_with(
        ids=expected_ids,
        embeddings=expected_embeddings,
        metadatas=expected_metadatas,
        documents=expected_documents,
    )


@patch("app.services.vector_db_service.get_chroma_client")
def test_store_chunks_in_vector_db_chroma_add_error(
    mock_get_chroma_client, sample_embedded_chunks: list[dict]
):
    """Test VectorDBError is raised if collection.upsert fails."""
    mock_chroma_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.upsert.side_effect = Exception("Simulated ChromaDB upsert error")
    mock_get_chroma_client.return_value = mock_chroma_client
    mock_chroma_client.get_or_create_collection.return_value = mock_collection

    agent_id = "test_agent_error"
    with pytest.raises(
        VectorDBError,
        match="ChromaDB upsert failed with unexpected error: Simulated ChromaDB upsert error",
    ):
        store_chunks_in_vector_db(agent_id, sample_embedded_chunks)


@patch("app.services.vector_db_service.get_chroma_client")
def test_store_chunks_in_vector_db_chroma_get_collection_error(
    mock_get_chroma_client, sample_embedded_chunks: list[dict]
):
    """Test VectorDBError is raised if get_or_create_collection fails."""
    mock_chroma_client = MagicMock()
    mock_chroma_client.get_or_create_collection.side_effect = Exception(
        "Simulated ChromaDB get_collection error"
    )
    mock_get_chroma_client.return_value = mock_chroma_client

    agent_id = "test_agent_get_error"
    with pytest.raises(
        VectorDBError,
        match="ChromaDB collection operation failed with unexpected error: Simulated ChromaDB get_collection error",
    ):
        store_chunks_in_vector_db(agent_id, sample_embedded_chunks)


@patch("app.services.vector_db_service.get_chroma_client")
def test_store_chunks_in_vector_db_empty_list(mock_get_chroma_client):
    """Test store_chunks_in_vector_db with an empty list of chunks."""
    # The function should return early if chunks_to_store is empty.
    store_chunks_in_vector_db("test_agent_empty", [])
    mock_get_chroma_client.assert_not_called()
