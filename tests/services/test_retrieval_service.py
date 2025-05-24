"""
Tests for the retrieval service.
"""

from unittest.mock import MagicMock, patch

import pytest
from FlagEmbedding import FlagReranker

from app.core.config import settings as app_settings
from app.services.retrieval_service import (
    RetrievalError,
    _format_semantic_results,
    hybrid_retrieve,
    keyword_search_bm25,
    reciprocal_rank_fusion,
    semantic_search,
)

# --- Tests for reciprocal_rank_fusion ---


def test_reciprocal_rank_fusion_empty_input():
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[]], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_reciprocal_rank_fusion_single_list():
    results1 = [
        {"id": "doc1", "score": 0.9, "text": "text1"},
        {"id": "doc2", "score": 0.8, "text": "text2"},
    ]
    fused = reciprocal_rank_fusion([results1], k=60)
    assert len(fused) == 2
    assert fused[0]["id"] == "doc1"
    assert fused[1]["id"] == "doc2"
    # Scores should be RRF scores, not original.
    # RRF score for rank 0 (doc1) = 1/(60+0) = 1/60
    # RRF score for rank 1 (doc2) = 1/(60+1) = 1/61
    assert fused[0]["score"] == pytest.approx(1 / 60)
    assert fused[1]["score"] == pytest.approx(1 / 61)


def test_reciprocal_rank_fusion_two_lists_no_overlap():
    results1 = [
        {"id": "doc1", "score": 0.9, "text": "text1"},
        {"id": "doc2", "score": 0.8, "text": "text2"},
    ]
    results2 = [
        {"id": "doc3", "score": 0.95, "text": "text3"},
        {"id": "doc4", "score": 0.75, "text": "text4"},
    ]
    fused = reciprocal_rank_fusion([results1, results2], k=60)
    assert len(fused) == 4
    # doc3 from results2 (rank 0) -> score 1/60
    # doc1 from results1 (rank 0) -> score 1/60
    # doc2 from results1 (rank 1) -> score 1/61
    # doc4 from results2 (rank 1) -> score 1/61
    # Sorted by score (desc), then by original order (doc3 before doc1 if scores equal)
    fused_ids = [r["id"] for r in fused]
    assert "doc3" in fused_ids
    assert "doc1" in fused_ids
    assert "doc2" in fused_ids
    assert "doc4" in fused_ids

    # RRF scores: (k=60)
    # doc1 (rank 0 in list1) = 1/60
    # doc3 (rank 0 in list2) = 1/60
    # doc2 (rank 1 in list1) = 1/61
    # doc4 (rank 1 in list2) = 1/61

    # Check that the top scores are 1/60 and next scores are 1/61
    assert fused[0]["score"] == pytest.approx(1 / 60)
    assert fused[1]["score"] == pytest.approx(1 / 60)
    assert fused[2]["score"] == pytest.approx(1 / 61)
    assert fused[3]["score"] == pytest.approx(1 / 61)
    # Check that the top two docs are doc1 and doc3 (order between them doesn't strictly matter for this test as scores are same)
    top_two_ids = {fused[0]["id"], fused[1]["id"]}
    assert top_two_ids == {"doc1", "doc3"}


def test_reciprocal_rank_fusion_with_overlap_different_ranks():
    results1 = [
        {"id": "docA", "score": 0.9, "text": "textA"},  # rank 0
        {"id": "docB", "score": 0.8, "text": "textB"},  # rank 1
    ]
    results2 = [
        {"id": "docC", "score": 0.95, "text": "textC"},  # rank 0
        {
            "id": "docA",
            "score": 0.85,
            "text": "textA_variant",
        },  # rank 1, overlaps with results1
    ]
    # RRF Scores (k=60):
    # docA: from results1 (rank 0) = 1/60. from results2 (rank 1) = 1/61. Total = 1/60 + 1/61
    # docB: from results1 (rank 1) = 1/61. Total = 1/61
    # docC: from results2 (rank 0) = 1/60. Total = 1/60

    # Expected order by RRF score: docA, docC, docB
    # docA_score = 1/60 + 1/61
    # docC_score = 1/60
    # docB_score = 1/61

    fused = reciprocal_rank_fusion([results1, results2], k=60)
    assert len(fused) == 3
    assert fused[0]["id"] == "docA"
    assert fused[0]["score"] == pytest.approx(1 / 60 + 1 / 61)
    assert fused[0]["text"] == "textA"  # Should keep text from first appearance

    assert fused[1]["id"] == "docC"
    assert fused[1]["score"] == pytest.approx(1 / 60)

    assert fused[2]["id"] == "docB"
    assert fused[2]["score"] == pytest.approx(1 / 61)


def test_reciprocal_rank_fusion_k_value_effect():
    results1 = [
        {"id": "doc1", "score": 0.9, "text": "text1"},
    ]
    fused_k1 = reciprocal_rank_fusion([results1], k=1)
    assert fused_k1[0]["score"] == pytest.approx(1 / (1 + 0))  # 1/1 = 1.0

    fused_k100 = reciprocal_rank_fusion([results1], k=100)
    assert fused_k100[0]["score"] == pytest.approx(1 / (100 + 0))  # 1/100


def test_reciprocal_rank_fusion_preserves_extra_metadata():
    """Ensure that extra fields in chunk dicts are preserved."""
    results1 = [
        {
            "id": "doc1",
            "score": 0.9,
            "text": "text1",
            "original_document_id": "csv_doc_A",
            "custom_field": "value1",
        }
    ]
    fused = reciprocal_rank_fusion([results1], k=60)
    assert len(fused) == 1
    assert fused[0]["id"] == "doc1"
    assert fused[0]["text"] == "text1"
    assert fused[0]["original_document_id"] == "csv_doc_A"
    assert fused[0]["custom_field"] == "value1"
    assert fused[0]["score"] == pytest.approx(1 / (60 + 0))


# --- Tests for semantic_search ---


@pytest.fixture
def mock_retrieval_embed_model():
    mock_model = MagicMock()
    # Simulate get_text_embedding returning a list (embedding for one query)
    mock_model.get_text_embedding.return_value = [0.1, 0.2, 0.3]
    return mock_model


@pytest.fixture
def mock_retrieval_chroma_client():
    mock_client = MagicMock()
    mock_collection = MagicMock()

    # Simulate a successful query_texts result from ChromaDB
    # Note: semantic_search uses query_embeddings, but test structure for query_texts is similar for response mocking.
    # The actual call inside semantic_search is collection.query(query_embeddings=...)
    # The _format_semantic_results helper expects a dict like this.
    mock_collection.query.return_value = {
        "ids": [["chunk1", "chunk2"]],
        "documents": [["text of chunk1", "text of chunk2"]],
        "metadatas": [[{"title": "Title1"}, {"title": "Title2"}]],
        "distances": [[0.1, 0.2]],  # distance = 1 - similarity_score
    }
    mock_client.get_collection.return_value = mock_collection
    return mock_client


@patch("app.services.retrieval_service.get_chroma_client")
@patch("app.services.retrieval_service.get_embedding_model")
@pytest.mark.asyncio
async def test_semantic_search_successful(
    mock_get_embedding_model,
    mock_get_chroma_client,
    mock_retrieval_embed_model,
    mock_retrieval_chroma_client,
):
    mock_get_embedding_model.return_value = mock_retrieval_embed_model
    mock_get_chroma_client.return_value = mock_retrieval_chroma_client

    agent_id = "test_agent_semantic"
    query_text = "find related docs"

    with patch.object(app_settings, "OPENAI_API_KEY", "fake_key_present"):
        results = await semantic_search(agent_id, query_text)

    assert len(results) == 2
    assert results[0]["id"] == "chunk1"
    assert results[0]["text"] == "text of chunk1"
    assert results[0]["score"] == pytest.approx(1 - 0.1)  # 0.9
    assert results[1]["id"] == "chunk2"
    assert results[1]["score"] == pytest.approx(1 - 0.2)  # 0.8

    mock_get_embedding_model.assert_called_once()
    mock_retrieval_embed_model.get_text_embedding.assert_called_once_with(query_text)
    mock_get_chroma_client.assert_called_once()
    expected_collection_name = f"agent_{agent_id}_vectors"
    mock_retrieval_chroma_client.get_collection.assert_called_once_with(
        name=expected_collection_name
    )
    mock_retrieval_chroma_client.get_collection.return_value.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2, 0.3]],  # Matches mock_retrieval_embed_model output
        n_results=app_settings.DEFAULT_SEMANTIC_TOP_K,
        include=[
            "metadatas",
            "documents",
            "distances",
        ],  # Corrected order to match code
    )


@patch("app.services.retrieval_service.get_chroma_client")
@patch("app.services.retrieval_service.get_embedding_model")
@pytest.mark.asyncio
async def test_semantic_search_embedding_failure(
    mock_get_embedding_model,
    mock_get_chroma_client,  # Not strictly needed for this error path, but patch is there
):
    # Test case 1: get_embedding_model itself raises ValueError (e.g. no API key)
    mock_get_embedding_model.side_effect = ValueError("No API Key for embeddings")
    with pytest.raises(
        RetrievalError,
        match="Embedding model configuration error: No API Key for embeddings",
    ):
        await semantic_search("agent1", "query")

    # Test case 2: embed_model.get_text_embedding raises an error
    mock_embed_model_instance = MagicMock()
    mock_embed_model_instance.get_text_embedding.side_effect = Exception(
        "Embedding API down"
    )
    mock_get_embedding_model.side_effect = None  # Reset side effect
    mock_get_embedding_model.return_value = mock_embed_model_instance

    with patch.object(app_settings, "OPENAI_API_KEY", "fake_key_present"):
        with pytest.raises(
            RetrievalError,
            match="Query embedding generation failed: Embedding API down",
        ):
            await semantic_search("agent1", "query")


@patch("app.services.retrieval_service.get_chroma_client")
@patch("app.services.retrieval_service.get_embedding_model")
@pytest.mark.asyncio
async def test_semantic_search_chroma_get_collection_failure(
    mock_get_embedding_model, mock_get_chroma_client, mock_retrieval_embed_model
):
    mock_get_embedding_model.return_value = mock_retrieval_embed_model
    mock_get_chroma_client.side_effect = Exception("Chroma client init failed")

    with patch.object(app_settings, "OPENAI_API_KEY", "fake_key_present"):
        with pytest.raises(
            RetrievalError,
            match="ChromaDB connection failed for semantic search: Chroma client init failed",
        ):
            await semantic_search("agent1", "query")

    # Test specific failure on get_collection
    mock_chroma_instance = MagicMock()
    mock_chroma_instance.get_collection.side_effect = Exception("Cannot get collection")
    mock_get_chroma_client.side_effect = None  # Reset
    mock_get_chroma_client.return_value = mock_chroma_instance

    with patch.object(app_settings, "OPENAI_API_KEY", "fake_key_present"):
        assert await semantic_search("agent1", "query") == []


@patch("app.services.retrieval_service.get_chroma_client")
@patch("app.services.retrieval_service.get_embedding_model")
@pytest.mark.asyncio
async def test_semantic_search_chroma_query_failure(
    mock_get_embedding_model,
    mock_get_chroma_client,
    mock_retrieval_embed_model,
    mock_retrieval_chroma_client,  # Using the fixture that has a collection
):
    mock_get_embedding_model.return_value = mock_retrieval_embed_model
    # Modify the successful chroma client fixture to make its collection.query fail
    mock_retrieval_chroma_client.get_collection.return_value.query.side_effect = (
        Exception("Chroma query error")
    )
    mock_get_chroma_client.return_value = mock_retrieval_chroma_client

    with patch.object(app_settings, "OPENAI_API_KEY", "fake_key_present"):
        with pytest.raises(
            RetrievalError,
            match="ChromaDB query failed for collection 'agent_agent1_vectors': Chroma query error",
        ):
            await semantic_search("agent1", "query")


@pytest.mark.asyncio
async def test_semantic_search_empty_query_or_agent_id():
    with pytest.raises(
        RetrievalError, match="Agent ID cannot be empty for semantic search."
    ):
        await semantic_search("", "query")
    # Query text empty now returns [], does not raise RetrievalError
    # with pytest.raises(RetrievalError, match="Query text cannot be empty for semantic search."):
    #     await semantic_search("agent1", "")
    assert await semantic_search("agent1", "") == []  # Verify it returns empty list


@patch("app.services.retrieval_service.get_chroma_client")
@patch("app.services.retrieval_service.get_embedding_model")
@pytest.mark.asyncio
async def test_semantic_search_empty_chroma_results(
    mock_get_embedding_model, mock_get_chroma_client, mock_retrieval_embed_model
):
    mock_get_embedding_model.return_value = mock_retrieval_embed_model

    mock_chroma_client_empty = MagicMock()
    mock_collection_empty = MagicMock()
    # Simulate empty or malformed results from ChromaDB
    mock_collection_empty.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }  # Empty lists within the structure
    mock_chroma_client_empty.get_collection.return_value = mock_collection_empty
    mock_get_chroma_client.return_value = mock_chroma_client_empty

    with patch.object(app_settings, "OPENAI_API_KEY", "fake_key_present"):
        results = await semantic_search("agent1", "query")
    assert results == []

    # Test with None results from Chroma (should also lead to empty list)
    mock_collection_none = MagicMock()
    mock_collection_none.query.return_value = None  # ChromaDB might return None
    mock_chroma_client_none = MagicMock()
    mock_chroma_client_none.get_collection.return_value = mock_collection_none
    mock_get_chroma_client.return_value = mock_chroma_client_none
    with patch.object(app_settings, "OPENAI_API_KEY", "fake_key_present"):
        results_none = await semantic_search("agent1", "query_none")
    assert results_none == []


# --- Tests for keyword_search_bm25 ---
# For keyword_search_bm25, we need to mock ChromaDB get_collection and its .get() method
# to return a corpus of documents. We also need rank_bm25 available.


@pytest.fixture
def mock_chroma_client_for_bm25():
    mock_client = MagicMock()
    mock_collection = MagicMock()

    # Sample corpus for BM25
    mock_collection.get.return_value = {
        "ids": ["bm25_doc1", "bm25_doc2", "bm25_doc3"],
        "documents": [
            "The quick brown fox jumps over the lazy dog.",
            "A fast brown dog jumps quickly.",
            "Exploring the universe with a new fox telescope.",
        ],
        "metadatas": [
            {"title": "Fox Story"},
            {"title": "Dog Story"},
            {"title": "Universe Story"},
        ],
    }
    mock_client.get_collection.return_value = mock_collection
    return mock_client


@patch("app.services.retrieval_service.get_chroma_client")
@pytest.mark.asyncio
async def test_keyword_search_bm25_successful(
    mock_get_chroma_client_bm25,
    mock_chroma_client_for_bm25,  # Fixture providing mock data
):
    mock_get_chroma_client_bm25.return_value = mock_chroma_client_for_bm25

    agent_id = "test_agent_bm25"
    query_text = "quick fox"

    results = await keyword_search_bm25(agent_id, query_text, top_k=2)

    assert len(results) <= 2  # Can be less if fewer than top_k have score > 0
    assert len(results) > 0  # Expecting at least one match for "quick fox"

    # Check that results are sorted by score (desc)
    if len(results) > 1:
        assert results[0]["score"] >= results[1]["score"]

    # Check content of results (IDs are from mock_chroma_client_for_bm25)
    result_ids = [r["id"] for r in results]
    assert "bm25_doc1" in result_ids  # "The quick brown fox..."
    # "bm25_doc3" ("Exploring the universe with a new fox telescope") might also appear

    # Verify mocks
    mock_get_chroma_client_bm25.assert_called_once()
    expected_collection_name = f"agent_{agent_id}_vectors"
    mock_chroma_client_for_bm25.get_collection.assert_called_once_with(
        name=expected_collection_name
    )
    mock_chroma_client_for_bm25.get_collection.return_value.get.assert_called_once_with(
        include=["documents", "metadatas", "ids"]
    )


@patch("app.services.retrieval_service.get_chroma_client")
@pytest.mark.asyncio
async def test_keyword_search_bm25_no_results(
    mock_get_chroma_client_bm25, mock_chroma_client_for_bm25
):
    mock_get_chroma_client_bm25.return_value = mock_chroma_client_for_bm25
    results = await keyword_search_bm25(
        "agent_bm25", "non_existent_term_xyzabc", top_k=2
    )
    assert results == []


@patch("app.services.retrieval_service.get_chroma_client")
@pytest.mark.asyncio
async def test_keyword_search_bm25_chroma_get_collection_failure(
    mock_get_chroma_client_bm25,
):
    mock_get_chroma_client_bm25.side_effect = Exception(
        "Chroma client init error for BM25"
    )
    with pytest.raises(
        RetrievalError,
        match="ChromaDB connection failed for keyword search: Chroma client init error for BM25",
    ):
        await keyword_search_bm25("agent1", "query")

    mock_chroma_instance = MagicMock()
    mock_chroma_instance.get_collection.side_effect = Exception(
        "BM25 Cannot get collection"
    )
    mock_get_chroma_client_bm25.side_effect = None  # Reset
    mock_get_chroma_client_bm25.return_value = mock_chroma_instance
    # This case (collection not found) should return empty list, not raise error, as per code logic
    results = await keyword_search_bm25("agent1", "query")
    assert results == []


@patch("app.services.retrieval_service.get_chroma_client")
@pytest.mark.asyncio
async def test_keyword_search_bm25_chroma_collection_get_data_failure(
    mock_get_chroma_client_func: MagicMock,  # This is the MagicMock for the get_chroma_client function
):
    """Test BM25 search when ChromaDB collection.get() fails."""

    # Setup the mock for get_chroma_client() -> client -> collection -> get()
    mock_collection_instance = MagicMock()  # Removed spec
    mock_collection_instance.get.side_effect = Exception(
        "Simulated error: Cannot get data from Chroma collection"
    )

    mock_client_instance = MagicMock()  # Removed spec
    mock_client_instance.get_collection.return_value = mock_collection_instance

    mock_get_chroma_client_func.return_value = mock_client_instance

    results = await keyword_search_bm25(
        "agent_col_get_fail", "query for collection get failure"
    )

    # The keyword_search_bm25 function should catch the exception from collection.get()
    # and return an empty list.
    assert results == []
    mock_get_chroma_client_func.assert_called_once()
    mock_client_instance.get_collection.assert_called_once_with(
        name="agent_agent_col_get_fail_vectors"
    )
    mock_collection_instance.get.assert_called_once_with(
        include=["documents", "metadatas", "ids"]
    )


@patch("app.services.retrieval_service.get_chroma_client")
@pytest.mark.asyncio
async def test_keyword_search_bm25_empty_corpus(mock_get_chroma_client_bm25):
    mock_client_empty_corpus = MagicMock()
    mock_collection_empty_corpus = MagicMock()
    mock_collection_empty_corpus.get.return_value = {
        "ids": [],
        "documents": [],
        "metadatas": [],
    }  # Empty corpus
    mock_client_empty_corpus.get_collection.return_value = mock_collection_empty_corpus
    mock_get_chroma_client_bm25.return_value = mock_client_empty_corpus

    results = await keyword_search_bm25("agent_empty_corpus", "any query")
    assert results == []


@pytest.mark.asyncio
async def test_keyword_search_bm25_empty_agent_id_or_query():
    with pytest.raises(
        RetrievalError, match="Agent ID cannot be empty for keyword search."
    ):
        await keyword_search_bm25("", "query")
    # Empty query text should return empty list
    results = await keyword_search_bm25("agent1", "")
    assert results == []


# --- Tests for _format_semantic_results (helper function) ---


def test_format_semantic_results_valid():
    chroma_results = {
        "ids": [["id1", "id2"]],
        "documents": [["doc text 1", "doc text 2"]],
        "metadatas": [[{"meta": "m1"}, {"meta": "m2"}]],
        "distances": [[0.2, 0.4]],  # scores will be 0.8, 0.6
    }
    formatted = _format_semantic_results(chroma_results)
    assert len(formatted) == 2
    assert formatted[0] == {
        "id": "id1",
        "text": "doc text 1",
        "metadata": {"meta": "m1"},
        "score": pytest.approx(0.8),
        "retrieval_strategy": "semantic",
    }
    assert formatted[1] == {
        "id": "id2",
        "text": "doc text 2",
        "metadata": {"meta": "m2"},
        "score": pytest.approx(0.6),
        "retrieval_strategy": "semantic",
    }


def test_format_semantic_results_empty_or_malformed():
    assert _format_semantic_results({}) == []
    assert (
        _format_semantic_results(
            {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        )
        == []
    )
    # Malformed - e.g., documents missing
    assert (
        _format_semantic_results(
            {"ids": [["id1"]], "metadatas": [[{"m": 1}]], "distances": [[0.1]]}
        )
        == []
    )  # Expect empty list due to missing 'documents'


# --- Tests for hybrid_retrieve ---


@pytest.fixture
def mock_flag_reranker_fixture():
    mock_reranker_cls = MagicMock(spec=FlagReranker)  # Class mock
    mock_reranker_inst = MagicMock(spec=FlagReranker)  # Instance mock
    # mock_reranker_cls.return_value = mock_reranker_inst # If _get_reranker_model was patched to return the class
    return mock_reranker_cls, mock_reranker_inst


@patch("app.services.retrieval_service.semantic_search")
@patch("app.services.retrieval_service.keyword_search_bm25")
@patch("app.services.retrieval_service.reciprocal_rank_fusion")
@patch("app.services.ranking_service._get_reranker_model")
@patch.object(app_settings, "RERANKER_MODEL_NAME", "bge-reranker-base")
@patch(
    "app.services.retrieval_service.settings.RERANKER_TOP_N", 1
)  # Directly patch setting - THIS STAYS FOR THIS TEST
@pytest.mark.asyncio
async def test_hybrid_retrieve_reranker_fails_fallback(
    _get_reranker_model: MagicMock,  # Innermost patch injecting mock, after RERANKER_TOP_N (which doesn't inject)
    reciprocal_rank_fusion: MagicMock,
    keyword_search_bm25: MagicMock,
    semantic_search: MagicMock,  # Outermost patch injecting mock
    mock_flag_reranker_fixture: tuple,  # Pytest fixture
):
    _mock_reranker_cls, mock_reranker_inst = mock_flag_reranker_fixture
    _get_reranker_model.return_value = mock_reranker_inst
    mock_reranker_inst.compute_score.side_effect = Exception("Reranker API Error")

    agent_id = "test_agent_rerank_fail"
    query_text = "test query reranker failure"

    fused_candidates_for_reranker = [
        {"id": "docX", "text": "text X", "rrf_score": 0.7, "metadata": {}},
        {"id": "docY", "text": "text Y", "rrf_score": 0.6, "metadata": {}},
    ]

    semantic_search.return_value = [
        {"id": "docX", "text": "text X", "score": 0.7, "metadata": {}}
    ]
    keyword_search_bm25.return_value = [
        {"id": "docY", "text": "text Y", "score": 0.6, "metadata": {}}
    ]

    reciprocal_rank_fusion.return_value = fused_candidates_for_reranker

    results = await hybrid_retrieve(
        agent_id, query_text, final_top_k=len(fused_candidates_for_reranker)
    )

    assert len(results) == 1  # Expect 1 due to RERANKER_TOP_N = 1 patch
    expected_fallback = sorted(
        fused_candidates_for_reranker, key=lambda x: x.get("rrf_score", 0), reverse=True
    )[:1]
    assert results == expected_fallback

    _get_reranker_model.assert_called_once()
    mock_reranker_inst.compute_score.assert_called_once()
    reciprocal_rank_fusion.assert_called_once_with(
        [semantic_search.return_value, keyword_search_bm25.return_value],
        k=app_settings.RRF_K_CONSTANT,  # Use RRF_K_CONSTANT as per reciprocal_rank_fusion
    )


@patch.object(
    app_settings, "RERANKER_MODEL_NAME", "bge-reranker-base"
)  # This should be active for the reranker to be called
@patch.object(
    app_settings, "RERANKER_TOP_N", 2
)  # Set RERANKER_TOP_N to 2 for this test
@patch("app.services.ranking_service._get_reranker_model")
@patch("app.services.retrieval_service.keyword_search_bm25")
@patch("app.services.retrieval_service.semantic_search")
@pytest.mark.asyncio
async def test_hybrid_retrieve_successful_with_reranking(  # Renamed from _original_mocks
    semantic_search: MagicMock,  # Innermost patch
    keyword_search_bm25: MagicMock,
    _get_reranker_model: MagicMock,  # Outermost patch
    mock_flag_reranker_fixture: tuple,  # Pytest fixture
):
    """Test hybrid retrieval logic with reranker enabled and successful."""
    _mock_reranker_cls, mock_reranker_inst = mock_flag_reranker_fixture
    _get_reranker_model.return_value = mock_reranker_inst

    agent_id = "test_agent_hybrid_rerank_success"
    query_text = "test query for reranking"

    sem_results = [
        {
            "id": "doc1",
            "text": "semantic content doc1",
            "score": 0.9,
            "metadata": {"file": "f1"},
        },
        {
            "id": "doc3",
            "text": "shared content doc3",
            "score": 0.7,
            "metadata": {"file": "f3"},
        },
    ]
    key_results = [
        {
            "id": "doc2",
            "text": "keyword content doc2",
            "score": 25.0,
            "metadata": {"file": "f2"},
        },
        {
            "id": "doc3",
            "text": "shared content doc3",
            "score": 20.0,
            "metadata": {"file": "f3_key"},
        },  # Note: metadata might differ if doc3 is from different sources
    ]

    # Mock RRF to produce a combined list. The exact scores from RRF are less critical here
    # than the set of documents fed to the reranker.
    fused_from_rrf_for_reranker = [
        {
            "id": "doc1",
            "text": "semantic content doc1",
            "rrf_score": 0.0163,
            "metadata": {"file": "f1"},
        },
        {
            "id": "doc2",
            "text": "keyword content doc2",
            "rrf_score": 0.0162,
            "metadata": {"file": "f2"},
        },  # Made score slightly different for ordering
        {
            "id": "doc3",
            "text": "shared content doc3",
            "rrf_score": 0.0161,
            "metadata": {"file": "f3"},
        },  # Text from first appearance
    ]

    # Reranker mock: assume it reorders and possibly filters based on its scores
    # It should return RERANKER_TOP_N (which is 2 for this test)
    reranked_by_model = [
        {
            "id": "doc3",
            "text": "shared content doc3",
            "rrf_score": 0.0161,
            "rerank_score": 0.95,
            "metadata": {"file": "f3"},
        },  # Mocked reranker score
        {
            "id": "doc1",
            "text": "semantic content doc1",
            "rrf_score": 0.0163,
            "rerank_score": 0.90,
            "metadata": {"file": "f1"},
        },  # Mocked reranker score
    ]
    # The reranker is called with sentence pairs. Let's mock compute_score to return scores that would result in reranked_by_model
    # Format: list of [query, text] pairs
    # compute_score should return list of scores in the same order.

    # We need to ensure the `rerank_documents` function in `ranking_service.py` (which is called by hybrid_retrieve)
    # uses the mocked reranker instance and its compute_score correctly.
    # `rerank_documents` will call `mock_reranker_inst.compute_score`.
    # The number of calls to compute_score depends on its batching logic.
    # For simplicity, let's assume it makes one call for all candidates.
    # It will pass a list of [query, doc_text] pairs.

    # Expected pairs for reranker based on `fused_from_rrf_for_reranker` (up to DEFAULT_HYBRID_TOP_K, which is 10 by default)
    # Let's assume DEFAULT_HYBRID_TOP_K >= 3 for this test to include all fused_from_rrf_for_reranker items.
    # app_settings.DEFAULT_HYBRID_TOP_K is 10.

    # This is what `rerank_documents` will pass to `compute_score` if `fused_from_rrf_for_reranker` are the candidates.
    expected_sentence_pairs = [
        [query_text, "semantic content doc1"],
        [query_text, "keyword content doc2"],
        [query_text, "shared content doc3"],
    ]
    # These scores would make doc3, then doc1 the top 2.
    mock_scores_from_reranker = [0.90, 0.85, 0.95]
    mock_reranker_inst.compute_score.return_value = mock_scores_from_reranker

    semantic_search.return_value = sem_results
    keyword_search_bm25.return_value = key_results

    # We need to mock reciprocal_rank_fusion as it's called by hybrid_retrieve
    with patch(
        "app.services.retrieval_service.reciprocal_rank_fusion",
        return_value=fused_from_rrf_for_reranker,
    ) as mock_rrf_call:
        results = await hybrid_retrieve(
            agent_id, query_text
        )  # final_top_k (old name for initial_rerank_top_n) will use settings.DEFAULT_HYBRID_TOP_K

    assert len(results) == 2  # Due to RERANKER_TOP_N = 2
    assert (
        results == reranked_by_model
    )  # Check if the final result matches the expected reranked output

    semantic_search.assert_called_once_with(
        agent_id, query_text, top_k=app_settings.DEFAULT_SEMANTIC_TOP_K
    )
    keyword_search_bm25.assert_called_once_with(
        agent_id, query_text, top_k=app_settings.DEFAULT_KEYWORD_TOP_K
    )
    mock_rrf_call.assert_called_once_with(
        [sem_results, key_results], k=app_settings.RRF_K_CONSTANT
    )
    _get_reranker_model.assert_called_once()  # Ensure reranker model was fetched

    # Assert that compute_score was called with the correct sentence pairs
    # The call is made from within `rerank_documents` in `ranking_service.py`
    # The candidates fed to `rerank_documents` are `fused_from_rrf_for_reranker` (up to `settings.DEFAULT_HYBRID_TOP_K`)
    # `settings.DEFAULT_HYBRID_TOP_K` is 10. `len(fused_from_rrf_for_reranker)` is 3. So all are used.
    mock_reranker_inst.compute_score.assert_called_once_with(
        expected_sentence_pairs, normalize=True
    )


@patch.object(app_settings, "RERANKER_MODEL_NAME", "")  # Reranker disabled
@patch("app.services.retrieval_service.reciprocal_rank_fusion")
@patch("app.services.retrieval_service.keyword_search_bm25")
@patch("app.services.retrieval_service.semantic_search")
@pytest.mark.asyncio
async def test_hybrid_retrieve_successful_original_mocks(
    semantic_search: MagicMock,
    keyword_search_bm25: MagicMock,
    reciprocal_rank_fusion: MagicMock,
    # No mock_flag_reranker here as it shouldn't be called when RERANKER_MODEL_NAME is ""
):
    """Test hybrid retrieval logic with reranker explicitly disabled."""
    agent_id = "test_agent_hybrid_orig"
    query_text = "test query"

    sem_results = [
        {"id": "doc1", "text": "semantic content doc1", "score": 0.9, "metadata": {}},
    ]
    key_results = [
        {"id": "doc2", "text": "keyword content doc2", "score": 25.0, "metadata": {}},
    ]
    fused_from_rrf = [
        {
            "id": "doc1",
            "text": "semantic content doc1",
            "rrf_score": 0.0163,
            "metadata": {},
        },
        {
            "id": "doc2",
            "text": "keyword content doc2",
            "rrf_score": 0.0161,
            "metadata": {},
        },
    ]

    semantic_search.return_value = sem_results
    keyword_search_bm25.return_value = key_results
    reciprocal_rank_fusion.return_value = fused_from_rrf

    results = await hybrid_retrieve(agent_id, query_text)

    assert len(results) == len(fused_from_rrf)
    if len(fused_from_rrf) == len(
        results
    ):  # Check if truncation happened if DEFAULT_HYBRID_TOP_K is smaller
        assert results == fused_from_rrf
    else:
        # If hybrid_retrieve truncates based on DEFAULT_HYBRID_TOP_K before returning when reranker is off
        assert results == fused_from_rrf[: app_settings.DEFAULT_HYBRID_TOP_K]

    semantic_search.assert_called_once_with(
        agent_id, query_text, top_k=app_settings.DEFAULT_SEMANTIC_TOP_K
    )
    keyword_search_bm25.assert_called_once_with(
        agent_id, query_text, top_k=app_settings.DEFAULT_KEYWORD_TOP_K
    )
    reciprocal_rank_fusion.assert_called_once_with(
        [sem_results, key_results],
        k=app_settings.RRF_K_CONSTANT,  # Use RRF_K_CONSTANT
    )
    # mock_flag_reranker should not have been called


@patch.object(app_settings, "RERANKER_MODEL_NAME", "")  # Reranker disabled
@patch("app.services.retrieval_service.reciprocal_rank_fusion")
@patch("app.services.retrieval_service.keyword_search_bm25")
@patch("app.services.retrieval_service.semantic_search")
@pytest.mark.asyncio
async def test_hybrid_retrieve_one_search_fails(
    semantic_search: MagicMock,
    keyword_search_bm25: MagicMock,
    reciprocal_rank_fusion: MagicMock,
    # mock_flag_reranker: MagicMock, # Should not be here
):
    agent_id = "hybrid_fail_agent"
    query = "test query fail"

    # Scenario 1: Semantic search fails, keyword search succeeds
    semantic_search.side_effect = RetrievalError("Semantic search down")
    keyword_results_scenario1 = [{"id": "key1", "score": 1.2, "text": "text_key1"}]
    keyword_search_bm25.return_value = keyword_results_scenario1
    # Mock RRF to return what it would with keyword_results_scenario1
    # For this test, RRF's exact score transformation isn't the focus, but its call.
    # Let's assume it returns a list structurally similar to its input for simplicity of the mock.
    rrf_output_scenario1 = [
        {"id": "key1", "rrf_score": 0.016, "text": "text_key1"}
    ]  # Example RRF score
    reciprocal_rank_fusion.return_value = rrf_output_scenario1

    results_sem_fail = await hybrid_retrieve(agent_id, query)
    # Expect results from keyword search after RRF (potentially truncated by final_top_k / DEFAULT_HYBRID_TOP_K)
    # Check that the result is the RRF output, potentially truncated
    expected_results_sem_fail = rrf_output_scenario1[
        : app_settings.DEFAULT_HYBRID_TOP_K
    ]
    assert results_sem_fail == expected_results_sem_fail

    reciprocal_rank_fusion.assert_called_with(
        [[], keyword_results_scenario1],
        k=app_settings.RRF_K_CONSTANT,  # Use RRF_K_CONSTANT
    )

    # Scenario 2: Keyword search fails, semantic search succeeds
    semantic_search.side_effect = None  # Reset side effect
    semantic_results_scenario2 = [{"id": "sem1", "score": 0.8, "text": "text_sem1"}]
    semantic_search.return_value = semantic_results_scenario2
    keyword_search_bm25.side_effect = RetrievalError("Keyword search down")
    keyword_search_bm25.return_value = []  # Ensure it's empty on failure path for RRF call

    rrf_output_scenario2 = [
        {"id": "sem1", "rrf_score": 0.016, "text": "text_sem1"}
    ]  # Example RRF score
    reciprocal_rank_fusion.return_value = rrf_output_scenario2

    results_key_fail = await hybrid_retrieve(agent_id, query)
    expected_results_key_fail = rrf_output_scenario2[
        : app_settings.DEFAULT_HYBRID_TOP_K
    ]
    assert results_key_fail == expected_results_key_fail

    reciprocal_rank_fusion.assert_called_with(
        [semantic_results_scenario2, []],
        k=app_settings.RRF_K_CONSTANT,  # Use RRF_K_CONSTANT
    )

    # Scenario 3: Semantic search raises a non-RetrievalError (should be caught by hybrid_retrieve's generic Exception)
    semantic_search.reset_mock()
    keyword_search_bm25.reset_mock()
    reciprocal_rank_fusion.reset_mock()

    semantic_search.side_effect = ValueError("Unexpected issue")
    keyword_search_bm25.side_effect = (
        None  # Reset side effect from previous scenario if any
    )
    keyword_search_bm25.return_value = []
    # RRF mock is reset. It won't be called if hybrid_retrieve returns early.

    results_val_err = await hybrid_retrieve(agent_id, query)
    assert results_val_err == []  # Expect empty due to early exit in hybrid_retrieve
    # Assert RRF was NOT called in this scenario because hybrid_retrieve should return early
    # If RRF was called in Scenario 2, its call_args would persist unless reset.
    # After reset, if not called, call_args is None and call_count is 0.
    reciprocal_rank_fusion.assert_not_called()


@patch.object(app_settings, "RERANKER_MODEL_NAME", "")  # Reranker disabled
@patch(
    "app.services.retrieval_service.reciprocal_rank_fusion"
)  # Injected as reciprocal_rank_fusion
@patch(
    "app.services.retrieval_service.keyword_search_bm25"
)  # Injected as keyword_search_bm25
@patch("app.services.retrieval_service.semantic_search")  # Injected as semantic_search
@pytest.mark.asyncio
async def test_hybrid_retrieve_one_or_both_searches_return_empty(
    semantic_search: MagicMock,  # Corresponds to @patch("...semantic_search")
    keyword_search_bm25: MagicMock,  # Corresponds to @patch("...keyword_search_bm25")
    reciprocal_rank_fusion: MagicMock,  # Corresponds to @patch("...reciprocal_rank_fusion")
    # mock_flag_reranker: MagicMock, # Should not be here
):
    agent_id = "hybrid_empty_search"
    query = "query producing some empty results"

    semantic_search.return_value = []
    mock_keyword_results = [{"id": "key1", "score": 1.1, "text": "k1"}]
    keyword_search_bm25.return_value = mock_keyword_results

    # Expected RRF output if only keyword results are present
    rrf_output_phase1 = [
        {"id": "key1", "score": 1 / (app_settings.RRF_K_CONSTANT + 0), "text": "k1"}
    ]  # Use RRF_K_CONSTANT
    reciprocal_rank_fusion.return_value = rrf_output_phase1

    results = await hybrid_retrieve(agent_id, query)
    reciprocal_rank_fusion.assert_called_with(
        [[], mock_keyword_results], k=app_settings.RRF_K_CONSTANT
    )  # Use RRF_K_CONSTANT
    # Check that the result is the RRF output, potentially truncated
    expected_results_phase1 = rrf_output_phase1[: app_settings.DEFAULT_HYBRID_TOP_K]
    assert results == expected_results_phase1

    # --- Phase 2: Semantic has results, Keyword is empty ---
    semantic_search.reset_mock()
    keyword_search_bm25.reset_mock()
    reciprocal_rank_fusion.reset_mock()

    mock_semantic_results = [{"id": "sem1", "score": 0.7, "text": "s1"}]
    semantic_search.return_value = mock_semantic_results
    keyword_search_bm25.return_value = []

    rrf_output_phase2 = [
        {"id": "sem1", "score": 1 / (app_settings.RRF_K_CONSTANT + 0), "text": "s1"}
    ]  # Use RRF_K_CONSTANT
    reciprocal_rank_fusion.return_value = rrf_output_phase2

    results = await hybrid_retrieve(agent_id, query)
    reciprocal_rank_fusion.assert_called_with(
        [mock_semantic_results, []], k=app_settings.RRF_K_CONSTANT
    )  # Use RRF_K_CONSTANT
    expected_results_phase2 = rrf_output_phase2[: app_settings.DEFAULT_HYBRID_TOP_K]
    assert results == expected_results_phase2

    # --- Phase 3: Both searches return empty ---
    semantic_search.reset_mock()
    keyword_search_bm25.reset_mock()
    reciprocal_rank_fusion.reset_mock()

    semantic_search.return_value = []
    keyword_search_bm25.return_value = []
    # reciprocal_rank_fusion.return_value = [] # Not needed as it shouldn't be called if hybrid_retrieve returns early

    results = await hybrid_retrieve(agent_id, query)
    # Assert RRF was NOT called in this phase because hybrid_retrieve should return early
    reciprocal_rank_fusion.assert_not_called()
    assert results == []
