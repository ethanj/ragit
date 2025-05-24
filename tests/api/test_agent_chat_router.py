"""
Tests for the Agent Chat API Router.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.db import get_db as original_get_db
from app.main import app  # Main FastAPI application
from app.services.generation_service import GenerationError

# Use a TestClient that points to your FastAPI app
client = TestClient(app)

# TODO: Add more comprehensive tests for chat router endpoints.


@pytest.fixture
def mock_prisma_db_chat_router(monkeypatch):
    """Fixture to mock Prisma client methods used in agent chat router."""
    db_mock = MagicMock()

    # Mock agent lookup
    mock_agent_instance = MagicMock()
    mock_agent_instance.id = "test_agent_id"
    db_mock.agent.find_unique = AsyncMock(return_value=mock_agent_instance)

    # Mock chat message creation
    # This will be called twice: once for user, once for assistant
    created_messages = []

    async def mock_create_chat_message(data):
        msg = MagicMock()
        msg.id = f"message_id_{len(created_messages)}"
        msg.role = data.get("role")
        msg.content = data.get("content")
        msg.createdAt = datetime.utcnow()  # Use a real datetime
        msg.agentId = data.get("agentId")
        msg.citations = data.get("citations")  # JSON string or None
        msg.rating = data.get("rating")
        created_messages.append(msg)
        return msg

    db_mock.chatmessage.create = AsyncMock(side_effect=mock_create_chat_message)

    # Dependency override for get_db
    def override_get_db():
        return db_mock

    # Update the dependency override to point to the new location of get_db
    monkeypatch.setitem(app.dependency_overrides, original_get_db, override_get_db)

    yield db_mock, created_messages  # Return created_messages for assertions

    # Clean up the override after the test
    monkeypatch.delitem(app.dependency_overrides, original_get_db, raising=False)


@patch("app.api.routers.agent_chat_router.generation_service.answer_query_with_rag")
def test_send_chat_message_success(
    mock_answer_query_with_rag, mock_prisma_db_chat_router
):
    """Test successful chat message sending and RAG response."""
    db_mock, created_messages_list = mock_prisma_db_chat_router

    # Mock structure should align with actual output of generation_service.answer_query_with_rag
    # where 'answer' is a dict containing 'response_text' and 'citations'.
    mock_rag_response = {
        "answer": {  # 'answer' is now a dictionary
            "response_text": "This is a RAG response.",
            "citations": [
                {
                    "id": "chunk1",
                    "text_snippet": "Relevant text snippet 1",
                    "title": "Doc 1",
                    "url": None,
                    "source_document_id": "doc_source_1",
                    "score": 0.9,
                },
                {
                    "id": "chunk2",
                    "text_snippet": "Relevant text snippet 2",
                    "title": "Doc 2",
                    "url": "http://example.com/doc2",
                    "source_document_id": "doc_source_2",
                    "score": 0.8,
                },
            ],
        },
        # These keys might be returned by answer_query_with_rag at top level for other purposes
        "retrieved_context_count": 2,
        "query": "User query",
        "agent_id": "test_agent_id",
        # "retrieved_chunks": [] # If needed for the 'else' branch, but this test hits the 'if'
    }
    mock_answer_query_with_rag.return_value = mock_rag_response

    agent_id = "test_agent_id"
    response = client.post(
        f"/api/v1/agents/{agent_id}/chat/", json={"content": "Hello, RAG system!"}
    )

    assert response.status_code == 201
    data = response.json()
    assert data["content"] == "This is a RAG response."
    assert data["role"] == "assistant"
    assert data["agentId"] == agent_id

    expected_http_citations = [
        {
            "id": "chunk1",
            "title": "Doc 1",
            "text_snippet": "Relevant text snippet 1",
            "url": None,
            "source_document_id": "doc_source_1",
            "score": 0.9,
            "marker": None,
        },
        {
            "id": "chunk2",
            "title": "Doc 2",
            "text_snippet": "Relevant text snippet 2",
            "url": "http://example.com/doc2",
            "source_document_id": "doc_source_2",
            "score": 0.8,
            "marker": None,
        },
    ]
    assert data["citations"] == expected_http_citations

    mock_answer_query_with_rag.assert_called_once_with(
        agent_id=agent_id, user_query="Hello, RAG system!"
    )

    # Check DB interactions
    assert db_mock.agent.find_unique.call_count == 1
    assert (
        db_mock.chatmessage.create.call_count == 2
    )  # User message + Assistant message

    # First created message is user
    assert created_messages_list[0].role == "user"
    assert created_messages_list[0].content == "Hello, RAG system!"
    # Second created message is assistant
    assert created_messages_list[1].role == "assistant"
    assert created_messages_list[1].content == "This is a RAG response."
    db_citations_expected = [
        {
            "id": "chunk1",
            "title": "Doc 1",
            "text_snippet": "Relevant text snippet 1",
            "source_document_id": "doc_source_1",
            "score": 0.9,
        },
        {
            "id": "chunk2",
            "title": "Doc 2",
            "text_snippet": "Relevant text snippet 2",
            "url": "http://example.com/doc2",
            "source_document_id": "doc_source_2",
            "score": 0.8,
        },
    ]
    assert json.loads(created_messages_list[1].citations) == db_citations_expected


@patch("app.api.routers.agent_chat_router.generation_service.answer_query_with_rag")
def test_send_chat_message_agent_not_found(
    mock_answer_query_with_rag, mock_prisma_db_chat_router
):
    """Test chat message sending when agent is not found (404)."""
    db_mock, _ = mock_prisma_db_chat_router
    db_mock.agent.find_unique.return_value = None  # Simulate agent not found

    agent_id = "non_existent_agent"
    response = client.post(
        f"/api/v1/agents/{agent_id}/chat/", json={"content": "Hello?"}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == f"Agent with ID '{agent_id}' not found."
    mock_answer_query_with_rag.assert_not_called()


def test_send_chat_message_empty_content(
    mock_prisma_db_chat_router,
):  # No RAG call if content empty
    """Test chat message sending with empty content (400)."""
    agent_id = "test_agent_id"
    response = client.post(
        f"/api/v1/agents/{agent_id}/chat/",
        json={"content": "    "},  # Empty/whitespace content
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Message content cannot be empty."


@patch("app.api.routers.agent_chat_router.generation_service.answer_query_with_rag")
def test_send_chat_message_rag_generation_error(
    mock_answer_query_with_rag, mock_prisma_db_chat_router
):
    """Test chat message sending when RAG service raises GenerationError (503)."""
    db_mock, created_messages_list = mock_prisma_db_chat_router
    mock_answer_query_with_rag.side_effect = GenerationError(
        "RAG pipeline failed: Test RAG error"
    )

    agent_id = "test_agent_id"
    response = client.post(
        f"/api/v1/agents/{agent_id}/chat/", json={"content": "Trigger error"}
    )

    assert response.status_code == 503
    # Check the user-friendly error message based on the specific GenerationError content
    assert (
        response.json()["detail"]
        == "I encountered an issue processing your request. Please try again."
    )

    # User message should be stored, then a fallback assistant message
    assert db_mock.chatmessage.create.call_count == 2
    assert created_messages_list[0].role == "user"
    assert created_messages_list[1].role == "assistant"
    assert (
        created_messages_list[1].content
        == "I encountered an issue processing your request. Please try again."
    )


@patch("app.api.routers.agent_chat_router.generation_service.answer_query_with_rag")
def test_send_chat_message_rag_openai_key_error(
    mock_answer_query_with_rag, mock_prisma_db_chat_router
):
    """Test specific user-friendly message for OpenAI key error."""
    db_mock, _ = mock_prisma_db_chat_router
    mock_answer_query_with_rag.side_effect = GenerationError(
        "OpenAI API Key not configured"
    )

    agent_id = "test_agent_id"
    response = client.post(
        f"/api/v1/agents/{agent_id}/chat/", json={"content": "Trigger OpenAI key error"}
    )
    assert response.status_code == 503
    assert (
        response.json()["detail"]
        == "I'm having trouble connecting to my knowledge core (configuration issue). Please contact support."
    )


@patch("app.api.routers.agent_chat_router.generation_service.answer_query_with_rag")
def test_send_chat_message_unexpected_error(
    mock_answer_query_with_rag, mock_prisma_db_chat_router
):
    """Test chat message sending with an unexpected error during RAG (500)."""
    db_mock, _ = mock_prisma_db_chat_router
    mock_answer_query_with_rag.side_effect = Exception("Totally unexpected error")

    agent_id = "test_agent_id"
    response = client.post(
        f"/api/v1/agents/{agent_id}/chat/", json={"content": "Trigger unexpected error"}
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "An unexpected error occurred."
    # User message stored, then fallback assistant message
    assert db_mock.chatmessage.create.call_count == 2


# Remove the placeholder test
# def test_chat_router_file_exists():
# assert True
