"""
API Router for Agent Chat operations.
"""

import json
import logging

# from datetime import datetime # No longer needed directly here if schemas handle it
from typing import Any, Dict, List  # Keep these for type hints

from fastapi import APIRouter, Depends, HTTPException, status

from app.db import get_db

# Import chat schemas
from app.schemas.chat_schemas import (
    ChatMessageCreateRequest,
    ChatMessageRatingRequest,
    ChatMessageResponse,
    CitationResponse,
)

# Ensure all necessary services are imported
from app.services import (  # retrieval_service is used by generation_service
    agent_service,
    generation_service,
)
from prisma import Prisma

# from prisma.models import ChatMessage as PrismaChatMessage # Likely no longer needed
from prisma.types import ChatMessageCreateInput  # Needed for helpers

logger = logging.getLogger(__name__)
# The router for agent_chat_router is APIRouter() and included in main.py.
# Routes here will be relative to /agents/{agent_id}/chat (e.g. / for POST, / for GET, /{message_id}/rating for PUT)
# This means the paths defined in the decorators for copied functions should be like "/", "/{message_id}/rating"
router = APIRouter(
    prefix="/agents/{agent_id}/chat", tags=["Agent Chat"]
)  # Explicitly set prefix and tags here for clarity


# --- Helper functions copied from chat_router.py ---
async def _get_rag_response_and_store_assistant_message(
    agent_id: str, user_query: str, db: Prisma
) -> ChatMessageResponse:
    """Helper to perform RAG, process citations, and store assistant message."""
    logger.info(f"Performing RAG for agent {agent_id}, query: '{user_query[:50]}...'")
    rag_response_data = await generation_service.answer_query_with_rag(
        agent_id=agent_id, user_query=user_query
    )

    # Ensure 'answer' is a string. If it's a dict (from new generation_service), extract 'response_text'.
    raw_assistant_response = rag_response_data["answer"]
    if (
        isinstance(raw_assistant_response, dict)
        and "response_text" in raw_assistant_response
    ):
        assistant_response_content = raw_assistant_response["response_text"]
        # The new generation_service also returns structured 'citations' in the 'answer' dict.
        # These are the ones to use for the response_citations_list.
        # The 'retrieved_chunks' might be different or a superset.
        response_citations_list = raw_assistant_response.get("citations", [])
        # Ensure response_citations_list items are CitationResponse instances if they are dicts
        # This structure comes from _build_citations_from_llm_response in generation_service
        # which should align with CitationResponse from chat_schemas.py if it includes 'id', 'title', 'text_snippet', etc.
        # Let's re-map them to ensure they are `CitationResponse` objects if they are dicts.
        # The `generation_service.answer_query_with_rag` wraps `generate_response`
        # and `generate_response`'s `_build_citations_from_llm_response` creates dicts.
        # So we should map them here.

        processed_response_citations = []
        for cit_data in response_citations_list:
            if isinstance(cit_data, dict):
                processed_response_citations.append(
                    CitationResponse(
                        id=cit_data.get("id", "unknown_id"),
                        title=cit_data.get("title"),
                        text_snippet=cit_data.get("text_snippet"),
                        url=cit_data.get("url"),
                        source_document_id=cit_data.get("source_document_id"),
                        score=cit_data.get("score"),
                        marker=cit_data.get("marker"),
                    )
                )
            elif isinstance(cit_data, CitationResponse):  # If already an instance
                processed_response_citations.append(cit_data)
        response_citations_list = processed_response_citations

    else:  # Fallback if 'answer' is just a string (older behavior)
        assistant_response_content = str(raw_assistant_response)
        # If 'answer' was just a string, then 'retrieved_chunks' are the source for citations
        retrieved_chunks_for_citation = rag_response_data.get("retrieved_chunks", [])
        response_citations_list = []  # Rebuild if not from structured answer
        if retrieved_chunks_for_citation:
            for chunk in retrieved_chunks_for_citation:
                chunk_id_val = chunk.get("id")
                metadata = chunk.get("metadata", {})
                text_snippet_val = chunk.get("text", "")[:150]
                if chunk_id_val:
                    response_citations_list.append(
                        CitationResponse(
                            id=chunk_id_val,
                            title=metadata.get("title"),
                            text_snippet=text_snippet_val,
                            url=metadata.get("url"),
                            source_document_id=metadata.get("original_document_id"),
                        )
                    )

    # Prepare db_citations_list (JSON string for DB) based on the final response_citations_list
    db_citations_list_for_json = []
    for cit_resp_obj in response_citations_list:
        db_citations_list_for_json.append(cit_resp_obj.model_dump(exclude_none=True))

    citations_json_str = (
        json.dumps(db_citations_list_for_json) if db_citations_list_for_json else None
    )

    logger.info(
        f"RAG response processed for agent {agent_id}. Assistant content: '{assistant_response_content[:50]}...'. Citations for response: {len(response_citations_list)}"
    )

    assistant_message_db_input = ChatMessageCreateInput(
        agentId=agent_id,
        role="assistant",
        content=assistant_response_content,
        citations=citations_json_str,  # Storing the JSON string of CitationResponse objects
    )
    assistant_message_db = await db.chatmessage.create(data=assistant_message_db_input)
    logger.info(
        f"Assistant message stored for agent {agent_id}, message ID: {assistant_message_db.id}"
    )

    return ChatMessageResponse(
        id=assistant_message_db.id,
        role=assistant_message_db.role,
        content=assistant_message_db.content,
        createdAt=assistant_message_db.createdAt,
        agentId=assistant_message_db.agentId,
        citations=response_citations_list,  # List of CitationResponse objects
        rating=assistant_message_db.rating,
    )


async def _handle_chat_generation_error(
    agent_id: str, db: Prisma, error: Exception, is_unexpected: bool = False
):
    """Handles errors during chat generation, logs them, stores a fallback message, and raises HTTPException."""
    logger.error(
        f"{'Unexpected error' if is_unexpected else 'RAG pipeline error'} for agent {agent_id}: {error}",
        exc_info=True,
    )

    if is_unexpected:
        fallback_content = (
            "Sorry, an unexpected error occurred while processing your message."
        )
        http_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        http_detail = "An unexpected error occurred."
    else:  # Specific handling for GenerationError
        fallback_content = (
            f"Sorry, I encountered an error processing your request: {str(error)}"
        )
        if "OpenAI API Key not configured" in str(error):
            fallback_content = "I'm having trouble connecting to my knowledge core (configuration issue). Please contact support."
        elif "RAG pipeline failed: LLM chat completion failed" in str(error):
            fallback_content = "I'm having trouble formulating a response right now. Please try again shortly."
        elif "RAG pipeline failed" in str(error):  # More generic RAG failure
            fallback_content = (
                "I encountered an issue processing your request. Please try again."
            )
        http_status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        http_detail = fallback_content

    try:
        await db.chatmessage.create(
            data=ChatMessageCreateInput(
                agentId=agent_id, role="assistant", content=fallback_content
            )
        )
        logger.info(
            f"Stored fallback assistant message for agent {agent_id} due to error."
        )
    except Exception as db_exc:
        logger.error(
            f"Failed to store fallback assistant message for agent {agent_id}: {db_exc}",
            exc_info=True,
        )
    raise HTTPException(status_code=http_status_code, detail=http_detail)


# --- API Endpoints ---
@router.post(
    "/",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_201_CREATED,  # Path relative to prefix
)
async def send_chat_message(  # Renamed from post_new_chat_message to match chat_router.py
    agent_id: str, chat_request: ChatMessageCreateRequest, db: Prisma = Depends(get_db)
) -> ChatMessageResponse:
    """
    Send a new message to an agent and get a response.
    This involves retrieval, generation, and storing both messages.
    (Logic from chat_router.py)
    """
    agent = await db.agent.find_unique(where={"id": agent_id})
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID '{agent_id}' not found.",
        )

    user_message_content = chat_request.content
    if not user_message_content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message content cannot be empty.",
        )

    try:  # Store User Message first
        user_message_db_input = ChatMessageCreateInput(
            agentId=agent_id, role="user", content=user_message_content
        )
        user_message_db = await db.chatmessage.create(data=user_message_db_input)
        logger.info(
            f"User message stored for agent {agent_id}, message ID: {user_message_db.id}"
        )
    except Exception as e:
        logger.error(
            f"Failed to store user message for agent {agent_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store user message.",
        )

    try:  # Then attempt to get RAG response and store assistant message
        return await _get_rag_response_and_store_assistant_message(
            agent_id=agent_id, user_query=user_message_content, db=db
        )
    except generation_service.GenerationError as e:
        await _handle_chat_generation_error(agent_id=agent_id, db=db, error=e)
    except Exception as e:  # Catch any other unexpected errors
        await _handle_chat_generation_error(
            agent_id=agent_id, db=db, error=e, is_unexpected=True
        )


@router.get("/", response_model=List[ChatMessageResponse])  # Path relative to prefix
async def get_chat_history(  # Renamed from get_chat_messages_for_agent
    agent_id: str,
    db: Prisma = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
) -> List[ChatMessageResponse]:
    """Fetches chat history for a selected agent."""
    logger.info(
        f"Fetching chat history for agent_id: {agent_id}, skip: {skip}, limit: {limit}"
    )
    try:
        # Verify agent exists (optional, but good practice if not strictly enforced by DB constraints elsewhere)
        agent = await db.agent.find_unique(where={"id": agent_id})
        if not agent:
            logger.warning(f"Agent with ID '{agent_id}' not found for chat history.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{agent_id}' not found.",
            )

        messages_from_db = await db.chatmessage.find_many(
            where={"agentId": agent_id},
            order={"createdAt": "asc"},  # Typically show oldest first
            skip=skip,
            take=limit,
        )
        logger.info(
            f"Retrieved {len(messages_from_db)} messages for agent_id: {agent_id}"
        )

        response_messages = [
            ChatMessageResponse.from_prisma_message(msg) for msg in messages_from_db
        ]
        return response_messages

    except HTTPException:  # Re-raise known HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            f"Error fetching chat history for agent '{agent_id}': {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch chat history.",
        )


@router.put(
    "/{message_id}/rating", response_model=ChatMessageResponse
)  # Path relative to prefix
async def rate_chat_message(  # Function name and logic from chat_router.py
    agent_id: str,
    message_id: str,
    rating_request: ChatMessageRatingRequest,
    db: Prisma = Depends(get_db),
) -> ChatMessageResponse:
    """Allows a user to rate an assistant's message."""
    logger.info(
        f"Rating message_id: {message_id} for agent_id: {agent_id} with rating: {rating_request.rating}"
    )
    try:
        # First, check if the message exists and belongs to the agent (optional, update might fail anyway)
        # message_to_rate = await db.chatmessage.find_first(
        #     where={"id": message_id, "agentId": agent_id}
        # )
        # if not message_to_rate:
        #     raise HTTPException(
        #         status_code=status.HTTP_404_NOT_FOUND,
        #         detail=f"Message with ID '{message_id}' not found for agent '{agent_id}'."
        #     )
        # if message_to_rate.role != "assistant":
        #     raise HTTPException(
        #         status_code=status.HTTP_400_BAD_REQUEST,
        #         detail="Only assistant messages can be rated."
        #     )

        updated_message = await db.chatmessage.update(
            where={
                "id": message_id,
                "agentId": agent_id,
            },  # Ensure it belongs to the agent
            data={"rating": rating_request.rating},
        )
        if not updated_message:  # Should be caught by Prisma error if not found
            logger.warning(
                f"Message with ID '{message_id}' not found for agent '{agent_id}' during rating update."
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Message with ID '{message_id}' not found for agent '{agent_id}' or does not belong to this agent.",
            )

        logger.info(
            f"Message '{message_id}' rated successfully with {rating_request.rating}."
        )
        return ChatMessageResponse.from_prisma_message(updated_message)

    except HTTPException:  # Re-raise known HTTP exceptions
        raise
    except Exception as e:  # Catch Prisma RecordNotFound, etc.
        # A more specific check for Prisma RecordNotFoundError would be better
        if "RecordNotFoundError" in str(type(e)) or (
            "prisma.errors.MissingRequiredValueError" in str(type(e))
            and "where" in str(e)
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Message with ID '{message_id}' not found for agent '{agent_id}' or does not belong to this agent.",
            )
        logger.error(
            f"Error rating message '{message_id}' for agent '{agent_id}': {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rate message.",
        )


@router.delete(
    "/",  # Path relative to prefix, for deleting all messages for agent_id
    status_code=status.HTTP_200_OK,
    summary="Delete all chat messages for a specific agent",
    response_description="Number of chat messages deleted",
)
async def delete_agent_chat_history(
    agent_id: str, db: Prisma = Depends(get_db)
) -> Dict[str, Any]:
    """Deletes all chat history for a specific agent."""
    logger.info(f"Attempting to delete chat history for agent_id: {agent_id}")
    try:
        # Ensure agent exists
        agent = await agent_service.get_agent_by_id(db, agent_id)
        if not agent:
            logger.warning(f"Agent with ID '{agent_id}' not found for chat deletion.")
            raise HTTPException(status_code=404, detail="Agent not found")

        deleted_count = await agent_service.delete_chat_history_for_agent(db, agent_id)
        logger.info(
            f"Successfully deleted {deleted_count} messages for agent_id: {agent_id}"
        )
        return {
            "message": f"Successfully deleted {deleted_count} chat messages for agent {agent_id}",
            "deleted_count": deleted_count,
        }
    except HTTPException:  # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        logger.error(
            f"Error deleting chat history for agent '{agent_id}': {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to delete chat history: {str(e)}"
        )


# We will also need to ensure Prisma client is properly managed (connected/disconnected)
# in the main application lifecycle (e.g., app/main.py).
