"""
Pydantic Schemas for Chat Operations.
"""

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


# --- Citation Schemas ---
class CitationBase(BaseModel):
    id: str = Field(
        ..., description="The unique identifier of the source chunk/document."
    )
    title: Optional[str] = Field(None, description="The title of the source document.")
    text_snippet: Optional[str] = Field(
        None, description="A relevant snippet of text from the source."
    )
    url: Optional[str] = Field(
        None, description="URL of the source, if available."
    )  # Added from analysis
    source_document_id: Optional[str] = Field(
        None,
        description="The ID of the original document from which the chunk was derived.",
    )  # Added from analysis
    score: Optional[float] = Field(None, description="Relevance score of the citation.")
    marker: Optional[str] = Field(None, description="The citation marker, e.g., '[1]'")


class CitationResponse(CitationBase):
    class Config:
        from_attributes = True


# --- Chat Message Schemas ---
class ChatMessageBase(BaseModel):
    content: str = Field(..., description="Content of the chat message")
    role: str = Field(
        ..., description="Role of the message sender: 'user' or 'assistant'"
    )


class ChatMessageCreateRequest(BaseModel):
    """Schema for the request body when a user sends a new chat message."""

    content: str = Field(..., description="Content of the user's chat message")


class ChatMessageResponse(ChatMessageBase):
    """Schema for representing a chat message in API responses."""

    id: str = Field(..., description="Unique identifier of the chat message.")
    createdAt: datetime = Field(
        ..., description="Timestamp of when the message was created."
    )
    agentId: str = Field(
        ..., description="Identifier of the agent associated with this message."
    )
    citations: Optional[List[CitationResponse]] = Field(
        None, description="List of citations for assistant messages."
    )
    rating: Optional[int] = Field(
        None,
        description="User-provided rating for the message (e.g., 1 for good, -1 for bad).",
    )

    class Config:
        from_attributes = True

    @classmethod
    def from_prisma_message(cls, message: Any) -> "ChatMessageResponse":
        import json
        import logging

        logger = logging.getLogger(__name__)

        parsed_citations_list = []
        if message.role == "assistant" and message.citations:
            citations_data_to_parse = message.citations
            if isinstance(citations_data_to_parse, str):
                try:
                    parsed_raw_citations = json.loads(citations_data_to_parse)
                except json.JSONDecodeError:
                    logger.error(
                        f"Could not parse citations JSON for message {message.id}: {citations_data_to_parse}",
                        exc_info=True,
                    )
                    parsed_raw_citations = []
            elif isinstance(citations_data_to_parse, list):
                parsed_raw_citations = citations_data_to_parse
            else:
                logger.warning(
                    f"Citations for message {message.id} is neither string nor list: {type(citations_data_to_parse)}. Skipping."
                )
                parsed_raw_citations = []

            if parsed_raw_citations:
                for cit_data in parsed_raw_citations:
                    if isinstance(cit_data, dict):
                        try:
                            parsed_citations_list.append(CitationResponse(**cit_data))
                        except Exception as e:
                            logger.error(
                                f"Could not validate citation data for message {message.id}: {cit_data}. Error: {e}",
                                exc_info=True,
                            )
                    else:
                        logger.warning(
                            f"Found non-dict item in parsed citations for message {message.id}: {cit_data}. Skipping."
                        )

        return cls(
            id=message.id,
            content=message.content,
            role=message.role,
            createdAt=message.createdAt,
            agentId=message.agentId,
            citations=parsed_citations_list if parsed_citations_list else None,
            rating=message.rating,
        )


class ChatMessageRatingRequest(BaseModel):
    """Schema for the request body when rating a chat message."""

    rating: int = Field(
        ...,
        description="Rating for the chat message (e.g., 1 for good, -1 for bad)",
        ge=-1,
        le=1,  # Assuming rating is -1, 0, or 1
    )
