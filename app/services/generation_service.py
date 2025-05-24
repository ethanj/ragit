"""
Service for generating responses using an LLM with retrieved context.
"""

import functools  # Added for lru_cache
import json  # Added for parsing LLM JSON output
import logging
import re  # Added for citation parsing
from typing import Any, Dict, List

from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import (
    OpenAI as LlamaOpenAI,  # Alias to avoid confusion with openai package
)

from app.core.config import settings
from app.services.retrieval_service import hybrid_retrieve  # For end-to-end RAG later

logger = logging.getLogger(__name__)

# DEFAULT_LLM_MODEL = "gpt-3.5-turbo" # Moved to settings
# Max context tokens to leave room for prompt and response (conservative for gpt-3.5-turbo 4k/16k context)
# This should be dynamic based on model later.
# DEFAULT_MAX_CONTEXT_TOKENS = 3000 # This constant is not directly used for truncation in the visible code.
# Context truncation is likely handled by LlamaIndex internals based on model limits.


class GenerationError(Exception):
    """Custom exception for generation errors."""

    pass


class LLMOutputParsingError(GenerationError):
    """Custom exception for errors when parsing LLM output."""

    pass


class PromptTooLargeError(GenerationError):
    """Custom exception for when the generated prompt exceeds the token limit."""

    pass


# --- Prompt Templates ---
# Using LlamaIndex ChatPromptTemplate for structured chat prompts

# System Prompt: General instructions for the RAG agent
# DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
#     "You are a helpful AI assistant. Your task is to answer questions accurately "
#     "based on the provided context. If the context does not contain the answer, "
#     "state that the information is not available in the provided documents.\n"
#     "Be concise and helpful. Do not make up information."
# ) # Moved to settings

# User Prompt: Includes context and the actual user query
# Note: LlamaIndex uses {{ placeholder }} syntax for variables in ChatPromptTemplate strings.
# DEFAULT_USER_PROMPT_TEMPLATE = (
#     "Here is some context to help you answer the question. Each context chunk is numbered (e.g., Context Chunk 1, Context Chunk 2).\\n"
#     "---------------------\\n"
#     "{context_str}\\n"
#     "---------------------\\n"
#     "Based on the context above, please answer the following question. "
#     "When you use information primarily from a specific context chunk in your answer, "
#     "please cite it at the end of the relevant sentence(s) using the format [N], where N is the context chunk number. "
#     "For example: The system is an innovative platform [1]. It offers various AI solutions [2].\\n"
#     "Your final output MUST be a single JSON object string with two keys: "
#     '1. "answer_text": Your narrative answer, including the [N] citation markers as described above. '
#     '2. "cited_context_numbers": A JSON list of integers representing the numbers of the context chunks you actually cited in your answer_text. For example: [1, 2]. If no chunks were cited, provide an empty list [].\\n'
#     "Example of the full JSON output format:\\\\n"
#     "```json\\\\n"
#     "{\\n"
#     '  "answer_text": "The system provides AI-powered solutions for enterprise [1]. Their platform is known for its scalability [3].",\\\\n'
#     '  "cited_context_numbers": [1, 3]\\\\n'
#     "}\\n"
#     "```\\\\n"
#     "Now, please answer the question based on the provided context and adhere strictly to this JSON output format.\\\\n"
#     "Question: {query_str}"
# ) # Moved to settings


@functools.lru_cache(maxsize=None)  # Added cache decorator
def get_llm(model_name: str = None, temperature: float = 0.1):
    """Initializes and returns the LlamaIndex OpenAI LLM."""
    if (
        not settings.OPENAI_API_KEY
        or settings.OPENAI_API_KEY == "your_openai_api_key_here"
    ):
        logger.error("OpenAI API Key is not configured. Cannot initialize LLM.")
        raise GenerationError("OpenAI API Key not configured for LLM generation.")

    llm_model_to_use = model_name or settings.LLM_MODEL_NAME
    # settings.LLM_MODEL_NAME has a Pydantic default, so it should always have a string value.
    # This check is for extreme robustness if it was somehow None or empty.
    if not llm_model_to_use:
        llm_model_to_use = "gpt-4o"  # Hardcoded ultimate fallback
        logger.warning(
            f"LLM model name resolved to None/empty. Using hardcoded fallback: {llm_model_to_use}"
        )

    try:
        llm = LlamaOpenAI(
            model=llm_model_to_use,
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
            # max_tokens can be set if needed, but often better to let model decide based on context/request
        )
        logger.info(f"Initialized LlamaIndex OpenAI LLM: {llm_model_to_use}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LlamaIndex OpenAI LLM: {e}")
        raise GenerationError(f"LLM initialization failed: {e}")


def format_context_chunks(
    context_chunks: List[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]]]:
    """Formats a list of context chunks into a single string for the prompt and a list of referenceable chunk details."""
    if not context_chunks:
        return "No context provided.", []

    formatted_contexts_str_parts = []
    chunk_references = []
    for i, chunk in enumerate(context_chunks):
        chunk_number = i + 1
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})

        if i < 3:  # Log for the first 3 chunks
            logger.info(
                f"Formatting chunk {chunk_number}: Original text length: {len(text)}, First 50 chars: '{text[:50]}'"
            )

        source_info_parts = []
        title = metadata.get("title")
        original_doc_id = metadata.get("original_document_id")
        url = metadata.get("url")

        if title:
            source_info_parts.append(f"Title: {title}")
        if original_doc_id:
            source_info_parts.append(f"Source ID: {original_doc_id}")
        if url:
            source_info_parts.append(f"URL: {url}")

        source_str_display = ", ".join(source_info_parts)

        header = f"Context Chunk {chunk_number}"
        if source_str_display:
            header += f" (Source: {source_str_display})"

        formatted_contexts_str_parts.append(f"{header}:\n{text}")

        chunk_references.append(
            {
                "number": chunk_number,
                "id": chunk.get(
                    "id", f"unidentified_chunk_{chunk_number}"
                ),  # Fallback ID
                "text_snippet": text[:150] + "..."
                if len(text) > 150
                else text,  # Keep snippet concise
                "title": title,
                "original_document_id": original_doc_id,
                "url": url,
            }
        )

        if i < 3:  # Log for the first 3 chunks
            logger.info(
                f"Formatted chunk {chunk_number}: Generated text_snippet: '{chunk_references[-1].get('text_snippet')}'"
            )

    full_context_str = "\n\n---\n\n".join(formatted_contexts_str_parts)
    return full_context_str, chunk_references


def _parse_llm_json_output(raw_llm_output: str, agent_id: str) -> Dict[str, Any]:
    """Parses the LLM's raw output string, expecting a JSON object.

    Handles potential markdown code block wrapping (e.g., ```json ... ```).

    Args:
        raw_llm_output: The raw string output from the LLM.
        agent_id: The agent ID, for logging.

    Returns:
        A dictionary parsed from the JSON string, expecting 'answer_text' and 'cited_context_numbers'.

    Raises:
        LLMOutputParsingError: If the JSON cannot be parsed or is missing expected keys.
    """
    logger.debug(f"Raw LLM output for agent {agent_id} to parse: {raw_llm_output}")

    json_str = raw_llm_output
    # Try to extract from ```json ... ``` markdown block
    match = re.search(r"```json\n(.*?)\n```", raw_llm_output, re.DOTALL)
    if match:
        json_str = match.group(1)
        logger.debug(
            f"Extracted JSON string from markdown block for agent {agent_id}: {json_str}"
        )
    else:
        # Check if it might be a simple ``` wrapped string
        match_simple = re.search(r"```\n(.*?)\n```", raw_llm_output, re.DOTALL)
        if match_simple:
            json_str = match_simple.group(1)
            logger.debug(
                f"Extracted JSON string from simple markdown block for agent {agent_id}: {json_str}"
            )
        else:
            logger.debug(
                f"No markdown block found for agent {agent_id}, assuming direct JSON string."
            )

    try:
        parsed_json = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse JSON from LLM output for agent {agent_id}. Content: '{json_str}'. Error: {e}",
            exc_info=True,
        )
        raise LLMOutputParsingError(
            f"LLM output was not valid JSON. Raw: '{raw_llm_output[:200]}...'"
        ) from e

    if not isinstance(parsed_json, dict):
        logger.error(
            f"Parsed JSON from LLM output for agent {agent_id} is not a dictionary. Content: '{parsed_json}'"
        )
        raise LLMOutputParsingError(
            f"LLM output, when parsed, was not a dictionary. Raw: '{raw_llm_output[:200]}...'"
        )

    # Validate expected keys
    expected_keys = ["answer_text", "cited_context_numbers"]
    missing_keys = [key for key in expected_keys if key not in parsed_json]
    if missing_keys:
        logger.error(
            f"Parsed JSON from LLM output for agent {agent_id} is missing expected keys: {missing_keys}. Content: '{parsed_json}'"
        )
        raise LLMOutputParsingError(
            f"LLM output JSON is missing keys: {', '.join(missing_keys)}. Raw: '{raw_llm_output[:200]}...'"
        )

    if not isinstance(parsed_json.get("cited_context_numbers"), list):
        logger.error(
            f"'cited_context_numbers' in LLM output for agent {agent_id} is not a list. Content: '{parsed_json}'"
        )
        raise LLMOutputParsingError(
            f"'cited_context_numbers' in LLM output must be a list. Raw: '{raw_llm_output[:200]}...'"
        )

    # Log the successfully parsed components before returning
    log_msg_answer_text = parsed_json.get("answer_text", "")[:100]
    log_msg_cited_numbers = parsed_json.get("cited_context_numbers")
    logger.info(
        f"Successfully parsed LLM output for agent {agent_id}. "
        f"Answer Text: '{log_msg_answer_text}...' "
        f"Cited Context Numbers: {log_msg_cited_numbers}"
    )

    return parsed_json


def _build_citations_from_llm_response(
    cited_numbers_from_llm: List[Any],
    chunk_references: List[Dict[str, Any]],
    agent_id: str,
) -> List[Dict[str, Any]]:
    """Builds a list of citation detail dictionaries based on numbers provided by the LLM.

    Args:
        cited_numbers_from_llm: A list of numbers (or strings of numbers) from LLM's output.
        chunk_references: The list of original chunk reference details.
        agent_id: The agent ID, for logging.

    Returns:
        A list of citation dictionaries, matching the structure expected by the API.
    """
    built_citations = []
    if not isinstance(cited_numbers_from_llm, list):
        logger.warning(
            f"'cited_context_numbers' from LLM was not a list for agent {agent_id}: {cited_numbers_from_llm}. No citations will be built."
        )
        return built_citations

    for num_str in cited_numbers_from_llm:
        try:
            num = int(num_str)  # LLM might return numbers as strings or ints
            # Find the corresponding chunk reference by its 'number' field
            found_ref = False
            for ref_chunk in chunk_references:
                if ref_chunk.get("number") == num:
                    # Construct the citation object as expected (matching old structure if needed or a new standard)
                    # This structure should align with what ChatMessageResponse expects for its `citations` field.
                    # The `parse_text_for_citations` utility produces a similar structure, so we align with that.
                    built_citations.append(
                        {
                            "id": ref_chunk.get("id"),
                            "text_snippet": ref_chunk.get("text_snippet"),
                            "title": ref_chunk.get("title"),
                            "marker": f"[{num}]",  # The marker used in text
                            "original_document_id": ref_chunk.get(
                                "original_document_id"
                            ),
                            "url": ref_chunk.get("url"),
                            "number": num,  # Also include the number for direct reference
                        }
                    )
                    found_ref = True
                    break
            if not found_ref:
                logger.warning(
                    f"LLM cited context number {num} which was not found in original chunk_references for agent {agent_id}."
                )
        except ValueError:
            logger.warning(
                f"Could not parse citation number '{num_str}' from LLM output for agent {agent_id}."
            )
    return built_citations


async def generate_response(
    query: str,
    context_chunks: List[Dict[str, Any]],
    agent_id: str,  # For logging/future agent-specific prompts
    llm_model_name: str = None,  # Allow overriding default model
    system_prompt_template: str = settings.DEFAULT_SYSTEM_PROMPT,  # Use settings
    user_prompt_template: str = settings.DEFAULT_USER_PROMPT,  # Use settings
) -> Dict[str, Any]:
    """
    Generates a response using the LLM based on the query and provided context chunks.
    Does not perform retrieval itself.
    """
    if not query:
        logger.error(
            f"generate_response called for agent {agent_id} with an empty query."
        )
        raise GenerationError("Query cannot be empty.")

    logger.info(
        f"generate_response called for agent {agent_id} with {len(context_chunks)} context chunks. Query: '{query[:50]}...'"
    )
    # Log the content of context_chunks for debugging snippet issue
    if context_chunks:
        logger.debug(
            f"Context chunks received by generate_response for agent {agent_id}: {context_chunks}"
        )
    else:
        logger.warning(
            f"generate_response called for agent {agent_id} with NO context chunks."
        )

    llm = get_llm(model_name=llm_model_name)

    # Attempt to set LLM to JSON mode if available and relevant
    # This is a placeholder for actual LlamaIndex/OpenAI SDK capabilities
    # For example, for OpenAI directly: response_format={ "type": "json_object" }
    # LlamaIndex might wrap this differently or it might be model-specific.
    # For now, we rely on the prompt to enforce JSON string output.
    if hasattr(llm, "response_format"):  # Simplistic check
        try:
            # This syntax is hypothetical for LlamaIndex's OpenAI wrapper
            # llm.response_format = {"type": "json_object"}
            # Or if it's a parameter to the predict/chat call:
            # llm_kwargs={"response_format": {"type": "json_object"}}
            logger.info(
                "LLM might support structured output, but relying on prompt for JSON string."
            )
        except Exception as e:
            logger.warning(f"Could not set LLM to JSON mode: {e}")

    context_str, chunk_references = format_context_chunks(context_chunks)

    chat_template = ChatPromptTemplate(
        message_templates=[
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt_template),
            ChatMessage(role=MessageRole.USER, content=user_prompt_template),
        ]
    )

    # query_bundle = QueryBundle(query_str=query) # Not directly used if formatting into prompt string

    try:
        # Format the messages with actual context and query
        messages = chat_template.format_messages(
            context_str=context_str, query_str=query
        )

        # Use llm.chat for chat models
        llm_response = await llm.achat(messages=messages)

        response_text = "Sorry, I encountered an issue generating a response."
        parsed_citations = []

        if llm_response and llm_response.message:
            raw_llm_output = llm_response.message.content
            # logger.debug(f"Raw LLM output for agent {agent_id}: {raw_llm_output}") # Moved to helper

            try:
                # The LLM is prompted to return a JSON string, so we parse it.
                parsed_llm_data = _parse_llm_json_output(raw_llm_output, agent_id)
                response_text = parsed_llm_data.get(
                    "answer_text", "Error: Missing answer_text from LLM."
                )
                cited_numbers_from_llm = parsed_llm_data.get(
                    "cited_context_numbers", []
                )

                # Citation objects are built based on LLM-provided numbers and original chunk_references
                parsed_citations = _build_citations_from_llm_response(
                    cited_numbers_from_llm, chunk_references, agent_id
                )

            except LLMOutputParsingError:  # Re-raise errors from _parse_llm_json_output
                raise
            except (
                Exception
            ) as e:  # Catch any other unexpected errors during this block
                logger.error(
                    f"Unexpected error processing LLM output for agent {agent_id}: {e}",
                    exc_info=True,
                )
                # Fallback response text already set, parsed_citations will be empty
        else:
            logger.warning(
                f"LLM response or message content was empty for agent {agent_id}."
            )
            # Fallback response text already set, parsed_citations will be empty

        return {"response_text": response_text, "citations": parsed_citations}

    except Exception as e:
        logger.error(
            f"Error during LLM chat completion for agent '{agent_id}': {e}",
            exc_info=True,
        )
        raise GenerationError(f"LLM chat completion failed: {e}")


# Example for a full RAG pipeline (can be moved to an API router or higher-level service)
async def answer_query_with_rag(
    agent_id: str,
    user_query: str,
    # Allow passing retrieval and generation params if needed
) -> Dict[str, Any]:
    """Answers a query using RAG: retrieves context, then generates a response."""
    logger.info(
        f"Answering query for agent {agent_id} using RAG. Query: '{user_query[:50]}...'"
    )
    try:
        # 1. Retrieve context
        # Convert agent_id to string if it's not already, for hybrid_retrieve
        str_agent_id = str(agent_id)
        retrieved_chunks = await hybrid_retrieve(
            query_text=user_query, agent_id=str_agent_id
        )
        logger.info(
            f"Retrieved {len(retrieved_chunks)} chunks for agent {agent_id} with query '{user_query[:50]}...'"
        )
        if not retrieved_chunks:
            logger.warning(
                f"No chunks retrieved for agent {agent_id}, query: '{user_query[:50]}...'. Proceeding without retrieved context."
            )
            # Fallback:  Attempt to generate response without RAG context
            # This might still use general LLM knowledge or a pre-defined persona if the prompt supports it.
            # For now, we'll pass an empty list, which format_context_chunks handles.

        # 2. Generate response using retrieved context
        # The generate_response function expects a list of dicts, where each dict has 'text' and 'metadata'
        # The hybrid_retrieve function should return data in this format.
        # Example chunk structure from hybrid_retrieve:
        # {
        # 'id': 'chunk_id_xyz',
        # 'text': 'The actual text content of the chunk...',
        # 'metadata': {'title': 'Document Title', 'source_document_id': 'file_abc', ...},
        # 'score': 0.85 (optional relevance score)
        # }
        generation_result = await generate_response(
            query=user_query, context_chunks=retrieved_chunks, agent_id=agent_id
        )

        return {
            "answer": generation_result,  # This will be the dict like {"answer_text": "...", "citations": [...]}
            "retrieved_chunks": retrieved_chunks,  # For transparency, pass along what was retrieved
            "retrieved_context_count": len(retrieved_chunks),
            "query": user_query,
            "agent_id": agent_id,
        }
    except Exception as e:
        logger.error(
            f"RAG pipeline failed for agent {agent_id}, query '{user_query[:50]}...': {e}",
            exc_info=True,
        )
        # Consider if a more specific error should be raised or if GenerationError is sufficient.
        # For now, wrapping in GenerationError to be caught by API layer.
        raise GenerationError(f"RAG pipeline failed: {e}") from e
