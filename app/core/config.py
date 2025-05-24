"""
Application configuration settings.
"""

import logging
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env file if it exists
load_dotenv()


class Settings(BaseSettings):
    APP_NAME: str = "RAG API"
    LOG_LEVEL: str = (
        "INFO"  # Default, pydantic-settings will override from env var LOG_LEVEL
    )
    SERVER_PORT: int = (
        8000  # Default, pydantic-settings will override from env var SERVER_PORT
    )
    SERVER_HOST: str = "0.0.0.0"  # Will add later if needed by uvicorn
    RELOAD_APP: bool = False  # Will add later if needed by uvicorn

    ALLOWED_ORIGINS: list[str] = ["*"]  # Added for CORS

    # OpenAI API Key
    OPENAI_API_KEY: str = (
        "your_openai_api_key_here"  # Default, pydantic-settings will override
    )

    # Embedding Model Configuration
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    # For LlamaIndex, this often translates to:
    # OpenAIEmbedding(model="text-embedding-3-small")

    # Ingestion Settings
    DEFAULT_CHUNK_SIZE: int = 512
    DEFAULT_CHUNK_OVERLAP: int = 50
    MIN_METADATA_FIELD_LENGTH: int = 3
    MAX_METADATA_FIELD_LENGTH: int = 256
    METADATA_TRUNCATION_SUFFIX: str = "..."

    # Vector DB Configuration (Chroma)
    CHROMA_PERSIST_DIR: str = "data/chroma_db"

    # LLM Configuration
    LLM_MODEL_NAME: str = "gpt-4o"  # As per PRD FR-05
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: Optional[int] = 512

    # Retrieval settings
    DEFAULT_SEMANTIC_TOP_K: int = 10
    DEFAULT_KEYWORD_TOP_K: int = 10
    DEFAULT_HYBRID_TOP_K: int = (
        5  # Number of results after RRF / initial retrieval for reranker
    )
    RRF_K_CONSTANT: int = 60  # Constant for Reciprocal Rank Fusion
    DEFAULT_RETRIEVAL_STRATEGY: str = "hybrid"

    # Reranker settings
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    RERANKER_TOP_N: int = 3  # Final number of results after reranking

    # Max context tokens for fitting into LLM (conservative for gpt-3.5-turbo 4k/16k context)
    # Should be dynamic based on model later.
    DEFAULT_MAX_CONTEXT_TOKENS_FOR_PROMPT: int = 3000

    DEFAULT_SYSTEM_PROMPT: str = (
        "You are a helpful AI assistant that answers questions based on provided documentation. Your task is to answer questions accurately "
        "based on the provided context. If the context doesn't contain enough information to "
        "answer the question, please say so clearly. When you use information from the context, "
        "cite the relevant parts using [N] format where N is the chunk number.\n\n"
        "Please provide comprehensive and detailed answers when the context allows it. "
        "Be specific and include relevant examples or details from the context when available."
    )

    DEFAULT_USER_PROMPT: str = (
        "Here is some context to help you answer the question. Each context chunk is numbered (e.g., Context Chunk 1, Context Chunk 2).\\n"
        "---------------------\\n"
        "{context_str}\\n"
        "---------------------\\n"
        "Based on the context above, please answer the following question. "
        "When you use information primarily from a specific context chunk in your answer, "
        "please cite it at the end of the relevant sentence(s) using the format [N], where N is the context chunk number. "
        "For example: The system is an innovative platform [1]. It offers various AI solutions [2].\\n"
        "Your final output MUST be a single JSON object string with two keys: "
        '1. "answer_text": Your narrative answer, including the [N] citation markers as described above. '
        '2. "cited_context_numbers": A JSON list of integers representing the numbers of the context chunks you actually cited in your answer_text. For example: [1, 2]. If no chunks were cited, provide an empty list [].\\n'
        "Example of the full JSON output format:\\\\n"
        "```json\\\\n"
        "{\\n"
        '  "answer_text": "The system provides AI-powered solutions for enterprise [1]. Their platform is known for its scalability [3].",\\\\n'
        '  "cited_context_numbers": [1, 3]\\\\n'
        "}\\n"
        "```\\\\n"
        "Now, please answer the question based on the provided context and adhere strictly to this JSON output format.\\\\n"
        "Question: {query_str}"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Allow extra fields in .env not defined in Settings

    # Post-processing for LOG_LEVEL if needed, or handle .upper() where consumed.
    # For now, main.py uses .upper() so keeping the default as uppercase is fine.
    # If we wanted to enforce uppercase via pydantic:
    # @field_validator("LOG_LEVEL")
    # def uppercase_log_level(cls, value):
    #     return value.upper()


settings = Settings()

# Basic logging setup (can be more sophisticated) - REMOVED
# logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

if settings.OPENAI_API_KEY == "your_openai_api_key_here":
    logger.warning(
        "OPENAI_API_KEY is not configured. Please set it in your environment or .env file. "
        "Embedding generation and LLM calls will fail."
    )
