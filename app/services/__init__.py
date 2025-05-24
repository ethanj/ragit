"""
Service layer for the RAG application.

This __init__.py file makes the 'services' directory a Python package and can
selectively expose service modules or their key components.
"""

# Expose individual service modules to allow imports like:
# from app.services import agent_service
# from app.services import ingestion_orchestrator as ingestion_service

from . import (
    agent_service,
    embedding_service,
    file_service,
    file_validation_service,
    generation_service,
    ingestion_orchestrator,
    ranking_service,
    retrieval_service,
    text_processing_service,
    vector_db_service,
)

# Optionally, define __all__ to specify what `from app.services import *` imports,
# though explicit imports are generally preferred.
# For now, we rely on direct imports of the modules as shown above,
# which are then used like `agent_service.create_agent(...)`.

# Example of exposing specific functions if a flatter API is desired:
# from .agent_service import create_agent, get_agent_by_id
# from .file_service import create_uploaded_file, get_uploaded_file_by_id
# from .ingestion_orchestrator import process_file_for_agent
# ... etc.

# For now, exposing modules is consistent with current usage in routers.
