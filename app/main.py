"""
Main FastAPI application for RAG API.
"""

import logging
import os
import sys

# Add project root to sys.path
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

# Import settings first as it's used by the logger configuration
from app.core.config import settings

# Configure logging for the 'app' module
# This needs to be done BEFORE other 'app' modules might be imported and try to log.
app_module_logger = logging.getLogger(
    "app"
)  # Use "app" as the root for app-specific logs
if not app_module_logger.handlers:  # Avoid adding multiple handlers if uvicorn reloads
    stream_handler = logging.StreamHandler()
    # Use LOG_FORMAT from settings if available, otherwise a sensible default
    log_format_str = (
        settings.LOG_FORMAT
        if hasattr(settings, "LOG_FORMAT")
        else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_formatter = logging.Formatter(log_format_str)
    stream_handler.setFormatter(log_formatter)
    app_module_logger.addHandler(stream_handler)

# Set level based on settings, default to INFO if not set
# Ensure LOG_LEVEL from settings is valid, otherwise default to INFO
log_level_setting = settings.LOG_LEVEL if hasattr(settings, "LOG_LEVEL") else "INFO"
numeric_log_level = getattr(logging, log_level_setting.upper(), logging.INFO)
app_module_logger.setLevel(numeric_log_level)

# Now import other app components that might use logging.getLogger("app.something")
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# from app.core.config import settings # Already imported above
from app.db import close_db, connect_db, get_db
from prisma import Prisma

# Root logger for general FastAPI/Uvicorn, but our app logs are handled by 'app_module_logger'
# logger = logging.getLogger(__name__) # This would be for main.py specific logs, less critical now

# Initialize Prisma Client
# init_db_client()

# Configure logging
# logging.basicConfig(
#     level=settings.LOG_LEVEL.upper(),
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout)],
# )
# logger.info(f"Logging level set to {settings.LOG_LEVEL.upper()}")
# if not settings.OPENAI_API_KEY:
# logger.warning("OPENAI_API_KEY is not set. OpenAI functionalities will fail.")


# Lifespan events for Prisma client
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to the database
    app_module_logger.info("Application startup: Connecting to database...")
    db_client = await connect_db()
    app.state.prisma_client = db_client
    app_module_logger.info("Application startup: Database connection established.")
    yield
    # Shutdown: Disconnect from the database
    app_module_logger.info("Application shutdown: Disconnecting from database...")
    if hasattr(app.state, "prisma_client") and app.state.prisma_client:
        await close_db(app.state.prisma_client)
    app_module_logger.info("Application shutdown: Database connection closed.")


app = FastAPI(
    title="RAG API",
    description="API for interacting with the RAG system, managing agents, and chatting.",
    version="0.1.0",
    lifespan=lifespan,
    # swagger_ui_parameters={"tryItOutEnabled": True},
)

# Configure CORS
if settings.ALLOWED_ORIGINS:
    app_module_logger.info(f"Configuring CORS for origins: {settings.ALLOWED_ORIGINS}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app_module_logger.warning(
        "ALLOWED_ORIGINS is not set. CORS will not be configured."
    )

# Import routers
from app.api.routers import agent_chat_router, agent_crud_router, file_router

# Include API routers
app.include_router(agent_crud_router.router, prefix="/api/v1")
app.include_router(agent_chat_router.router, prefix="/api/v1")
app.include_router(file_router.router, prefix="/api/v1")


# Redirect root to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


# Health check endpoint (example)
@app.get("/health", tags=["Server Health"])
async def health_check(db: Prisma = Depends(get_db)):
    try:
        if db:
            app_module_logger.debug("Health check: Database connection seems OK.")
            return {"status": "healthy", "database_connection": "ok"}
        else:
            app_module_logger.error("Health check: Database client not available.")
            return {"status": "unhealthy", "database_connection": "unavailable"}

    except Exception as e:
        app_module_logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    app_module_logger.info(
        f"Starting Uvicorn server for RAG API on port {settings.SERVER_PORT}..."
    )
    # Pass the app as an import string to enable reload
    uvicorn.run(
        "app.main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=settings.RELOAD_APP,
        log_level=settings.LOG_LEVEL.lower(),
    )
