"""
Database connection management using Prisma.

This __init__.py makes the db module a package and exposes
the core database utility functions and the prisma_client instance.
"""

import logging
from typing import AsyncGenerator, Optional

from fastapi import HTTPException, status  # For raising error in get_db

from prisma import Prisma

logger = logging.getLogger(__name__)

_db_client: Optional[Prisma] = None


async def get_db() -> AsyncGenerator[Prisma, None]:
    """FastAPI dependency to get a Prisma client session."""
    global _db_client
    if _db_client is None or not _db_client.is_connected():
        # This state should ideally not be reached if lifespan manager works correctly
        logger.error(
            "CRITICAL: Prisma client accessed via get_db() but is not connected. Lifespan issue?"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database client is not connected. Please check server logs.",
        )
    yield _db_client


async def connect_db() -> Prisma:
    """Connects to the database. To be called on application startup."""
    global _db_client
    if _db_client is None:
        logger.info("Prisma client is None, creating a new instance.")
        _db_client = Prisma(
            log_queries=False
        )  # Disable verbose query logging by default

    if not _db_client.is_connected():
        logger.info("Prisma client not connected. Attempting to connect...")
        try:
            await _db_client.connect(timeout=10)
            logger.info("Prisma client connected successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Prisma: {e}", exc_info=True)
            raise
    else:
        logger.info("Prisma client is already connected.")
    return _db_client  # Return the client


async def close_db(client: Optional[Prisma] = None):
    """Disconnects the Prisma client."""
    global _db_client
    target_client = client if client is not None else _db_client

    if target_client and target_client.is_connected():
        logger.info("Disconnecting Prisma client...")
        try:
            await target_client.disconnect()
            logger.info("Prisma client disconnected successfully.")
        except Exception as e:
            logger.error(f"Error disconnecting Prisma client: {e}", exc_info=True)
            # Potentially set _db_client to None here if it was the global one
            if target_client is _db_client:
                _db_client = None  # Or re-initialize to a disconnected state
    elif target_client:
        logger.info("Prisma client was already disconnected.")
    else:
        logger.info("No active Prisma client to disconnect.")

    # If the global client was closed (either directly or by passing it as arg)
    if target_client is _db_client:
        _db_client = None  # Ensure global is None after closing


__all__ = ["connect_db", "close_db", "get_db", "_db_client"]
