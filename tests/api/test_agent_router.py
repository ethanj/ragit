"""
Tests for the Agent API Router.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.db import get_db as original_get_db  # For dependency override
from app.main import app  # Main FastAPI application

client = TestClient(app)


@pytest.fixture
def mock_prisma_db_agent_router(monkeypatch):
    """Fixture to mock Prisma client methods used in agent router."""
    db_mock = MagicMock()
    created_agents_store = {}

    async def mock_agent_create(data):
        agent_id = f"agent_{len(created_agents_store) + 1}"
        new_agent = MagicMock()
        new_agent.id = agent_id
        new_agent.name = data.get("name")
        new_agent.modelConfig = data.get("modelConfig", "{}")  # JSON string
        new_agent.createdAt = datetime.utcnow()
        new_agent.updatedAt = datetime.utcnow()
        created_agents_store[agent_id] = new_agent
        return new_agent

    async def mock_agent_find_unique(where):
        return created_agents_store.get(where.get("id"))

    async def mock_agent_find_many(*args, **kwargs):
        # Simplified find_many, doesn't handle where/order_by for now for this test
        # It will just return all agents in the store for simplicity of list_agents test
        # Actual call from router: db.agent.find_many()
        # If specific filtering/pagination were tested, this mock would need to be smarter
        return list(created_agents_store.values())

    async def mock_agent_update(where, data):
        agent_id = where.get("id")
        if agent_id in created_agents_store:
            agent = created_agents_store[agent_id]
            if "name" in data:
                agent.name = data["name"]
            if "modelConfig" in data:
                agent.modelConfig = data["modelConfig"]
            agent.updatedAt = datetime.utcnow()
            return agent
        return None

    async def mock_agent_delete(where):
        agent_id = where.get("id")
        return created_agents_store.pop(agent_id, None)

    db_mock.agent.create = AsyncMock(side_effect=mock_agent_create)
    db_mock.agent.find_unique = AsyncMock(side_effect=mock_agent_find_unique)
    db_mock.agent.find_many = AsyncMock(side_effect=mock_agent_find_many)
    db_mock.agent.update = AsyncMock(side_effect=mock_agent_update)
    db_mock.agent.delete = AsyncMock(side_effect=mock_agent_delete)

    def override_get_db():
        return db_mock

    monkeypatch.setitem(app.dependency_overrides, original_get_db, override_get_db)
    yield db_mock, created_agents_store  # expose store for assertions
    monkeypatch.delitem(app.dependency_overrides, original_get_db, raising=False)


def test_create_agent_success(mock_prisma_db_agent_router):
    db_mock, store = mock_prisma_db_agent_router
    model_config_dict = {"llm": "gpt-4"}
    agent_data_payload = {"name": "Test Agent", "modelConfig": model_config_dict}

    response = client.post("/api/v1/agents/", json=agent_data_payload)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Agent"
    assert data["modelConfig"] == model_config_dict
    assert "id" in data
    assert len(store) == 1
    created_agent_id = data["id"]
    assert store[created_agent_id].modelConfig == json.dumps(model_config_dict)


def test_create_agent_duplicate_name(mock_prisma_db_agent_router):
    db_mock, store = mock_prisma_db_agent_router

    # First agent creation
    first_agent_name = "Unique Agent Name"
    client.post("/api/v1/agents/", json={"name": first_agent_name, "modelConfig": {}})
    assert len(store) == 1  # Ensure first one is in our mock store

    # Setup mock for db.agent.find_unique to simulate finding an existing agent by name
    original_find_unique = db_mock.agent.find_unique

    async def mock_find_unique_for_dupe_check(where):
        if where.get("name") == first_agent_name:
            existing_agent_mock = MagicMock()
            existing_agent_mock.name = first_agent_name
            return existing_agent_mock
        return await original_find_unique(where=where)

    db_mock.agent.find_unique = AsyncMock(side_effect=mock_find_unique_for_dupe_check)

    # Attempt to create another agent with the same name
    response = client.post(
        "/api/v1/agents/", json={"name": first_agent_name, "modelConfig": {}}
    )
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]
    assert len(store) == 1  # Should not have created a second agent

    db_mock.agent.find_unique = original_find_unique


def test_list_agents_success(mock_prisma_db_agent_router):
    db_mock, store = mock_prisma_db_agent_router
    client.post("/api/v1/agents/", json={"name": "Agent Alpha", "modelConfig": {}})
    client.post("/api/v1/agents/", json={"name": "Agent Beta", "modelConfig": {}})

    response = client.get("/api/v1/agents/")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["name"] == "Agent Alpha"
    assert data[1]["name"] == "Agent Beta"


def test_get_agent_details_success(mock_prisma_db_agent_router):
    db_mock, store = mock_prisma_db_agent_router
    create_response = client.post(
        "/api/v1/agents/", json={"name": "Detail Agent", "modelConfig": {}}
    )
    agent_id = create_response.json()["id"]

    response = client.get(f"/api/v1/agents/{agent_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == agent_id
    assert data["name"] == "Detail Agent"


def test_get_agent_details_not_found(mock_prisma_db_agent_router):
    response = client.get("/api/v1/agents/non_existent_id")
    assert response.status_code == 404


def test_update_agent_details_success(mock_prisma_db_agent_router):
    db_mock, store = mock_prisma_db_agent_router
    initial_model_config = {"old_setting": 1}
    create_response = client.post(
        "/api/v1/agents/",
        json={"name": "Old Name", "modelConfig": initial_model_config},
    )
    assert create_response.status_code == 201
    agent_id = create_response.json()["id"]
    assert store[agent_id].modelConfig == json.dumps(initial_model_config)

    updated_model_config_dict = {"temperature": 0.8, "new_setting": True}
    update_data_payload = {"name": "New Name", "modelConfig": updated_model_config_dict}

    response = client.put(f"/api/v1/agents/{agent_id}", json=update_data_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "New Name"
    assert data["modelConfig"] == updated_model_config_dict
    assert store[agent_id].name == "New Name"
    assert store[agent_id].modelConfig == json.dumps(updated_model_config_dict)


def test_update_agent_not_found(mock_prisma_db_agent_router):
    response = client.put("/api/v1/agents/non_existent_id", json={"name": "Ghost Name"})
    assert response.status_code == 404


@patch(
    "app.api.routers.agent_crud_router.agent_service.delete_agent_comprehensively",
    new_callable=AsyncMock,  # Ensure it's an AsyncMock
)
def test_delete_agent_success(
    mock_delete_comprehensive, mock_prisma_db_agent_router
):  # Changed mock_prisma_db to mock_prisma_db_agent_router
    db_mock, store = (
        mock_prisma_db_agent_router  # Assuming this fixture provides the mock db and a way to check store
    )
    # If not, this needs to align with how agent_id is obtained for deletion check

    # Create an agent to delete - this part should be real or appropriately mocked for ID generation
    # For this test, we assume an agent with a known ID that mock_delete_comprehensive will act upon.
    # The original test created an agent. If we purely test the router's reaction to the service,
    # we just need an ID.
    agent_id_to_delete = "agent_id_that_will_be_deleted"  # Example ID

    mock_delete_comprehensive.return_value = (
        True  # Mock successful comprehensive delete
    )

    response = client.delete(f"/api/v1/agents/{agent_id_to_delete}")

    assert response.status_code == 204
    mock_delete_comprehensive.assert_called_once_with(
        db_mock,
        agent_id_to_delete,  # agent_id is positional
    )
    # Assertions about 'store' might need to be removed if we don't create/delete from a real mock store here.


@patch(
    "app.api.routers.agent_crud_router.agent_service.delete_agent_comprehensively",
    new_callable=AsyncMock,  # Ensure AsyncMock
)
def test_delete_agent_not_found(
    mock_delete_comprehensive,
    mock_prisma_db_agent_router,  # Changed fixture name
):
    db_mock, _ = mock_prisma_db_agent_router  # Get the db_mock

    agent_id_not_found = "non_existent_agent_id_for_delete"
    # Configure the mock to raise the specific ValueError the router expects
    mock_delete_comprehensive.side_effect = ValueError(
        f"Agent with ID '{agent_id_not_found}' not found, cannot delete."
    )

    response = client.delete(f"/api/v1/agents/{agent_id_not_found}")

    assert response.status_code == 404  # Test expects 404

    mock_delete_comprehensive.assert_called_once_with(
        db_mock,
        agent_id_not_found,  # agent_id is a positional argument in the actual call
    )
