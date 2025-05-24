"""
Streamlit Chat Interface for the RAG Prototype.
"""

import asyncio  # Import asyncio
import os
import re
import sys  # Add sys import
from typing import Any, Dict, List, Optional

# Add project root to sys.path to allow importing app modules
project_root_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)  # Go up two levels (ui -> app -> root)
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

import httpx  # For making API calls to FastAPI backend
import streamlit as st

from app.core.config import settings  # Import settings

# --- Configuration & State Management ---
st.set_page_config(page_title="RAG Chat", layout="wide")

if "api_base_url" not in st.session_state:
    # Default to the port FastAPI is configured to run on (e.g. 8010 from your logs)
    # Ideally, this would come from a shared config or env var if consistently different from 8000
    st.session_state.api_base_url = "http://localhost:8010"
if "selected_agent_id" not in st.session_state:
    st.session_state.selected_agent_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of ChatMessageResponse like dicts
if "agents_list" not in st.session_state:
    st.session_state.agents_list = []  # List of AgentResponse like dicts
if "editing_agent_id" not in st.session_state:
    st.session_state.editing_agent_id = (
        None  # Tracks which agent's config is being edited
    )

# --- API Client Functions ---
# These functions will interact with the FastAPI backend.


async def get_agents(api_url: str) -> List[Dict[str, Any]]:
    """Fetches a list of available agents from the API."""
    # st.write(f"DEBUG UI: Attempting to fetch agents from {api_url}/api/v1/agents") # REMOVED DEBUG
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/api/v1/agents")
            response.raise_for_status()  # Raise an exception for bad status codes
            agents_data = response.json()
            # st.write(f"DEBUG UI: Agents data received from API: {agents_data}") # REMOVED DEBUG
            return agents_data
    except httpx.RequestError as e:
        st.error(f"Error fetching agents: {e}. Is the backend running at {api_url}?")
        # st.write(f"DEBUG UI: httpx.RequestError fetching agents: {e}") # REMOVED DEBUG
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching agents: {e}")
        # st.write(f"DEBUG UI: Unexpected error fetching agents: {e}") # REMOVED DEBUG
        return []


async def get_chat_history_for_agent(
    api_url: str, agent_id: str
) -> List[Dict[str, Any]]:
    """Fetches chat history for a selected agent."""
    if not agent_id:
        return []
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{api_url}/api/v1/agents/{agent_id}/chat/?limit=100"
            )  # Fetch last 100
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        st.error(f"Error fetching chat history: {e}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching chat history: {e}")
        return []


async def post_chat_message(
    api_url: str, agent_id: str, message_content: str
) -> Optional[Dict[str, Any]]:
    """Posts a new chat message and gets the assistant's response."""
    if not agent_id or not message_content:
        return None
    try:
        payload = {"content": message_content}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{api_url}/api/v1/agents/{agent_id}/chat/", json=payload, timeout=120.0
            )
            response.raise_for_status()
            return response.json()  # This should be the assistant's response message
    except httpx.HTTPStatusError as e:
        st.error(f"Error sending message: {e.response.status_code} - {e.response.text}")
        # Try to parse error detail from FastAPI
        try:
            error_detail = e.response.json().get("detail", "Unknown error")
            st.session_state.chat_history.append(
                {"role": "system", "content": f"Error from backend: {error_detail}"}
            )
        except:
            st.session_state.chat_history.append(
                {"role": "system", "content": f"Error: {e.response.text}"}
            )
        return None
    except httpx.RequestError as e:
        st.error(f"Error sending message: {e}")
        st.session_state.chat_history.append(
            {"role": "system", "content": f"Request Error: {e}"}
        )
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while sending message: {e}")
        return None


async def upload_file_for_agent_api(
    api_url: str, agent_id: str, file_to_upload
) -> Optional[Dict[str, Any]]:
    """Uploads a file for the specified agent."""
    if not agent_id or not file_to_upload:
        return None

    files = {"file": (file_to_upload.name, file_to_upload, file_to_upload.type)}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{api_url}/api/v1/agents/{agent_id}/files/", files=files, timeout=30.0
            )
            response.raise_for_status()
            return response.json()  # Returns the initial UploadedFile record
    except httpx.HTTPStatusError as e:
        st.error(f"Error uploading file: {e.response.status_code} - {e.response.text}")
        try:
            error_detail = e.response.json().get("detail", "File upload failed.")
            st.toast(f"Upload Error: {error_detail}", icon="🚨")
        except:
            st.toast(f"Upload Error: {e.response.text}", icon="🚨")
        return None
    except httpx.RequestError as e:
        st.error(f"Error uploading file: {e}")
        st.toast(f"Upload Request Error: {e}", icon="🚨")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during file upload: {e}")
        st.toast(f"Unexpected UploadError: {e}", icon="🚨")
        return None


async def get_files_for_agent_api(api_url: str, agent_id: str) -> List[Dict[str, Any]]:
    """Fetches the list of files for the specified agent."""
    if not agent_id:
        return []
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/api/v1/agents/{agent_id}/files/")
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        st.error(f"Error fetching files for agent {agent_id}: {e}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching files: {e}")
        return []


async def create_agent_api(
    api_url: str, agent_name: str, model_config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Creates a new agent via the API."""
    try:
        payload = {"name": agent_name, "modelConfig": model_config}
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{api_url}/api/v1/agents", json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        st.error(f"Error creating agent: {e.response.status_code} - {e.response.text}")
        try:
            error_detail = e.response.json().get("detail", "Agent creation failed.")
            st.toast(f"Creation Error: {error_detail}", icon="🚨")
        except:
            st.toast(f"Creation Error: {e.response.text}", icon="🚨")
        return None
    except httpx.RequestError as e:
        st.error(f"Error creating agent: {e}")
        st.toast(f"Creation Request Error: {e}", icon="🚨")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during agent creation: {e}")
        st.toast(f"Unexpected Creation Error: {e}", icon="🚨")
        return None


async def update_agent_api(
    api_url: str,
    agent_id: str,
    agent_name: Optional[str],
    model_config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Updates an existing agent via the API."""
    payload = {}
    if agent_name is not None:
        payload["name"] = agent_name
    if model_config is not None:
        payload["modelConfig"] = model_config

    if not payload:  # Nothing to update
        st.toast("No changes to update.", icon="ℹ️")
        return None  # Or perhaps return current agent data if needed

    try:
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{api_url}/api/v1/agents/{agent_id}", json=payload
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        st.error(f"Error updating agent: {e.response.status_code} - {e.response.text}")
        try:
            error_detail = e.response.json().get("detail", "Agent update failed.")
            st.toast(f"Update Error: {error_detail}", icon="🚨")
        except:
            st.toast(f"Update Error: {e.response.text}", icon="🚨")
        return None
    except httpx.RequestError as e:
        st.error(f"Error updating agent: {e}")
        st.toast(f"Update Request Error: {e}", icon="🚨")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during agent update: {e}")
        st.toast(f"Unexpected Update Error: {e}", icon="🚨")
        return None


async def delete_chat_history_api(
    api_url: str, agent_id: str
) -> Optional[Dict[str, Any]]:
    """Deletes chat history for the specified agent via the API."""
    if not agent_id:
        st.error("Agent ID is required to delete chat history.")
        return None
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{api_url}/api/v1/agents/{agent_id}/chat/")
            response.raise_for_status()
            return (
                response.json()
            )  # Should contain { "message": str, "deleted_count": int }
    except httpx.HTTPStatusError as e:
        st.error(
            f"Error deleting chat history: {e.response.status_code} - {e.response.text}"
        )
        try:
            error_detail = e.response.json().get(
                "detail", "Chat history deletion failed."
            )
            st.toast(f"Deletion Error: {error_detail}", icon="🚨")
        except:
            st.toast(f"Deletion Error: {e.response.text}", icon="🚨")
        return None
    except httpx.RequestError as e:
        st.error(f"Error deleting chat history: {e}")
        st.toast(f"Deletion Request Error: {e}", icon="🚨")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during chat history deletion: {e}")
        st.toast(f"Unexpected Deletion Error: {e}", icon="🚨")
        return None


async def rate_message_api(
    api_url: str, agent_id: str, message_id: str, rating: int
) -> Optional[Dict[str, Any]]:
    """Rates a specific chat message."""
    if not agent_id or not message_id:
        return None
    try:
        payload = {"rating": rating}
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{api_url}/api/v1/agents/{agent_id}/chat/{message_id}/rating",
                json=payload,
            )
            response.raise_for_status()
            return response.json()  # Returns the updated message
    except httpx.HTTPStatusError as e:
        st.error(
            f"Error rating message {message_id}: {e.response.status_code} - {e.response.text}"
        )
        return None
    except httpx.RequestError as e:
        st.error(f"Error rating message {message_id}: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while rating message {message_id}: {e}")
        return None


# --- UI Layout & Logic ---

# st.sidebar.title("⚙️ Configuration") # REMOVED
# st.session_state.api_base_url = st.sidebar.text_input( # REMOVED
#     "FastAPI Backend URL", value=st.session_state.api_base_url # REMOVED
# ) # REMOVED


# --- Callback Functions ---
def handle_agent_selection_change():
    """
    Callback function for when the agent selection changes in the sidebar.
    Updates the selected agent ID and fetches/clears chat history accordingly.
    """
    placeholder = "Select an agent..."  # Must match the placeholder in selectbox
    selected_name_from_widget = st.session_state.agent_selector  # Get value using key

    if selected_name_from_widget and selected_name_from_widget != placeholder:
        newly_selected_id = next(
            (
                agent["id"]
                for agent in st.session_state.agents_list
                if agent["name"] == selected_name_from_widget
            ),
            None,
        )
        if st.session_state.selected_agent_id != newly_selected_id:
            st.session_state.selected_agent_id = newly_selected_id
            if newly_selected_id:
                st.session_state.chat_history = asyncio.run(
                    get_chat_history_for_agent(
                        st.session_state.api_base_url, newly_selected_id
                    )
                )
            else:  # Placeholder selected or agent became invalid
                st.session_state.chat_history = []
            # st.rerun() # Streamlit usually reruns automatically after on_change completes
    elif selected_name_from_widget == placeholder:
        # If placeholder is selected, clear current agent context
        if st.session_state.selected_agent_id is not None:
            st.session_state.selected_agent_id = None
            st.session_state.chat_history = []
            # st.rerun()


# --- Sidebar Agent Selection & Management ---
st.sidebar.subheader("🤖 Select Agent")
placeholder = "Select an agent..."  # Or any other placeholder text
# Initialize agents_list if it's not already (e.g., on first run)
if not st.session_state.agents_list:
    st.session_state.agents_list = asyncio.run(
        get_agents(st.session_state.api_base_url)
    )
    # st.write(f"DEBUG UI: Populated st.session_state.agents_list: {st.session_state.agents_list}") # REMOVED DEBUG

# This is the selectbox that needs to be present
selected_agent_name_variable_for_widget = (
    st.sidebar.selectbox(  # Renamed variable to avoid conflict if used elsewhere
        "Available Agents:",
        options=[placeholder]
        + [
            agent.get("name", "Unnamed Agent") for agent in st.session_state.agents_list
        ],
        index=0,  # Default to placeholder
        key="agent_selector",  # Key used by the on_change callback
        on_change=handle_agent_selection_change,
    )
)

# Display selected agent's configuration if an agent is selected
if st.session_state.selected_agent_id:
    selected_agent_details = next(
        (
            agent
            for agent in st.session_state.agents_list
            if agent["id"] == st.session_state.selected_agent_id
        ),
        None,
    )
    if selected_agent_details:
        agent_id_for_form = selected_agent_details["id"]
        is_editing_this_agent = st.session_state.editing_agent_id == agent_id_for_form

        st.sidebar.subheader(f"⚙️ Config: {selected_agent_details.get('name', 'N/A')}")
        with st.sidebar.expander(
            "Manage Configuration", expanded=is_editing_this_agent
        ):
            model_config = selected_agent_details.get("modelConfig", {})
            # Ensure model_config is a dict for safe .get() usage
            if not isinstance(model_config, dict):
                model_config = {}

            edited_agent_name = st.text_input(
                "Agent Name*",
                value=selected_agent_details.get("name", ""),
                disabled=not is_editing_this_agent,
                key=f"edit_name_{agent_id_for_form}",
            )

            # Model Configuration Fields
            edited_llm_model_name = st.text_input(
                "LLM Model Name",
                value=model_config.get("llm_model_name", settings.LLM_MODEL_NAME),
                disabled=not is_editing_this_agent,
                key=f"edit_llm_{agent_id_for_form}",
            )
            edited_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(model_config.get("temperature", settings.LLM_TEMPERATURE)),
                step=0.01,
                disabled=not is_editing_this_agent,
                key=f"edit_temp_{agent_id_for_form}",
            )

            retrieval_options = ["hybrid", "vector", "keyword"]
            current_retrieval_strategy = model_config.get(
                "retrieval_strategy", settings.DEFAULT_RETRIEVAL_STRATEGY
            )
            try:
                retrieval_index = retrieval_options.index(current_retrieval_strategy)
            except (
                ValueError
            ):  # Should not happen if DEFAULT_RETRIEVAL_STRATEGY is in options
                retrieval_index = retrieval_options.index(
                    settings.DEFAULT_RETRIEVAL_STRATEGY
                )

            edited_retrieval_strategy = st.selectbox(
                "Retrieval Strategy",
                options=retrieval_options,
                index=retrieval_index,
                disabled=not is_editing_this_agent,
                key=f"edit_retrieval_{agent_id_for_form}",
            )
            edited_embedding_model_name = st.text_input(
                "Embedding Model Name",
                value=model_config.get(
                    "embedding_model_name", settings.EMBEDDING_MODEL_NAME
                ),
                disabled=not is_editing_this_agent,
                key=f"edit_embedding_{agent_id_for_form}",
            )
            edited_prompt_template = st.text_area(
                "Prompt Template",
                value=model_config.get("prompt_template", settings.DEFAULT_USER_PROMPT),
                height=150,
                disabled=not is_editing_this_agent,
                key=f"edit_prompt_{agent_id_for_form}",
            )

            if is_editing_this_agent:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "💾 Save Changes",
                        key=f"save_agent_{agent_id_for_form}",
                        type="primary",
                    ):
                        if not edited_agent_name or len(edited_agent_name) < 3:
                            st.error(
                                "Agent Name is required and must be at least 3 characters long."
                            )
                        else:
                            updated_model_config = {
                                "llm_model_name": edited_llm_model_name,
                                "temperature": edited_temperature,
                                "retrieval_strategy": edited_retrieval_strategy,
                                "embedding_model_name": edited_embedding_model_name,
                                "prompt_template": edited_prompt_template,
                            }
                            # Determine what actually changed to avoid unnecessary updates
                            name_changed = (
                                edited_agent_name != selected_agent_details.get("name")
                            )
                            config_changed = (
                                updated_model_config != model_config
                            )  # Compare with original parsed model_config

                            with st.spinner("Saving changes..."):
                                updated_agent = asyncio.run(
                                    update_agent_api(
                                        st.session_state.api_base_url,
                                        agent_id_for_form,
                                        agent_name=edited_agent_name
                                        if name_changed
                                        else None,  # Send name only if changed
                                        model_config=updated_model_config
                                        if config_changed
                                        else None,  # Send config only if changed
                                    )
                                )
                            if updated_agent:
                                st.success(
                                    f"Agent '{updated_agent.get('name')}' updated successfully!"
                                )
                                st.session_state.editing_agent_id = (
                                    None  # Exit edit mode
                                )
                                # Refresh agent list to get the latest data
                                st.session_state.agents_list = asyncio.run(
                                    get_agents(st.session_state.api_base_url)
                                )
                                st.rerun()
                            # update_agent_api handles its own error toasts/messages

                with col2:
                    if st.button(
                        "❌ Cancel", key=f"cancel_edit_agent_{agent_id_for_form}"
                    ):
                        st.session_state.editing_agent_id = None
                        st.rerun()
            else:
                if st.button(
                    "✏️ Edit Agent Configuration",
                    key=f"edit_agent_btn_{agent_id_for_form}",
                ):
                    st.session_state.editing_agent_id = agent_id_for_form
                    st.rerun()

# Agent Creation Form
st.sidebar.subheader("➕ Create New Agent")
with st.sidebar.expander("Agent Creation Form", expanded=False):
    with st.form("new_agent_form"):
        new_agent_name = st.text_input("Agent Name*", key="new_agent_name")
        # We can decide later if/how to store it (e.g., in modelConfig or a new field)
        new_agent_description = st.text_area(
            "Agent Description (Optional)", key="new_agent_description"
        )

        st.markdown("**Model Configuration**")
        # Define operational defaults - REMOVED, will use settings
        # OPERATIONAL_LLM_MODEL = "gpt-4o"
        # OPERATIONAL_TEMPERATURE = 0.1
        # OPERATIONAL_RETRIEVAL_STRATEGY = "hybrid"
        # OPERATIONAL_EMBEDDING_MODEL = "text-embedding-3-small"
        # OPERATIONAL_PROMPT_TEMPLATE = ( ... )

        mc_llm_model_name = st.text_input(
            "LLM Model Name",
            value=settings.LLM_MODEL_NAME,
            key="mc_llm_model",  # Use settings
        )
        mc_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=settings.LLM_TEMPERATURE,  # Use settings
            step=0.01,
            key="mc_temp",
        )
        mc_retrieval_strategy = st.selectbox(
            "Retrieval Strategy",
            options=[
                "hybrid",
                "vector",
                "keyword",
            ],
            index=["hybrid", "vector", "keyword"].index(
                settings.DEFAULT_RETRIEVAL_STRATEGY
            ),  # Use settings
            key="mc_retrieval",
        )
        mc_embedding_model_name = st.text_input(
            "Embedding Model Name",
            value=settings.EMBEDDING_MODEL_NAME,  # Use settings
            key="mc_embedding",
        )
        mc_prompt_template = st.text_area(
            "Prompt Template",
            value=settings.DEFAULT_USER_PROMPT,  # Use settings
            height=200,
            key="mc_prompt",
        )

        submitted = st.form_submit_button("Create Agent")
        if submitted:
            if not new_agent_name or len(new_agent_name) < 3:
                st.error(
                    "Agent Name is required and must be at least 3 characters long."
                )
            else:
                model_config_payload = {
                    "llm_model_name": mc_llm_model_name,
                    "temperature": mc_temperature,
                    "retrieval_strategy": mc_retrieval_strategy,
                    "embedding_model_name": mc_embedding_model_name,
                    "prompt_template": mc_prompt_template,
                    # Add description here if we want to store it in modelConfig
                    # "description": new_agent_description
                }
                # Show a spinner while creating
                with st.spinner("Creating agent..."):
                    created_agent = asyncio.run(
                        create_agent_api(
                            st.session_state.api_base_url,
                            new_agent_name,
                            model_config_payload,
                        )
                    )
                if created_agent:
                    st.success(
                        f"Agent '{created_agent.get('name')}' created successfully!"
                    )
                    # Refresh agent list
                    st.session_state.agents_list = asyncio.run(
                        get_agents(st.session_state.api_base_url)
                    )
                    # Optionally, clear the form or select the new agent
                    # For now, just rerun to update the selectbox
                    st.rerun()
                # Error messages are handled by create_agent_api using st.error/st.toast

# --- Main Content Area ---
if st.session_state.selected_agent_id:
    # Agent is selected, show chat and file management for that agent
    selected_agent_details = next(
        (
            agent
            for agent in st.session_state.agents_list
            if agent["id"] == st.session_state.selected_agent_id
        ),
        None,
    )
    selected_agent_name = (
        selected_agent_details["name"] if selected_agent_details else "Unknown Agent"
    )

    st.title(f"🗣️ Agent: {selected_agent_name}")

    # Create tabs for Chat and Documents
    tab_chat, tab_documents = st.tabs(["💬 Chat", "📄 Documents"])

    with tab_chat:
        st.header(f"Chat with {selected_agent_name}")

        # Add Clear Chat History button
        if st.button(
            "🗑️ Clear Chat",
            key=f"clear_chat_{st.session_state.selected_agent_id}",
        ):
            if st.session_state.selected_agent_id:
                # Confirmation dialog would be nice, but st.confirm is not async friendly directly
                # For now, direct deletion upon click
                with st.spinner("Clearing chat..."):
                    delete_response = asyncio.run(
                        delete_chat_history_api(
                            st.session_state.api_base_url,
                            st.session_state.selected_agent_id,
                        )
                    )
                if delete_response and "deleted_count" in delete_response:
                    st.success(
                        delete_response.get("message", "Chat cleared successfully.")
                    )
                    st.session_state.chat_history = []  # Clear local state
                    st.rerun()
                else:
                    # Error toasts are handled by delete_chat_history_api
                    st.error(
                        "Failed to clear chat. Check logs or backend status."
                    )  # Fallback message
            else:
                st.warning("No agent selected to clear chat for.")
        st.markdown("---")  # Visual separator

        # Display chat messages
        # Ensure chat history is loaded if empty
        if not st.session_state.chat_history:
            st.session_state.chat_history = asyncio.run(
                get_chat_history_for_agent(
                    st.session_state.api_base_url, st.session_state.selected_agent_id
                )
            )

        # chat_container = st.container()
        # with chat_container:
        for message_idx, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                elif message["role"] == "assistant":
                    # Render message content with interactive citations
                    message_content = message.get("content", "")
                    citations_data = message.get(
                        "citations", []
                    )  # This should be a list of dicts

                    if not message_content:
                        st.write("Assistant provided an empty response.")
                    else:
                        # Pattern to find [N] where N is one or more digits
                        pattern = r"(\[\d+\])"  # Corrected regex pattern
                        segments = re.split(pattern, message_content)

                        processed_html_segments = []
                        for segment_idx, segment in enumerate(segments):
                            # Check if the segment is a marker
                            if re.fullmatch(pattern, segment):
                                try:
                                    citation_info = next(
                                        (
                                            c
                                            for c in citations_data
                                            if c.get("marker") == segment
                                        ),
                                        None,
                                    )

                                    if citation_info:
                                        title = (
                                            str(citation_info.get("title", "N/A"))
                                            .replace('"', "&quot;")
                                            .replace("\\n", " ")
                                        )
                                        c_id = str(
                                            citation_info.get("id", "N/A")
                                        ).replace('"', "&quot;")
                                        snippet = (
                                            str(
                                                citation_info.get("text_snippet", "N/A")
                                            )
                                            .replace('"', "&quot;")
                                            .replace("\\n", " ")
                                        )

                                        tooltip_text = f"Source: {title}\\nID: {c_id}\\nSnippet: {snippet}"

                                        # Simple styled span for tooltip
                                        processed_html_segments.append(
                                            f'<span title="{tooltip_text}" style="text-decoration: underline; color: #1E90FF; cursor: help;">{segment}</span>'
                                        )
                                    else:
                                        processed_html_segments.append(segment)
                                except Exception as e:
                                    # st.error(f"Error processing citation {segment}: {e}") # Avoid erroring out UI
                                    print(
                                        f"Error processing citation marker {segment} for message {message.get('id', 'N/A')}: {e}"
                                    )
                                    processed_html_segments.append(segment)
                            else:
                                processed_html_segments.append(segment)

                        st.markdown(
                            "".join(processed_html_segments), unsafe_allow_html=True
                        )

                    # Display citations below the message if they exist
                    if citations_data:
                        with st.expander("References", expanded=False):
                            for cit_idx, citation in enumerate(citations_data):
                                title = citation.get("title", "Unknown Source")
                                snippet = citation.get("text_snippet", "N/A")
                                # chunk_id = citation.get("id", "N/A") # This is the long ChromaDB ID
                                original_doc_id = citation.get(
                                    "original_document_id"
                                )  # User-facing file ID
                                url = citation.get("url")
                                marker = citation.get("marker", "")

                                display_title = title
                                if url:
                                    display_title = f"[{title}]({url})"

                                id_info_parts = []
                                if original_doc_id:
                                    id_info_parts.append(
                                        f"File ID: `{original_doc_id}`"
                                    )
                                # else:
                                # id_info_parts.append(f"Chunk ID: `{chunk_id}`") # Fallback if no original_doc_id

                                id_display_str = (
                                    " (" + ", ".join(id_info_parts) + ")"
                                    if id_info_parts
                                    else ""
                                )

                                help_text_parts = [f"Source: {title}"]
                                if original_doc_id:
                                    help_text_parts.append(
                                        f"File ID: {original_doc_id}"
                                    )
                                # help_text_parts.append(f"Chunk ID: {chunk_id}")
                                if url:
                                    help_text_parts.append(f"URL: {url}")
                                help_text_parts.append(f"Snippet: {snippet}")

                                st.markdown(
                                    f"""**{marker} {display_title}**{id_display_str}
                                    > _{snippet if snippet and snippet != "N/A" else "Snippet not available."}_
                                    """,
                                    help=" | ".join(help_text_parts),
                                )
                                if cit_idx < len(citations_data) - 1:
                                    st.divider()
                elif (
                    message["role"] == "system"
                ):  # Handle system messages (e.g., errors)
                    st.warning(message["content"])

                # Add rating buttons for assistant messages
                if message["role"] == "assistant" and message.get("id"):
                    message_id = message["id"]
                    current_rating = message.get("rating")

                    cols = st.columns([1, 1, 10])  # Adjust column ratios as needed
                    with cols[0]:
                        if st.button(
                            "👍",
                            key=f"thumb_up_{message_id}_{message_idx}",
                            help="Mark as helpful",
                            type="primary" if current_rating == 1 else "secondary",
                        ):
                            # Original logic: set to 1, or if already 1, could optionally clear, but simple set is fine
                            # For this revert, we go back to simple set, not toggle-off
                            updated_msg = asyncio.run(
                                rate_message_api(
                                    st.session_state.api_base_url,
                                    st.session_state.selected_agent_id,
                                    message_id,
                                    1,
                                )
                            )
                            if updated_msg:
                                st.session_state.chat_history[message_idx] = updated_msg
                                st.rerun()
                    with cols[1]:
                        if st.button(
                            "👎",
                            key=f"thumb_down_{message_id}_{message_idx}",
                            help="Mark as not helpful",
                            type="primary" if current_rating == -1 else "secondary",
                        ):
                            # Original logic: set to -1
                            updated_msg = asyncio.run(
                                rate_message_api(
                                    st.session_state.api_base_url,
                                    st.session_state.selected_agent_id,
                                    message_id,
                                    -1,
                                )
                            )
                            if updated_msg:
                                st.session_state.chat_history[message_idx] = updated_msg
                                st.rerun()

        # Input for new chat message
        if prompt := st.chat_input("What would you like to discuss?"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            # Immediately display user message
            # st.rerun() # This might be too quick, let's wait for assistant

            with st.spinner("Assistant is thinking..."):
                assistant_response_message = asyncio.run(
                    post_chat_message(
                        st.session_state.api_base_url,
                        st.session_state.selected_agent_id,
                        prompt,
                    )
                )
            if assistant_response_message:
                st.session_state.chat_history.append(assistant_response_message)
            else:
                # Error already handled by post_chat_message with st.error and system message
                pass
            st.rerun()

    with tab_documents:
        st.header(f"Documents for {selected_agent_name}")

        # File Uploader
        # Ensure this session state variable is initialized for tracking upload
        if "ui_file_tracker" not in st.session_state:
            st.session_state.ui_file_tracker = {
                "current_file_name_processing": None,
                "lock_active_upload": False,
            }

        uploaded_file_obj = st.file_uploader(  # Renamed for clarity from uploaded_file
            "Upload a document to this agent's knowledge base",
            type=[".pdf", ".txt", ".md", ".html", ".docx", ".csv"],
            key="document_uploader_key",  # Added a key
        )

        if uploaded_file_obj is not None:
            # Check if this is a new file or the same file that was just processed
            # and if an upload isn't already locked
            if (
                st.session_state.ui_file_tracker["current_file_name_processing"]
                != uploaded_file_obj.name
                and not st.session_state.ui_file_tracker["lock_active_upload"]
            ):
                st.session_state.ui_file_tracker["current_file_name_processing"] = (
                    uploaded_file_obj.name
                )
                st.session_state.ui_file_tracker["lock_active_upload"] = (
                    True  # Lock to prevent re-entry
                )

                with st.spinner(f"Uploading {uploaded_file_obj.name}..."):
                    upload_response = asyncio.run(
                        upload_file_for_agent_api(
                            st.session_state.api_base_url,
                            st.session_state.selected_agent_id,
                            uploaded_file_obj,  # Use the renamed variable
                        )
                    )
                    if upload_response and upload_response.get("id"):
                        st.success(
                            f"File '{upload_response.get('filename', uploaded_file_obj.name)}' uploaded successfully with ID: {upload_response['id']}. It will be processed shortly."
                        )
                        # File processed, name remains in current_file_name_processing
                        # to prevent re-upload of the same displayed file if script reruns.
                        # Lock is released below.
                    else:
                        st.error("File upload failed. Check backend logs.")
                        # If failed, clear the name so user can retry the same file
                        st.session_state.ui_file_tracker[
                            "current_file_name_processing"
                        ] = None

                st.session_state.ui_file_tracker["lock_active_upload"] = (
                    False  # Release lock
                )

        # If the file uploader becomes empty (e.g., user clears it), reset our tracker
        elif (
            uploaded_file_obj is None
            and st.session_state.ui_file_tracker["current_file_name_processing"]
            is not None
        ):
            st.session_state.ui_file_tracker["current_file_name_processing"] = None
            st.session_state.ui_file_tracker["lock_active_upload"] = (
                False  # Ensure lock is also reset
            )

        st.divider()

        # Display list of files
        st.subheader("Uploaded Documents")
        if st.button("Refresh Files List"):
            # No specific state for files list, just trigger a rerun which will call get_files... if needed
            pass  # The get_files_for_agent_api call will happen if the display logic needs it

        files_list = asyncio.run(
            get_files_for_agent_api(
                st.session_state.api_base_url, st.session_state.selected_agent_id
            )
        )

        if files_list:
            for file_data in files_list:
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                with col1:
                    st.markdown(f"**{file_data.get('fileName', 'N/A')}**")
                with col2:
                    st.caption(
                        f"ID: {file_data.get('id', 'N/A')[:8]}..."
                    )  # Shortened ID
                with col3:
                    status = file_data.get("status", "N/A")
                    if status == "completed":
                        st.success(status, icon="✅")
                    elif status == "processing" or status == "pending":
                        st.info(status, icon="⏳")
                    elif status == "failed":
                        st.error(status, icon="🚨")
                    else:
                        st.write(status)
                with col4:
                    file_size_bytes = file_data.get("fileSize")
                    if file_size_bytes is not None:
                        if file_size_bytes == 0:
                            display_size_str = "0.00 MB"
                        else:
                            size_mb = file_size_bytes / (1024 * 1024)
                            if size_mb < 0.01:
                                display_size_str = "0.01 MB"  # Minimum display for non-zero files < 0.01MB
                            else:
                                display_size_str = f"{size_mb:.2f} MB"
                        st.caption(f"Size: {display_size_str}")
                    else:
                        st.caption("Size: N/A")

                # Expander for more details, e.g., error message if failed
                if file_data.get("error"):  # Condition simplified, only check for error
                    with st.expander("Details", expanded=False):
                        if file_data.get("error"):
                            st.error(
                                f"Error: {file_data.get('error', 'No specific error message.')}"
                            )
                        # chunk_count and url are no longer in UploadedFileResponse based on current schema
                        # Restoring other details that were present
                        st.caption(f"Full ID: {file_data.get('id', 'N/A')}")
                        st.caption(f"Created: {file_data.get('createdAt', 'N/A')}")
                        st.caption(f"Updated: {file_data.get('updatedAt', 'N/A')}")
                st.divider()
        else:
            st.info("No documents found for this agent yet.")

else:
    st.info("👈 Select an agent from the sidebar to begin.")

# Add some vertical space at the bottom
from streamlit_extras.add_vertical_space import add_vertical_space

add_vertical_space(3)
