# RAG Chat System Prototype

This project implements a Retrieval-Augmented Generation (RAG) chat system prototype for document-based question answering.

## Overview

The system allows users to chat with an AI agent that can answer questions based on knowledge from uploaded CSV documents. It features:

- A FastAPI backend providing APIs for agent management, file ingestion, and chat.
- A data ingestion pipeline that processes CSV files, chunks text, generates embeddings (using OpenAI), and stores them in a ChromaDB vector store.
- A hybrid retrieval system combining semantic search (ChromaDB) and keyword search (BM25), with results fused using Reciprocal Rank Fusion (RRF).
- An LLM (OpenAI's GPT-4o by default) for generating responses based on retrieved context, with support for citing sources.
- A Streamlit user interface for interacting with the system: selecting agents, uploading files, chatting, and rating messages.
- Prisma ORM for managing metadata in an SQLite database.

## Project Structure

- `app/`: Contains the main application code.
  - `api/`: FastAPI routers and API logic.
  - `core/`: Configuration (e.g., API keys, model names).
  - `services/`: Business logic for ingestion, retrieval, generation, file management.
  - `ui/`: Streamlit application code (`chat_app.py`).
  - `main.py`: FastAPI application entry point.
- `data/`: For SQLite database, ChromaDB persistence, and uploaded files.
  - `uploads/`: Storage for uploaded CSVs.
  - `chroma_db/`: ChromaDB persistent storage.
  - `langchain_com_pages_4.csv`: Sample CSV containing LangChain documentation for seeding.
- `docs/`: Project documentation (PRD, task list).
- `prisma/`: Prisma schema and migration files.
- `scripts/`: Utility scripts (e.g., `seed.py`).
- `Makefile`: For common development tasks.
- `pyproject.toml`, `uv.lock`: Python project and dependency management.
- `package.json`, `package-lock.json`: For Node.js dependencies (Prisma CLI).
- `.env`: (User-created) For environment variables like API keys.

## Setup Instructions

1. **Prerequisites:**
    - Python 3.11 (as specified in `.python-version`). Consider using `pyenv` to manage Python versions.
    - `uv` (Python packaging tool): [Installation Guide](https://github.com/astral-sh/uv#installation)
    - Node.js and npm (for Prisma CLI): [Download Node.js](https://nodejs.org/)

2. **Clone the Repository:**

    ```bash
    git clone <repository_url>
    cd ragit
    ```

3. **Install Node.js Dependencies (for Prisma CLI):**

    ```bash
    npm install
    ```

4. **Create and Configure Environment File:**
    - Create a `.env` file in the project root: `cp .env.example .env` (if an example file is provided, otherwise create it manually).
    - Add your OpenAI API key to the `.env` file:

        ```env
        OPENAI_API_KEY="your_openai_api_key_here"
        ```

5. **Install Python Dependencies and Setup Database:**
    - Use the Makefile target (recommended):

        ```bash
        make setup
        ```

    - This command will:
        - Install Python dependencies using `uv sync`.
        - Generate the Prisma client.
        - Push the Prisma schema to the SQLite database (creating `data/dev.db`).

## Running the Application

### 1. Seed the Database (Optional but Recommended for Initial Setup)

   To populate the system with a default agent and sample data from `data/langchain_com_pages_4.csv`:

   ```bash
   make seed
   ```

   Ensure your `OPENAI_API_KEY` is set in `.env` before running this, as it performs embedding generation.

### 2. Run the FastAPI Backend

   ```bash
   make run
   ```

   Or directly:

   ```bash
   uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

   The API will be available at `http://localhost:8000`. You can access the OpenAPI docs at `http://localhost:8000/docs`.

### 3. Run the Streamlit UI

   ```bash
   make ui
   ```

   Or directly:

   ```bash
   uv run streamlit run app/ui/chat_app.py
   ```

   The Streamlit application will typically open in your browser, or you can access it at `http://localhost:8501` (or the port Streamlit indicates).

## Makefile Targets

- `make setup`: Installs all dependencies, generates Prisma client, and sets up the database.
- `make run`: Starts the FastAPI backend server.
- `make ui`: Starts the Streamlit UI application.
- `make test`: Runs pytest tests (tests need to be written).
- `make lint`: Runs ruff linter and formatter.
- `make seed`: Seeds the database with a default agent and sample file data.
- `make prisma-generate`: Regenerates the Prisma client.
- `make prisma-migrate-dev`: Creates a new database migration (for schema changes in development).
- `make prisma-db-push`: Pushes the current Prisma schema to the database without creating a migration file (useful for development with SQLite).
- `make prisma-studio`: Opens Prisma Studio in the browser to view/manage database content.
- `make help`: Displays a list of all available targets.

## Further Development

- Implement robust testing (unit, integration).
- Enhance citation parsing and display.
- Optimize BM25 indexing for larger datasets.
- Implement deletion of vector embeddings when a file is deleted.
- Add more comprehensive error handling and user feedback in the UI.
