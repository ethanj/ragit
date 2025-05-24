.PHONY: setup run test lint help seed ui

# ====================================================================================
# HELP
# ====================================================================================

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  setup          Install dependencies, generate Prisma client, and apply database migrations."
	@echo "  run            Run the FastAPI application."
	@echo "  ui             Run the Streamlit UI application."
	@echo "  test           Run pytest tests."
	@echo "  lint           Run ruff linter and formatter."
	@echo "  seed           Seed the database with initial data."
	@echo "  prisma-studio  Open Prisma Studio."

# ====================================================================================
# DEVELOPMENT
# ====================================================================================

setup:
	@echo "📦 Installing dependencies..."
	uv sync
	@echo "✨ Generating Prisma client..."
	uv run prisma generate
	@echo "💾 Applying database migrations..."
	uv run prisma db push
	@echo "✅ Setup complete."

run: ## Run the FastAPI application
	@echo "🚀 Starting FastAPI application..."
	uv run python app/main.py

test: ## Run pytest tests
	@echo "🧪 Running tests..."
	uv run pytest

lint: ## Run ruff linter and formatter
	@echo "🎨 Linting Python files..."
	uv run ruff check .
	uv run ruff format .

seed: ## Seed the database with initial data
	@echo "🌱 Seeding database..."
	uv run python scripts/seed.py

# ====================================================================================
# PRISMA
# ====================================================================================

prisma-generate: ## Generate Prisma client
	@echo "✨ Generating Prisma client..."
	uv run prisma generate

prisma-migrate-dev: ## Create a new migration (development)
	@echo "💾 Creating new migration... (dev)"
	uv run prisma migrate dev

prisma-db-push: ## Push schema changes to the database (development - no migrations)
	@echo "💾 Pushing schema to DB... (dev - no migrations)"
	uv run prisma db push

prisma-studio: ## Open Prisma Studio
	@echo "🎨 Opening Prisma Studio..."
	uv run prisma studio

ui: ## Run the Streamlit UI application
	@echo "🎨 Starting Streamlit UI application..."
	uv run streamlit run app/ui/chat_app.py
