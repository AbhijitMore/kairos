# KAIROS – Developer Onboarding

Welcome to the **KAIROS** project. This guide will help you set up your local development environment and understand our core engineering workflows.

---

## 1. Prerequisites

Ensure you have the following installed on your host machine:

- **Docker & Docker Compose** (version 24.0.0 or higher recommended)
- **Python 3.10+** (if running backend services outside Docker)
- **Node.js 18+ & npm** (if running frontend outside Docker)
- **Git**

---

## 2. Quick Start (Containerized)

The fastest way to get KAIROS running is using Docker Compose. This spins up the FastAPI backend, the React/JS frontend, and any necessary persistent storage.

```bash
# Clone the repository
git clone https://github.com/AbhijitMore/kairos.git
cd kairos

# Launch the full stack
docker compose up --build
```

- **Frontend:** http://localhost:5000
- **Backend API:** http://localhost:8000
- **API Docs (Swagger):** http://localhost:8000/docs

---

## 3. Local Development Workflow

### Backend (Python)

If you are developing features for the ML pipeline or the API, you can run the backend locally with hot-reload:

```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API with hot-reload
export PYTHONPATH=$PYTHONPATH:.
uvicorn app.main:app --reload --port 8000
```

### Frontend (React/Vanilla JS)

For UI changes, navigate to the frontend directory:

```bash
cd frontend
npm install
npm run dev
```

---

## 4. Engineering Standards

### Testing

We enforce high test coverage for all logic changes.

- **Backend:** `pytest tests/`
- **Coverage Reports:** `pytest --cov=src --cov=app tests/`

### Linting & Formatting

We use `ruff` and `black` for Python, and `eslint` for JavaScript.

- **Run Python Linting:** `ruff check .`
- **Run Python Formatting:** `black .`

### Pre-commit Hooks

Please install the pre-commit hooks to ensure code meets standards before every commit:

```bash
pip install pre-commit
pre-commit install
```

---

## 5. Branching & Contributions

1. **Fork** the repository and create your branch from `main`.
2. **Branch Naming:** `feature/description` or `fix/description`.
3. **Draft PR:** Open a Draft PR early to get feedback on the design.
4. **Commit Messages:** Follow [Conventional Commits](https://www.conventionalcommits.org/).

---

## 6. Resources

- [Architecture Overview](./ARCHITECTURE.md)
- [Research & Methodology](./RESEARCH.md)
- [API Specification](./openapi.yaml)
