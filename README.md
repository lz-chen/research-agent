# Research Agent

This repository contains a research agent application designed to
facilitate paper research and slide generation. It consists of a
backend powered by FastAPI and a frontend built with Streamlit.

**Note:**
the frontend part of this project is still under development.

## Project Structure

```
├── backend                     # Backend code using FastAPI
│   ├── prompts                 # Prompts used in the workflow 
│   ├── services                # Services for LLMs and embeddings
│   ├── utils                   # Utility functions
│   ├── workflows               # Workflow difinitions
│   ├── config.py               # Configuration settings
│   ├── models.py               # Pydantic models
│   ├── main.py                 # Main entry point for FastAPI
│   ├── Dockerfile              # Dockerfile for backend
│   ├── pyproject.toml          # Backend dependencies
│   └── __init__.py             # Package initialization
├── frontend                    # Frontend code using Streamlit
│   ├── pages                   # Streamlit pages
│   ├── Dockerfile              # Dockerfile for frontend
│   ├── pyproject.toml          # Frontend dependencies
└── └── app.py                  # Main entry point for Streamlit
```


## Prerequisites

- Python >= 3.12
- Poetry
- Docker
- Docker Compose


## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   # install dependencies
   poetry install
   ```

2. Set up the environment variables by creating a `.env` file in the root directory,
    and add the following environment variables as those listed in `.env.example`

## Usage

To run the research agent workflow, from root directory, run the following command:

```bash
poetry run python backend/workflows/summarize_and_generate_slides.py -q <YOUR_INPUT_QUERY>
```

