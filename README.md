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

## Features

- **Slide Generation**: Generate slides from research papers using a workflow agent.
- **Streaming Data**: Real-time updates from the backend to the frontend.
- **Azure CLI Integration**: Utilize Azure services within the backend.
- **Document Conversion**: Convert documents using `unoconv` and LibreOffice.

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
   ```

2. **Build and run the Docker containers**:
   ```bash
   docker-compose up --build
   ```

3. **Access the application**:
   - Frontend: Open your browser and go to `http://localhost:8501`
   - Backend: API documentation available at `http://localhost:8000/docs`

## Usage

- **Slide Generation**: Navigate to the "Slide Generation" page in the Streamlit app, enter the directory path for slide generation, and submit the form to start the process.

## Frontend

- **Streamlit Application**: The frontend is built using Streamlit, providing an interactive interface for users to generate slides and view research summaries.
- **Pages**: Located in the `frontend/pages` directory, each page corresponds to a different functionality of the application.
- **Assets**: Static files such as images and stylesheets are stored in the `frontend/assets` directory.
