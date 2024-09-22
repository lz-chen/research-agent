# Research Agent

This repository contains a research agent application designed to
facilitate paper research and slide generation. It consists of a
backend powered by FastAPI and a frontend built with Streamlit.

![ui-recording.gif](docs/assets/ui-recording.gif)./assets/research-agent.png)

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
│   └── pyproject.toml          # Backend dependencies
│  
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

2. **Set up environment variables**
    
    Create a `.env` file in the root directory,
    and add the following environment variables as those listed in `.env.example`


3. **Build and run the Docker containers**:
   ```bash
   docker-compose up --build
   ```

4. **Access the application**:
   - Frontend: Open your browser and go to `http://localhost:8501`
   - Backend: API documentation available at `http://localhost:8000/docs`
## Usage

- **Summary and Slide Generation**: Navigate to the "Slide Generation" page in the Streamlit app, enter 
the research topic query, and click the submit button to start the process.


