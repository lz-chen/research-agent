# Research Agent

This repository contains a research agent application designed to facilitate paper research and slide generation. It consists of a backend powered by FastAPI and a frontend built with Streamlit.

## Project Structure

```
├── backend                     # Backend code using FastAPI
│   ├── app                     # FastAPI application files
│   ├── config.py               # Configuration settings
│   ├── models.py               # Pydantic models
│   ├── services                # Services for LLMs and embeddings
│   ├── utils                   # Utility functions
│   ├── workflows               # Workflow scripts
│   ├── Dockerfile              # Dockerfile for backend
│   ├── pyproject.toml          # Backend dependencies
│   ├── main.py                 # Main entry point for FastAPI
│   ├── prompts                 # Prompt templates
│   └── __init__.py             # Package initialization
├── frontend                    # Frontend code using Streamlit
│   ├── pages                   # Streamlit pages
│   ├── Dockerfile              # Dockerfile for frontend
│   ├── pyproject.toml          # Frontend dependencies
│   └── app.py                  # Main entry point for Streamlit
├── docker                      # Docker-related files
└── config                      # Configuration files
├── backend                     # Backend code using FastAPI
│   ├── app                     # FastAPI application files
│   ├── config.py               # Configuration settings
│   ├── models.py               # Pydantic models
│   ├── services                # Services for LLMs and embeddings
│   ├── utils                   # Utility functions
│   ├── workflows               # Workflow scripts
│   ├── Dockerfile              # Dockerfile for backend
│   ├── pyproject.toml          # Backend dependencies
│   └── main.py                 # Main entry point for FastAPI
├── frontend                    # Frontend code using Streamlit
│   ├── pages                   # Streamlit pages
│   ├── Dockerfile              # Dockerfile for frontend
│   ├── pyproject.toml          # Frontend dependencies
│   └── app.py                  # Main entry point for Streamlit
├── docker                      # Docker-related files
└── config                      # Configuration files
├── backend                     # Backend code using FastAPI
│   ├── app                     # FastAPI application files
│   └── tests                   # Tests for the backend
├── frontend                    # Frontend code using Streamlit
│   ├── pages                   # Streamlit pages
│   └── assets                  # Static assets for the frontend
├── backend                     # Backend code using FastAPI
│   ├── app                     # FastAPI application files
│   └── tests                   # Tests for the backend
├── frontend                    # Frontend code using Streamlit
│   ├── pages                   # Streamlit pages
│   └── assets                  # Static assets for the frontend
├── docker                      # Docker-related files
└── config                      # Configuration files
```

## Features

- **Slide Generation**: Generate slides from research papers using a workflow agent.
- **Streaming Data**: Real-time updates from the backend to the frontend.
- **Azure CLI Integration**: Utilize Azure services within the backend.
- **Document Conversion**: Convert documents using `unoconv` and LibreOffice.

## Prerequisites

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
- **Feedback**: Provide feedback on the generated slide outlines directly in the app.

## Frontend

- **Streamlit Application**: The frontend is built using Streamlit, providing an interactive interface for users to generate slides and view research summaries.
- **Pages**: Located in the `frontend/pages` directory, each page corresponds to a different functionality of the application.
- **Assets**: Static files such as images and stylesheets are stored in the `frontend/assets` directory.

## Development

- **Backend**: Located in the `backend` directory, using FastAPI.
- **Frontend**: Located in the `frontend` directory, using Streamlit.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Project Structure

```
├── backend                     # Backend code using FastAPI
│   ├── app                     # FastAPI application files
│   └── tests                   # Tests for the backend
├── frontend                    # Frontend code using Streamlit
│   ├── pages                   # Streamlit pages
│   └── assets                  # Static assets for the frontend
├── docker                      # Docker-related files
└── config                      # Configuration files
```

- **Slide Generation**: Generate slides from research papers using a workflow agent.
- **Streaming Data**: Real-time updates from the backend to the frontend.
- **Azure CLI Integration**: Utilize Azure services within the backend.
- **Document Conversion**: Convert documents using `unoconv` and LibreOffice.

## Prerequisites

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
- **Feedback**: Provide feedback on the generated slide outlines directly in the app.

## Development

- **Backend**: Located in the `backend` directory, using FastAPI.
- **Frontend**: Located in the `frontend` directory, using Streamlit.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
# Project Name

## Description

A brief description of the project, its purpose, and what it does.

## Project Structure

- **src/**: Contains the source code for the project.
- **tests/**: Contains test cases for the project.
- **docker/**: Docker-related files for containerization.
- **config/**: Configuration files for the project.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the environment:
   ```bash
   cp .env.example .env
   ```

## Usage

Instructions on how to use the project, including any commands or scripts to run.

## Docker

To run the project using Docker:

1. Build the Docker image:
   ```bash
   docker-compose build
   ```

2. Run the Docker container:
   ```bash
   docker-compose up
   ```

## Contributing

Guidelines for contributing to the project.

## License

Information about the project's license.
