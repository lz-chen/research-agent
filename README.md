# Research Agent

This repository contains a research agent application designed to facilitate paper research and slide generation. It consists of a backend powered by FastAPI and a frontend built with Streamlit.

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

## Development

- **Backend**: Located in the `backend` directory, using FastAPI.
- **Frontend**: Located in the `frontend` directory, using Streamlit.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
