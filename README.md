# Multi-Agent API Interaction System

This application uses a multi-agent system to interact with APIs through their OpenAPI specifications. It leverages LLMs to understand API endpoints and execute requests.

## Features

- Endpoint retrieval agent to find the right API endpoints
- Request executor agent to make API calls
- Manager agent to coordinate the workflow
- Gradio UI for easy interaction

## Prerequisites

- Docker and Docker Compose installed on your system
- Gemini API key for the main LLM agent
- OpenAI API key for embeddings and vector search

## Setup

1. Clone this repository
2. Create a `.env` file based on the `.env.example` template:

```bash
cp .env.example .env
```
3. Edit the `.env` file and add your API keys:
4. Start the application using Docker:

```bash
docker-compose up --build
```

This will:
- Build the Docker image
- Start the container
- Map port 7860 to your host machine
- Mount the necessary volumes
- Set environment variables from your .env file

Access the Gradio UI at: http://localhost:7860

## Running without Docker

If you prefer to run the application without Docker:

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r new_requirements.txt
```

3. Run the application:

```bash
python main.py
```

## Project Structure

- `main.py`: Entry point of the application
- `agents/`: Contains the agent implementations
- `utils/`: Utility functions
- `openapi_docs/`: OpenAPI specifications for APIs
- `db_chroma_cache/`: Cache for vector embeddings

## Customization

To add new API specifications, place the OpenAPI JSON files in the `openapi_docs/` directory and update the `docs_path_to_endpoint_summary` dictionary in `main.py`.