from smolagents import LiteLLMModel, GradioUI
import os
from dotenv import load_dotenv
import sys

# Print Python version and path for debugging
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

# Try to import smolagents and print version
try:
    import smolagents
    print(f"smolagents version: {smolagents.__version__ if hasattr(smolagents, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"Error importing smolagents: {e}")

# Import from our modular structure
from utils.file_utils import load_openapi_docs
from agents.endpoint_agents import create_endpoint_retriever_agent, create_manager_agent, create_request_executor_agent

# Load environment variables
load_dotenv()

# TODO: HAVE ENVIRONMENT VARIABLES FOR BASE_URLS OR MAYBE ADD THEM IN docs_path_to_endpoint_summary

# Load OpenAPI documents
openapi_docs_path = "openapi_docs"
openapi_docs = load_openapi_docs(openapi_docs_path)

# Define document summaries
docs_path_to_endpoint_summary = {
    "webslayer.json": """The WebSlayer API facilitates web scraping and data extraction. Define data schemas, 
hen launch scraping jobs using LLMs (Ollama, Claude, OpenAI) against URLs based on those schemas. Monitor job status and 
manage the resulting reports: list, retrieve, delete, or download the structured data as JSON files."""
}

# Initialize the model
model = LiteLLMModel(
    model_id="gemini/gemini-2.0-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

# Create the endpoint retriever agent
endpoint_retriever_agent = create_endpoint_retriever_agent(model)

# Create request executor agent
request_executor_agent = create_request_executor_agent(model)

# Create the manager agent
manager_agent = create_manager_agent(model, endpoint_retriever_agent, request_executor_agent, docs_path_to_endpoint_summary)

# Launch the UI
# TODO: Investigate if it can be customized
GradioUI(manager_agent).launch(server_name="0.0.0.0")