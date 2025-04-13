from smolagents import (
ToolCallingAgent,
CodeAgent,
LiteLLMModel,
GradioUI,
UserInputTool,
tool
)
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import Literal
import ast
import hashlib
import json

class EndpointSchema(BaseModel):
    url: str = Field(
        description="The URL endpoint path",
        pattern="^/.*",  # Must start with /
        example="/webslayer/schema/{schema_name}"
    )
    type: Literal["GET", "POST", "PUT", "DELETE"] = Field(
        description="The HTTP method type",
        example="GET"
    )


load_dotenv()

def get_file_hash(file_path):
    """Calculates the SHA256 hash of a file's content."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as file:
            while chunk := file.read(8192): # Read in chunks
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error hashing file {file_path}: {e}")
        return None

def load_metadata(metadata_path):
    """Loads filehash metadata from a JSON file."""
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading metadata {metadata_path}: {e}")
        return None

def save_metadata(metadata_path, data):
    """Saves filehash metadata to a JSON file."""
    try:
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving metadata {metadata_path}: {e}")

@tool
def semantic_openapi_search(filename: str, query: str) -> str:
    """
    Performs semantic search on a specified OpenAPI document within the 'openapi_docs' directory. You may call this multiple times
     with different queries to refine results if the initial search does not yield satisfactory outcomes.

    Args:
        filename: The name of the OpenAPI file (e.g., 'webslayer.json') located in the 'openapi_docs' directory.
        query: The natural language query to search for within the document.

    Returns:
        A string containing the most relevant parts of the document based on the query,
        or an error message if the file is not found or processing fails.
    """
    source_file_path = os.path.join("openapi_docs", filename)
    persist_directory = os.path.join("db_chroma_cache", os.path.splitext(filename)[0]) # Unique dir per file
    metadata_path = os.path.join(persist_directory, "metadata.json")
    if not os.path.exists(file_path):
        return f"Error: File '{filename}' not found in 'openapi_docs'."

    try:
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            # Ensure API key is available, e.g., via environment variable
            # openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        vectorstore = None
        current_file_hash = get_file_hash(source_file_path)
        if not current_file_hash:
             return f"Error: Could not calculate hash for '{filename}'."

        # Check cache validity
        metadata = load_metadata(metadata_path)
        cache_is_valid = False
        if metadata and os.path.exists(persist_directory):
            stored_hash = metadata.get("source_file_hash")
            if stored_hash == current_file_hash:
                cache_is_valid = True
                print(f"Cache hit for '{filename}'. Loading existing vector store.")
            else:
                 print(f"Cache invalid for '{filename}'. Source file changed. Rebuilding...")
        else:
            print(f"Cache miss for '{filename}'. Building new vector store...")


        if cache_is_valid:
            # Load from existing persisted directory
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_model
            )
            print(f"Loaded Chroma vector store.")
        else:
            print(f"Processing and embedding '{filename}'...")
            # Load the document
            loader = TextLoader(source_file_path)
            documents = loader.load()

            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            # Create a Chroma vector store FROM the documents and persist
            vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=embedding_model,
                persist_directory=persist_directory # Saves the index
            )
            vectorstore.persist()

            print(f"Chroma vector store created and persisted with {vectorstore._collection.count()} documents.")

            # Save metadata including the new hash
            save_metadata(metadata_path, {"source_file_hash": current_file_hash})


        # Proceed with search using the loaded or newly created vectorstore
        if not vectorstore:
             return "Error: Vector store could not be loaded or created."

        # Create a retriever from the vector store
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 2}
        )

        # Perform search
        print(f"Performing search for query: '{query}'")
        docs = retriever.invoke(query)

        return "\nRetrieved documents:\n" + "".join(
                [
                    f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                    for i, doc in enumerate(docs)
                ]
            )

    except Exception as e:
        # TODO: Consider more specific error handling
        import traceback
        print(traceback.format_exc())
        return f"Error processing file '{filename}': {e}"

@tool
def validate_endpoint_format(endpoints: str) -> str:
    """
    Run this when you found the endpoints before returning them to validate. Parses a string containing a Python dictionary literal 
    and validates it against the EndpointSchema.

    Args:
        endpoints: A string representation of a Python dictionary expected to match EndpointSchema.

    Returns:
        model dump of validated endpoints
    """
    try: 
        # Safely parse the string dictionary
        parsed_dict = ast.literal_eval(endpoints)

        if not isinstance(parsed_dict, dict):
                raise ValueError(f"Input string did not evaluate to a Python dictionary. Expected to match EndpointSchema: {str(EndpointSchema)}")

        # --- VALIDATION STEP ---
        validated_schema = EndpointSchema(**parsed_dict)

        return validated_schema.model_dump()
    except Exception as e:
        return f"Expected dictionary to match EndpointSchema: {str(EndpointSchema)}. Full stacktrace: {str(e)} "


openapi_docs = {}
openapi_docs_path = "openapi_docs"
for file_name in os.listdir(openapi_docs_path):
    file_path = os.path.join(openapi_docs_path, file_name)
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            openapi_docs[file_name] = file.read()

docs_path_to_endpoint_summary = {"webslayer.json": """The WebSlayer API facilitates web scraping and data extraction. Define data schemas, 
then launch scraping jobs using LLMs (Ollama, Claude, OpenAI) against URLs based on those schemas. Monitor job status and 
manage the resulting reports: list, retrieve, delete, or download the structured data as JSON files."""}

model = LiteLLMModel(model_id="gemini/gemini-2.0-flash",
                     api_key=os.getenv("GEMINI_API_KEY"))
                     
endpoint_retreiever_agent = ToolCallingAgent(
    tools=[semantic_openapi_search, validate_endpoint_format],
    model=model,
    max_steps=10,
    name="endpoint_retreiever_agent",
    description="This is an agent that gets appropriate endpoint to make the request to. Provide it user query and the file to be searched as one parameter in natural language.",
)

endpoint_retreiever_agent.prompt_templates["managed_agent"]["task"] = endpoint_retreiever_agent.prompt_templates["managed_agent"]["task"] + ".\nvalidate_endpoint_format tool expects dictionary to adhere to EndpointSchema: " + EndpointSchema.schema_json(indent=2)
endpoint_retreiever_agent.prompt_templates["final_answer"]["pre_messages"] = endpoint_retreiever_agent.prompt_templates["final_answer"]["pre_messages"] + ".\nYour answer should adhere to EndpointSchema: " + EndpointSchema.schema_json(indent=2)

userInputTool = UserInputTool()

manager_agent = CodeAgent(
    tools=[userInputTool],
    model=model,
    managed_agents=[endpoint_retreiever_agent],
)

manager_agent.prompt_templates["system_prompt"] = manager_agent.prompt_templates["system_prompt"] + f"""
    Context:
    Help user find the endpoiint they need based on user query. Endpoints are described by files as below, 
    you will see the json files with their description:
    {docs_path_to_endpoint_summary}
    
    Based on this information identify which file is likely to contain the endpoint needed by user. Once you know the file, use endpoint_retreiever_agent
    it will help retrieve the specific endpoint details required. Once identified, provide the endpoint details to the user.
"""

GradioUI(manager_agent).launch(pwa=True)