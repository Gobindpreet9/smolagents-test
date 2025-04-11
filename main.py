import re
from smolagents import (
ToolCallingAgent,
CodeAgent,
LiteLLMModel,
tool
)
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()


@tool
def semantic_openapi_search(filename: str, query: str) -> str:
    """
    Performs semantic search on a specified OpenAPI document within the 'openapi_docs' directory.

    Args:
        filename: The name of the OpenAPI file (e.g., 'webslayer.json') located in the 'openapi_docs' directory.
        query: The natural language query to search for within the document.

    Returns:
        A string containing the most relevant parts of the document based on the query,
        or an error message if the file is not found or processing fails.
    """
    file_path = os.path.join("openapi_docs", filename)
    if not os.path.exists(file_path):
        return f"Error: File '{filename}' not found in 'openapi_docs'."

    try:
        # 1. Load the document
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()

        # 2. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # 3. Create embeddings
        # Using a smaller, faster model suitable for CPU inference
        model_name = "sentence-transformers/all-MiniLM-L6-v2" 
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'}) # Explicitly use CPU

        # 4. Create FAISS vector store
        db = FAISS.from_documents(texts, embeddings)

        # 5. Perform similarity search
        results = db.similarity_search(query, k=3) # Get top 3 results

        # 6. Format and return results
        return "\n\n".join([doc.page_content for doc in results])

    except Exception as e:
        return f"Error processing file '{filename}': {e}"

@tool
def ask_user_for_clarification(question: str) -> str:
    """
    Asks the user a question to clarify ambiguity or gather more information.
    Use this when the next step is unclear, multiple options exist (e.g., multiple API endpoints),
    or required information is missing.

    Args:
        question: The question to ask the user.

    Returns:
        The user's response as a string.
    """
    print(f"\n{question}")
    user_response = input("Your response: ")
    return user_response

@tool
def return_api_endpoints(query: str, openapi_context: str) -> list[dict[str, str]]:
    """
    Selects the most relevant API endpoint(s) based on a user query and OpenAPI context.

    Args:
        query: The original user query.
        openapi_context: Relevant snippets from the OpenAPI documentation (obtained via semantic_openapi_search).

    Returns:
        A list of dictionaries, where each dictionary represents a selected endpoint
        with its URL and HTTP method, e.g., [{'url': '/webslayer/schema/', 'method': 'GET'}].
        Returns an empty list if no suitable endpoint is found or an error occurs.
    """
    prompt = f"""
    Based on the user query: "{query}"
    And the following relevant OpenAPI documentation context:
    ---
    {openapi_context}
    ---
    Identify the most appropriate API endpoint URL and its corresponding HTTP method (e.g., GET, POST, PUT, DELETE) to fulfill the user's request.
    Provide ONLY a JSON list containing one dictionary with 'url' and 'method' keys for the best matching endpoint.
    Example format: [{'url': '/path/to/endpoint', 'method': 'POST'}]
    If no single suitable endpoint can be clearly identified from the context, return an empty JSON list [].
    """
    try:
        # Use the globally defined model to analyze the context and select the endpoint
        response = model.generate([{"role": "user", "content": prompt}])

        # Extract the JSON part from the response (LLMs might add explanations)
        # Use regex to find the list structure [...]
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
        if json_match:
            endpoints_str = json_match.group(0)
            endpoints = json.loads(endpoints_str)
            # Basic validation: ensure it's a list of dicts with 'url' and 'method'
            if isinstance(endpoints, list) and all(isinstance(item, dict) and 'url' in item and 'method' in item for item in endpoints):
                 # Ensure method is uppercase as per HTTP standards
                for item in endpoints:
                    item['method'] = item.get('method', '').upper()
                print(f"Selected endpoints: {endpoints}") # Debug print
                return endpoints
            else:
                print(f"Warning: LLM response for endpoint selection was not in the expected list-of-dicts format: {endpoints_str}")
                return []
        else:
            print(f"Warning: Could not extract JSON endpoint list from LLM response: {response}")
            return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from LLM for endpoint selection: {e}\nResponse was: {response}")
        return []
    except Exception as e:
        print(f"Error during endpoint selection: {e}")
        return []


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
                     api_key="")
                     
endpoint_retreiever_agent = ToolCallingAgent(
    tools=[semantic_openapi_search, return_api_endpoints],
    model=model,
    max_steps=10,
    name="endpoint_retreiever_agent",
    description="Gets appropriate endpoint to make the request. Provide it user query and the file to be searched.",
)

manager_agent = CodeAgent(
    tools=[ask_user_for_clarification],
    model=model,
    managed_agents=[endpoint_retreiever_agent],
)

answer = manager_agent.run(
    f"""User query: What endpoint should I use to get status of running job?

    Context:
    Help user find the endpoiint they need based on user query. Endpoints are described by files as below, 
    you will see the json files with their description:
    {docs_path_to_endpoint_summary}
    
    Based on this information identify which file is likely to contain the endpoint needed by user. Once you know the file, use endpoint_retreiever_agent
    it will help retrieve the specific endpoint details required. Once identified, provide the endpoint details to the user.
    """
)

print(answer)