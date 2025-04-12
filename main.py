from smolagents import (
ToolCallingAgent,
CodeAgent,
LiteLLMModel,
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
    file_path = os.path.join("openapi_docs", filename)
    if not os.path.exists(file_path):
        return f"Error: File '{filename}' not found in 'openapi_docs'."

    try:
        embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

        # Load the document
        loader = TextLoader(file_path)
        documents = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Create a Chroma vector store FROM the documents    
        persist_directory = 'db_chroma_similarity'
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embedding_model,
            persist_directory=persist_directory # Optional: Saves the index
        )
        print(f"Chroma vector store created with {vectorstore._collection.count()} documents.")

        # Create a retriever from the vector store
        retriever = vectorstore.as_retriever(
            search_type="similarity", # Explicitly stating, though it's often the default
            search_kwargs={'k': 2}     # Retrieve the top 2 most similar documents
        )

        # Perform search
        docs = retriever.invoke(query)

        return "\nRetrieved documents:\n" + "".join(
                [
                    f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                    for i, doc in enumerate(docs)
                ]
            )

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
    description="Gets appropriate endpoint to make the request. Provide it user query and the file to be searched.",
)

endpoint_retreiever_agent.prompt_templates["managed_agent"]["task"] = endpoint_retreiever_agent.prompt_templates["managed_agent"]["task"] + ".\nvalidate_endpoint_format tool expects dictionary to adhere to EndpointSchema: " + str(EndpointSchema)

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