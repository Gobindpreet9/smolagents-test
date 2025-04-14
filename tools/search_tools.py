import json
import os
import ast
from smolagents import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models.schemas import EndpointSchema
from utils.file_utils import get_file_hash, load_metadata, save_metadata

#TODO: Create a tool to ctrl+f into the open_api file fully and return +-30 lines

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
    if not os.path.exists(source_file_path):
        return f"Error: File '{filename}' not found in 'openapi_docs'."

    try:
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
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
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
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
        #todo: increase the search content size
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
    Run this when you found the endpoints before returning them to validate. Parses a string containing either a JSON object
    or a Python dictionary literal and validates it against the EndpointSchema.

    Args:
        endpoints: A string representation of a JSON object or Python dictionary expected to match EndpointSchema.

    Returns:
        model dump of validated endpoints
    """
    try:
        try:
            parsed_dict = json.loads(endpoints)
        except json.JSONDecodeError:
            parsed_dict = ast.literal_eval(endpoints)

        if not isinstance(parsed_dict, dict):
            raise ValueError(f"Input string did not evaluate to a dictionary. Expected to match EndpointSchema: {json.dumps(EndpointSchema.model_json_schema(), indent=2)}")

        # --- VALIDATION STEP ---
        validated_schema = EndpointSchema(**parsed_dict)

        return validated_schema.model_dump()
    except Exception as e:
        return f"Expected dictionary to match EndpointSchema: {json.dumps(EndpointSchema.model_json_schema(), indent=2)}. Full stacktrace: {str(e)} "