import os
import logging
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings
from langchain_core.tools import tool
import requests
# Set up logging for better visibility
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Set your Gemini API key from the environment variable (or hardcode for testing if you must)
# It's better to set this as an environment variable once, outside of the script
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"

# IMPORTANT: Configure the LLM and Embedding Model
# This is the new part that fixes your error
Settings.llm = Gemini(model="gemini-2.5-flash")
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")

def get_rag_index():
    try:
        # Load the index from the 'storage' directory
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        print("RAG index loaded from disk.")
    except Exception:
        # If no index is found, build a new one from the 'data' directory
        print("No RAG index found. Building a new one...")
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir="./storage")
        print("RAG index built and saved to disk.")
    return index

# Get the RAG index
rag_index = get_rag_index()

@tool
def get_product_details(id: str) -> str:
    """
    Retrieves a user's information from the company's internal API.
    Input should be the user's ID.
    """
    print(f"Executing API call to get product details for ID: {id}")
    
    # Construct the API endpoint URL with the product ID
    api_url = f"http://54.238.204.246:8080/api/products/{id}"  # Updated to match the correct API endpoint
    
    try:
        # Make the GET request to the API
        response = requests.get(api_url)
        response.raise_for_status() # This will raise an exception for bad status codes (4xx or 5xx)
        
        # Get the JSON data from the response
        product_data = response.json()
        
        # Return a formatted string with the product details
        # Based on the actual API response structure
        return (
            f"Product found! Details:\n"
            f"Product ID: {product_data.get('productId')}\n"
            f"Product Name: {product_data.get('productName')}\n"
            f"Version: {product_data.get('version')}\n"
            f"Description: {product_data.get('productDescription')}\n"
            f"Status: {product_data.get('status')}\n"
            f"Product Type: {product_data.get('productType')}\n"
            f"Internal SKU Code: {product_data.get('internalSkuCode')}\n"
            f"Created On: {product_data.get('createdOn')}\n"
            f"Last Updated: {product_data.get('lastUpdated')}"
        )
    except requests.exceptions.HTTPError as errh:
        return f"HTTP Error: {errh} - The API returned an error for product ID {id}. The product may not exist."
    except requests.exceptions.RequestException as err:
        return f"An error occurred while making the API request: {err}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# tools.py (updated section)

...
@tool
def search_company_docs(query: str) -> str:
    """
    Searches the internal company documents (PDFs) for information.
    Use this tool for questions about company policies, products, or guidelines.
    """
    print(f"Executing RAG search for query: {query}")
    # Use as_query_engine to create the query engine
    query_engine = rag_index.as_query_engine()

    # The query engine will now perform the RAG process:
    # 1. Retrieve the most relevant chunks from the PDF.
    # 2. Pass those chunks and the user's query to the LLM to synthesize a response.
    try:
        response = query_engine.query(query)
        # If the response indicates no info found, we'll return a more explicit message.
        # This part depends on the LLM's output format.
        if "not contain details" in str(response).lower() or "not have information" in str(response).lower():
            return "I couldn't find any information on that specific topic in our documents. Would you like me to try another query or is there something else I can help with?"
        return str(response)
    except Exception as e:
        return f"An error occurred while searching the documents: {e}"

# ... rest of the script remains the same

@tool
def get_current_time() -> str:
    """Returns the current time. Useful for time-related questions."""
    return "The current time is 10:00 AM."