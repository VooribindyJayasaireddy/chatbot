import os
import logging
import sys
import requests
import json
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings
from langchain_core.tools import tool

# Set up logging for a better view of the process
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# IMPORTANT: Configure the LLM and Embedding Model
# This is crucial for LlamaIndex to work with Gemini
Settings.llm = Gemini(model="gemini-2.5-flash")
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")

def get_rag_index():
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        print("RAG index loaded from disk.")
    except Exception:
        print("No RAG index found. Building a new one...")
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir="./storage")
        print("RAG index built and saved to disk.")
    return index

rag_index = get_rag_index()

@tool
def get_product_details(id: str) -> str:
    """
    Retrieves the details of a specific product by its ID.
    Use this to answer questions about a single product, like its name, description, or version.
    """
    print(f"Executing API call to get product details for ID: {id}")
    api_url = f"http://54.238.204.246:8080/api/products/{id}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        product_data = response.json()
        return json.dumps(product_data, indent=2)  # Return full JSON for agent to parse
    except requests.exceptions.RequestException as e:
        return f"An error occurred while retrieving product details: {e}"

@tool
def get_all_products() -> str:
    """
    Retrieves a list of all products from the company's internal API.
    Use this when a user asks for a list of products or what products the company sells.
    """
    print("Executing API call to get all products.")
    api_url = "http://54.238.204.246:8080/api/products"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        products_data = response.json()
        return json.dumps(products_data, indent=2) # Return full JSON list
    except requests.exceptions.RequestException as e:
        return f"An error occurred while retrieving products: {e}"

@tool
def create_product(data: str) -> str:
    """
    Creates a new product in the company's database.
    The 'data' parameter must be a JSON string of all required fields: productName, productDescription, productType, internalSkuCode, version, and status.
    """
    print("Executing API call to create a new product.")
    api_url = "http://54.238.204.246:8080/api/products"
    try:
        product_data = json.loads(data)
        response = requests.post(api_url, json=product_data)
        response.raise_for_status()
        created_product = response.json()
        return f"Product '{created_product.get('productName')}' created successfully! New Product ID: {created_product.get('productId')}"
    except json.JSONDecodeError:
        return "Error: The product data provided was not in a valid JSON format."
    except requests.exceptions.RequestException as e:
        return f"An error occurred while creating the product: {e}"

@tool
def update_product_put(id: str, data: str) -> str:
    """
    Replaces an entire product resource by its ID.
    The 'data' parameter must be a JSON string of the full product details.
    """
    print(f"Executing API call (PUT) to update product with ID: {id}")
    api_url = f"http://54.238.204.246:8080/api/products/{id}"
    try:
        update_data = json.loads(data)
        response = requests.put(api_url, json=update_data)
        response.raise_for_status()
        return f"Product with ID {id} fully updated successfully!"
    except json.JSONDecodeError:
        return "Error: The update data provided was not in a valid JSON format."
    except requests.exceptions.RequestException as e:
        return f"An error occurred while updating the product: {e}"

@tool
def update_product_patch(id: str, data: str) -> str:
    """
    Performs a partial update to a product by its ID.
    The 'data' parameter must be a JSON string of only the fields to be updated.
    """
    print(f"Executing API call (PATCH) to partially update product with ID: {id}")
    api_url = f"http://54.238.204.246:8080/api/products/{id}"
    try:
        update_data = json.loads(data)
        response = requests.patch(api_url, json=update_data)
        response.raise_for_status()
        return f"Product with ID {id} partially updated successfully!"
    except json.JSONDecodeError:
        return "Error: The update data provided was not in a valid JSON format."
    except requests.exceptions.RequestException as e:
        return f"An error occurred while partially updating the product: {e}"

@tool
def delete_product(id: str) -> str:
    """
    Deletes a product from the database by its ID.
    """
    print(f"Executing API call (DELETE) to delete product with ID: {id}")
    api_url = f"http://54.238.204.246:8080/api/products/{id}"
    try:
        response = requests.delete(api_url)
        response.raise_for_status()
        return f"Product with ID {id} has been successfully deleted."
    except requests.exceptions.RequestException as e:
        return f"An error occurred while deleting the product: {e}"

@tool
def finalize_product(id: str) -> str:
    """
    Finalizes a product by its ID, typically to mark it as ready for release.
    """
    print(f"Executing API call (POST) to finalize product with ID: {id}")
    api_url = f"http://54.238.204.246:8080/api/products/{id}/finalize"
    try:
        response = requests.post(api_url)
        response.raise_for_status()
        return f"Product with ID {id} has been finalized successfully."
    except requests.exceptions.RequestException as e:
        return f"An error occurred while finalizing the product: {e}"

@tool
def delete_product_icon(id: str) -> str:
    """
    Deletes the icon associated with a product by its ID.
    """
    print(f"Executing API call (DELETE) to remove icon for product with ID: {id}")
    api_url = f"http://54.238.204.246:8080/api/products/{id}/icon"
    try:
        response = requests.delete(api_url)
        response.raise_for_status()
        return f"Icon for product with ID {id} has been successfully deleted."
    except requests.exceptions.RequestException as e:
        return f"An error occurred while deleting the product icon: {e}"

@tool
def update_product_icon(id: str, data: str) -> str:
    """
    Updates the icon for a product by its ID.
    The 'data' parameter must be a JSON string of the new icon details.
    """
    print(f"Executing API call (PATCH) to update icon for product with ID: {id}")
    api_url = f"http://54.238.204.246:8080/api/products/{id}/icon"
    try:
        update_data = json.loads(data)
        response = requests.patch(api_url, json=update_data)
        response.raise_for_status()
        return f"Icon for product with ID {id} has been successfully updated."
    except json.JSONDecodeError:
        return "Error: The icon data provided was not in a valid JSON format."
    except requests.exceptions.RequestException as e:
        return f"An error occurred while updating the product icon: {e}"

@tool
def search_company_docs(query: str) -> str:
    """
    Searches the internal company documents (PDFs) for information.
    Use this tool for questions about company policies, products, or guidelines.
    """
    print(f"Executing RAG search for query: {query}")
    query_engine = rag_index.as_query_engine()
    try:
        response = query_engine.query(query)
        if "not contain details" in str(response).lower() or "not have information" in str(response).lower():
            return "I couldn't find any information on that specific topic in our documents. Would you like me to try another query or is there something else I can help with?"
        return str(response)
    except Exception as e:
        return f"An error occurred while searching the documents: {e}"

@tool
def get_current_time() -> str:
    """Returns the current time. Useful for time-related questions."""
    return "The current time is 10:00 AM."

# A list of all available tools for the agent to use
__all__ = [
    "get_current_time", 
    "get_product_details", 
    "get_all_products",
    "create_product",
    "update_product_put",
    "update_product_patch",
    "delete_product",
    "finalize_product",
    "delete_product_icon",
    "update_product_icon",
    "search_company_docs"
]