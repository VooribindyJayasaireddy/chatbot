import os
import logging
import sys
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# Load environment variables from .env file (if it exists)
load_dotenv()

# Set up logging for a better view of the process.
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Set the Google API key from the environment variable.
# This is the recommended practice for security.
api_key = os.getenv("GOOGLE_API_KEY")

# If not found in environment, use the hardcoded key for testing
if not api_key:
    api_key = "use your own api key"
    print("Using hardcoded API key for testing. Please set GOOGLE_API_KEY environment variable for production.")

# --- The Key Changes ---

# Configure the LLM for your RAG system.
# The `Gemini()` class is still a valid way to initialize the model.
# Note: A future-proof alternative is to use `GoogleGenerativeAI` from the `llama-index-llms-google-genai` package.
Settings.llm = Gemini(model="gemini-2.5-flash", api_key=api_key)

# Configure the embedding model. This is the crucial step to resolve your error.
# We explicitly tell LlamaIndex to use the Gemini embedding model.
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=api_key)

# --- End of Key Changes ---

# Load the PDF documents from the `data` folder.
# The `SimpleDirectoryReader` automatically handles this.
print("Loading documents from the 'data' directory...")
documents = SimpleDirectoryReader("data").load_data()
print(f"Loaded {len(documents)} documents.")

# Create a VectorStoreIndex from the documents.
# Now, with the `Settings.embed_model` configured, this step will correctly use the Gemini embedding model.
print("Building the index from documents...")
index = VectorStoreIndex.from_documents(documents)
print("Index built successfully!")

# Persist the index to disk to avoid rebuilding it every time.
# This saves time and API calls for future runs.
print("Persisting the index to disk...")
index.storage_context.persist(persist_dir="./storage")
print("Index saved to the './storage' folder.")

# Create a query engine to ask questions about the PDF.
# This engine will retrieve relevant document chunks and pass them to the LLM.
query_engine = index.as_query_engine()

# Let's ask a question that the LLM can only answer by reading the PDF.
print("\nAsking a question to the RAG system...")
question = "What are the common pitfalls to avoid when implementing AI projects?"
response = query_engine.query(question)

# Print the response.
print(f"Question: {question}")
print(f"Response: {response}")
