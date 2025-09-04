import os
import logging
import sys
from flask import Flask, request, jsonify, render_template
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
# Import the new agent type
from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain.agents.react.base import create_react_agent # Use this if you want to try an alternative
# Update the imports to include the necessary prompt classes
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Import all your tools as before
from tools import get_current_time, get_product_details, get_all_products, create_product, update_product_put, update_product_patch, delete_product, finalize_product, delete_product_icon, update_product_icon, search_company_docs

# Optional: Set up logging for a better view of the process
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Initialize Flask app
app = Flask(__name__)

# --- Agent Setup (with a conversational agent) ---
# Your existing setup code remains the same
if "GOOGLE_API_KEY" not in os.environ:
    print("Please set the GOOGLE_API_KEY environment variable.")
    exit()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# A list of all available tools for the agent to use
tools = [
    get_current_time,
    get_product_details,
    get_all_products,
    create_product,
    update_product_put,
    update_product_patch,
    delete_product,
    finalize_product,
    delete_product_icon,
    update_product_icon,
    search_company_docs
]

# We will now use a more explicit prompt with instructions
system_message = """
You are a conversational AI assistant. Your purpose is to help users by answering questions and performing actions using the tools at your disposal.

You have access to various tools for product management, document search, and general assistance.

For multi-step requests, like creating a product, you must gather all the required information from the user before you call the tool. Do not call the tool until you have all the necessary parameters.

When you need more information from the user, explicitly ask for it. Be polite and clear about what you need.

If a user asks to perform a task, first check if you have all the required parameters for the corresponding tool. If you do not, ask for the missing parameters one by one until you have everything you need.

You should always respond conversationally.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Use create_tool_calling_agent
agent = create_tool_calling_agent(llm, tools, prompt)

# We still use the same AgentExecutor
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    input_variables=["input"],
    verbose=True
)

# --- Conversational Loop and API Endpoint ---

# No longer need the simple chat_history list, the memory object handles it.

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        # The agent executor now handles the chat history automatically
        response = agent_executor.invoke({"input": user_input})
        agent_response = response["output"]

        return jsonify({"response": agent_response})

    except Exception as e:
        logging.error(f"Error during agent execution: {e}")
        return jsonify({"error": "An error occurred during processing."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)