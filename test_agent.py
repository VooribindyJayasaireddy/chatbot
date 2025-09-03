# test_agent.py
import os
import logging
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# Import our new tool from the tools.py file
from tools import search_company_docs, get_current_time, get_product_details # <-- Add get_product_details

# Optional: Set up logging for a better view of the process
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Make sure your API key is set
if "GOOGLE_API_KEY" not in os.environ:
    print("Please set the GOOGLE_API_KEY environment variable.")
    exit()

# Define the LLM (our agent's brain)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Create a prompt that tells the LLM about ALL of its tools
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. You can access internal company documents to answer questions and get the current time."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# List all the tools our agent can use
tools = [search_company_docs, get_current_time, get_product_details] # <-- Add the new tool to the list

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an agent executor to run the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Now, we'll test all three types of questions
print("--- Testing General Question ---")
result1 = agent_executor.invoke({"input": "What is the current time?"})
print("Final response:")
print(result1["output"])

print("\n--- Testing Company-Specific Question ---")
result2 = agent_executor.invoke({"input": "What are some common pitfalls to avoid when implementing AI projects?"})
print("Final response:")
print(result2["output"])

print("\n--- Testing API-Driven Question ---")
result3 = agent_executor.invoke({"input": "Can you get the details for product ID 1?"})
print("Final response:")
print(result3["output"])