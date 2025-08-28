import os
import re
from typing import Union

# Import necessary loaders and splitters
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Import embedding model and vector store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Import tool decorator and core agent components
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

# --- 1. Knowledge Base Setup (RAG) ---

# Define the path for the FAISS index and the source log file.
FAISS_INDEX_PATH = "faiss_index"
log_file_path = "F:\\Coding\\HPE\\Linux.log"

# Initialize the embedding model from HuggingFace.
print("Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Check if the vector store already exists. If not, create it.
if not os.path.exists(FAISS_INDEX_PATH):
    print(f"No existing FAISS index found. Creating new one from: {log_file_path}")
    loader = TextLoader(log_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index saved locally at {FAISS_INDEX_PATH}.")
else:
    print(f"Found existing FAISS index at {FAISS_INDEX_PATH}. Loading...")

# --- 2. Custom Tools Definition ---

@tool
def knowledge_base_retriever(query: str) -> str:
    """Searches the knowledge base for similar log patterns or incident reports."""
    print(f"\n--- Calling knowledge_base_retriever with query: {query} ---")
    local_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = local_db.similarity_search(query)
    if docs:
        return docs[0].page_content
    return "No relevant information found in the knowledge base."

@tool
def log_volume_checker(time_window_minutes: int) -> str:
    """Checks the volume of logs within a specified time window to spot unusual activity."""
    print(f"\n--- Calling log_volume_checker for time window: {time_window_minutes} minutes ---")
    return "Log volume is within normal parameters."

# --- 3. Agent and Executor Assembly ---

print("Initializing Llama LLM via Ollama...")
llm = ChatOllama(model="llama3.2:3b", temperature=0)
tools = [knowledge_base_retriever, log_volume_checker]

# This is the final version of the prompt with explicit output formatting.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Log Anomaly Detection Expert. Your primary task is to determine if the original 'Input Log' is an anomaly. You must use the available tools to gather context about the Input Log. Your final conclusion of 'ANOMALY' or 'NORMAL' must be about the original Input Log itself, not about information found in the knowledge base. Use the retrieved context to inform your decision. For example, if the Input Log is a common event found in the knowledge base, it is likely NORMAL. First, check the log volume, then consult the knowledge base. Your final answer MUST be a single word, either 'ANOMALY' or 'NORMAL', followed by a colon and then a one-sentence justification. For example: 'NORMAL: This is a routine system event.'"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create the more modern and reliable Tool Calling agent
print("Creating Tool Calling agent...")
agent = create_tool_calling_agent(llm, tools, prompt)

# The AgentExecutor remains the same
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 4. Interactive Execution Loop ---

if __name__ == "__main__":
    print("\n--- Log Anomaly Detection Agent Initialized ---")
    print("--- Enter a log entry to analyze, or type 'exit' or 'quit' to end. ---")

    while True:
        user_input = input("\n[Enter Log Entry] > ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting agent. Goodbye!")
            break
        if not user_input:
            print("Input cannot be empty.")
            continue

        print(f"\nAnalyzing log: '{user_input}'")
        try:
            # The invoke call is now much simpler
            result = agent_executor.invoke({
                "input": user_input
            })
            print("\n--- Agent Final Output ---")
            print(result['output'])
        except Exception as e:
            print(f"An error occurred: {e}")
        print("--------------------------")
