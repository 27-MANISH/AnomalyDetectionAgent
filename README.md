# Log Anomaly Detection Agent

This project contains a Python-based agent designed to detect anomalies in log files using a combination of Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) via Ollama.

## Features

-   **Knowledge Base (FAISS):** Utilizes a FAISS vector store to create a searchable knowledge base from your log data, enabling the agent to retrieve relevant log patterns.
-   **Custom Tools:** Implements custom tools for the LLM to interact with, such as `knowledge_base_retriever` for searching the log knowledge base and `log_volume_checker` for analyzing log activity.
-   **Tool Calling Agent:** Employs a Tool Calling agent architecture, which leverages models specifically fine-tuned for tool/function calling (e.g., `llama 3.2:3b`).

## Prerequisites

Before running the agent, ensure you have the following installed:

-   **Python 3.x:** Download and install from [python.org](https://www.python.org/).
-   **Ollama:** Install Ollama from [ollama.ai](https://ollama.ai/).
-   **Log File:** A log file named `Linux.log` in the root directory of your project (`f:\Coding\HPE\`). If your log file has a different name or path, you will need to update the `log_file_path` variable in `log_anomaly_detection.py`.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>/Anomaly Detection Agent
    ```

2.  **Install Python Dependencies:**
    Navigate to the `Anomaly Detection Agent` directory and install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Pull Ollama Model:**
    The agent is configured to use the `llama 3.2:3b` Ollama model. Pull it using the Ollama CLI:
    ```bash
    ollama pull llama 3.2:3b
    ```

4.  **FAISS Index Creation:**
    The first time you run the script, it will automatically create a FAISS vector store (knowledge base) from your `Linux.log` file. This may take some time depending on the size of your log file. The index will be saved in the `faiss_index/` directory.

## How to Run

Navigate to the `Anomaly Detection Agent` directory in your terminal.

### Running the Tool Calling Agent

This agent (`log_anomaly_detection.py`) is designed to work best with models that have strong tool-calling capabilities, such as `llama 3.2:3b`.

```bash
python log_anomaly_detection.py
```

After running the script, you will be prompted to `[Enter Log Entry] >`. Type a log entry and press Enter to analyze it. Type `exit` or `quit` to end the session.

## Project Structure

```
Anomaly Detection Agent/
├── log_anomaly_detection.py          # Tool Calling Agent implementation
├── faiss_index/                      # Directory for the FAISS vector store (created on first run)
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation (this file)
└── .gitignore                        # Specifies intentionally untracked files to ignore
```

## Troubleshooting

-   **`An error occurred: model is required`:** Ensure the `llama 3.2:3b` Ollama model is pulled and running in your Ollama instance.
-   **Dependency Errors:** If you encounter `ModuleNotFoundError`, make sure all dependencies are installed via `pip install -r requirements.txt`.
-   **FAISS Index Issues:** If the `faiss_index/` directory is corrupted or you wish to rebuild the knowledge base, simply delete the `faiss_index/` directory, and it will be recreated on the next run.
