# AskDocs Project Documentation

**AskDocs** is a Retrieval-Augmented Generation (RAG) application that allows users to chat with their PDF documents. It leverages local Large Language Models (LLMs) via **Ollama** to ensure data privacy and offline capability.

## 1. System Overview

The system is designed to process PDF documents, convert them into vector embeddings, and allow users to ask natural language questions. It uses a **Multi-Query Retrieval** approach to improve search accuracy by generating multiple perspectives of a user's question.

### Key Technologies
-   **Frontend**: [Streamlit](https://streamlit.io/) for the interactive web interface.
-   **LLM & Embeddings**: [Ollama](https://ollama.com/) running local models (e.g., `llama2`, `mistral`) and embeddings (`nomic-embed-text`).
-   **Orchestration**: [LangChain](https://www.langchain.com/) for the RAG pipeline.
-   **Vector Database**: [ChromaDB](https://www.trychroma.com/) for persistent storage of document embeddings.

## 2. Tech Stack

-   **Language**: Python 3.8+
-   **Frontend**: [Streamlit](https://streamlit.io/)
-   **LLM Orchestration**: [LangChain](https://www.langchain.com/)
-   **Local LLM Runner**: [Ollama](https://ollama.com/)
-   **Vector Database**: [ChromaDB](https://www.trychroma.com/)
-   **PDF Processing**: [Unstructured](https://unstructured.io/), `pdfplumber`
-   **Embeddings**: `nomic-embed-text` (via Ollama)


## 2. Architecture

The application follows a standard RAG pipeline:

1.  **Ingestion**:
    -   **Loading**: `UnstructuredPDFLoader` extracts text from uploaded PDFs.
    -   **Splitting**: `RecursiveCharacterTextSplitter` chunks text (7500 chars, 100 overlap).
    -   **Embedding**: `OllamaEmbeddings` converts chunks into vectors using `nomic-embed-text`.
    -   **Storage**: Vectors are stored in a local `Chroma` database (`data/vectors`).

2.  **Retrieval**:
    -   **Multi-Query**: The `MultiQueryRetriever` (from `langchain_experimental`) uses an LLM to generate 2 alternative versions of the user's question.
    -   **Search**: These queries are executed against the Chroma database to find relevant context.

3.  **Generation**:
    -   **Synthesis**: The retrieved context and the original question are fed into the selected local LLM.
    -   **Response**: The LLM generates a concise answer based *only* on the provided context.

## 3. Directory Structure

The project contains two distinct implementations of the logic:

```
AskDocs/
├── run.py                  # Helper script to launch the app
├── requirements.txt        # Project dependencies
├── src/
│   ├── app/
│   │   └── main.py         # [ACTIVE] The standalone Streamlit application
│   └── core/               # [INACTIVE] Modularized components (refactoring candidates)
│       ├── rag.py
│       ├── llm.py
│       ├── embeddings.py
│       └── document.py
```

> [!NOTE]
> **Active Implementation**: The currently running application logic resides entirely within `src/app/main.py`. The files in `src/core/` appear to be a modularized version of the logic that is **not currently used** by the main application.

## 4. Component Analysis

### Application Logic (`src/app/main.py`)
This single file contains the end-to-end logic for the Streamlit app.

-   **`main()`**: The entry point. It handles the UI layout, model selection, file upload, and chat interface.
-   **`create_vector_db(file_upload)`**:
    -   Saves the uploaded file temporarily.
    -   Loads and splits the PDF.
    -   Creates and persists the Chroma vector database.
-   **`process_question(question, vector_db, selected_model)`**:
    -   Sets up the `MultiQueryRetriever` with a custom prompt to generate alternative questions.
    -   Constructs the RAG chain using `RunnablePassthrough` and `StrOutputParser`.
    -   Invokes the chain to get the answer.
-   **`extract_all_pages_as_images(file_upload)`**: Uses `pdfplumber` to render PDF pages as images for the sidebar display.

### Core Modules (`src/core/`)
These files represent a structured approach to the same logic, likely intended for future refactoring or scalability.

-   **`rag.py`**: Defines a `RAGPipeline` class to manage the retrieval and generation flow.
-   **`llm.py`**: `LLMManager` for handling Ollama configurations and prompts.
-   **`embeddings.py`**: `VectorStore` class for managing ChromaDB operations.
-   **`document.py`**: `DocumentProcessor` for loading and splitting PDFs.

## 5. Installation & Setup

### Prerequisites
1.  **Python 3.8+** installed.
2.  **Ollama** installed and running.
    -   Pull the required models:
        ```bash
        ollama pull llama2
        ollama pull nomic-embed-text
        ```

### Installation
1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd AskDocs
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 6. Usage Guide

1.  **Start the App**:
    ```bash
    streamlit run src/app/main.py
    # OR
    python run.py
    ```
2.  **Select a Model**: Choose a local LLM from the dropdown (e.g., `llama2`).
3.  **Upload Data**:
    -   Upload a PDF file via the "Upload a PDF" button.
    -   OR toggle "Use sample PDF" to test with the provided sample.
4.  **Chat**: Type your question in the input box. The system will retrieve relevant context and generate an answer.
5.  **Reset**: Click "Delete collection" to clear the database and upload a new file.

## 7. Troubleshooting

-   **`ImportError: cannot import name 'MultiQueryRetriever'`**: Ensure you have `langchain-experimental` installed, as this class was moved in recent LangChain versions.
-   **Ollama Connection Issues**: Make sure `ollama serve` is running in a separate terminal.
-   **Missing Models**: If the dropdown is empty or models fail, run `ollama list` to verify installed models and `ollama pull <model>` to get them.
