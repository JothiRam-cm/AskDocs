"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.
"""

import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
import warnings

from typing import List, Tuple, Any, Optional

# Suppress noisy torch warnings some packages emit
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.classes.*")

# -------------------------------
# LANGCHAIN IMPORTS (WINDOWS-FRIENDLY, 0.1.x STYLE)
# -------------------------------
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# MultiQueryRetriever lives here in langchain 0.0.x–0.1.x
from langchain.retrievers.multi_query import MultiQueryRetriever

# Use pure-Python protobuf to avoid some native issues
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Persistent directory for Chroma vectors
PERSIST_DIRECTORY = os.path.join("data", "vectors")

# Streamlit page configuration
st.set_page_config(
    page_title="AskDocs",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_model_names(models_info: Any) -> Tuple[str, ...]:
    """Extract model names from the response of ollama.list()."""
    try:
        if hasattr(models_info, "models"):
            return tuple(model.model for model in models_info.models)
        return tuple()
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()


def create_vector_db(file_upload) -> Chroma:
    """
    Create a vector database from an uploaded PDF file.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()
    try:
        path = os.path.join(temp_dir, file_upload.name)
        with open(path, "wb") as f:
            f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")

        loader = UnstructuredPDFLoader(path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        logger.info("Document split into chunks")

        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=f"pdf_{hash(file_upload.name)}",  # unique per file name
        )
        logger.info("Vector DB created with persistent storage")
        return vector_db
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"Temporary directory {temp_dir} removed")


def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected Ollama model.
    """
    logger.info(f"Processing question with model={selected_model}: {question!r}")

    llm = ChatOllama(model=selected_model)

    # Prompt used to generate multiple variants of the user query
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=(
            "You are an AI assistant. Generate 2 different reformulations of the "
            "user question to help retrieve more relevant documents from a vector "
            "database. Return each variant on a new line.\n\n"
            "Original question: {question}"
        ),
    )

    # Multi-query retriever over the Chroma retriever
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT,
    )

    rag_template = """You are a helpful assistant. Use ONLY the context below to answer.

Context:
{context}

Question: {question}

If the answer is not in the context, say you don't know based on the document.
"""

    prompt = ChatPromptTemplate.from_template(rag_template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Response generated successfully")
    return response


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """Extract all pages from a PDF file as images."""
    with pdfplumber.open(file_upload) as pdf:
        return [page.to_image().original for page in pdf.pages]


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """Delete the vector database and clear related session state."""
    if not vector_db:
        st.error("No vector database found to delete.")
        return

    try:
        vector_db.delete_collection()
        st.session_state.pop("pdf_pages", None)
        st.session_state.pop("file_upload", None)
        st.session_state.pop("vector_db", None)
        st.success("Collection and temporary files deleted successfully.")
        st.rerun()
    except Exception as e:
        st.error(f"Error deleting collection: {str(e)}")
        logger.error(f"Error deleting collection: {e}")


def main() -> None:
    """Main function to run the Streamlit application."""
    st.subheader("🧠 AskDocs playground - (Upload, Ask, Get Answers)", divider="gray")

    # Get available local Ollama models
    try:
        models_info = ollama.list()
        available_models = extract_model_names(models_info)
    except Exception as e:
        available_models = ()
        st.error(f"Error listing Ollama models: {e}")
        logger.error(f"Error listing Ollama models: {e}")

    col1, col2 = st.columns([1.5, 2])

    # Session state init
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "use_sample" not in st.session_state:
        st.session_state["use_sample"] = False

    # Model selection
    selected_model = None
    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓",
            available_models,
            key="model_select",
        )
    else:
        col2.warning("No Ollama models found. Make sure Ollama is running and has models pulled.")

    # Sample PDF toggle
    use_sample = col1.toggle(
        "Use sample PDF (Scammer Agent Paper)",
        key="sample_checkbox",
    )

    # Clear vector DB if switching between sample/upload
    if use_sample != st.session_state.get("use_sample"):
        if st.session_state["vector_db"] is not None:
            st.session_state["vector_db"].delete_collection()
            st.session_state["vector_db"] = None
            st.session_state["pdf_pages"] = None
        st.session_state["use_sample"] = use_sample

    # --- Sample PDF path ---
    if use_sample:
        sample_path = "data/pdfs/sample/scammer-agent.pdf"
        if os.path.exists(sample_path):
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing sample PDF..."):
                    loader = UnstructuredPDFLoader(sample_path)
                    documents = loader.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
                    chunks = splitter.split_documents(documents)

                    st.session_state["vector_db"] = Chroma.from_documents(
                        documents=chunks,
                        embedding=OllamaEmbeddings(model="nomic-embed-text"),
                        persist_directory=PERSIST_DIRECTORY,
                        collection_name="sample_pdf",
                    )

                    with pdfplumber.open(sample_path) as pdf:
                        st.session_state["pdf_pages"] = [
                            page.to_image().original for page in pdf.pages
                        ]
        else:
            st.error("Sample PDF file not found at data/pdfs/sample/scammer-agent.pdf")
    else:
        # Regular upload
        file_upload = col1.file_uploader(
            "Upload a PDF file ↓",
            type="pdf",
            accept_multiple_files=False,
            key="pdf_uploader",
        )

        if file_upload and st.session_state["vector_db"] is None:
            with st.spinner("Processing uploaded PDF..."):
                st.session_state["vector_db"] = create_vector_db(file_upload)
                with pdfplumber.open(file_upload) as pdf:
                    st.session_state["pdf_pages"] = [
                        page.to_image().original for page in pdf.pages
                    ]

    # Display PDF pages
    if st.session_state.get("pdf_pages"):
        zoom = col1.slider(
            "Zoom Level",
            min_value=100,
            max_value=1000,
            value=700,
            step=50,
            key="zoom_slider",
        )
        with col1:
            with st.container(height=410, border=True):
                for img in st.session_state["pdf_pages"]:
                    st.image(img, width=zoom)

    # Delete collection button
    if col1.button("⚠️ Delete collection", type="secondary", key="delete_button"):
        delete_vector_db(st.session_state["vector_db"])

    # Chat interface
    with col2:
        message_container = st.container(height=500, border=True)

        # Show history
        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Input
        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    if st.session_state["vector_db"] is not None and selected_model:
                        with st.spinner(":green[processing...]"):
                            response = process_question(
                                prompt,
                                st.session_state["vector_db"],
                                selected_model,
                            )
                            st.markdown(response)
                            st.session_state["messages"].append(
                                {"role": "assistant", "content": response}
                            )
                    elif not selected_model:
                        st.warning("Please select an Ollama model first.")
                    else:
                        st.warning("Please upload a PDF file or enable the sample PDF first.")
            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file or use the sample PDF to begin chat...")


if __name__ == "__main__":
    main()
