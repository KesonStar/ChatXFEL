# ChatXFEL

ChatXFEL is an intelligent Question & Answer system designed for the X-ray Free-Electron Laser (XFEL) community. It leverages Retrieval-Augmented Generation (RAG) to provide accurate answers based on a vast collection of scientific publications, technical reports, and theses.

## Overview

ChatXFEL assists researchers and engineers by retrieving relevant information from a specialized corpus of XFEL-related documents. The system processes PDF documents, extracts metadata, vectorizes the content for semantic search, and uses Large Language Models (LLMs) to synthesize answers with source citations.

**Current Version:** Beta 1.0

## Features

*   **Domain-Specific RAG:** Tailored for XFEL scientific literature and engineering reports.
*   **Dual Database Architecture:**
    *   **MongoDB:** Stores document metadata (DOI, title, authors, publication year).
    *   **Milvus:** Stores vector embeddings for high-performance semantic search.
*   **Advanced Retrieval:** Utilizes the `BGE-M3` model for dense and sparse vector retrieval, with support for reranking.
*   **Automated Pipeline:** Tools for batch processing PDFs, extracting metadata (via `pdf2doi`, `pdf2bib`), and vectorization.
*   **Interactive UI:** A user-friendly web interface built with **Streamlit**.

## Architecture

The system consists of three main stages:

1.  **Data Ingestion (`process_bibs.py`):** Scans directories for PDF files, extracts metadata (DOI, Title, Abstract) using online services (Crossref) or local extraction, and stores the metadata in MongoDB.
2.  **Vectorization (`vectorize_bibs.py`):** Reads processed documents, splits text into chunks, generates embeddings using `BGE-M3`, and indexes them in the Milvus vector database.
3.  **Inference (`chatxfel_app.py` & `rag.py`):** The Streamlit app captures user queries, retrieves relevant context from Milvus, and prompts the LLM to generate an answer with citations.

## Prerequisites

*   **Python:** 3.10 or higher recommended
*   **Services:**
    *   [MongoDB](https://www.mongodb.com/) (for metadata storage)
    *   [Milvus](https://milvus.io/) (for vector storage)
    *   [Ollama](https://ollama.com/) (for LLM inference API)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ChatXFEL
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (conda or venv).
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Before running the scripts, ensure your database connections are configured. 

*   **Database Credentials:** Check `utils.py` (or the script arguments) to configure your MongoDB and Milvus host, port, and credentials.
*   **LLM Endpoint:** The system connects to an Ollama instance. Ensure the `base_url` in `rag.py` or `chatxfel_app.py` points to your running Ollama server.

## Usage

### 1. Process Bibliographies (PDFs)
Scan a directory of PDFs, extract metadata, and save to MongoDB.

```bash
python process_bibs.py --file_path /path/to/pdfs --db_name your_db --collection bibs
```
*Use `--help` for more options like duplicate detection or manual metadata entry.*

### 2. Vectorize Documents
Embed the processed documents and insert them into Milvus.

```bash
# Generate a file list from MongoDB first
python vectorize_bibs.py --generate-list --mongo-db your_db --mongo-col bibs --file-path ./data

# Run vectorization (creating a new Milvus collection if needed)
python vectorize_bibs.py --new --collection chatxfel_vectors --file-path ./data --model bge
```

### 3. Run the Chat Application
Launch the web interface.

```bash
streamlit run chatxfel_app.py
```

## Roadmap

We are actively working on the following enhancements (see `IMPLEMENTATION_TODO.md` for details):

*   **Query Rewrite:** Improving retrieval by rewriting ambiguous user queries based on conversation history.
*   **Chat History Management:** Enabling multi-turn conversations with context awareness.
*   **Agent-based Research:** Implementing a ReAct Agent to perform deep research tasks, multi-step reasoning, and cross-referencing.



