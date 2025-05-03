# reasoning-rag-retrival
# Local RAG Pipeline with LangChain and Ollama

This project implements a simple Retrieval-Augmented Generation (RAG) pipeline that runs locally using LangChain, FAISS for vector storage, Sentence Transformers for embeddings, and a local Ollama instance for LLM inference.

## Setup

1.  **Install Ollama:**
    *   Download and install Ollama from [https://ollama.com/](https://ollama.com/).
    *   Ensure the Ollama application is running.
    *   Pull the LLM model you want to use (the script defaults to `llama3`). Open your terminal and run:
        ```bash
        ollama pull llama3
        ```
        (Replace `llama3` if you want to use a different model like `mistral`, `phi3`, etc., and update the `OLLAMA_MODEL` variable in `app.py` accordingly).

2.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd reasoning-rag-retrival
    ```

3.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4.  **Install dependencies:**
    *   Ensure you have the necessary build tools if `faiss-cpu` requires compilation (this can vary by OS).
    *   Install the Python packages:
        ```bash
        pip install -r requirements.txt
        ```

5.  **Add Documents:**
    *   Place the PDF documents you want to query into the `data/` directory.

## Running the Pipeline

1.  **Ensure Ollama is running.**

2.  **Run the application:**
    ```bash
    python app.py
    ```

3.  **First Run (Ingestion):**
    *   On the first run, the script will:
        *   Load documents from the `data/` directory.
        *   Split the documents into chunks.
        *   Generate embeddings using the `all-MiniLM-L6-v2` model.
        *   Create a FAISS vector store and save it to the `vectorstore/db_faiss` directory.
    *   This process might take some time depending on the number and size of your documents and your hardware.

4.  **Subsequent Runs:**
    *   On subsequent runs, the script will load the existing vector store from `vectorstore/db_faiss`, making startup much faster.

5.  **Querying:**
    *   Once the RAG chain is set up, you'll be prompted to enter your questions.
    *   Type your question and press Enter. The script will retrieve relevant document chunks, send them along with your question to your local Ollama LLM, and print the answer.
    *   Type `quit` to exit the application.

## How it Works

1.  **Document Loading:** PDFs from the `data/` folder are loaded.
2.  **Text Splitting:** Documents are split into smaller, manageable chunks.
3.  **Embedding:** Each chunk is converted into a numerical vector (embedding) using Sentence Transformers.
4.  **Vector Storage:** Embeddings are stored in a FAISS index for efficient similarity search.
5.  **Retrieval:** When you ask a question, it's embedded, and FAISS finds the most relevant document chunks based on embedding similarity.
6.  **Generation:** The relevant chunks (context) and your original question are passed to the locally running Ollama LLM via a prompt template.
7.  **Answer:** The Ollama LLM generates an answer based *only* on the provided context and question.