# reasoning-rag-retrival
# Local RAG Pipeline with LangChain and Groq

This project implements a simple Retrieval-Augmented Generation (RAG) pipeline that runs locally using LangChain, FAISS for vector storage, Sentence Transformers for embeddings, and Groq for fast LLM inference.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd reasoning-rag-retrival
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Groq API Key:**
    *   Sign up for a free Groq account at [https://console.groq.com/](https://console.groq.com/).
    *   Create an API key.
    *   Create a file named `.env` in the project root directory.
    *   Add your Groq API key to the `.env` file:
        ```env
        GROQ_API_KEY=your_groq_api_key_here
        ```

5.  **Add Documents:**
    *   Place the PDF documents you want to query into the `data/` directory.

## Running the Pipeline

1.  **Run the application:**
    ```bash
    python app.py
    ```

2.  **First Run (Ingestion):**
    *   On the first run, the script will:
        *   Load documents from the `data/` directory.
        *   Split the documents into chunks.
        *   Generate embeddings using the `all-MiniLM-L6-v2` model.
        *   Create a FAISS vector store and save it to the `vectorstore/db_faiss` directory.
    *   This process might take some time depending on the number and size of your documents and your hardware.

3.  **Subsequent Runs:**
    *   On subsequent runs, the script will load the existing vector store from `vectorstore/db_faiss`, making startup much faster.

4.  **Querying:**
    *   Once the RAG chain is set up, you'll be prompted to enter your questions.
    *   Type your question and press Enter. The script will retrieve relevant document chunks, send them along with your question to the Groq LLM (llama3-70b-8192), and print the answer.
    *   Type `quit` to exit the application.

## How it Works

1.  **Document Loading:** PDFs from the `data/` folder are loaded.
2.  **Text Splitting:** Documents are split into smaller, manageable chunks.
3.  **Embedding:** Each chunk is converted into a numerical vector (embedding) using Sentence Transformers.
4.  **Vector Storage:** Embeddings are stored in a FAISS index for efficient similarity search.
5.  **Retrieval:** When you ask a question, it's embedded, and FAISS finds the most relevant document chunks based on embedding similarity.
6.  **Generation:** The relevant chunks (context) and your original question are passed to the Groq LLM via a prompt template.
7.  **Answer:** The LLM generates an answer based *only* on the provided context and question.