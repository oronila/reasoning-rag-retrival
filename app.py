import os
import sys
import time
# Removed dotenv import as Groq key is no longer needed
# from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
# Removed ChatGroq import
# from langchain_groq import ChatGroq
# Added Ollama import
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# --- Configuration ---
DATA_PATH = "data/"
VECTORSTORE_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Common sentence transformer model
OLLAMA_MODEL = "llama3" # Specify the Ollama model you have pulled

# Removed load_config function
# def load_config():
#     """Load environment variables."""
#     load_dotenv()
#     groq_api_key = os.getenv("GROQ_API_KEY")
#     if not groq_api_key:
#         print("ERROR: GROQ_API_KEY environment variable not set.")
#         sys.exit(1)
#     return groq_api_key

def load_documents(data_path):
    """Load documents from the specified directory."""
    print(f"Loading documents from '{data_path}'...")
    # Using DirectoryLoader with PyPDFLoader for PDFs
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True)
    documents = loader.load()
    if not documents:
        print(f"No PDF documents found in '{data_path}'. Please add some PDF files.")
        sys.exit(1)
    print(f"Loaded {len(documents)} document(s).")
    return documents

def split_documents(documents):
    """Split documents into smaller chunks."""
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")
    return texts

def create_vector_store(texts, embedding_model_name, vectorstore_path):
    """Create or load the vector store."""
    print(f"Initializing embedding model '{embedding_model_name}'...")
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    if os.path.exists(vectorstore_path):
        print(f"Loading existing vector store from '{vectorstore_path}'...")
        db = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True) # Allow deserialization
        # Optionally, add new texts if needed (more complex logic for updates)
        # print("Adding new texts to existing vector store (if any)...")
        # db.add_documents(texts) # This might create duplicates if not handled carefully
        print("Vector store loaded.")
    else:
        print(f"Creating new vector store at '{vectorstore_path}'...")
        start_time = time.time()
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(vectorstore_path)
        end_time = time.time()
        print(f"Vector store created and saved in {end_time - start_time:.2f} seconds.")
    return db

def setup_rag_chain(vector_store):
    """Setup the RAG chain using Ollama."""
    print(f"Setting up RAG chain with Ollama model '{OLLAMA_MODEL}'...")
    # Instantiate Ollama
    # Assumes Ollama is running on the default http://localhost:11434
    # Add base_url="<your_ollama_url>" if it's running elsewhere
    llm = Ollama(model=OLLAMA_MODEL)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 5}) # Retrieve top 5 docs

    # Define the prompt template
    template = """SYSTEM: You are a helpful assistant. Answer the question based only on the following context. If you don't know the answer, just say you don't know. Don't make up an answer.

CONTEXT:
{context}

QUESTION:
{input}

ANSWER:"""
    prompt = ChatPromptTemplate.from_template(template)

    # Create the chain that combines documents into a context string
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create the main retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print("RAG chain setup complete.")
    return retrieval_chain


def ask_question(chain, question):
    """Ask a question to the RAG chain and print the response."""
    print(f"Asking question: {question}")
    start_time = time.time()
    response = chain.invoke({"input": question})
    end_time = time.time()

    print("--- Response ---")
    print(response["answer"])
    print("----------------")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    # print("--- Context Documents ---")
    # for i, doc in enumerate(response["context"]):
    #     print(f"Document {i+1}:")
    #     # print(doc.page_content) # Uncomment to see full context
    #     print(f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
    #     print("-" * 20)


def main():
    """Main function to run the RAG pipeline."""
    # Removed Groq key loading
    # groq_api_key = load_config()

    # --- Data Ingestion and Vector Store Creation/Loading ---
    # Check if vector store exists, if not, create it
    if not os.path.exists(VECTORSTORE_PATH):
        print("Vector store not found. Starting ingestion process...")
        documents = load_documents(DATA_PATH)
        texts = split_documents(documents)
        vector_store = create_vector_store(texts, EMBEDDING_MODEL, VECTORSTORE_PATH)
    else:
        print(f"Found existing vector store at '{VECTORSTORE_PATH}'. Loading...")
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded successfully.")

    # --- Setup RAG Chain ---
    # Updated call to setup_rag_chain (no API key needed)
    rag_chain = setup_rag_chain(vector_store)

    # --- Querying ---
    print("Enter your questions (type 'quit' to exit):")
    while True:
        user_question = input("> ")
        if user_question.lower() == 'quit':
            break
        if user_question:
            ask_question(rag_chain, user_question)

if __name__ == "__main__":
    main() 