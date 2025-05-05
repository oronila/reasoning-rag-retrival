import os
import sys
import time
# Removed dotenv import as Groq key is no longer needed
# from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Removed ChatGroq import
# from langchain_groq import ChatGroq
# Added Ollama import
# from langchain_community.llms import Ollama # Old import
from langchain_ollama import OllamaLLM # New recommended import
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# Need itemgetter for chain manipulation
from operator import itemgetter
# Need RunnableLambda for custom functions in the chain
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# Import Document type hint
from langchain_core.documents import Document


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
    # Use HuggingFaceEmbeddings instead of SentenceTransformerEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

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

# --- Helper functions for Query Expansion Chain ---
def parse_expanded_queries(llm_output: str) -> list[str]:
    """Parses the LLM output (newline-separated questions) into a list."""
    queries = [q.strip() for q in llm_output.split('\n') if q.strip()]
    # Add the original query implicitly later if needed, or ensure prompt asks for it
    print(f"--- Expanded Queries ---\n{queries}\n----------------------")
    return queries

def unique_documents(documents: list[Document]) -> list[Document]:
    """Filters a list of documents to keep only unique ones based on page_content."""
    seen_contents = set()
    unique_docs = []
    for doc in documents:
        if doc.page_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(doc.page_content)
    print(f"Retrieved {len(documents)} docs, reduced to {len(unique_docs)} unique docs.")
    return unique_docs
# ---

def setup_rag_chain(vector_store):
    """Setup the RAG chain using Ollama with Query Expansion."""
    print(f"Setting up RAG chain with Ollama model '{OLLAMA_MODEL}' and Query Expansion...")
    # Instantiate Ollama LLM
    llm = OllamaLLM(model=OLLAMA_MODEL)

    # Instantiate Retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 5}) # Retrieve top 5 docs per query

    # 1. Query Expansion Chain
    expansion_prompt_template = """SYSTEM: You are an AI assistant. Your task is to rephrase the user's question in 3 different ways to improve document retrieval.
Maintain the core meaning of the question. Include the original question as one of the outputs.
Output *only* the questions, each on a new line. Do not add any commentary or numbering.

USER QUESTION:
{question}

REPHRASED QUESTIONS:"""
    expansion_prompt = PromptTemplate(input_variables=["question"], template=expansion_prompt_template)

    expansion_chain = (
        expansion_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(parse_expanded_queries)
    )

    # 2. Retrieval Chain (using expanded queries)
    # Takes a list of queries, runs retriever.batch, flattens the list of lists, and deduplicates
    retrieval_chain_expanded = (
        RunnableLambda(lambda queries: retriever.batch(queries, config={"max_concurrency": 5})) # Run retrievals in parallel
        | RunnableLambda(lambda list_of_lists: [doc for sublist in list_of_lists for doc in sublist]) # Flatten list
        | RunnableLambda(unique_documents) # Deduplicate
    )


    # 3. Final Answer Chain (using `create_stuff_documents_chain`)
    # This chain combines the retrieved documents into a context string for the final LLM call.
    final_answer_prompt_template = """SYSTEM: You are a helpful assistant. Answer the question based *only* on the following context. If you don't know the answer based on the context, just say you don't know. Don't make up an answer. Be concise.

CONTEXT:
{context}

QUESTION:
{input}

ANSWER:"""
    final_answer_prompt = ChatPromptTemplate.from_template(final_answer_prompt_template)
    answer_chain = create_stuff_documents_chain(llm, final_answer_prompt)

    # 4. Full RAG Chain
    # Ties Expansion -> Retrieval -> Final Answer Generation together
    # The structure ensures the final answer chain receives the deduplicated documents ('context')
    # and the original user question ('input').
    full_rag_chain = (
        RunnablePassthrough.assign(expanded_queries=itemgetter("input") | expansion_chain) # Expand query
        | RunnablePassthrough.assign( # Retrieve based on expanded queries
              context=itemgetter("expanded_queries") | retrieval_chain_expanded
          )
        | answer_chain # Generate answer using retrieved context and original input
    )


    print("RAG chain with Query Expansion setup complete.")
    return full_rag_chain


def ask_question(chain, question):
    """Ask a question to the RAG chain and print the response."""
    print(f"\nAsking original question: {question}")
    start_time = time.time()
    # Invoke the full chain with the original question under the key "input"
    response = chain.invoke({"input": question})
    end_time = time.time()

    print("\n--- Final Response ---")
    # The response from create_stuff_documents_chain is directly the answer string
    print(response)
    print("--------------------")
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
        # Use HuggingFaceEmbeddings here as well when loading the store
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded successfully.")

    # --- Setup RAG Chain ---
    # Updated call to setup_rag_chain (no API key needed)
    rag_chain = setup_rag_chain(vector_store)

    # --- Querying ---
    print("\nEnter your questions (type 'quit' to exit):")
    while True:
        user_question = input("> ")
        if user_question.lower() == 'quit':
            break
        if user_question:
            ask_question(rag_chain, user_question)

if __name__ == "__main__":
    main() 