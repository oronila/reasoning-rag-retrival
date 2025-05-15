import os
import sys
# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed. If you need to load environment variables from a .env file, please install it with 'pip install python-dotenv'.")

import argparse

try:
    import pandas as pd
    import json
    from difflib import SequenceMatcher
    import tiktoken
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI, AzureChatOpenAI
    from langchain_ollama import OllamaLLM
except ImportError as e:
    print(f"Missing dependency: {e}. Please install pandas, tiktoken, langchain-groq, langchain-openai, langchain-ollama, and python-dotenv.")
    sys.exit(1)

def grade(response, answer, random_string_to_prepend) -> float:
    if not response.startswith(random_string_to_prepend):
        return 0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())

def n_tokens(messages: list[dict], enc) -> int:
    return sum([len(enc.encode(m["content"])) for m in messages])

def main():
    parser = argparse.ArgumentParser(description="Run MRCR eval on a local parquet file with OpenAI, Groq, Ollama, or Azure OpenAI via LangChain.")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "groq", "ollama", "azure"],
        default="openai",
        help="Which provider to use: openai, groq, ollama, or azure. Default: openai",
    )
    parser.add_argument(
        "--needle",
        type=int,
        choices=[2, 4, 8],
        default=2,
        help="Which needle file to use (2, 4, or 8). Default: 2",
    )
    parser.add_argument(
        "--max-context-window",
        type=int,
        default=1000000,
        help="Max context window (tokens). Default: 1000000",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. For OpenAI: gpt-4.1 (default), for Groq: llama3-70b-8192 (default), for Ollama: llama3 (default), for Azure: your deployment name (required)",
    )
    args = parser.parse_args()

    parquet_file = os.path.join("mrcr_dataset_files", f"{args.needle}needle.parquet")
    if not os.path.exists(parquet_file):
        print(f"File not found: {parquet_file}")
        sys.exit(1)

    print(f"Loading dataset from {parquet_file} ...")
    dataset = pd.read_parquet(parquet_file)
    enc = tiktoken.get_encoding("o200k_base")

    if args.provider == "openai":
        model_name = args.model or "gpt-4.1"
        llm = ChatOpenAI(model=model_name)
    elif args.provider == "groq":
        model_name = args.model or "llama3-70b-8192"
        llm = ChatGroq(model=model_name)
    elif args.provider == "ollama":
        model_name = args.model or "llama3"
        llm = OllamaLLM(model=model_name)
    elif args.provider == "azure":
        if not args.model:
            print("For Azure, you must specify --model as your Azure deployment name.")
            sys.exit(1)
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
        llm = AzureChatOpenAI(
            azure_deployment=args.model,
            api_version=api_version,
        )
    else:
        print(f"Unknown provider: {args.provider}")
        sys.exit(1)

    # Iterate over the dataset in reverse order
    for index, row in dataset.iloc[::-1].iterrows():
        messages = json.loads(row["prompt"])
        num_tokens = n_tokens(messages, enc)
        if num_tokens > args.max_context_window:
            continue
        # Convert OpenAI-style messages to LangChain format
        lc_messages = []
        for m in messages:
            if m["role"] == "user":
                lc_messages.append({"type": "human", "content": m["content"]})
            elif m["role"] == "assistant":
                lc_messages.append({"type": "ai", "content": m["content"]})
        # LangChain expects a list of messages, but the API may differ by version
        try:
            response = llm.invoke(lc_messages)
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)
        except Exception as e:
            print(f"Sample {index}: ERROR - {e}")
            continue
        score = grade(response_text, row["answer"], row["random_string_to_prepend"])
        print(f"Sample {index}: {score}")
        print(f"  Tokens in prompt: {num_tokens}")
        print(f"  Model response (first 200 chars): {response_text[:200]!r}\n")

if __name__ == "__main__":
    main() 