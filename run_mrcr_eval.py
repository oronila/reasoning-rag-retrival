import argparse
import os
import sys

try:
    import pandas as pd
    import json
    from openai import OpenAI
    from difflib import SequenceMatcher
    import tiktoken
except ImportError as e:
    print(f"Missing dependency: {e}. Please install openai, pandas, and tiktoken.")
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
    parser = argparse.ArgumentParser(description="Run MRCR eval on a local parquet file.")
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
        default="gpt-4.1",
        help="OpenAI model name. Default: gpt-4.1",
    )
    args = parser.parse_args()

    parquet_file = os.path.join("mrcr_dataset_files", f"{args.needle}needle.parquet")
    if not os.path.exists(parquet_file):
        print(f"File not found: {parquet_file}")
        sys.exit(1)

    print(f"Loading dataset from {parquet_file} ...")
    dataset = pd.read_parquet(parquet_file)
    client = OpenAI()
    enc = tiktoken.get_encoding("o200k_base")

    for index, row in dataset.iterrows():
        messages = json.loads(row["prompt"])
        if n_tokens(messages, enc) > args.max_context_window:
            continue
        completion = client.chat.completions.create(
            model=args.model,
            messages=messages,
        )
        response = completion.choices[0].message.content
        score = grade(response, row["answer"], row["random_string_to_prepend"])
        print(f"Sample {index}: {score}")

if __name__ == "__main__":
    main() 