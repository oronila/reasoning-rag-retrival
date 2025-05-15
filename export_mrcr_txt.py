import argparse
import os
import sys

try:
    import pandas as pd
except ImportError as e:
    print(f"Missing dependency: {e}. Please install pandas.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Export prompts and answers from MRCR parquet to txt.")
    parser.add_argument(
        "--needle",
        type=int,
        choices=[2, 4, 8],
        default=2,
        help="Which needle file to use (2, 4, or 8). Default: 2",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output txt file. Default: mrcr_export_{needle}needle.txt",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Number of rows to export. Default: all",
    )
    args = parser.parse_args()

    parquet_file = os.path.join("mrcr_dataset_files", f"{args.needle}needle.parquet")
    if not os.path.exists(parquet_file):
        print(f"File not found: {parquet_file}")
        sys.exit(1)

    output_file = args.output or f"mrcr_export_{args.needle}needle.txt"
    print(f"Loading dataset from {parquet_file} ...")
    dataset = pd.read_parquet(parquet_file)
    if args.rows:
        dataset = dataset.head(args.rows)

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, row in dataset.iterrows():
            f.write(f"=== SAMPLE {idx} ===\n")
            f.write("PROMPT:\n")
            f.write(str(row["prompt"]))
            f.write("\n\nANSWER:\n")
            f.write(str(row["answer"]))
            f.write("\n\n====================\n\n")
    print(f"Exported {len(dataset)} samples to {output_file}")

if __name__ == "__main__":
    main() 