import argparse
import os
import sys

try:
    import pandas as pd
except ImportError as e:
    print(f"Missing dependency: {e}. Please install pandas.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Read and inspect a local MRCR parquet file.")
    parser.add_argument(
        "--needle",
        type=int,
        choices=[2, 4, 8],
        default=2,
        help="Which needle file to use (2, 4, or 8). Default: 2",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of rows to display. Default: 5",
    )
    args = parser.parse_args()

    parquet_file = os.path.join("mrcr_dataset_files", f"{args.needle}needle.parquet")
    if not os.path.exists(parquet_file):
        print(f"File not found: {parquet_file}")
        sys.exit(1)

    print(f"Loading dataset from {parquet_file} ...")
    dataset = pd.read_parquet(parquet_file)
    print(dataset.head(args.rows))
    print("\nColumns:", list(dataset.columns))

if __name__ == "__main__":
    main() 