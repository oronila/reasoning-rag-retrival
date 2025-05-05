from datasets import load_dataset
import os

# Ensure you are logged in to Hugging Face CLI for potential private/gated datasets
# (openai/mrcr might be public, but it's good practice)
# Run `huggingface-cli login` in your terminal if you haven't.

# Define the dataset repository ID
REPO_ID = "openai/mrcr"

print(f"Attempting to load dataset: {REPO_ID}")

# Set HF_HUB_ENABLE_HF_TRANSFER for potentially faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Load the dataset. This will download it to your cache if not already present.
# It might download specific configurations/splits if available.
# You might need to specify a split like 'train' or configuration if the default doesn't work.
try:
    # Try loading the default configuration/split first
    # Set download_mode to force_redownload if you suspect cache issues, otherwise defaults are usually fine.
    # Set trust_remote_code=True if the dataset loading script requires it (use with caution)
    ds = load_dataset(REPO_ID)
    print("\nDataset loaded successfully!")
    print("\nDataset structure:")
    print(ds)

    # --- Accessing Dataset Info (includes README/description) ---
    print("\nDataset Information (including README/description):")
    print(ds.info)
    # You can often access specific fields like description directly if needed:
    # print("\nDescription:")
    # print(ds.info.description)
    # -----------------------------------------------------------

    # Example: Access the first example from the 'train' split (if it exists)
    if "train" in ds:
        print("\nFirst example from 'train' split:")
        print(ds["train"][0])
    else:
        # If 'train' split doesn't exist, try accessing the first available split
        first_split_name = list(ds.keys())[0]
        print(f"\n'train' split not found. Accessing first example from '{first_split_name}' split:")
        print(ds[first_split_name][0])

except Exception as e:
    print(f"\nError loading dataset: {e}")
    print("\nPlease check the dataset name and ensure you have the necessary permissions.")
    print("You might need to specify a configuration or split, e.g., load_dataset('openai/mrcr', 'some_config_or_split')") 