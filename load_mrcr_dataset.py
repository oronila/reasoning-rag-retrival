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
    ds = load_dataset(REPO_ID)
    print("\nDataset loaded successfully!")
    print("\nDataset structure:")
    print(ds)

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