import os
import sys
import re
import argparse
import warnings
from huggingface_hub import HfApi, hf_hub_download
from requests.exceptions import HTTPError


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download first N parquet files from the elsaEU/ELSA_D3 dataset"
    )
    parser.add_argument(
        "N",
        type=int,
        help="Number of files to download",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="data",
        help="Destination folder for saving files (default: data/)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Branch or revision in the repository (default: main)",
    )
    return parser.parse_args()


def natural_sort_key(s: str):
    """
    Generate a key for natural sorting: split text into digit and non-digit parts,
    converting digits to integers for proper ordering.
    """
    parts = re.split(r"(\d+)", s)
    return [int(text) if text.isdigit() else text.lower() for text in parts]


def main():
    # Suppress all warnings
    warnings.filterwarnings("ignore")

    args = parse_args()
    api = HfApi()
    token = os.getenv("HF_TOKEN", None)

    # Hardcoded repository ID
    repo_id = "elsaEU/ELSA_D3"

    # Create destination directory if it doesn't exist
    os.makedirs(args.dest, exist_ok=True)

    try:
        # List all files in the dataset repository
        all_files = api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision=args.revision,
            token=token,
        )
    except HTTPError as e:
        print(f"Error fetching file list: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

    # Filter for parquet files with train- prefix
    parquet_files = [
        fname
        for fname in all_files
        if fname.startswith("data/train-") and fname.endswith(".parquet")
    ]

    if not parquet_files:
        print("No files matching pattern train-*.parquet found.", file=sys.stderr)
        sys.exit(1)

    # Sort files naturally (e.g., train-2 before train-10)
    parquet_files.sort(key=natural_sort_key)

    # Select first N files
    to_download = parquet_files[: args.N]

    for idx, remote_path in enumerate(to_download, start=1):
        filename = os.path.basename(remote_path)
        dest_path = os.path.join(args.dest, filename)
        print(f"[{idx}/{len(to_download)}] Downloading {filename} -> {dest_path}")
        try:
            # Directly download into dest without using HF cache
            local_file = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=remote_path,
                revision=args.revision,
                use_auth_token=token,
                local_dir=args.dest,
                force_filename=filename,
                local_dir_use_symlinks=False,
            )
            if not os.path.exists(local_file):
                print(f"Download failed, file not found: {local_file}", file=sys.stderr)
        except HTTPError as e:
            print(f"Error downloading {filename}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Unexpected error downloading {filename}: {e}", file=sys.stderr)

    print(f"\nDone! Downloaded {len(to_download)} file(s) into '{args.dest}/'.")


if __name__ == "__main__":
    main()
