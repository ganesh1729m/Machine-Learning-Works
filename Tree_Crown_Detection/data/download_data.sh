#!/usr/bin/env bash
# download_data.sh
# Script to download the NEON Tree Crown dataset from Kaggle

set -e

# Ensure Kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "Error: Kaggle CLI not found. Install with 'pip install kaggle' and configure API credentials." >&2
    exit 1
fi

# Directory to store raw data
DATA_DIR="data/tree-crown-dataset"
mkdir -p "${DATA_DIR}"

echo "Downloading NEON Tree Crown dataset to ${DATA_DIR}..."
kaggle datasets download -d asilva1691/tree-crown-dataset -p "${DATA_DIR}" --unzip

echo "Download complete."
echo "Directory structure:"
find "${DATA_DIR}" -maxdepth 2 | sed 's/^/  /'

echo "You can now run preprocessing and training scripts pointing to '${DATA_DIR}/neon-dataset'."
