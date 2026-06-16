#!/bin/bash

# setup_model.sh
# Script to prepare the faster-whisper model (MediaTek-Research/Breeze-ASR-25)
# for CTranslate2 inference.
# Optimized for Tesla V100 (float16 quantization).

set -e

echo "=> Installing required packages..."
pip install ctranslate2 transformers huggingface_hub

MODEL_ID="MediaTek-Research/Breeze-ASR-25"
OUTPUT_DIR="./models/breeze-25-ct2"

echo "=> Converting ${MODEL_ID} to CTranslate2 format..."
echo "   Applying float16 quantization optimized for Tesla V100."

# ct2-transformers-converter will download the model weights and convert them.
# The --force flag will overwrite the output directory if it already exists.
ct2-transformers-converter --model "${MODEL_ID}" \
    --output_dir "${OUTPUT_DIR}" \
    --quantization float16 \
    --force

echo "=> Copying essential Whisper configuration files..."
# faster-whisper requires the tokenizer.json and preprocessor_config.json
# Sometimes the converter doesn't include all necessary whisper files,
# so we ensure they are explicitly downloaded into the output directory.
# Use python to download configuration files to avoid issues with deprecated huggingface-cli/hf wrapper commands
python -c "
import os
from huggingface_hub import hf_hub_download
for filename in ['tokenizer.json', 'preprocessor_config.json']:
    hf_hub_download(repo_id='${MODEL_ID}', filename=filename, local_dir='${OUTPUT_DIR}')
"

echo "=> Setup complete! The model is ready at: ${OUTPUT_DIR}"
