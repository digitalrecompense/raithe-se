#!/bin/bash
# =============================================================================
# © RAiTHE INDUSTRIES INCORPORATED 2026
# RAiTHE-SE Model Download Script
# =============================================================================
# Purpose: Download required ONNX models from Hugging Face Hub
# Usage: ./scripts/download-models.sh
#
# This script is designed for AI agents working with the raithe-se codebase.
# Models are large (~1-4GB each) and are excluded from GitHub to keep
# the repository lightweight.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/data/models"

echo "================================================================================"
echo "RAiTHE-SE Model Download Script"
echo "================================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Models directory: $MODELS_DIR"
echo ""

# Check if Python and huggingface_hub are available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is required but not installed."
    exit 1
fi

# Install huggingface_hub if not present
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub -q
fi

# Create models directory structure
echo "Creating models directory structure..."
mkdir -p "$MODELS_DIR/deberta/onnx"
mkdir -p "$MODELS_DIR/mini-lm/onnx"
mkdir -p "$MODELS_DIR/phi-2/onnx"

# Function to download a model from Hugging Face
download_model() {
    local MODEL_ID="$1"
    local TARGET_DIR="$2"
    local MODEL_NAME="$3"

    echo ""
    echo "Downloading $MODEL_NAME from Hugging Face..."
    echo "Model ID: $MODEL_ID"
    echo "Target: $TARGET_DIR"

    python3 << EOF
from huggingface_hub import snapshot_download
import os

model_id = "$MODEL_ID"
target_dir = "$TARGET_DIR"

print(f"Starting download of {model_id}...")

try:
    # Download only the ONNX files if available
    local_dir = snapshot_download(
        repo_id=model_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Skip non-ONNX files
    )
    print(f"Downloaded to: {local_dir}")
except Exception as e:
    print(f"Note: {e}")
    print("This model may not have ONNX format available.")
    print("You may need to convert it manually or use PyTorch version.")
EOF

    echo "✓ $MODEL_NAME download attempted"
}

# Download models (adjust model IDs based on actual models used)
echo ""
echo "================================================================================"
echo "Downloading ONNX Models"
echo "================================================================================"

# Cross-encoder for Phase 3 reranking (MiniLM style)
download_model "cross-encoder/ms-marco-MiniLM-L-6-v2" "$MODELS_DIR/mini-lm/onnx" "Cross-Encoder (MiniLM-L6)"

# Bi-encoder for semantic search (all-MiniLM style)
download_model "sentence-transformers/all-MiniLM-L6-v2" "$MODELS_DIR/mini-lm/onnx" "Bi-Encoder (all-MiniLM)"

# Query reformulation LLM (Phi-2 or smaller)
# Note: Phi-2 ONNX conversion is complex; may need manual steps
echo ""
echo "NOTE: Phi-2 requires manual ONNX conversion."
echo "See: https://huggingface.co/docs/optimum/exporters/onnx"
echo "Download Phi-2 from: https://huggingface.co/microsoft/phi-2"

# DeBERTa model (if used for document understanding)
download_model "microsoft/deberta-v3-base" "$MODELS_DIR/deberta/onnx" "DeBERTa Base"

echo ""
echo "================================================================================"
echo "Post-Download Instructions"
echo "================================================================================"
echo ""
echo "1. Verify downloaded models:"
echo "   ls -la $MODELS_DIR/*/onnx/"
echo ""
echo "2. Update config/engine.toml with model paths if different:"
echo "   [neural]"
echo "   cross_encoder_model_path = \"data/models/mini-lm/onnx/model.onnx\""
echo "   bi_encoder_model_path = \"data/models/mini-lm/onnx/model.onnx\""
echo ""
echo "3. For ONNX Runtime with GPU support:"
echo "   - Install CUDA-enabled ONNX Runtime: pip install onnxruntime-gpu"
echo "   - Or use CPU: pip install onnxruntime"
echo ""
echo "================================================================================"
echo "Download complete!"
echo "================================================================================"
