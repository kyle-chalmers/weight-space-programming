#!/usr/bin/env bash
# One-time setup. Written for an Apple Silicon Mac; also runs on an NVIDIA GPU or CPU. Run from this folder: ./setup.sh
# Creates .venv, installs packages, fetches the paper's logs (for its published Qwen edit),
# and downloads the models into models/. Nothing here accepts a license or creates a token.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q "torch>=2.6" "transformers>=5.0" accelerate safetensors datasets numpy "huggingface_hub[cli]"
python -c "import torch; print('torch', torch.__version__, '| Apple GPU (MPS):', torch.backends.mps.is_available())"

# The paper's public logs: apply_edit.py, the edit files and the text-only Qwen3.5 config.
if [ ! -d vendor/lm_understand-logs ]; then
  mkdir -p vendor
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/RikiyaT/lm_understand-logs vendor/lm_understand-logs
fi

mkdir -p models
# 1) Qwen3.5-0.8B: the paper's model, pinned to the exact revision its edit was made on.
# (hf >= 1.0 takes one pattern per --include; extra bare patterns are read as file names.)
if ! ls models/Qwen3.5-0.8B-text/*.safetensors >/dev/null 2>&1; then
  hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
    --include "*.safetensors" --include "*.json" --include "tokenizer*" --include "*.jinja" --local-dir models/Qwen3.5-0.8B-text
fi
# Text-only config and chat template from the authors' export, so the kit loads the language model alone.
cp vendor/lm_understand-logs/exp8_math/base_model/Qwen3.5-0.8B-text/{config.json,generation_config.json,tokenizer_config.json,chat_template.jinja} models/Qwen3.5-0.8B-text/

# 2) Qwen2.5-0.5B-Instruct: ungated fallback from another release.
if ! ls models/Qwen2.5-0.5B-Instruct/*.safetensors >/dev/null 2>&1; then
  hf download Qwen/Qwen2.5-0.5B-Instruct --include "*.safetensors" --include "*.json" --include "tokenizer*" --include "*.txt" --local-dir models/Qwen2.5-0.5B-Instruct
fi

# 3) Llama 3.2 1B Instruct: gated by Meta. Accept Meta's license on Hugging Face and log in yourself.
if ! ls models/Llama-3.2-1B-Instruct/*.safetensors >/dev/null 2>&1; then
  if hf download meta-llama/Llama-3.2-1B-Instruct --include "*.safetensors" --include "*.json" --include "tokenizer*" \
       --local-dir models/Llama-3.2-1B-Instruct; then
    echo "Llama 3.2 1B downloaded."
  else
    echo ""
    echo "LLAMA NOT DOWNLOADED. It is gated. To use it:"
    echo "  1. Open the meta-llama/Llama-3.2-1B-Instruct page on huggingface.co while signed in and accept Meta's license."
    echo "  2. Run: hf auth login   (paste your own read token)"
    echo "  3. Re-run ./setup.sh"
    echo "Until then the kit runs with qwen2.5-0.5b as the second model."
  fi
fi
echo "Setup done. Models in models/:"; ls models
