#!/bin/bash
# =============================================================================
# Hugging Face Hub Push Script for TinyLLM
#
# Usage:
#   chmod +x push_to_hub.sh
#   ./push_to_hub.sh                  # Push all configured models
#   ./push_to_hub.sh nano             # Push only nano model
#   ./push_to_hub.sh --dry-run        # Dry run (don't actually push)
# =============================================================================

set -euo pipefail

HF_NAMESPACE="${HF_NAMESPACE:-RyoOtani}"
DRY_RUN=false
MODEL_FILTER="${1:-all}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Check prerequisites ──────────────────────────────────────────────────────

check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v huggingface-cli &> /dev/null; then
        log_error "huggingface-cli not found. Install it with:"
        echo "  pip install huggingface-hub"
        exit 1
    fi

    if ! huggingface-cli whoami &> /dev/null; then
        log_error "Not logged in to Hugging Face. Run:"
        echo "  huggingface-cli login"
        exit 1
    fi

    log_ok "All prerequisites satisfied"
}

# ── Push a single model ─────────────────────────────────────────────────────

push_model() {
    local scale="$1"
    local repo_id="${HF_NAMESPACE}/tinyllm-${scale}"

    log_info "============================================"
    log_info "Pushing model: ${repo_id}"
    log_info "============================================"

    # Create repo if it doesn't exist
    if ! huggingface-cli repo info "${repo_id}" &> /dev/null; then
        log_info "Creating repository: ${repo_id}"
        if [ "$DRY_RUN" = false ]; then
            huggingface-cli repo create "tinyllm-${scale}" \
                --type model \
                --organization "${HF_NAMESPACE}" \
                --yes 2>/dev/null || \
            huggingface-cli repo create "tinyllm-${scale}" \
                --type model \
                --yes 2>/dev/null || true
        fi
    fi

    # Build the list of files to upload
    local upload_files=()

    # GGUF model file
    if ls tinyllm-${scale}*.gguf 2>/dev/null; then
        for f in tinyllm-${scale}*.gguf; do
            upload_files+=("$f")
            log_info "Found GGUF: $f ($(du -h "$f" | cut -f1))"
        done
    else
        log_warn "No GGUF file found for ${scale} (not yet trained)"
    fi

    # SafeTensors
    if [ -f "checkpoints/tinyllm-${scale}/model.safetensors" ]; then
        upload_files+=("checkpoints/tinyllm-${scale}/model.safetensors")
        log_info "Found SafeTensors checkpoint"
    fi

    # Config and tokenizer
    if [ -f "model_config_${scale}.json" ]; then
        upload_files+=("model_config_${scale}.json")
    fi
    if [ -f "tokenizer.json" ]; then
        upload_files+=("tokenizer.json")
    fi

    # Model card (README.md)
    cp HUGGINGFACE_MODEL_CARD.md README_HF.md
    upload_files+=("README_HF.md")

    if [ ${#upload_files[@]} -eq 0 ]; then
        log_warn "No files to upload for ${scale}. Run training first."
        return 0
    fi

    # Upload
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would upload to ${repo_id}:"
        for f in "${upload_files[@]}"; do
            echo "  - $f"
        done
    else
        log_info "Uploading ${#upload_files[@]} files to ${repo_id}..."
        for f in "${upload_files[@]}"; do
            log_info "  Uploading: $f"
            huggingface-cli upload "${repo_id}" "$f" "$(basename "$f")" --repo-type model
        done
        log_ok "Successfully pushed ${repo_id}"
    fi
}

# ── Push config only (for pre-trained models) ────────────────────────────────

push_config() {
    local scale="$1"
    local repo_id="${HF_NAMESPACE}/tinyllm-${scale}"

    log_info "Pushing model config for: ${repo_id}"

    # Generate model config JSON
    python3 -c "
import json, sys
sys.path.insert(0, '.')
from model.config import get_config

cfg = get_config('${scale}')
config_dict = {
    'model_type': 'tinyllm',
    'architectures': ['TinyLLMModel'],
    'hidden_size': cfg.hidden_dim,
    'num_hidden_layers': cfg.n_layers,
    'num_attention_heads': cfg.n_heads,
    'num_key_value_heads': cfg.n_kv_heads,
    'head_dim': cfg.head_dim,
    'intermediate_size': cfg.ffn_inter_dim,
    'vocab_size': cfg.vocab_size,
    'max_position_embeddings': cfg.max_seq_len,
    'use_moe': cfg.use_moe,
    'num_experts': cfg.n_experts,
    'num_active_experts': cfg.n_active_experts,
    'expert_intermediate_size': cfg.expert_inter_dim,
    'use_mla': cfg.use_mla,
    'kv_latent_dim': cfg.kv_latent_dim,
    'use_mtp': cfg.use_mtp,
    'mtp_depth': cfg.mtp_depth,
    'norm_type': cfg.norm_type,
    'activation_function': cfg.ffn_activation,
    'rope_theta': cfg.rope_theta,
    'tie_word_embeddings': cfg.tie_word_embeddings,
    'torch_dtype': 'bfloat16',
    'transformers_version': '4.x',
}

with open('config_tmp_${scale}.json', 'w') as f:
    json.dump(config_dict, f, indent=2)
print(f'Config generated: {cfg.name} ({cfg.total_params_estimate//1e6:.0f}M params)')
"

    if [ "$DRY_RUN" = false ]; then
        huggingface-cli upload "${repo_id}" "config_tmp_${scale}.json" "config.json" --repo-type model
        rm -f "config_tmp_${scale}.json"
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────

main() {
    check_prerequisites

    # Parse arguments
    for arg in "$@"; do
        case "$arg" in
            --dry-run) DRY_RUN=true ;;
            *) MODEL_FILTER="$arg" ;;
        esac
    done

    MODELS=(
        "nano" "small" "medium" "large" "dense-7b"
        "xlarge" "xxlarge" "mega" "giga"
    )

    for scale in "${MODELS[@]}"; do
        if [ "$MODEL_FILTER" = "all" ] || [ "$MODEL_FILTER" = "$scale" ]; then
            # Push config (always works, no training needed)
            push_config "$scale"

            # Push model weights (only if files exist)
            push_model "$scale"

            echo ""
        fi
    done

    log_ok "============================================"
    log_ok "All done! Visit your models at:"
    for scale in "${MODELS[@]}"; do
        if [ "$MODEL_FILTER" = "all" ] || [ "$MODEL_FILTER" = "$scale" ]; then
            echo "  https://huggingface.co/${HF_NAMESPACE}/tinyllm-${scale}"
        fi
    done
    log_ok "============================================"
}

main "$@"
