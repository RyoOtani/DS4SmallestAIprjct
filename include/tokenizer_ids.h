/*
 * tokenizer_ids.h — Canonical special token IDs (Single Source of Truth)
 *
 * MUST match:
 *   1. tokenizer/tokenizer_config.json (Python HuggingFace)
 *   2. tokenizer.tokbin (binary export for C runtime)
 *   3. create_tokenizer.py (tokenizer generation)
 *
 * These IDs are RESERVED (0..NUM_SPECIALS-1) and must NEVER be used
 * for regular BPE tokens in any vocab.
 */

#ifndef TINYLLM_TOKENIZER_IDS_H
#define TINYLLM_TOKENIZER_IDS_H

#define TL_NUM_SPECIAL_TOKENS  21

/* ── Canonical special token IDs ─────────────────────────────── */
#define TL_BOS_ID              0
#define TL_EOS_ID              1
#define TL_PAD_ID              2
#define TL_UNK_ID              3

/* FIM (Fill-in-the-Middle) */
#define TL_FIM_PREFIX_ID       4
#define TL_FIM_SUFFIX_ID       5
#define TL_FIM_MIDDLE_ID       6
#define TL_FIM_HOLE_ID         7
#define TL_FIM_PAD_ID          8

/* Repository / File context */
#define TL_REPO_NAME_ID        9
#define TL_FILE_SEP_ID         10
#define TL_FILE_PATH_ID        11

/* Tool calling */
#define TL_TOOL_CALL_ID        12
#define TL_TOOL_CALL_END_ID    13
#define TL_TOOL_RESPONSE_ID    14
#define TL_TOOL_RESPONSE_END_ID 15

/* Scratchpad (chain-of-thought) */
#define TL_SCRATCHPAD_ID       16
#define TL_SCRATCHPAD_END_ID   17

/* Chat role tokens */
#define TL_SYSTEM_ID           18
#define TL_USER_ID             19
#define TL_ASSISTANT_ID        20

/* ── Canonical special token strings ─────────────────────────── */
#define TL_SPECIAL_STR_BOS           "<s>"
#define TL_SPECIAL_STR_EOS           "</s>"
#define TL_SPECIAL_STR_PAD           "<pad>"
#define TL_SPECIAL_STR_UNK           "<unk>"
#define TL_SPECIAL_STR_FIM_PREFIX    "<fim_prefix>"
#define TL_SPECIAL_STR_FIM_SUFFIX    "<fim_suffix>"
#define TL_SPECIAL_STR_FIM_MIDDLE    "<fim_middle>"
#define TL_SPECIAL_STR_FIM_HOLE      "<fim_hole>"
#define TL_SPECIAL_STR_FIM_PAD       "<fim_pad>"
#define TL_SPECIAL_STR_REPO_NAME     "<repo_name>"
#define TL_SPECIAL_STR_FILE_SEP      "<file_sep>"
#define TL_SPECIAL_STR_FILE_PATH     "<file_path>"
#define TL_SPECIAL_STR_TOOL_CALL     "<tool_call>"
#define TL_SPECIAL_STR_TOOL_CALL_END "</tool_call>"
#define TL_SPECIAL_STR_TOOL_RESP     "<tool_response>"
#define TL_SPECIAL_STR_TOOL_RESP_END "</tool_response>"
#define TL_SPECIAL_STR_SCRATCHPAD    "<scratchpad>"
#define TL_SPECIAL_STR_SCRATCHPAD_END "</scratchpad>"
#define TL_SPECIAL_STR_SYSTEM        "<|system|>"
#define TL_SPECIAL_STR_USER          "<|user|>"
#define TL_SPECIAL_STR_ASSISTANT     "<|assistant|>"

/* ── Helper: check if a token ID is a special token ───────────── */
static inline int tl_is_special(tl_token_t id) {
    return (id >= 0 && id < TL_NUM_SPECIAL_TOKENS);
}

static inline int tl_is_fim(tl_token_t id) {
    return (id >= TL_FIM_PREFIX_ID && id <= TL_FIM_PAD_ID);
}

#endif /* TINYLLM_TOKENIZER_IDS_H */
