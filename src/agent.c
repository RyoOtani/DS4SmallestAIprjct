/*
 * agent.c — Self-correcting agent loop.
 *
 * ds4 philosophy: the model thinks, calls tools, observes results,
 * and retries.  No complex planning framework — just a loop.
 *
 * Flow:
 *   PLAN → THINK → ACT(tool call) → OBSERVE → (repeat or DONE)
 *
 * Maximum self-correction iterations: TL_MAX_SELF_CORRECT (5)
 */
#include "tinyllm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── System prompt template ──────────────────────────────────────── */
static const char *SYSTEM_PROMPT =
    "You are tinyllm, a coding AI assistant inspired by the ds4 philosophy. "
    "You think step by step, use tools when needed, and self-correct errors.\n\n"
    "## Available Tools\n"
    "You can call tools using this format:\n"
    "<tool_call>\n"
    "<name>tool_name</name>\n"
    "<params>JSON parameters</params>\n"
    "</tool_call>\n\n"
    "## Self-Correction\n"
    "When a tool returns an error, fix the issue and retry up to 3 times.\n"
    "Write your thinking in a <scratchpad> block.\n\n"
    "## Output Format\n"
    "Final answers should be clear and complete.\n";

/* ── Build tool definitions into prompt ──────────────────────────── */
static void append_tools_to_prompt(char *buf, size_t *pos, size_t cap) {
    tl_tool_def_t defs[16];
    int n = tl_tools_list(defs, 16);

    *pos += snprintf(buf + *pos, cap - *pos, "\n## Tools\n");
    for (int i = 0; i < n; i++) {
        *pos += snprintf(buf + *pos, cap - *pos,
                        "- **%s**: %s\n  Params: %s\n",
                        defs[i].name, defs[i].description, defs[i].params_json);
    }
}

/* ═══════════════════════════════════════════════════════════════════
   Agent context
   ═══════════════════════════════════════════════════════════════════ */

tl_agent_t *tl_agent_create(tl_infer_t *infer) {
    tl_agent_t *agent = tl_calloc(1, sizeof(tl_agent_t));
    agent->infer         = infer;
    agent->max_tool_calls = TL_MAX_TOOL_CALLS;
    agent->max_retries    = TL_MAX_SELF_CORRECT;
    agent->state          = TL_AGENT_IDLE;
    agent->tool_calls     = tl_calloc(TL_MAX_TOOL_CALLS, sizeof(tl_tool_call_t));
    return agent;
}

void tl_agent_free(tl_agent_t *agent) {
    if (!agent) return;
    for (int i = 0; i < agent->n_tool_calls; i++) {
        tl_free(agent->tool_calls[i].tool_name);
        tl_free(agent->tool_calls[i].params);
        tl_free(agent->tool_calls[i].result);
    }
    tl_free(agent->tool_calls);
    tl_free(agent->task);
    tl_free(agent->plan);
    tl_free(agent);
}

/* ═══════════════════════════════════════════════════════════════════
   Parse tool calls from model output
   ═══════════════════════════════════════════════════════════════════ */

static int parse_tool_calls(const char *text, tl_tool_call_t *calls, int max_calls) {
    int n = 0;
    const char *p = text;

    while (n < max_calls && (p = strstr(p, "<tool_call>"))) {
        p += 11; /* skip <tool_call> */

        /* Find <name> */
        const char *name_start = strstr(p, "<name>");
        const char *name_end   = strstr(p, "</name>");
        if (name_start && name_end && name_end > name_start) {
            name_start += 6;
            size_t name_len = name_end - name_start;
            calls[n].tool_name = tl_alloc(name_len + 1);
            memcpy(calls[n].tool_name, name_start, name_len);
            calls[n].tool_name[name_len] = '\0';
        }

        /* Find <params> */
        const char *param_start = strstr(p, "<params>");
        const char *param_end   = strstr(p, "</params>");
        if (param_start && param_end && param_end > param_start) {
            param_start += 8;
            size_t param_len = param_end - param_start;
            calls[n].params = tl_alloc(param_len + 1);
            memcpy(calls[n].params, param_start, param_len);
            calls[n].params[param_len] = '\0';
        }

        /* Match tool name to type */
        if (calls[n].tool_name) {
            if (strcmp(calls[n].tool_name, "run_cmd") == 0) calls[n].type = TL_TOOL_RUN_CMD;
            else if (strcmp(calls[n].tool_name, "read_file") == 0) calls[n].type = TL_TOOL_READ_FILE;
            else if (strcmp(calls[n].tool_name, "write_file") == 0) calls[n].type = TL_TOOL_WRITE_FILE;
            else if (strcmp(calls[n].tool_name, "search_code") == 0) calls[n].type = TL_TOOL_SEARCH_CODE;
            else if (strcmp(calls[n].tool_name, "run_test") == 0) calls[n].type = TL_TOOL_RUN_TEST;
            else if (strcmp(calls[n].tool_name, "web_search") == 0) calls[n].type = TL_TOOL_WEB_SEARCH;
            else if (strcmp(calls[n].tool_name, "rag_retrieve") == 0) calls[n].type = TL_TOOL_RAG_RETRIEVE;
            else if (strcmp(calls[n].tool_name, "mem_store") == 0) calls[n].type = TL_TOOL_MEM_STORE;
            else if (strcmp(calls[n].tool_name, "mem_recall") == 0) calls[n].type = TL_TOOL_MEM_RECALL;
            else if (strcmp(calls[n].tool_name, "sandbox_exec") == 0) calls[n].type = TL_TOOL_SANDBOX_EXEC;
            else if (strcmp(calls[n].tool_name, "browser") == 0) calls[n].type = TL_TOOL_BROWSER;
        }
        n++;
        p = strstr(p, "</tool_call>");
        if (!p) break;
        p += 12;
    }
    return n;
}

/* ═══════════════════════════════════════════════════════════════════
   Agent run: plan → think → act → observe loop
   ═══════════════════════════════════════════════════════════════════ */

char *tl_agent_run(tl_agent_t *agent, const char *task) {
    tl_tokenizer_t *tok = agent->infer->tokenizer;
    tl_infer_t *inf     = agent->infer;

    /* Build full system + task prompt */
    size_t buf_cap = 65536;
    char *prompt_text = tl_alloc(buf_cap);
    size_t pos = 0;

    pos += snprintf(prompt_text + pos, buf_cap - pos, "%s", SYSTEM_PROMPT);
    append_tools_to_prompt(prompt_text, &pos, buf_cap);
    pos += snprintf(prompt_text + pos, buf_cap - pos,
                    "\n## Task\n%s\n\n## Response\n", task);

    /* Tokenize */
    int prompt_tokens_max = TL_MAX_SEQ_LEN;
    tl_token_t *prompt_tokens = tl_alloc(prompt_tokens_max * sizeof(tl_token_t));
    int prompt_len = tl_tokenize(tok, prompt_text, prompt_tokens, prompt_tokens_max);
    tl_free(prompt_text);

    /* Reset KV cache for new turn */
    tl_kv_cache_clear(inf);

    /* Generation buffer */
    size_t out_cap = 65536;
    char *output = tl_alloc(out_cap);
    size_t out_len = 0;

    /* Agent loop: up to max_retries iterations */
    int iteration;
    for (iteration = 0; iteration < agent->max_retries; iteration++) {
        agent->state = TL_AGENT_THINKING;

        /* Run generation */
        int new_tokens = tl_generate(inf, prompt_tokens, prompt_len,
                                      2048, NULL, NULL);

        /* Build text from generated tokens */
        tl_token_t *gen = inf->generated + prompt_len;
        char *gen_text = tl_detokenize(tok, gen, new_tokens);
        if (!gen_text) continue;

        /* Parse tool calls */
        tl_tool_call_t calls[16] = {0};
        int n_calls = parse_tool_calls(gen_text, calls, 16);

        if (n_calls == 0) {
            /* No tool calls → agent is done */
            agent->state = TL_AGENT_DONE;
            /* Accumulate output */
            size_t add_len = strlen(gen_text);
            if (out_len + add_len + 1 > out_cap) {
                out_cap *= 2;
                output = realloc(output, out_cap);
            }
            memcpy(output + out_len, gen_text, add_len);
            out_len += add_len;
            tl_free(gen_text);
            break;
        }

        /* Execute tools */
        agent->state = TL_AGENT_CALLING_TOOL;
        for (int i = 0; i < n_calls && i < agent->max_tool_calls; i++) {
            int ec = tl_tool_execute(&calls[i], &calls[i].result);
            calls[i].exit_code = ec;
            calls[i].executed = true;

            /* Accumulate tool results */
            size_t add = snprintf(NULL, 0, "\n[Tool: %s]\n%s\n",
                                  calls[i].tool_name ? calls[i].tool_name : "?",
                                  calls[i].result ? calls[i].result : "");
            if (out_len + add + 1 > out_cap) {
                out_cap = out_cap * 2 + add;
                output = realloc(output, out_cap);
            }
            out_len += snprintf(output + out_len, out_cap - out_len,
                                "\n[Tool: %s]\n%s\n",
                                calls[i].tool_name ? calls[i].tool_name : "?",
                                calls[i].result ? calls[i].result : "");
        }

        agent->state = TL_AGENT_OBSERVING;
        agent->n_tool_calls = n_calls;

        /* Append tool results to prompt for next iteration */
        /* (In a full implementation, we'd extend the prompt with
           tool results and re-generate) */

        /* Cleanup this iteration's calls */
        for (int i = 0; i < n_calls; i++) {
            tl_free(calls[i].tool_name);
            tl_free(calls[i].params);
            tl_free(calls[i].result);
        }
        tl_free(gen_text);
    }

    tl_free(prompt_tokens);
    output[out_len] = '\0';
    return output;
}
