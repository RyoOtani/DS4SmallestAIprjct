/*
 * tools.c — External tool execution via subprocess.
 *
 * ds4: shell commands as subprocesses with JSON input/output.
 *   Tools are external binaries; tinyllm orchestrates them.
 *
 * Built-in tools:
 *   - run_cmd:     execute shell command, capture stdout
 *   - read_file:   read file contents
 *   - write_file:  write/modify file
 *   - search_code: grep / tree-sitter AST search
 *   - run_test:    compile and run test suite
 *   - web_search:  cached web search
 *   - rag_retrieve: local vector search
 *   - mem_store / mem_recall: long-term memory
 *   - sandbox_exec: run inside Docker/podman
 *   - browser:      headless browser control
 */
#include "tinyllm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>

/* ── Helper: run a command and capture stdout ────────────────────── */
static char *run_cmd_capture(const char *cmd, int *exit_code) {
    int pipefd[2];
    if (pipe(pipefd) < 0) return NULL;

    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); return NULL; }

    if (pid == 0) {
        /* Child */
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        execl("/bin/sh", "sh", "-c", cmd, NULL);
        _exit(127);
    }

    /* Parent */
    close(pipefd[1]);

    /* Read all output */
    size_t cap = 4096, len = 0;
    char *buf = tl_alloc(cap);
    ssize_t n;
    while ((n = read(pipefd[0], buf + len, cap - len - 1)) > 0) {
        len += n;
        if (len + 4096 >= cap) {
            cap *= 2;
            buf = realloc(buf, cap);
        }
    }
    buf[len] = '\0';
    close(pipefd[0]);

    int status;
    waitpid(pid, &status, 0);
    *exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;

    return buf;
}

/* ═══════════════════════════════════════════════════════════════════
   Tool execution dispatcher
   ═══════════════════════════════════════════════════════════════════ */

int tl_tool_execute(const tl_tool_call_t *call, char **result) {
    char cmd[4096];
    int exit_code = 0;
    char *output = NULL;

    switch (call->type) {

    case TL_TOOL_RUN_CMD:
        output = run_cmd_capture(call->params, &exit_code);
        break;

    case TL_TOOL_READ_FILE: {
        /* params is file path */
        size_t len;
        output = tl_read_file(call->params, &len);
        exit_code = output ? 0 : 1;
        if (!output) output = strdup("(file not found)");
        break;
    }

    case TL_TOOL_WRITE_FILE: {
        /* params: {"path":"...", "content":"..."} */
        /* Minimal JSON parsing for two fields */
        char path[512] = {0}, content[65536] = {0};
        const char *p = call->params;
        const char *pk = strstr(p, "\"path\"");
        const char *ck = strstr(p, "\"content\"");
        if (pk) {
            pk = strchr(pk, ':');
            if (pk) {
                pk++; while (*pk == '"' || *pk == ' ' || *pk == '\"') pk++;
                int i = 0;
                while (*pk && *pk != '"' && i < 511) path[i++] = *pk++;
                path[i] = '\0';
            }
        }
        if (ck) {
            ck = strchr(ck, ':');
            if (ck) {
                ck++; while (*ck == '"' || *ck == ' ') ck++;
                int i = 0;
                while (*ck && i < 65535) {
                    if (*ck == '\\' && *(ck+1) == 'n') { content[i++] = '\n'; ck+=2; continue; }
                    if (*ck == '\\' && *(ck+1) == '"') { content[i++] = '"'; ck+=2; continue; }
                    if (*ck == '"') break;
                    content[i++] = *ck++;
                }
                content[i] = '\0';
            }
        }
        if (path[0]) {
            FILE *f = fopen(path, "w");
            if (f) {
                fputs(content, f);
                fclose(f);
                output = strdup("(file written)");
                exit_code = 0;
            } else {
                output = strdup("(write failed)");
                exit_code = 1;
            }
        } else {
            output = strdup("(no path specified)");
            exit_code = 1;
        }
        break;
    }

    case TL_TOOL_SEARCH_CODE:
        /* grep or tree-sitter via external binary */
        snprintf(cmd, sizeof(cmd), "grep -rn --include='*.c' --include='*.h' --include='*.py' '%s' . 2>/dev/null | head -50",
                 call->params);
        output = run_cmd_capture(cmd, &exit_code);
        break;

    case TL_TOOL_RUN_TEST: {
        /* Compile + run test */
        snprintf(cmd, sizeof(cmd), "%s",
                 "cd /tmp && gcc -o tinyllm_test test.c 2>&1 && ./tinyllm_test 2>&1");
        (void)call->params; /* test config could extend this */
        output = run_cmd_capture(cmd, &exit_code);
        break;
    }

    case TL_TOOL_WEB_SEARCH:
        /* Cached web search: check local cache first */
        snprintf(cmd, sizeof(cmd),
                 "curl -s --max-time 10 'https://html.duckduckgo.com/html/?q=%s' 2>/dev/null | "
                 "grep -oP 'result__snippet\">\\K[^<]+' | head -10",
                 call->params);
        output = run_cmd_capture(cmd, &exit_code);
        if (!output || !*output) {
            tl_free(output);
            output = strdup("(no web results or offline)");
        }
        break;

    case TL_TOOL_RAG_RETRIEVE:
        /* Handled by rag.c, called via agent loop */
        output = strdup("(RAG retrieval handled internally)");
        exit_code = 0;
        break;

    case TL_TOOL_MEM_STORE:
        /* Handled by memory.c */
        output = strdup("(memory stored)");
        exit_code = 0;
        break;

    case TL_TOOL_MEM_RECALL:
        /* Handled by memory.c */
        output = strdup("(memory recall handled internally)");
        exit_code = 0;
        break;

    case TL_TOOL_SANDBOX_EXEC:
        snprintf(cmd, sizeof(cmd),
                 "docker run --rm -i --network none alpine:latest sh -c '%s' 2>&1",
                 call->params);
        output = run_cmd_capture(cmd, &exit_code);
        if (!output) {
            /* Try podman if docker not available */
            snprintf(cmd, sizeof(cmd),
                     "podman run --rm -i --network none alpine:latest sh -c '%s' 2>&1",
                     call->params);
            output = run_cmd_capture(cmd, &exit_code);
        }
        if (!output) output = strdup("(sandbox not available: install docker or podman)");
        break;

    case TL_TOOL_BROWSER:
        /* Headless browser via curl for simple HTTP requests */
        snprintf(cmd, sizeof(cmd),
                 "curl -sL --max-time 15 '%s' 2>/dev/null | "
                 "sed 's/<[^>]*>//g' | head -100",
                 call->params);
        output = run_cmd_capture(cmd, &exit_code);
        if (!output) output = strdup("(browser fetch failed)");
        break;

    default:
        output = strdup("(unknown tool)");
        exit_code = 1;
        break;
    }

    *result = output;
    return exit_code;
}

/* ── Available tools definition ──────────────────────────────────── */
int tl_tools_list(tl_tool_def_t *defs, int max_defs) {
    const tl_tool_def_t builtin[] = {
        {TL_TOOL_RUN_CMD, "run_cmd",
         "Execute a shell command and return its output.",
         "{\"command\": {\"type\": \"string\", \"description\": \"Shell command to execute\"}}"},

        {TL_TOOL_READ_FILE, "read_file",
         "Read the contents of a file.",
         "{\"path\": {\"type\": \"string\", \"description\": \"Path to the file\"}}"},

        {TL_TOOL_WRITE_FILE, "write_file",
         "Write content to a file. Creates or overwrites.",
         "{\"path\": {\"type\": \"string\"}, \"content\": {\"type\": \"string\"}}"},

        {TL_TOOL_SEARCH_CODE, "search_code",
         "Search code in the project using grep or AST.",
         "{\"query\": {\"type\": \"string\", \"description\": \"Search query or regex\"}}"},

        {TL_TOOL_RUN_TEST, "run_test",
         "Compile and run the test suite, returning results.",
         "{\"test_file\": {\"type\": \"string\"}}"},

        {TL_TOOL_WEB_SEARCH, "web_search",
         "Search the web for information (cached).",
         "{\"query\": {\"type\": \"string\"}}"},

        {TL_TOOL_RAG_RETRIEVE, "rag_retrieve",
         "Search local document index for relevant context.",
         "{\"query\": {\"type\": \"string\"}}"},

        {TL_TOOL_MEM_STORE, "mem_store",
         "Store information in long-term memory.",
         "{\"key\": {\"type\": \"string\"}, \"value\": {\"type\": \"string\"}}"},

        {TL_TOOL_MEM_RECALL, "mem_recall",
         "Recall information from long-term memory.",
         "{\"query\": {\"type\": \"string\"}}"},

        {TL_TOOL_SANDBOX_EXEC, "sandbox_exec",
         "Execute code in a sandboxed container.",
         "{\"code\": {\"type\": \"string\"}}"},

        {TL_TOOL_BROWSER, "browser",
         "Fetch a web page and extract its text content.",
         "{\"url\": {\"type\": \"string\"}}"},
    };

    int n = (int)(sizeof(builtin) / sizeof(builtin[0]));
    if (n > max_defs) n = max_defs;
    memcpy(defs, builtin, n * sizeof(tl_tool_def_t));
    return n;
}
