/*
 * http.c — Minimal HTTP server (POSIX sockets, no external libs).
 *
 * ds4: a ~200 line HTTP server.  POST /v1/completions for generation,
 *   GET /health for health check.
 */
#include "tinyllm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>

/* ── Minimal HTTP response helpers ───────────────────────────────── */
static void http_respond(int fd, int code, const char *status,
                         const char *content_type, const char *body) {
    char buf[8192];
    int len = snprintf(buf, sizeof(buf),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n%s",
        code, status, content_type, strlen(body), body);
    write(fd, buf, len);
}

/* ── Extremely minimal JSON parsing for completion request ───────── */
static int extract_json_field(const char *json, const char *key,
                              char *out, int max_len) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    p++; while (*p == ' ' || *p == '"') p++;

    int i = 0;
    while (*p && *p != '"' && *p != '\n' && *p != ',' && *p != '}' && i < max_len-1) {
        if (*p == '\\' && *(p+1) == 'n') { out[i++] = '\n'; p += 2; continue; }
        out[i++] = *p++;
    }
    out[i] = '\0';
    return i;
}

/* ── Parse completion request body ───────────────────────────────── */
static void handle_completion(tl_infer_t *inf, int fd, const char *body) {
    char prompt[16384] = {0};
    extract_json_field(body, "prompt", prompt, sizeof(prompt));

    int max_tokens = 512;
    char mt_str[32];
    if (extract_json_field(body, "max_tokens", mt_str, sizeof(mt_str)))
        max_tokens = atoi(mt_str);

    float temp = TL_DEFAULT_TEMP;
    char t_str[32];
    if (extract_json_field(body, "temperature", t_str, sizeof(t_str)))
        temp = strtof(t_str, NULL);

    /* Safety: validate tokenizer and model */
    if (!inf || !inf->tokenizer || !inf->model) {
        http_respond(fd, 503, "Service Unavailable",
            "application/json",
            "{\"error\":\"Model or tokenizer not loaded\"}");
        return;
    }

    if (strlen(prompt) == 0) {
        http_respond(fd, 400, "Bad Request",
            "application/json",
            "{\"error\":\"Empty prompt\"}");
        return;
    }

    /* Set temperature */
    inf->sampler.temperature = temp;

    /* Tokenize prompt */
    tl_token_t *tokens = tl_alloc(TL_MAX_SEQ_LEN * sizeof(tl_token_t));
    int prompt_len = tl_tokenize(inf->tokenizer, prompt, tokens, TL_MAX_SEQ_LEN);

    /* Clear KV cache for new request */
    tl_kv_cache_clear(inf);

    /* Collect generated text */
    size_t cap = 65536;
    char *result = tl_alloc(cap);

    /* Run generation */
    int n_gen = tl_generate(inf, tokens, prompt_len, max_tokens, NULL, NULL);

    /* Detokenize generated part */
    tl_token_t *gen = inf->generated + prompt_len;
    char *gen_text = tl_detokenize(inf->tokenizer, gen, n_gen);

    /* Build JSON response */
    /* Minimal: {"text": "..."} */
    /* Escape gen_text for JSON */
    size_t gt_len = strlen(gen_text);
    char *escaped = tl_alloc(gt_len * 2 + 1);
    size_t epos = 0;
    for (size_t i = 0; i < gt_len; i++) {
        if (gen_text[i] == '"' || gen_text[i] == '\\')
            escaped[epos++] = '\\';
        if (gen_text[i] == '\n')
            escaped[epos++] = '\\', escaped[epos++] = 'n';
        else
            escaped[epos++] = gen_text[i];
    }
    escaped[epos] = '\0';

    snprintf(result, cap,
        "{\"text\":\"%s\",\"tokens_generated\":%d}",
        escaped, n_gen);

    http_respond(fd, 200, "OK", "application/json", result);

    tl_free(tokens);
    tl_free(result);
    tl_free(escaped);
    tl_free(gen_text);
}

/* ── Handle agent task ───────────────────────────────────────────── */
static void handle_agent_task(tl_agent_t *agent, int fd, const char *body) {
    char task[16384] = {0};
    extract_json_field(body, "task", task, sizeof(task));

    if (!task[0]) {
        http_respond(fd, 400, "Bad Request", "application/json",
                     "{\"error\":\"Missing 'task' field\"}");
        return;
    }

    char *result = tl_agent_run(agent, task);

    /* Escape for JSON */
    size_t len = strlen(result);
    char *escaped = tl_alloc(len * 2 + 1);
    size_t ep = 0;
    for (size_t i = 0; i < len; i++) {
        if (result[i] == '"' || result[i] == '\\')
            escaped[ep++] = '\\';
        if (result[i] == '\n')
            escaped[ep++] = '\\', escaped[ep++] = 'n';
        else
            escaped[ep++] = result[i];
    }
    escaped[ep] = '\0';

    char *resp = tl_alloc(len * 2 + 256);
    snprintf(resp, len * 2 + 256, "{\"result\":\"%s\"}", escaped);

    http_respond(fd, 200, "OK", "application/json", resp);

    tl_free(result);
    tl_free(escaped);
    tl_free(resp);
}

/* ═══════════════════════════════════════════════════════════════════
   HTTP Server
   ═══════════════════════════════════════════════════════════════════ */

tl_http_server_t *tl_http_create(int port, tl_infer_t *infer, tl_agent_t *agent) {
    tl_http_server_t *srv = tl_calloc(1, sizeof(tl_http_server_t));
    srv->port  = port;
    srv->infer = infer;
    srv->agent = agent;

    srv->fd = socket(AF_INET, SOCK_STREAM, 0);
    if (srv->fd < 0) { tl_log("socket() failed"); tl_free(srv); return NULL; }

    /* Reuse address */
    int opt = 1;
    setsockopt(srv->fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port   = htons((uint16_t)port),
        .sin_addr.s_addr = INADDR_ANY,
    };

    if (bind(srv->fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        tl_log("bind() failed on port %d", port);
        close(srv->fd); tl_free(srv); return NULL;
    }

    if (listen(srv->fd, 8) < 0) {
        tl_log("listen() failed");
        close(srv->fd); tl_free(srv); return NULL;
    }

    srv->running = true;
    tl_log("HTTP server listening on http://localhost:%d", port);
    return srv;
}

void tl_http_free(tl_http_server_t *srv) {
    if (srv) { srv->running = false; close(srv->fd); tl_free(srv); }
}

int tl_http_listen(tl_http_server_t *srv) {
    /* Ignore SIGPIPE from broken connections */
    signal(SIGPIPE, SIG_IGN);

    while (srv->running) {
        struct sockaddr_in client;
        socklen_t client_len = sizeof(client);
        int cfd = accept(srv->fd, (struct sockaddr*)&client, &client_len);

        if (cfd < 0) { if (srv->running) continue; else break; }

        /* Read request (very simple, not production-grade) */
        char buf[65536];
        ssize_t n = read(cfd, buf, sizeof(buf) - 1);
        if (n <= 0) { close(cfd); continue; }
        buf[n] = '\0';

        /* Parse method and path */
        char method[16] = {0}, path[256] = {0};
        sscanf(buf, "%15s %255s", method, path);

        /* Find body (after \r\n\r\n) */
        char *body = strstr(buf, "\r\n\r\n");
        if (body) body += 4;

        /* Route */
        if (strcmp(path, "/health") == 0) {
            http_respond(cfd, 200, "OK", "application/json",
                         "{\"status\":\"ok\",\"model\":\"tinyllm\"}");
        } else if (strcmp(path, "/v1/completions") == 0 && strcmp(method, "POST") == 0) {
            if (body) handle_completion(srv->infer, cfd, body);
            else http_respond(cfd, 400, "Bad Request", "text/plain", "Missing body");
        } else if (strcmp(path, "/v1/agent") == 0 && strcmp(method, "POST") == 0) {
            if (body) handle_agent_task(srv->agent, cfd, body);
            else http_respond(cfd, 400, "Bad Request", "text/plain", "Missing body");
        } else {
            http_respond(cfd, 404, "Not Found", "application/json",
                         "{\"error\":\"Not found\"}");
        }
        close(cfd);
    }
    return 0;
}
