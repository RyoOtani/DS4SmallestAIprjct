/*
 * main.c — tinyllm entry point.
 *
 * Usage:
 *   tinyllm run <model.gguf>              # CLI interactive mode
 *   tinyllm serve <model.gguf> [port]     # HTTP API server
 *   tinyllm agent <model.gguf> <task>      # Single agent task
 *   tinyllm index <dir>                   # Index directory for RAG
 *   tinyllm info <model.gguf>             # Print model info
 *
 * ds4: one binary, zero config, auto-adapts.
 */
#include "tinyllm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>

static void print_usage(const char *prog) {
    fprintf(stderr,
        "tinyllm — ds4 personal AI v1.0\n"
        "Usage:\n"
        "  %s run    <model.gguf>                  Interactive CLI\n"
        "  %s serve  <model.gguf> [port]           HTTP API server (default port %d)\n"
        "  %s agent  <model.gguf> <task>           Execute a single agent task\n"
        "  %s info   <model.gguf>                  Print model information\n"
        "  %s index  <directory>                   Index directory for RAG\n"
        "  %s daemon <model.gguf> [interval_s]     Long-running daemon mode\n"
        "\n"
        "Environment:\n"
        "  TINYLLM_TEMP=temperature                Sampling temperature (default %.1f)\n"
        "  TINYLLM_TOP_K=N                        Top-k sampling (default %d)\n"
        "  TINYLLM_MAX_TOKENS=N                   Max new tokens (default 2048)\n"
        "  TINYLLM_DAEMON_TASK=<file>              Daemon task file (default: daemon_task.txt)\n",
        prog, prog, TL_HTTP_PORT, prog, prog, prog, prog,
        TL_DEFAULT_TEMP, TL_DEFAULT_TOP_K);
}

/* ── Daemon mode: long-running autonomous agent ──────────────────── */
static int run_daemon(tl_agent_t *agent, int interval_s) {
    const char *task_file = getenv("TINYLLM_DAEMON_TASK");
    if (!task_file) task_file = "daemon_task.txt";

    tl_log("Daemon mode: checking %s every %ds", task_file, interval_s);
    tl_log("PID: %d", getpid());

    int64_t last_task_mtime = 0;
    char prev_summary[512] = "";

    while (1) {
        /* Check task file for updates */
        struct stat st;
        if (stat(task_file, &st) != 0) {
            sleep(interval_s);
            continue;
        }

        if (st.st_mtime == last_task_mtime) {
            sleep(interval_s);
            continue;
        }
        last_task_mtime = st.st_mtime;

        /* Read task file */
        size_t len;
        char *task = tl_read_file(task_file, &len);
        if (!task || len == 0) {
            tl_free(task);
            sleep(interval_s);
            continue;
        }

        /* Skip if same as previous */
        if (strcmp(task, prev_summary) == 0) {
            tl_free(task);
            sleep(interval_s);
            continue;
        }

        tl_log("New task detected (%zu bytes)", len);

        /* Execute agent */
        double t0 = tl_time_now();
        char *result = tl_agent_run(agent, task);
        double dt = tl_time_now() - t0;

        tl_log("Task completed in %.1fs", dt);

        /* Save result */
        char result_file[640];
        snprintf(result_file, sizeof(result_file), "%s.result", task_file);
        FILE *rf = fopen(result_file, "w");
        if (rf) {
            fprintf(rf, "## Task\n%s\n## Result\n%s\n## Duration\n%.1fs\n",
                    task, result, dt);
            fclose(rf);
        }

        /* Save summary for dedup */
        strncpy(prev_summary, task, sizeof(prev_summary)-1);
        tl_free(task);
        tl_free(result);

        sleep(interval_s);
    }
    return 0;
}

/* ── Interactive CLI mode ────────────────────────────────────────── */
static int run_cli(tl_infer_t *inf, tl_agent_t *agent) {
    printf("╔══════════════════════════════════╗\n");
    printf("║   tinyllm — ds4 personal AI     ║\n");
    printf("║   Type /quit to exit            ║\n");
    printf("║   Type /clear to reset context  ║\n");
    printf("╚══════════════════════════════════╝\n\n");

    char line[4096];

    while (1) {
        printf(">>> ");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) break;

        /* Strip newline */
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';

        if (len == 0) continue;

        /* Commands */
        if (strcmp(line, "/quit") == 0 || strcmp(line, "/q") == 0) break;
        if (strcmp(line, "/clear") == 0 || strcmp(line, "/c") == 0) {
            tl_kv_cache_clear(inf);
            printf("[Context cleared]\n");
            continue;
        }

        /* Run agent on the user's task */
        printf("\n");  /* spacing */

        double t0 = tl_time_now();
        char *result = tl_agent_run(agent, line);
        double dt = tl_time_now() - t0;

        printf("%s\n", result);
        printf("\n[%.1fs, %.1f GB RSS]\n\n",
               dt, tl_memory_usage() / (1024.0*1024.0*1024.0));

        tl_free(result);
    }

    printf("\nGoodbye.\n");
    return 0;
}

/* ── Main ────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char *cmd   = argv[1];
    const char *model_path = argv[2];

    /* ─── info ─────────────────────────────────────────────────── */
    if (strcmp(cmd, "info") == 0) {
        tl_model_t *m = tl_model_load(model_path);
        if (!m) return 1;
        tl_model_print_info(m);
        tl_model_free(m);
        return 0;
    }

    /* ─── index ────────────────────────────────────────────────── */
    if (strcmp(cmd, "index") == 0) {
        tl_rag_index_t *rag = tl_rag_create(512);
        int n = tl_rag_index_dir(rag, model_path); /* argv[2] is dir */
        tl_log("Indexed %d files from %s", n, model_path);
        tl_rag_free(rag);
        return 0;
    }

    /* ─── Load model for run/serve/agent ───────────────────────── */
    tl_log("Loading model: %s", model_path);
    tl_model_t *model = tl_model_load(model_path);
    if (!model) {
        tl_log("Failed to load model. Make sure the file is a valid GGUF.");
        return 1;
    }
    tl_model_print_info(model);

    /* Tokenizer */
    tl_tokenizer_t *tok = tl_tokenizer_load(model_path);

    /* Sampler (from env or defaults) */
    tl_sampler_t sampler = tl_sampler_default();
    const char *env_temp = getenv("TINYLLM_TEMP");
    if (env_temp) sampler.temperature = strtof(env_temp, NULL);
    const char *env_topk = getenv("TINYLLM_TOP_K");
    if (env_topk) sampler.top_k = atoi(env_topk);

    /* Inference context */
    tl_infer_t *infer = tl_infer_create(model, tok, sampler);
    if (!infer) { tl_model_free(model); return 1; }

    /* Agent */
    tl_agent_t *agent = tl_agent_create(infer);

    int ret = 0;

    /* ─── run (CLI) ────────────────────────────────────────────── */
    if (strcmp(cmd, "run") == 0) {
        ret = run_cli(infer, agent);
    }
    /* ─── serve (HTTP) ─────────────────────────────────────────── */
    else if (strcmp(cmd, "serve") == 0) {
        int port = TL_HTTP_PORT;
        if (argc >= 4) port = atoi(argv[3]);
        tl_http_server_t *http = tl_http_create(port, infer, agent);
        if (!http) { ret = 1; }
        else {
            ret = tl_http_listen(http);
            tl_http_free(http);
        }
    }
    /* ─── agent (single task) ──────────────────────────────────── */
    else if (strcmp(cmd, "agent") == 0) {
        if (argc < 4) {
            tl_log("Usage: %s agent <model.gguf> <task>", argv[0]);
            ret = 1;
        } else {
            char *result = tl_agent_run(agent, argv[3]);
            printf("%s\n", result);
            tl_free(result);
        }
    }
    /* ─── daemon (long-running) ────────────────────────────────── */
    else if (strcmp(cmd, "daemon") == 0) {
        int interval = 60; /* default: check every 60s */
        if (argc >= 4) interval = atoi(argv[3]);
        if (interval < 5) interval = 5;
        ret = run_daemon(agent, interval);
    }
    else {
        print_usage(argv[0]);
        ret = 1;
    }

    /* Cleanup */
    tl_agent_free(agent);
    tl_infer_free(infer);
    tl_tokenizer_free(tok);
    tl_model_free(model);

    return ret;
}
