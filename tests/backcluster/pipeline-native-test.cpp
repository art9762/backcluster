#include "llama.h"
#include "ggml.h"
#include "tail-server.h"
#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * reason) { if (!ok) throw std::runtime_error(reason); }
static int greedy(const float * logits, int n) { return (int) (std::max_element(logits, logits + n) - logits); }

int main(int argc, char ** argv) {
    if (argc == 4 && std::string(argv[1]) == "--worker") {
        llama_backend_init();
        llama_log_set([](ggml_log_level level, const char * text, void *) { if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr); }, nullptr);
        spt::server_status status;
        spt::tail_server server(argv[3], &status, [](const std::string & text) { fprintf(stderr, "%s\n", text.c_str()); });
        server.on_hello = [](llama_context * context, int) { llama_set_n_threads(context, 1, 1); return std::string(); };
        const auto load_error = server.load();
        if (!load_error.empty()) { fprintf(stderr, "%s\n", load_error.c_str()); return 1; }
        const auto serve_error = server.serve(std::stoi(argv[2]));
        fprintf(stderr, "%s\n", serve_error.c_str()); return 1;
    }
    if (argc != 9) { fprintf(stderr, "usage: test FULL MIDDLE TAIL L M SHA256 MIDDLE_PID TAIL_PID\n"); return 2; }
    llama_log_set([](ggml_log_level level, const char * text, void *) { if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr); }, nullptr);
    llama_backend_init();
    llama_model * model = nullptr;
    llama_context * reference = nullptr, * split = nullptr;
    try {
        unsetenv("LLAMA_SPLIT_DECODE"); unsetenv("LLAMA_SPLIT_TAIL"); unsetenv("LLAMA_SPLIT_MIDDLE");
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
        model = llama_model_load_from_file(argv[1], mp);
        require(model, "full model failed to load");
        auto cp = llama_context_default_params();
        cp.n_ctx = 128; cp.n_batch = 32; cp.n_ubatch = 4; cp.n_seq_max = 1;
        cp.n_threads = 1; cp.n_threads_batch = 1; cp.no_perf = true;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        reference = llama_init_from_model(model, cp);
        require(reference, "reference context failed");
        setenv("LLAMA_SPLIT_DECODE", "1", 1);
        if (std::atoi(argv[5]) > 0) setenv("LLAMA_SPLIT_MIDDLE", argv[2], 1);
        setenv("LLAMA_SPLIT_TAIL", argv[3], 1); setenv("LLAMA_SPLIT_L", argv[4], 1);
        setenv("LLAMA_SPLIT_MIDDLE_END", argv[5], 1); setenv("LLAMA_SPLIT_MODEL_SHA256", argv[6], 1);
        setenv("LLAMA_SPLIT_TIMEOUT_S", "5", 1); setenv("LLAMA_SPLIT_VERBOSE", "1", 1);
        split = llama_init_from_model(model, cp);
        require(split, "pipeline context failed");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        int pos = 0, checks = 0; float max_difference = 0;
        auto append = [&](int n, bool output, int seed) {
            llama_batch b = llama_batch_init(n, 0, 1); b.n_tokens = n;
            for (int i = 0; i < n; ++i) {
                b.token[i] = (seed + i * 3) % vocab; b.pos[i] = pos + i;
                b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = output && i == n - 1;
            }
            const int a = llama_decode(reference, b), c = llama_decode(split, b);
            llama_batch_free(b); require(a == 0 && c == 0, "decode failed"); pos += n;
            require(llama_memory_seq_pos_max(llama_get_memory(split), 0) == pos - 1, "head watermark mismatch");
            if (output) {
                const float * r = llama_get_logits_ith(reference, -1), * s = llama_get_logits_ith(split, -1);
                for (int j = 0; j < vocab; ++j) {
                    require(std::isfinite(r[j]) && std::isfinite(s[j]), "nonfinite logits");
                    max_difference = std::max(max_difference, std::fabs(r[j] - s[j]));
                    if (std::fabs(r[j] - s[j]) > 2e-4f + 2e-4f * std::fabs(r[j])) {
                        fprintf(stderr, "pos %d vocab %d reference %.9g split %.9g diff %.9g\n", pos, j, r[j], s[j], r[j] - s[j]);
                        throw std::runtime_error("logits numerical mismatch");
                    }
                }
                require(greedy(r, vocab) == greedy(s, vocab), "greedy token mismatch"); checks++;
            }
        };
        append(13, true, 1);
        for (int i = 0; i < 8; ++i) append(1, true, greedy(llama_get_logits_ith(reference, -1), vocab));
        append(7, false, 5); append(1, true, 9); append(9, true, 3);
        // Reset all owners, then check the same prefix and decode sequence again.
        llama_memory_clear(llama_get_memory(reference), true); llama_memory_clear(llama_get_memory(split), true); pos = 0;
        append(11, true, 4); append(1, true, 2);
        for (int p = std::atoi(argv[5]) > 0 ? 7 : 8; p <= 8; ++p) {
            const pid_t pid = (pid_t) std::stoi(argv[p]);
            require(kill(pid, SIGSTOP) == 0, "worker suspend failed");
            llama_token token = 1;
            const auto before = std::chrono::steady_clock::now();
            const int rc = llama_decode(split, llama_batch_get_one(&token, 1));
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - before).count();
            kill(pid, SIGCONT);
            require(rc != 0 && elapsed >= 4.5 && elapsed < 8, "worker failure did not honor its 5-second timeout");
            require(llama_decode(split, llama_batch_get_one(&token, 1)) != 0, "lost state silently continued");
            llama_memory_clear(llama_get_memory(reference), true); llama_memory_clear(llama_get_memory(split), true); pos = 0;
            append(13, true, 1); append(1, true, 6);
        }
        const pid_t tail_pid = (pid_t) std::stoi(argv[8]);
        require(kill(tail_pid, SIGSTOP) == 0, "cancel test worker suspend failed");
        std::atomic<bool> canceled{false};
        llama_set_abort_callback(split, [](void * data) { return ((std::atomic<bool> *) data)->load(); }, &canceled);
        std::thread trigger([&] { std::this_thread::sleep_for(std::chrono::milliseconds(100)); canceled.store(true); });
        llama_batch cancel_batch = llama_batch_init(17, 0, 1); cancel_batch.n_tokens = 17;
        for (int i = 0; i < 17; ++i) {
            cancel_batch.token[i] = i % vocab; cancel_batch.pos[i] = pos + i;
            cancel_batch.n_seq_id[i] = 1; cancel_batch.seq_id[i][0] = 0; cancel_batch.logits[i] = i == 16;
        }
        const auto cancel_start = std::chrono::steady_clock::now();
        const int canceled_rc = llama_decode(split, cancel_batch);
        llama_batch_free(cancel_batch);
        const double cancel_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - cancel_start).count();
        trigger.join(); kill(tail_pid, SIGCONT);
        llama_set_abort_callback(split, nullptr, nullptr);
        require(canceled_rc != 0 && cancel_s < 0.75, "native cancellation waited for the socket timeout");
        llama_memory_clear(llama_get_memory(reference), true); llama_memory_clear(llama_get_memory(split), true); pos = 0;
        append(5, true, 2);
        printf("PASS: %s prompt, append, greedy, reset, cancellation and worker failures; %d logits checks, max_abs_diff %.9g\n", std::atoi(argv[5]) > 0 ? "three-stage" : "legacy two-stage", checks, max_difference);
        llama_free(split); llama_free(reference); llama_model_free(model); llama_backend_free(); return 0;
    } catch (const std::exception & e) {
        for (int p = 7; p < argc; ++p) kill((pid_t) std::atoi(argv[p]), SIGCONT);
        fprintf(stderr, "FAIL: %s\n", e.what());
        if (split) llama_free(split);
        if (reference) llama_free(reference);
        if (model) llama_model_free(model);
        llama_backend_free(); return 1;
    }
}
