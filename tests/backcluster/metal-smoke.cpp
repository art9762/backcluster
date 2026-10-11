// Require shader initialization and a real Metal graph, even if --help exits 0.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <cmath>
#include <cstdio>

struct resources {
    ggml_backend_t backend = nullptr;
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ~resources() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
        if (backend) ggml_backend_free(backend);
    }
};

int main() {
    const auto reg = ggml_backend_metal_reg();
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        std::fprintf(stderr, "Metal smoke failed: shader library has no usable device\n");
        return 1;
    }

    resources state;
    state.backend = ggml_backend_metal_init();
    if (!state.backend || !ggml_backend_is_metal(state.backend)) {
        std::fprintf(stderr, "Metal smoke failed: backend initialization\n");
        return 1;
    }
    const ggml_init_params params = {2 * 1024 * 1024, nullptr, true};
    state.context = ggml_init(params);
    if (!state.context) return 1;

    auto * a = ggml_new_tensor_2d(state.context, GGML_TYPE_F32, 4, 4);
    auto * b = ggml_new_tensor_2d(state.context, GGML_TYPE_F32, 4, 4);
    // Hosted Macs expose a paravirtual GPU without SIMD-group matrix multiply.
    // Addition still initializes and executes a real Metal compute pipeline.
    auto * result = ggml_add(state.context, a, b);
    if (!ggml_backend_supports_op(state.backend, result)) {
        std::fprintf(stderr, "Metal smoke failed: f32 addition is unsupported\n");
        return 1;
    }
    auto * graph = ggml_new_graph_custom(state.context, 16, false);
    ggml_build_forward_expand(graph, result);
    state.buffer = ggml_backend_alloc_ctx_tensors(state.context, state.backend);
    if (!state.buffer) {
        std::fprintf(stderr, "Metal smoke failed: tensor allocation\n");
        return 1;
    }

    float increment[16];
    float input[16];
    for (int i = 0; i < 16; ++i) {
        increment[i] = 1.5f;
        input[i] = float(i - 8);
    }
    ggml_backend_tensor_set(a, increment, 0, sizeof(increment));
    ggml_backend_tensor_set(b, input, 0, sizeof(input));
    if (ggml_backend_graph_compute(state.backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "Metal smoke failed: graph execution\n");
        return 1;
    }
    float output[16];
    ggml_backend_tensor_get(result, output, 0, sizeof(output));
    for (int i = 0; i < 16; ++i) {
        if (!std::isfinite(output[i]) || std::fabs(output[i] - (input[i] + 1.5f)) > 1e-5f) {
            std::fprintf(stderr, "Metal smoke failed: result[%d]=%g, expected %g\n",
                         i, double(output[i]), double(input[i] + 1.5f));
            return 1;
        }
    }
    std::puts("Metal shader initialization and f32 addition graph passed");
    return 0;
}
