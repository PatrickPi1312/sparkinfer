#include "sparkinfer/hf_config.h"
#include <cstdio>
#include <fstream>
#include <string>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { std::printf("FAIL: %s line %d\n", #x, __LINE__); return 1; } } while (0)

int main() {
    const char* dir = "/tmp/nvfp4_hf_cfg_test";
    mkdir(dir, 0755);
    {
        std::ofstream o(std::string(dir) + "/config.json");
        o << R"({
          "model_type": "qwen3_5",
          "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248320,
            "intermediate_size": 17408,
            "rms_norm_eps": 1e-6,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "partial_rotary_factor": 0.25,
            "rope_parameters": { "rope_theta": 10000000 }
          },
          "vision_config": { "hidden_size": 1152 },
          "quantization_config": { "quant_algo": "NVFP4" }
        })";
    }
    {
        std::ofstream o(std::string(dir) + "/hf_quant_config.json");
        o << R"({"quantization":{"quant_algo":"NVFP4"}})";
    }
    CHECK(sparkinfer::path_is_nvfp4_dir(dir));
    sparkinfer::Qwen35Config cfg;
    CHECK(sparkinfer::qwen3_config_from_hf_dir(dir, cfg));
    CHECK(cfg.hidden == 5120);
    CHECK(cfg.n_layers == 64);
    CHECK(cfg.n_q_heads == 24);
    CHECK(cfg.n_kv_heads == 4);
    CHECK(cfg.head_dim == 256);
    CHECK(cfg.vocab == 248320);
    CHECK(cfg.moe_ffn == 17408);
    CHECK(cfg.linear_q_heads == 16);
    CHECK(cfg.linear_v_heads == 48);
    CHECK(cfg.linear_head_dim == 128);
    CHECK(cfg.rope_dim == 64);
    CHECK(cfg.dense_ffn);
    CHECK(cfg.hybrid);
    CHECK(cfg.rope_theta > 1e6f);
    std::printf("hf_config_cpu_test: OK\n");
    return 0;
}
