#include "sparkinfer/hf_config.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>

namespace sparkinfer {
namespace {

std::string slurp(const std::string& path) {
    std::ifstream in(path);
    if (!in) return {};
    std::stringstream ss; ss << in.rdbuf();
    return ss.str();
}

bool file_exists(const std::string& p) {
    struct stat st{};
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

size_t find_key_object(const std::string& s, const std::string& key) {
    const std::string pat = "\"" + key + "\"";
    size_t pos = 0;
    while ((pos = s.find(pat, pos)) != std::string::npos) {
        size_t i = pos + pat.size();
        while (i < s.size() && isspace((unsigned char)s[i])) i++;
        if (i < s.size() && s[i] == ':') {
            i++;
            while (i < s.size() && isspace((unsigned char)s[i])) i++;
            if (i < s.size() && s[i] == '{') return i;
        }
        pos++;
    }
    return std::string::npos;
}

std::string extract_object_at(const std::string& s, size_t brace) {
    if (brace >= s.size() || s[brace] != '{') return {};
    int depth = 0;
    size_t i = brace;
    for (; i < s.size(); i++) {
        if (s[i] == '{') depth++;
        else if (s[i] == '}') {
            depth--;
            if (depth == 0) return s.substr(brace, i - brace + 1);
        }
    }
    return {};
}

int get_int(const std::string& obj, const std::string& key, int def) {
    const std::string pat = "\"" + key + "\"";
    size_t pos = obj.find(pat);
    if (pos == std::string::npos) return def;
    size_t i = pos + pat.size();
    while (i < obj.size() && obj[i] != ':') i++;
    if (i >= obj.size()) return def;
    i++;
    while (i < obj.size() && isspace((unsigned char)obj[i])) i++;
    if (i < obj.size() && obj[i] == 'n') return def;  // null
    char* end = nullptr;
    long v = strtol(obj.c_str() + i, &end, 10);
    if (end == obj.c_str() + i) return def;
    return (int)v;
}

double get_float(const std::string& obj, const std::string& key, double def) {
    const std::string pat = "\"" + key + "\"";
    size_t pos = obj.find(pat);
    if (pos == std::string::npos) return def;
    size_t i = pos + pat.size();
    while (i < obj.size() && obj[i] != ':') i++;
    if (i >= obj.size()) return def;
    i++;
    while (i < obj.size() && isspace((unsigned char)obj[i])) i++;
    char* end = nullptr;
    double v = strtod(obj.c_str() + i, &end);
    if (end == obj.c_str() + i) return def;
    return v;
}

bool get_bool(const std::string& obj, const std::string& key, bool def) {
    const std::string pat = "\"" + key + "\"";
    size_t pos = obj.find(pat);
    if (pos == std::string::npos) return def;
    size_t i = pos + pat.size();
    while (i < obj.size() && obj[i] != ':') i++;
    if (i >= obj.size()) return def;
    i++;
    while (i < obj.size() && isspace((unsigned char)obj[i])) i++;
    if (obj.compare(i, 4, "true") == 0) return true;
    if (obj.compare(i, 5, "false") == 0) return false;
    return def;
}

} // namespace

bool path_is_nvfp4_dir(const std::string& path) {
    struct stat st{};
    if (stat(path.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return false;
    if (!file_exists(path + "/config.json")) return false;
    if (file_exists(path + "/hf_quant_config.json")) {
        const std::string q = slurp(path + "/hf_quant_config.json");
        if (q.find("NVFP4") != std::string::npos) return true;
    }
    const std::string cfg = slurp(path + "/config.json");
    return cfg.find("NVFP4") != std::string::npos || cfg.find("\"quant_algo\"") != std::string::npos;
}

bool qwen3_config_from_hf_dir(const std::string& dir, Qwen35Config& cfg) {
    const std::string js = slurp(dir + "/config.json");
    if (js.empty()) {
        fprintf(stderr, "[hf] cannot read %s/config.json\n", dir.c_str());
        return false;
    }
    size_t brace = find_key_object(js, "text_config");
    const std::string text = (brace == std::string::npos) ? js : extract_object_at(js, brace);
    if (text.empty()) {
        fprintf(stderr, "[hf] no text_config in %s/config.json\n", dir.c_str());
        return false;
    }

    cfg.hidden = get_int(text, "hidden_size", cfg.hidden);
    cfg.n_layers = get_int(text, "num_hidden_layers", cfg.n_layers);
    cfg.n_q_heads = get_int(text, "num_attention_heads", cfg.n_q_heads);
    cfg.n_kv_heads = get_int(text, "num_key_value_heads", cfg.n_kv_heads);
    cfg.head_dim = get_int(text, "head_dim", cfg.head_dim);
    cfg.vocab = get_int(text, "vocab_size", cfg.vocab);
    cfg.moe_ffn = get_int(text, "intermediate_size", cfg.moe_ffn);
    cfg.rms_eps = (float)get_float(text, "rms_norm_eps", cfg.rms_eps);
    cfg.eos_id = get_int(text, "eos_token_id", cfg.eos_id);
    cfg.full_attn_interval = get_int(text, "full_attention_interval", 4);
    cfg.linear_conv_kernel = get_int(text, "linear_conv_kernel_dim", 4);
    cfg.linear_q_heads = get_int(text, "linear_num_key_heads", 16);
    cfg.linear_v_heads = get_int(text, "linear_num_value_heads", cfg.linear_v_heads);
    cfg.linear_head_dim = get_int(text, "linear_key_head_dim", 128);
    const float pr = (float)get_float(text, "partial_rotary_factor", 0.25);
    cfg.rope_dim = (int)(pr * (float)cfg.head_dim + 0.5f);
    if (cfg.rope_dim <= 0) cfg.rope_dim = 64;

    size_t rp = find_key_object(text, "rope_parameters");
    if (rp != std::string::npos) {
        const std::string rope = extract_object_at(text, rp);
        cfg.rope_theta = (float)get_float(rope, "rope_theta", 10000000.0);
    } else {
        cfg.rope_theta = (float)get_float(text, "rope_theta", 10000000.0);
    }

    cfg.hybrid = true;
    cfg.dense_ffn = true;
    cfg.n_experts = 1;
    cfg.top_k = 1;
    cfg.n_shared = 0;
    (void)get_bool(text, "attn_output_gate", true);
    if (cfg.max_seq < 2048) cfg.max_seq = 2048;
    return cfg.hidden > 0 && cfg.n_layers > 0 && cfg.vocab > 0;
}

} // namespace sparkinfer
