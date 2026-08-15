#pragma once
#include "sparkinfer/models/qwen_config.h"
#include <string>

namespace sparkinfer {

// True when `path` is a Hugging Face directory with ModelOpt NVFP4 shards.
bool path_is_nvfp4_dir(const std::string& path);

// Fill Qwen35Config from config.json (uses text_config when present).
bool qwen3_config_from_hf_dir(const std::string& dir, Qwen35Config& cfg);

} // namespace sparkinfer
