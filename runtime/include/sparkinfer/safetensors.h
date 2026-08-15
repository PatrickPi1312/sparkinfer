#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sparkinfer {

struct STTensor {
    std::string dtype;                 // "BF16", "F32", "F8_E4M3", "U8", ...
    std::vector<int64_t> shape;
    const void* data = nullptr;
    size_t nbytes = 0;
};

// Read-only multi-shard safetensors (mmap). Looks at model.safetensors.index.json
// when present, otherwise every *.safetensors in the directory.
class SafeTensorsDir {
public:
    ~SafeTensorsDir();
    bool open(const std::string& dir);
    const STTensor* tensor(const std::string& name) const;
    bool has(const std::string& name) const { return tensor(name) != nullptr; }

    struct Map { void* base = nullptr; size_t size = 0; int fd = -1; };

private:
    std::vector<Map> maps_; // mmap'd shards
    std::unordered_map<std::string, STTensor> tensors_;
};

} // namespace sparkinfer
