#include "sparkinfer/safetensors.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace sparkinfer {
namespace {

bool map_file(const std::string& path, void*& base, size_t& size, int& fd) {
    fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "[st] open failed: %s\n", path.c_str()); return false; }
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "[st] stat failed: %s\n", path.c_str());
        close(fd); fd = -1; return false;
    }
    size = (size_t)st.st_size;
    base = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        fprintf(stderr, "[st] mmap failed: %s\n", path.c_str());
        close(fd); fd = -1; base = nullptr; return false;
    }
    return true;
}

// Parse the safetensors JSON header (one object of tensor descriptors).
bool parse_header(const uint8_t* json, size_t n, const uint8_t* data_base,
                  std::unordered_map<std::string, STTensor>& out) {
    std::string s((const char*)json, n);
    size_t i = 0;
    auto skip_ws = [&] {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) i++;
    };
    auto expect = [&](char c) {
        skip_ws();
        if (i >= s.size() || s[i] != c) return false;
        i++;
        return true;
    };
    auto parse_str = [&](std::string& o) -> bool {
        skip_ws();
        if (i >= s.size() || s[i] != '"') return false;
        i++;
        o.clear();
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) { o.push_back(s[i + 1]); i += 2; }
            else { o.push_back(s[i]); i++; }
        }
        if (i >= s.size()) return false;
        i++;
        return true;
    };
    if (!expect('{')) return false;
    while (true) {
        skip_ws();
        if (i < s.size() && s[i] == '}') { i++; break; }
        std::string name;
        if (!parse_str(name)) return false;
        if (!expect(':') || !expect('{')) return false;
        STTensor t;
        int64_t off0 = -1, off1 = -1;
        while (true) {
            skip_ws();
            if (i < s.size() && s[i] == '}') { i++; break; }
            std::string key;
            if (!parse_str(key) || !expect(':')) return false;
            skip_ws();
            if (key == "dtype") {
                if (!parse_str(t.dtype)) return false;
            } else if (key == "shape") {
                if (!expect('[')) return false;
                while (true) {
                    skip_ws();
                    if (i < s.size() && s[i] == ']') { i++; break; }
                    char* end = nullptr;
                    long v = strtol(s.c_str() + i, &end, 10);
                    if (end == s.c_str() + i) return false;
                    t.shape.push_back(v);
                    i = (size_t)(end - s.c_str());
                    skip_ws();
                    if (i < s.size() && s[i] == ',') i++;
                }
            } else if (key == "data_offsets") {
                if (!expect('[')) return false;
                char* end = nullptr;
                off0 = strtoll(s.c_str() + i, &end, 10);
                i = (size_t)(end - s.c_str());
                skip_ws();
                if (i < s.size() && s[i] == ',') i++;
                skip_ws();
                off1 = strtoll(s.c_str() + i, &end, 10);
                i = (size_t)(end - s.c_str());
                if (!expect(']')) return false;
            } else {
                // skip value
                if (s[i] == '"') { std::string tmp; if (!parse_str(tmp)) return false; }
                else if (s[i] == '{') {
                    int depth = 0;
                    do {
                        if (s[i] == '{') depth++;
                        else if (s[i] == '}') depth--;
                        i++;
                    } while (i < s.size() && depth > 0);
                } else if (s[i] == '[') {
                    int depth = 0;
                    do {
                        if (s[i] == '[') depth++;
                        else if (s[i] == ']') depth--;
                        i++;
                    } while (i < s.size() && depth > 0);
                } else {
                    while (i < s.size() && s[i] != ',' && s[i] != '}') i++;
                }
            }
            skip_ws();
            if (i < s.size() && s[i] == ',') i++;
        }
        if (name != "__metadata__" && off0 >= 0 && off1 >= off0) {
            t.data = data_base + off0;
            t.nbytes = (size_t)(off1 - off0);
            out.emplace(std::move(name), std::move(t));
        }
        skip_ws();
        if (i < s.size() && s[i] == ',') i++;
    }
    return true;
}

bool load_one_file(const std::string& path, std::vector<SafeTensorsDir::Map>& maps,
                   std::unordered_map<std::string, STTensor>& tensors) {
    SafeTensorsDir::Map m;
    if (!map_file(path, m.base, m.size, m.fd)) return false;
    if (m.size < 8) return false;
    uint64_t hdr_len = 0;
    memcpy(&hdr_len, m.base, 8);
    if (8 + hdr_len > m.size) {
        fprintf(stderr, "[st] bad header length in %s\n", path.c_str());
        return false;
    }
    const uint8_t* json = (const uint8_t*)m.base + 8;
    const uint8_t* data = json + hdr_len;
    if (!parse_header(json, (size_t)hdr_len, data, tensors)) {
        fprintf(stderr, "[st] header parse failed: %s\n", path.c_str());
        return false;
    }
    maps.push_back(m);
    return true;
}

} // namespace

SafeTensorsDir::~SafeTensorsDir() {
    for (auto& m : maps_) {
        if (m.base && m.size) munmap(m.base, m.size);
        if (m.fd >= 0) close(m.fd);
    }
}

bool SafeTensorsDir::open(const std::string& dir) {
    const std::string index = dir + "/model.safetensors.index.json";
    std::vector<std::string> files;
    std::ifstream in(index);
    if (in) {
        std::stringstream ss; ss << in.rdbuf();
        const std::string js = ss.str();
        // Collect unique shard filenames from "weight_map" string values ending in .safetensors
        size_t pos = 0;
        while ((pos = js.find(".safetensors", pos)) != std::string::npos) {
            size_t start = js.rfind('"', pos);
            if (start != std::string::npos && start < pos) {
                std::string fn = js.substr(start + 1, pos + 12 - (start + 1));
                if (fn.find('/') == std::string::npos) {
                    bool seen = false;
                    for (const auto& f : files) if (f == fn) { seen = true; break; }
                    if (!seen) files.push_back(fn);
                }
            }
            pos += 12;
        }
    }
    if (files.empty()) {
        DIR* d = opendir(dir.c_str());
        if (!d) { fprintf(stderr, "[st] cannot open dir %s\n", dir.c_str()); return false; }
        while (dirent* e = readdir(d)) {
            std::string n = e->d_name;
            if (n.size() > 12 && n.rfind(".safetensors") == n.size() - 12)
                files.push_back(n);
        }
        closedir(d);
    }
    if (files.empty()) {
        fprintf(stderr, "[st] no safetensors in %s\n", dir.c_str());
        return false;
    }
    for (const auto& fn : files) {
        if (!load_one_file(dir + "/" + fn, maps_, tensors_)) return false;
    }
    fprintf(stderr, "[st] loaded %zu tensors from %zu shards in %s\n",
            tensors_.size(), maps_.size(), dir.c_str());
    return true;
}

const STTensor* SafeTensorsDir::tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

} // namespace sparkinfer
