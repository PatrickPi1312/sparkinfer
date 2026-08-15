cmake_minimum_required(VERSION 3.20)

# Blackwell consumer/edge: sm_120 (RTX 5090, RTX PRO 6000). sm_121 (RTX Spark, Jetson
# Thor) is intentionally excluded — CUDA 12.8's nvcc (currently in use) hard-errors with
# "Unsupported gpu architecture 'compute_121'" during compiler identification, since that
# target requires CUDA 12.9+.
# Must be set BEFORE project(... CUDA) — project()/enable_language(CUDA) reads
# CMAKE_CUDA_ARCHITECTURES at that call and caches its own (environment-dependent,
# sometimes ancient e.g. "52") default if it isn't already set. A set() after
# project() is a CACHE no-op once that's happened, silently building tensor-core/wmma
# kernels for a target that can't support them.
set(CMAKE_CUDA_ARCHITECTURES "89;90;100;120" CACHE STRING "CUDA architectures")
project(sparkinfer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(BUILD_TESTS    "Build unit / integration tests" ON)
option(BUILD_EXAMPLES "Build example programs"          ON)

find_package(CUDAToolkit REQUIRED)

# Declared here, not in server/, because runtime/ is configured BEFORE server/ (see the
# superbuild ordering below) and runtime/examples/qwen3_gguf_bench needs this target to read a
# HuggingFace compressed-tensors checkpoint's config.json via qwen38_hf_config.h -- the same
# header the server uses, so the benchmark and the server can never disagree about how a
# checkpoint is configured. This links JSON into that one EXAMPLE EXECUTABLE only; the
# sparkinfer_runtime library itself still links no JSON dependency (runtime/src/safetensors.cpp
# keeps its own hand-rolled parser for exactly that reason). server/CMakeLists.txt reuses this
# target rather than re-declaring it.
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL      https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
)
FetchContent_MakeAvailable(nlohmann_json)

enable_testing()   # so `ctest` at the build root discovers the subdirs' add_test() registrations

# Superbuild. moe/ and runtime/ guard their sibling add_subdirectory(../kernels)
# with if(NOT TARGET ...), so adding kernels first here makes them reuse the
# already-defined targets instead of re-adding. Dependency order: kernels -> moe -> runtime.
add_subdirectory(kernels)
add_subdirectory(moe)
add_subdirectory(runtime)
# bench/ is benchmark scripts + frozen configs + results — not a CMake target.

option(BUILD_SERVER "Build sparkinfer-server (OpenAI-compatible HTTP API)" OFF)
if(BUILD_SERVER)
  add_subdirectory(server)
endif()
