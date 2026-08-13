cmake_minimum_required(VERSION 3.20)

include(FetchContent)
set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    tokenizers_cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG        c586c52f93f7b060753bd2388eb96a105cb7374d
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(tokenizers_cpp)

FetchContent_Declare(
    nlohmann_json
    URL      https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
)
FetchContent_MakeAvailable(nlohmann_json)

add_executable(sparkinfer_server
    src/sparkinfer_server.cpp
    src/model_engine.cpp
    src/chat_tokenizer.cpp
    src/chat_tools.cpp
)

target_include_directories(sparkinfer_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/examples
)

target_link_libraries(sparkinfer_server PRIVATE
    sparkinfer_runtime
    tokenizers_cpp
    nlohmann_json::nlohmann_json
    CUDA::cudart
    Threads::Threads
)

target_compile_features(sparkinfer_server PRIVATE cxx_std_17)

if(BUILD_TESTS)
    add_executable(chat_tools_test
        tests/chat_tools_test.cpp
        src/chat_tools.cpp
    )
    target_include_directories(chat_tools_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(chat_tools_test PRIVATE
        nlohmann_json::nlohmann_json
    )
    target_compile_definitions(chat_tools_test PRIVATE
        SPARKINFER_SERVER_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests"
    )
    target_compile_features(chat_tools_test PRIVATE cxx_std_17)
    add_test(NAME chat_tools_test COMMAND chat_tools_test)

    add_executable(chat_tokenizer_test
        tests/chat_tokenizer_test.cpp
        src/chat_tokenizer.cpp
        src/chat_tools.cpp
    )
    target_include_directories(chat_tokenizer_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(chat_tokenizer_test PRIVATE
        tokenizers_cpp
        nlohmann_json::nlohmann_json
    )
    target_compile_features(chat_tokenizer_test PRIVATE cxx_std_17)
    add_test(NAME chat_tokenizer_test COMMAND chat_tokenizer_test)
endif()

install(TARGETS sparkinfer_server RUNTIME DESTINATION bin)
