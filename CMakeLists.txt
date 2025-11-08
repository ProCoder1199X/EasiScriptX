cmake_minimum_required(VERSION 3.20)
project(EasiScriptX VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Include directories
include_directories(src)

# Find packages
find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

# Optional dependencies
find_package(MPI COMPONENTS CXX)
if(MPI_CXX_FOUND)
    add_definitions(-DUSE_MPI=1)
    message(STATUS "MPI found: ${MPI_CXX_VERSION}")
else()
    add_definitions(-DUSE_MPI=0)
    message(STATUS "MPI not found, distributed training disabled")
endif()

# CUDA support
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA=1)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
else()
    add_definitions(-DUSE_CUDA=0)
    message(STATUS "CUDA not found, GPU acceleration disabled")
endif()

# ONNX Runtime
find_package(onnxruntime QUIET)
if(onnxruntime_FOUND)
    add_definitions(-DUSE_ONNX=1)
    message(STATUS "ONNX Runtime found")
else()
    add_definitions(-DUSE_ONNX=0)
    message(STATUS "ONNX Runtime not found, ONNX support disabled")
endif()

# PyTorch C++ API
find_package(Torch QUIET)
if(Torch_FOUND)
    add_definitions(-DUSE_TORCH=1)
    message(STATUS "PyTorch C++ API found: ${Torch_VERSION}")
else()
    add_definitions(-DUSE_TORCH=0)
    message(STATUS "PyTorch C++ API not found, PyTorch support disabled")
endif()

# FetchContent for dependencies
include(FetchContent)

# Catch2 for testing
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

# nlohmann/json
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(nlohmann_json)

# libcurl
find_package(CURL REQUIRED)
find_package(OpenSSL QUIET)
find_package(spdlog QUIET)

# Source files
set(ESX_SOURCES
    src/main.cpp
    src/parser.cpp
    src/error_handler.cpp
    src/onnx_export.cpp
    src/repl.cpp
)

set(ESX_HEADERS
    src/ast.hpp
    src/config.hpp
    src/parser.hpp
    src/interpreter.hpp
    src/security.hpp
    src/tensor.hpp
    src/dataset.hpp
    src/distributed.hpp
    src/model.hpp
    src/runtime.hpp
    src/printer.hpp
    src/error_handler.hpp
    src/onnx_export.hpp
    src/repl.hpp
)

# Main executable
add_executable(esx ${ESX_SOURCES} ${ESX_HEADERS})

# Link libraries
target_link_libraries(esx
    Boost::system
    Boost::filesystem
    Eigen3::Eigen
    Threads::Threads
    CURL::libcurl
    nlohmann_json::nlohmann_json
)

if(MPI_CXX_FOUND)
    target_link_libraries(esx MPI::MPI_CXX)
endif()

if(Torch_FOUND)
    target_link_libraries(esx ${TORCH_LIBRARIES})
    target_compile_definitions(esx PRIVATE TORCH_API_INCLUDE_EXTRA_H)
endif()

if(onnxruntime_FOUND)
    target_link_libraries(esx onnxruntime::onnxruntime)
endif()
if(OpenSSL_FOUND)
    target_compile_definitions(esx PRIVATE USE_OPENSSL=1)
    target_link_libraries(esx OpenSSL::SSL OpenSSL::Crypto)
else()
    target_compile_definitions(esx PRIVATE USE_OPENSSL=0)
endif()
if(spdlog_FOUND)
    target_compile_definitions(esx PRIVATE USE_SPDLOG=1)
    target_link_libraries(esx spdlog::spdlog)
else()
    target_compile_definitions(esx PRIVATE USE_SPDLOG=0)
endif()

# Compiler definitions
target_compile_definitions(esx PRIVATE
    ESX_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    ESX_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    ESX_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    ESX_VERSION_STRING="${PROJECT_VERSION}"
)

# Tests
add_executable(test_esx tests/test_esx.cpp ${ESX_HEADERS})
target_link_libraries(test_esx
    Boost::system
    Boost::filesystem
    Eigen3::Eigen
    Threads::Threads
    CURL::libcurl
    nlohmann_json::nlohmann_json
    Catch2::Catch2WithMain
)

if(MPI_CXX_FOUND)
    target_link_libraries(test_esx MPI::MPI_CXX)
endif()

if(Torch_FOUND)
    target_link_libraries(test_esx ${TORCH_LIBRARIES})
endif()
if(onnxruntime_FOUND)
    target_link_libraries(test_esx onnxruntime::onnxruntime)
endif()
if(OpenSSL_FOUND)
    target_link_libraries(test_esx OpenSSL::SSL OpenSSL::Crypto)
endif()
if(spdlog_FOUND)
    target_link_libraries(test_esx spdlog::spdlog)
endif()

# Benchmarks
add_executable(benchmark benchmarks/benchmark.cpp ${ESX_HEADERS})
target_link_libraries(benchmark
    Boost::system
    Boost::filesystem
    Eigen3::Eigen
    Threads::Threads
    CURL::libcurl
    nlohmann_json::nlohmann_json
)

if(MPI_CXX_FOUND)
    target_link_libraries(benchmark MPI::MPI_CXX)
endif()

option(ESX_ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ESX_ENABLE_ASAN AND NOT MSVC)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

if(Torch_FOUND)
    target_link_libraries(benchmark ${TORCH_LIBRARIES})
endif()

# Examples
file(GLOB ESX_EXAMPLES "examples/*.esx")
foreach(example ${ESX_EXAMPLES})
    get_filename_component(example_name ${example} NAME_WE)
    add_custom_target(run_${example_name}
        COMMAND esx ${example}
        DEPENDS esx
        COMMENT "Running example: ${example_name}"
    )
endforeach()

# Install targets
install(TARGETS esx test_esx benchmark
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${ESX_HEADERS}
    DESTINATION include/esx
)

install(DIRECTORY examples/
    DESTINATION share/esx/examples
    FILES_MATCHING PATTERN "*.esx"
)

# CTest
enable_testing()
add_test(NAME esx_tests COMMAND test_esx)

# Custom targets
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_esx
    COMMENT "Running all tests"
)

add_custom_target(run_benchmarks
    COMMAND benchmark
    DEPENDS benchmark
    COMMENT "Running benchmarks"
)

add_custom_target(run_examples
    DEPENDS run_train run_matmul run_distributed run_autonomic run_pretrained run_custom_losses
    COMMENT "Running all examples"
)

# Documentation
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_SOURCE_DIR}/doxyfile ${CMAKE_BINARY_DIR}/doxyfile @ONLY)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/doxyfile
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    EasiScriptXConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/EasiScriptXConfig.cmake.in
    ${CMAKE_BINARY_DIR}/EasiScriptXConfig.cmake
    INSTALL_DESTINATION lib/cmake/EasiScriptX
)

install(FILES
    ${CMAKE_BINARY_DIR}/EasiScriptXConfig.cmake
    ${CMAKE_BINARY_DIR}/EasiScriptXConfigVersion.cmake
    DESTINATION lib/cmake/EasiScriptX
)

# CPack for packaging
include(CPack)
set(CPACK_PACKAGE_NAME "EasiScriptX")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "EasiScriptX - AI/ML DSL")
set(CPACK_PACKAGE_VENDOR "EasiScriptX Team")
set(CPACK_PACKAGE_CONTACT "contact@easiscriptx.org")

# Platform-specific settings
if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
elseif(APPLE)
    set(CPACK_GENERATOR "ZIP;TGZ")
else()
    set(CPACK_GENERATOR "ZIP;TGZ;DEB;RPM")
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "EasiScriptX Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  MPI support: ${MPI_CXX_FOUND}")
message(STATUS "  CUDA support: ${CUDA_FOUND}")
message(STATUS "  ONNX support: ${onnxruntime_FOUND}")
message(STATUS "  PyTorch support: ${Torch_FOUND}")
message(STATUS "")