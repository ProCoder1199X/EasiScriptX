#ifndef ESX_CONFIG_HPP
#define ESX_CONFIG_HPP

/**
 * @file config.hpp
 * @brief Configuration macros for EasiScriptX (ESX).
 * @details Defines build-time configuration options for enabling/disabling features
 * and optimizing performance for CPU or GPU execution.
 */

/// @brief Enables CUDA support for GPU acceleration (set via CMake -DUSE_CUDA=ON).
#ifndef USE_CUDA
#define USE_CUDA 0
#endif

/// @brief Enables MPI support for multi-node distributed training (set via CMake -DUSE_MPI=ON).
#ifndef USE_MPI
#define USE_MPI 1
#endif

/// @brief Enables ONNX Runtime support for pretrained model loading.
#define USE_ONNX 1

/// @brief Enables TensorFlow support for .pb model loading (via ONNX conversion).
#define USE_TENSORFLOW 1

/// @brief Enables profiling for execution time and memory usage.
#define USE_PROFILING 1

/// @brief Enables debug mode for AST printing and detailed logging.
#define DEBUG_MODE 0

/// @brief Tile size for GPU matrix operations (optimized for CUDA).
#define GPU_TILE_SIZE 128

/// @brief Tile size for CPU matrix operations (optimized for Eigen/SIMD).
#define CPU_TILE_SIZE 64

/// @brief Selects tile size based on CUDA availability.
#if USE_CUDA
#define MAX_TILE_SIZE GPU_TILE_SIZE
#else
#define MAX_TILE_SIZE CPU_TILE_SIZE
#endif

#endif // ESX_CONFIG_HPP