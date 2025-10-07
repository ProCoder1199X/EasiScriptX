#ifndef ESX_CONFIG_HPP
#define ESX_CONFIG_HPP

/**
 * @file config.hpp
 * @brief Configuration flags and build options for EasiScriptX (ESX).
 * @details Defines compile-time configuration options for enabling/disabling
 * various features and backends.
 */

// Backend support flags
#define USE_ONNX 1
#define USE_TENSORFLOW 0
#define USE_CUDA 0
#define USE_MPI 1
#define USE_NCCL 0

// Backend registry (bitmask)
enum Backend : unsigned int {
    BACKEND_CPU   = 1u << 0,
    BACKEND_CUDA  = 1u << 1,
    BACKEND_ROCM  = 1u << 2,
    BACKEND_METAL = 1u << 3,
    BACKEND_XLA   = 1u << 4
};

static inline unsigned int getDefaultBackends() {
    unsigned int mask = BACKEND_CPU;
#if USE_CUDA
    mask |= BACKEND_CUDA;
#endif
    return mask;
}

// Precision support
#define USE_BF16 1
#define USE_FP16 1
#define USE_FP32 1

// Optimization flags
#define USE_PROFILING 1
#define USE_ENERGY_TRACKING 1
#define USE_MEMORY_OPTIMIZATION 1

// Research features
#define USE_LORA 1
#define USE_QUANTIZATION 1
#define USE_SPECULATIVE_DECODING 1
#define USE_PATTERN_RECOGNITION 1
#define USE_GRADIENT_CHECKPOINTING 1
#define USE_KERNEL_FUSION 1
#define USE_SPARSE_ATTENTION 1

// Performance tuning
#define DEFAULT_BATCH_SIZE 32
#define MAX_MEMORY_GB 8
#define DEFAULT_LEARNING_RATE 0.001
#define DEFAULT_EPOCHS 10

// Error handling
#define ENABLE_DETAILED_ERRORS 1
#define ENABLE_STACK_TRACES 0

// Debugging
#define DEBUG_PARSER 0
#define DEBUG_INTERPRETER 0
#define DEBUG_TENSOR_OPS 0

// Version information
#define ESX_VERSION_MAJOR 1
#define ESX_VERSION_MINOR 0
#define ESX_VERSION_PATCH 0
#define ESX_VERSION_STRING "1.0.0"

// Platform detection
#ifdef _WIN32
    #define ESX_PLATFORM_WINDOWS 1
    #define ESX_PLATFORM_LINUX 0
    #define ESX_PLATFORM_MACOS 0
#elif __APPLE__
    #define ESX_PLATFORM_WINDOWS 0
    #define ESX_PLATFORM_LINUX 0
    #define ESX_PLATFORM_MACOS 1
#else
    #define ESX_PLATFORM_WINDOWS 0
    #define ESX_PLATFORM_LINUX 1
    #define ESX_PLATFORM_MACOS 0
#endif

// Architecture detection
#ifdef __aarch64__
    #define ESX_ARCH_ARM64 1
    #define ESX_ARCH_X86_64 0
#else
    #define ESX_ARCH_ARM64 0
    #define ESX_ARCH_X86_64 1
#endif

// Compiler detection
#ifdef __GNUC__
    #define ESX_COMPILER_GCC 1
    #define ESX_COMPILER_CLANG 0
    #define ESX_COMPILER_MSVC 0
#elif __clang__
    #define ESX_COMPILER_GCC 0
    #define ESX_COMPILER_CLANG 1
    #define ESX_COMPILER_MSVC 0
#elif _MSC_VER
    #define ESX_COMPILER_GCC 0
    #define ESX_COMPILER_CLANG 0
    #define ESX_COMPILER_MSVC 1
#else
    #define ESX_COMPILER_GCC 0
    #define ESX_COMPILER_CLANG 0
    #define ESX_COMPILER_MSVC 0
#endif

// Feature availability checks
#if USE_CUDA && !defined(__CUDACC__)
    #undef USE_CUDA
    #define USE_CUDA 0
#endif

#if USE_MPI && !defined(MPI_VERSION)
    #undef USE_MPI
    #define USE_MPI 0
#endif

#if USE_NCCL && !USE_CUDA
    #undef USE_NCCL
    #define USE_NCCL 0
#endif

// Performance optimization macros
#if ESX_COMPILER_GCC || ESX_COMPILER_CLANG
    #define ESX_LIKELY(x) __builtin_expect(!!(x), 1)
    #define ESX_UNLIKELY(x) __builtin_expect(!!(x), 0)
    #define ESX_FORCE_INLINE __attribute__((always_inline)) inline
#elif ESX_COMPILER_MSVC
    #define ESX_LIKELY(x) (x)
    #define ESX_UNLIKELY(x) (x)
    #define ESX_FORCE_INLINE __forceinline
#else
    #define ESX_LIKELY(x) (x)
    #define ESX_UNLIKELY(x) (x)
    #define ESX_FORCE_INLINE inline
#endif

// Memory alignment
#define ESX_ALIGN_64 __attribute__((aligned(64)))
#define ESX_ALIGN_32 __attribute__((aligned(32)))
#define ESX_ALIGN_16 __attribute__((aligned(16)))

// Assertion macros
#if ENABLE_DETAILED_ERRORS
    #include <cassert>
    #define ESX_ASSERT(condition, message) assert((condition) && (message))
#else
    #define ESX_ASSERT(condition, message) ((void)0)
#endif

// Logging macros
#if DEBUG_PARSER || DEBUG_INTERPRETER || DEBUG_TENSOR_OPS
    #include <iostream>
    #define ESX_DEBUG(module, message) std::cout << "[" << module << "] " << message << std::endl
#else
    #define ESX_DEBUG(module, message) ((void)0)
#endif

// Utility macros
#define ESX_UNUSED(x) ((void)(x))
#define ESX_ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

// Feature test macros
#define ESX_HAS_FEATURE(feature) (ESX_HAS_##feature)
#define ESX_HAS_ONNX USE_ONNX
#define ESX_HAS_CUDA USE_CUDA
#define ESX_HAS_MPI USE_MPI
#define ESX_HAS_NCCL USE_NCCL
#define ESX_HAS_BF16 USE_BF16
#define ESX_HAS_FP16 USE_FP16
#define ESX_HAS_LORA USE_LORA
#define ESX_HAS_QUANTIZATION USE_QUANTIZATION
#define ESX_HAS_SPECULATIVE_DECODING USE_SPECULATIVE_DECODING
#define ESX_HAS_PATTERN_RECOGNITION USE_PATTERN_RECOGNITION
#define ESX_HAS_GRADIENT_CHECKPOINTING USE_GRADIENT_CHECKPOINTING
#define ESX_HAS_KERNEL_FUSION USE_KERNEL_FUSION
#define ESX_HAS_SPARSE_ATTENTION USE_SPARSE_ATTENTION

#endif // ESX_CONFIG_HPP