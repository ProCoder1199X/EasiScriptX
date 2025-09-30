# EasiScriptX (ESX)

EasiScriptX (ESX) is a high-performance domain-specific language (DSL) designed for AI/ML workflows. It simplifies the process of defining models, datasets, and training routines with a declarative syntax. ESX supports distributed training, pretrained models, advanced tensor operations, and extensibility for both research and production.
--- 
## Features
- Declarative Syntax: Write AI/ML workflows with simple, human-readable code.
- Advanced Tensor Operations: Includes matrix multiplication, convolution, pooling, normalization, and more.
- Distributed Training: Multi-GPU (via NCCL) and multi-node (via MPI) support.
- Pretrained Models: Load pretrained models using ONNX Runtime or PyTorch.
- Custom Loss Functions and Optimizers: Define your own loss functions and optimizers.
- Profiling and Debugging: Built-in tools for profiling and debugging.
- Autonomic Optimization: Optimize models with autonomic compression and multi-agent tuning.
- Extensibility: Add new layers, operations, and research features.

## Installation
### Dependencies
- Required:
- C++20-compatible compiler (e.g., GCC, Clang, MSVC)
- Boost (1.75+)
- Eigen (3.4+)
- ONNX Runtime
- OpenMPI (for distributed training)

- Optional:
- CUDA (for GPU acceleration)

