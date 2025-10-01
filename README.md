# EasiScriptX (ESX)
#### EasiScriptX (ESX) is a high-performance domain-specific language (DSL) designed for AI/ML workflows. It simplifies the process of defining models, datasets, and training routines with a declarative syntax. ESX supports distributed training, pretrained models, advanced tensor operations, and extensibility for both research and production.
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



## Research Foundations and Implemented Features
EasiScriptX (ESX) is built on several key research papers and concepts to deliver a high-performance DSL for AI/ML workflows. Below are the research foundations and their implementations in ESX:

-  Transformers (Vaswani et al., "Attention is All You Need," 2017)
Relevance: Provides the foundation for the AttentionExpr in ast.hpp, enabling multi-head attention for small-to-medium LLMs.
Implementation:
The AttentionExpr struct (q, k, v, heads, dim) maps to PyTorch’s torch::nn::MultiheadAttention for efficient attention computation.
Used for transformer-based models in ESX scripts (e.g., attention(q, k, v, heads=8, dim=64)).

-  Distributed Training (Li et al., "PyTorch Distributed: Experiences on Accelerating Data Parallel Training," 2020)
Relevance: Guides the Distributed class in distributed.hpp, supporting multi-GPU (NCCL) and multi-node (MPI) training.
Implementation:
init_multi_gpu and init_multi_node use NCCL (ncclCommInitAll) and MPI (MPI_Init, MPI_Allreduce) for gradient aggregation, as seen in aggregate_gradients.
The new DistributeStmt in ast.hpp supports distribute(gpus: N) { ... }.

- Model Compression (Han et al., "Deep Compression: Compressing Deep Neural Networks," 2015)
Relevance: Underpins QuantizeExpr and PruneExpr in ast.hpp for model compression (quantization, pruning).
Implementation:
QuantizeExpr (bits, aware) supports post-training quantization (e.g., 8-bit).
PruneExpr (ratio) supports weight pruning.
These map to PyTorch/ONNX Runtime operations in interpreter.hpp.

- Autonomic Computing (Kephart and Chess, "The Vision of Autonomic Computing," 2003)
Relevance: Inspires AutonomicStmt and AgentTuneStmt in ast.hpp for multi-agent optimization.
Implementation:
AutonomicStmt wraps statements for self-optimizing execution (e.g., hyperparameter tuning).
AgentTuneStmt (fn, agents, target) simulates multi-agent tuning (stubbed for v1.0).

- ONNX (Bai et al., "ONNX: Open Neural Network Exchange," 2019)
Relevance: Enables model interoperability for .pt (PyTorch), .pb (TensorFlow via ONNX), and .onnx models.
Implementation:
ImportExpr in ast.hpp supports load_pretrained(name) for ONNX Runtime model loading.
USE_ONNX in config.hpp enables this feature.

### Implemented Features in ESX (v1.0)
- Core DSL:
Declarative syntax for models, tensors, and training (e.g., model mymodel { ... }, train(mymodel, mnist, ...)).
Supported by ast.hpp (ModelExpr, TrainStmt, Decl, Program).
- Tensor Operations:
Supported operations: matmul, conv2d, maxpool, batchnorm, layernorm, attention, tokenize, visualize.
Implemented via MatmulExpr, Conv2dExpr, MaxPoolExpr, BatchNormExpr, LayerNormExpr, AttentionExpr, TokenizeExpr, VisualizeExpr, DenseExpr in ast.hpp.
dataset.hpp supports tokenize with JSON vocab.
- Training:
Optimizers: SGD, Adam, LAMB, AdaFactor (via TrainStmt::opt and opt_params).
Loss functions: MSE, cross-entropy, Huber, hinge (TrainStmt::loss).
Devices: CPU, GPU (TrainStmt::device).
Distributed Training:
Multi-GPU (NCCL) and multi-node (MPI) via DistributeStmt and distributed.hpp (init_multi_gpu, init_multi_node, aggregate_gradients).
CPU fallback for unsupported GPUs in init_multi_gpu.
- Pretrained Models:
Load .pt, .pb (via ONNX), .onnx models using ImportExpr and USE_ONNX/USE_TENSORFLOW in config.hpp.
- Datasets:
Streaming datasets via DatasetExpr and dataset.hpp (load, next_batch).
Basic tokenization via TokenizeExpr and dataset.hpp::tokenize.
- Optimization:
Quantization (QuantizeExpr), pruning (PruneExpr), gradient checkpointing (stub in distributed.hpp::checkpoint_gradients).
Autonomic tuning (AutonomicStmt, AgentTuneStmt).
- Profiling:
Execution time and memory usage via ProfileStmt and USE_PROFILING in config.hpp.




### Installation
- Dependencies
   - Required:
      - C++20-compatible compiler (e.g., GCC, Clang, MSVC)
      - Boost (1.75+)
      - Eigen (3.4+)
      - ONNX Runtime
      - OpenMPI (for distributed training)
- Optional:
      - CUDA (for GPU acceleration)

### Language Syntax
-  Variable Declaration:
```
    let <variable_name> = <expression>;
```
 example: 
```
let a = tensor([[1, 2], [3, 4]]);
let b = tensor([[5, 6], [7, 8]]);
```

----
- Tensor Opertations
  - Matrix Multiplication
    ```
    let c = a @ b;
    ```
---     
- Convolutions
  ```
  let conv = conv2d(input, kernel, stride=1, padding=same);
  ```
---   
- Model Definition
```
model <model_name> {
    layer dense(units=128, activation=relu);
    layer dropout(rate=0.5);
    layer dense(units=10, activation=softmax);
}
```
--- 
- Training a Model

```
train(<model>, <dataset>, loss: <loss_function>, opt: <optimizer>, epochs: <number>, device: <device>);
```

example
```
train(mymodel, mnist, loss: ce, opt: adam(lr=0.001), epochs: 10, device: gpu);
```
---

- Distributed Training
```
distribute(gpus: <number>) {
    <training_code>
}
```

example
```
distribute(gpus: 2) {
    train(mymodel, mnist, loss: ce, opt: adam(lr=0.001), epochs: 5);
}
```
---
- Pretrained Models
  ```
  let model = load_pretrained("<model_name>");
  ```

  example
  
  ```
  let model = load_pretrained("resnet50");
  
  ```
---

- Custom Loss Functions

```
fn <loss_name>(<parameters>) {
    return <expression>;
}
```

example

```
fn custom_loss(pred, true_val) {
    return mean((pred - true_val)^2);
}
```

---

- Autonomic Optimization
 
  ```
  
  with autonomic {
    <training_code>
                  }

  ```

  example
  ```
  with autonomic {
    train(mymodel, mnist, loss: ce, opt: adam(lr=0.001), epochs: 10, device: gpu);
                 }
  ```
---
### License
EasiScriptX is licensed under the MIT License. See LICENSE.txt for details.

  

  

  
    

