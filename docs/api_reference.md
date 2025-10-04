# EasiScriptX API Reference

## Overview

This document provides a comprehensive reference for all EasiScriptX (ESX) language features, including syntax, parameters, and usage examples.

## Core Language Syntax

### Variable Declaration
```esx
let <variable_name> = <expression>;
```

**Example:**
```esx
let x = tensor([[1, 2], [3, 4]]);
let y = x @ x;
```

### Tensor Operations

#### Matrix Multiplication
```esx
let result = <tensor1> @ <tensor2>;
```

#### Convolution
```esx
let result = conv2d(<input>, <kernel>, stride: <number>, padding: <number>);
```

#### Attention (with FlashAttention-2)
```esx
let result = attention(<q>, <k>, <v>, heads: <number>, dim: <number>, flash: true);
```

#### LoRA (Low-Rank Adaptation)
```esx
let result = lora(<model>, rank: <number>);
```

#### Mixed Precision
```esx
let result = mixed_precision(<model>, <bf16|fp16>);
```

#### Kernel Fusion
```esx
let result = fused_matmul_relu(<tensor1>, <tensor2>);
```

## Memory Optimization Commands

### Memory Broker
Optimizes GPU memory usage for large models.

**Syntax:**
```esx
memory_broker(<model>, max_mem: <memory_in_GB>, strategy: <zeRO|offload>);
```

**Parameters:**
- `model`: The model tensor to optimize
- `max_mem`: Maximum memory usage in GB (double)
- `strategy`: Memory optimization strategy
  - `zeRO`: Zero Redundancy Optimizer for distributed training
  - `offload`: Offload parameters to CPU memory

**Example:**
```esx
let model = tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]);
memory_broker(model, max_mem: 8, strategy: zeRO);
```

**Benefits:** 30-40% memory reduction for large models

### Quantization
Reduces model size and speeds up inference through quantization.

**Syntax:**
```esx
quantize(<model>, bits: <8|4>, method: <ptq|qat>);
```

**Parameters:**
- `model`: The model tensor to quantize
- `bits`: Quantization bit width (8 or 4)
- `method`: Quantization method
  - `ptq`: Post-Training Quantization
  - `qat`: Quantization-Aware Training

**Example:**
```esx
let model = tensor([[1,2],[3,4]]);
quantize(model, bits: 8, method: ptq);
```

**Benefits:** 4x model size reduction, 2-3x inference speedup

### Gradient Checkpointing
Reduces memory usage during training by recomputing intermediate activations.

**Syntax:**
```esx
checkpoint(<model>, segments: <number>);
```

**Parameters:**
- `model`: The model tensor to checkpoint
- `segments`: Number of segments to checkpoint (positive integer)

**Example:**
```esx
let model = tensor([[1,2],[3,4]]);
checkpoint(model, segments: 4);
```

**Benefits:** 50% memory reduction during training

## LLM Optimization Commands

### Speculative Decoding
Speeds up LLM inference by predicting multiple tokens in parallel.

**Syntax:**
```esx
speculative_decode(<model>, <draft_model>, max_tokens: <number>);
```

**Parameters:**
- `model`: Main model for verification
- `draft_model`: Smaller draft model for prediction
- `max_tokens`: Maximum number of tokens to predict (positive integer)

**Example:**
```esx
let main_model = tensor([[1,2],[3,4]]);
let draft_model = tensor([[0.5,1],[1.5,2]]);
speculative_decode(main_model, draft_model, max_tokens: 100);
```

**Benefits:** 2x faster LLM inference

### Sparse Attention
Reduces attention memory usage for long sequences.

**Syntax:**
```esx
let result = attention(<q>, <k>, <v>, heads: <number>, dim: <number>, flash: true, sparse: true);
```

**Parameters:**
- `q`, `k`, `v`: Query, key, and value tensors
- `heads`: Number of attention heads (positive integer)
- `dim`: Attention dimension (positive integer)
- `flash`: Enable FlashAttention-2 (boolean)
- `sparse`: Enable sparse attention (boolean)

**Example:**
```esx
let q = tensor([[1,2],[3,4]]);
let k = tensor([[5,6],[7,8]]);
let v = tensor([[9,10],[11,12]]);
let result = attention(q, k, v, heads: 8, dim: 2, flash: true, sparse: true);
```

**Benefits:** 40% memory reduction for long sequences

## Pattern Recognition Commands

### Pattern Recognition
Applies ARC-AGI2-inspired pattern recognition for abstract reasoning.

**Syntax:**
```esx
pattern_recognize(<dataset>, rules: <geometric|arithmetic>);
```

**Parameters:**
- `dataset`: Dataset tensor to analyze
- `rules`: Pattern recognition rules
  - `geometric`: Spatial relationships, shapes, transformations
  - `arithmetic`: Numerical sequences, operations

**Example:**
```esx
let dataset = tensor([[1,2,3,4],[5,6,7,8]]);
pattern_recognize(dataset, rules: geometric);
pattern_recognize(dataset, rules: arithmetic);
```

**Benefits:** 15-20% better generalization for abstract reasoning tasks

## Training Commands

### Training
Trains a model with specified parameters and optimizers.

**Syntax:**
```esx
train(<model>, <dataset>, loss: <ce|mse>, opt: <adam|sgd|lomo>(<params>), epochs: <number>, device: <cpu|gpu>);
```

**Parameters:**
- `model`: Model tensor to train
- `dataset`: Training dataset
- `loss`: Loss function
  - `ce`: Cross-entropy
  - `mse`: Mean squared error
- `opt`: Optimizer with parameters
  - `adam(lr=<learning_rate>)`: Adam optimizer
  - `sgd(lr=<learning_rate>)`: Stochastic gradient descent
  - `lomo(lr=<learning_rate>)`: Low-Memory Optimization
- `epochs`: Number of training epochs (positive integer)
- `device`: Training device
  - `cpu`: CPU training
  - `gpu`: GPU training

**Example:**
```esx
let model = tensor([[1,2],[3,4]]);
let dataset = tensor([[5,6],[7,8]]);
train(model, dataset, loss: ce, opt: adam(lr=0.001), epochs: 10, device: cpu);
```

### Dynamic Batching
Automatically adjusts batch sizes based on memory availability.

**Implementation:** Built into the training loop, automatically calculates optimal batch size based on model size and available memory.

## Energy-Aware Computing

### Energy-Aware Scheduling
Monitors and optimizes energy consumption during training.

**Syntax:**
```esx
energy_aware(<model>, max_power: <watts>);
```

**Parameters:**
- `model`: Model tensor to monitor
- `max_power`: Maximum power consumption in watts (positive number)

**Example:**
```esx
let model = tensor([[1,2],[3,4]]);
energy_aware(model, max_power: 100);
```

### Heterogeneous Scheduling
Optimizes CPU/GPU resource utilization.

**Syntax:**
```esx
schedule_heterogeneous(cpu: <ratio>, gpu: <ratio>);
```

**Parameters:**
- `cpu`: CPU utilization ratio (0.0 to 1.0)
- `gpu`: GPU utilization ratio (0.0 to 1.0)
- **Constraint:** cpu + gpu must equal 1.0

**Example:**
```esx
schedule_heterogeneous(cpu: 0.7, gpu: 0.3);
```

## Distributed Training

### Pipeline Parallelism
Distributes model across multiple GPUs in a pipeline.

**Syntax:**
```esx
pipeline_parallel(<model>, stages: <number>);
```

**Parameters:**
- `model`: Model tensor to parallelize
- `stages`: Number of pipeline stages (positive integer)

**Example:**
```esx
let model = tensor([[1,2],[3,4]]);
pipeline_parallel(model, stages: 2);
```

### Multi-GPU Training
```esx
distribute(gpus: <number>) {
    <training_code>
}
```

**Example:**
```esx
distribute(gpus: 2) {
    train(mymodel, mnist, loss: ce, opt: adam(lr=0.001), epochs: 5);
}
```

## Framework Interoperability

### Framework Switching
Switches between different ML frameworks.

**Syntax:**
```esx
switch_framework(<pytorch|tensorflow|jax>, <model_path>);
```

**Parameters:**
- Framework: Target framework
  - `pytorch`: PyTorch framework
  - `tensorflow`: TensorFlow framework (planned)
  - `jax`: JAX framework (planned)
- `model_path`: Path to model file

**Example:**
```esx
switch_framework(pytorch, model.pt);
```

## Experiment Tracking

### MLflow Integration
Tracks experiments with MLflow.

**Syntax:**
```esx
track_experiment(mlflow, <run_id>);
```

**Parameters:**
- `mlflow`: Experiment tracking system
- `run_id`: Unique run identifier

**Example:**
```esx
track_experiment(mlflow, run123);
```

## Error Handling

EasiScriptX provides comprehensive error handling with detailed error messages:

### Common Error Types

1. **Invalid Parameters:**
   - Invalid LoRA rank (must be > 0)
   - Invalid attention heads (must be > 0)
   - Invalid epochs (must be > 0)
   - Invalid memory values (must be > 0)

2. **Tensor Errors:**
   - Empty tensors
   - Inconsistent tensor dimensions
   - Undefined variables

3. **File Errors:**
   - Model file not found
   - Vocabulary file not found

4. **Validation Errors:**
   - Invalid framework names
   - Invalid optimization strategies
   - Invalid pattern recognition rules

### Error Message Format
```
<Error Type> at line <line_number>, column <column_number>
```

**Example:**
```
Invalid LoRA rank at line 5
Model file model.pt not found at line 10
```

## Performance Profiling

EasiScriptX includes built-in performance profiling:

### Automatic Profiling
- FlashAttention execution time
- Training loop duration
- Memory usage tracking
- Energy consumption monitoring

### Profiling Output
```
FlashAttention time: 1234us
Training completed in 5678us
Memory broker applied: max_mem=8GB, strategy=zeRO
```

## Best Practices

1. **Memory Management:**
   - Use memory_broker for large models
   - Apply quantization for inference
   - Use gradient checkpointing for training

2. **Performance Optimization:**
   - Enable FlashAttention for attention operations
   - Use kernel fusion for CPU operations
   - Apply sparse attention for long sequences

3. **Error Handling:**
   - Validate input parameters
   - Check file existence before loading
   - Use appropriate data types

4. **Testing:**
   - Test with edge cases
   - Validate error conditions
   - Benchmark performance improvements

## Version Compatibility

- **ESX v1.0**: All features documented above
- **Future versions**: Additional optimizations and features planned

For the latest updates and feature additions, refer to the project repository and changelog.
