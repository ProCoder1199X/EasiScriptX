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



### Research Foundations and Implemented Features
EasiScriptX (ESX) is built on several key research papers and concepts to deliver a high-performance DSL for AI/ML workflows. Below are the research foundations and their implementations in ESX:

visit `references.md` file for information upon what research work ESX has implemented features.

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

  

  

  
    

