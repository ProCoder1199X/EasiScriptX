# EasiScriptX (ESX)

EasiScriptX (ESX) is a high-performance domain-specific language (DSL) designed for AI/ML workflows. It provides a declarative syntax for defining models, datasets, and training workflows, with support for distributed training, pretrained models, and advanced tensor operations.

## Features
- **Declarative Syntax:** Define models, datasets, and training workflows with simple, human-readable syntax.
- **Advanced Tensor Operations:** Includes `matmul`, `conv2d`, `maxpool`, `batchnorm`, and more.
- **Distributed Training:** Multi-GPU (via NCCL) and multi-node (via MPI) support.
- **Pretrained Models:** Load pretrained models using ONNX Runtime.
- **Custom Loss Functions and Optimizers:** Define your own loss functions and optimizers.
- **Profiling and Debugging:** Built-in tools for profiling and debugging.
- **Autonomic Optimization:** Optimize models with autonomic compression and multi-agent tuning.

## Quick Start

### Installation
1. Clone the repository:
  ```
   git clone https://github.com/yourusername/EasiScriptX.git
   cd EasiScriptX
   ```

2. Build the project:
```
mkdir build && cd build
cmake -DUSE_CUDA=ON ..
make  
```


### Running a Script
Run an ESX script:

./esx [train.esx](http://_vscodecontentref_/10)

### Debug Mode:
Enable debug mode to print the AST and log execution details:
./esx --debug [train.esx](http://_vscodecontentref_/11)

### Installing Pretrained Models
Install a pretrained model:

./esx install resnet50

### Examples
- Training a Model:
train(mymodel, mnist, loss: ce, opt: adam(lr=0.001), epochs=10, device: gpu)

- Matrix Multiplication:
let a = tensor([[1,2],[3,4]])
let b = tensor([[5,6],[7,8]])
let c = a @ b

- Distributed Training:
distribute(gpus: 2) {
    train(mymodel, mnist, loss: ce, opt: adam(lr=0.001), epochs=5)
}

## License 

EasiScriptX is licensed under MIT License. See LICENSE.txt
