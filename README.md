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
### Build Instructions

- Clone the repository:
 ```
git clone https://github.com/yourusername/EasiScriptX.git
cd EasiScriptX
```

- Create a build directory and configure the project:
```
mkdir build 77 cd build

cmake -DUSE_CUDA=ON ..
```

- Build the project
  ```
  make
  ```

- Run the executable
  ```
  ./esx examples/train.esx
  ```

## Language Syntax
- 1. Variable Decleration
 
   
    ```
    let <variable_name> = <expresssion>;
    ```
    
      - example:
    
        ```
            let a = tensor([[1, 2], [3, 4]]);
            let b = tensor([[5, 6], [7, 8]]);
         ```
  
    
-  2. Tensor Operations
    - Matrix Multiplication:

    
 
          let c = a @ b;
          

          
   - Convolution:
       
            let conv = conv2d(input, kernel, stride=1, padding=same);
            

       
- 3. Model Definition
 
   
    
    model <model_name> {
    layer dense(units=128, activation=relu);
    layer dropout(rate=0.5);
    layer dense(units=10, activation=softmax);
}
    

     
- 4. Training a Model
   
     train(<model>, <dataset>, loss: <loss_function>, opt: <optimizer>, epochs: <number>, device: <device>);
  
      - Example:
     
        train(mymodel, mnist, loss: ce, opt: adam(lr=0.001), epochs: 10, device: gpu);
       
 - 5. Distributed Training
  
   distribute(gpus: <number>) {
    <training_code>
}         

        - example:
             
              distribute(gpus: 2) {
    train(mymodel, mnist, loss: ce, opt: adam(lr=0.001), epochs: 5);
}
            

- 6 training Models

let model = load_pretrained("<model_name>");

    - example:
          
          let model = load_pretrained("resnet50");
          
- 7. Custom Loss Functions
         
fn <loss_name>(<parameters>) {
    return <expression>;
}
       
    - example:
           
           fn custom_loss(pred, true_val) {
    return mean((pred - true_val)^2);
} 
             


- 8. Autonomic Optimization


with autonomic {
    <training_code>
}

    - example
        
          with autonomic {
    train(mymodel, mnist, loss: ce, opt: adam(lr=0.001), epochs: 10, device: gpu);
}
          
