
# EasiScriptX Tutorials

## 1. Basic Tensor Operations
```
let a = tensor([[1,2],[3,4]])
let b = tensor([[5,6],[7,8]])
let c = a @ b
```

## 2. Training a Model 
```
train(mymodel, mnist, loss: ce, opt: adam(lr=0.001), epochs=10, device: gpu)
```
## 3. Using Pretrained Models:
```
let model = load_pretrained("resnet50")
```
## 4. Distributed Training:
```
distribute(gpus: 2) {
    train(mymodel, mnist, loss: ce, opt: adam(lr=0.001), epochs=5)
}
```
## 5. Custom Loss Functions:
```
fn custom_loss(pred, true_val) {
    return mean((pred - true_val)^2)
}
```
## 6. Autonomic Optimization:
```
with autonomic {
    train(mymodel, mnist, loss: ce, opt: adam(lr=0.001), epochs=10, device: gpu)
}
```



