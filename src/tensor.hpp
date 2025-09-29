
#ifndef ESX_TENSOR_HPP
#define ESX_TENSOR_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <torch/torch.h>
#include <vector>
#include <stdexcept>
#include <memory>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace esx::runtime {
// Memory pool for tensor allocation
class MemoryPool {
public:
    MemoryPool() = default;
    void* allocate(size_t size) {
        void* ptr = std::malloc(size);
        if (!ptr) throw std::runtime_error("Memory allocation failed");
        allocated.push_back(ptr);
        return ptr;
    }
    void deallocate_all() {
        for (void* ptr : allocated) std::free(ptr);
        allocated.clear();
    }
    ~MemoryPool() { deallocate_all(); }

private:
    std::vector<void*> allocated;
};

// Tensor class with PyTorch and Eigen backends
struct Tensor {
    torch::Tensor data;  // Primary storage (PyTorch)
    Eigen::MatrixXd dense_data;  // Fallback for CPU ops
    Eigen::SparseMatrix<double> sparse_data;  // For sparse ops
    bool is_sparse = false;
    std::vector<size_t> shape;
    static MemoryPool pool;

    Tensor() = default;

    Tensor(const std::vector<std::vector<double>>& input) {
        shape = {input.size(), input[0].size()};
        dense_data = Eigen::Map<const Eigen::MatrixXd>(&input[0][0], shape[0], shape[1]);
        data = torch::from_blob(dense_data.data(), {static_cast<int64_t>(shape[0]), static_cast<int64_t>(shape[1])}).clone();
    }

    Tensor(const std::vector<std::pair<std::vector<size_t>, double>>& indices_values, const std::vector<size_t>& shape_)
        : is_sparse(true), shape(shape_) {
        sparse_data.resize(shape[0], shape[1]);
        std::vector<Eigen::Triplet<double>> triplets;
        for (const auto& [indices, value] : indices_values) {
            triplets.emplace_back(indices[0], indices[1], value);
        }
        sparse_data.setFromTriplets(triplets.begin(), triplets.end());
        data = torch::sparse_coo_tensor(
            torch::tensor(indices_values, torch::kInt64),
            torch::tensor(std::vector<double>(indices_values.size(), 1.0)),
            {static_cast<int64_t>(shape[0]), static_cast<int64_t>(shape[1])}
        );
    }

    void to_device(const std::string& device) {
        if (device == "gpu" && USE_CUDA) {
            data = data.to(torch::kCUDA);
        } else {
            data = data.to(torch::kCPU);
        }
    }

    void to_half() {
        data = data.to(torch::kHalf);
    }
};

MemoryPool Tensor::pool;

// Abstract tensor operations
class TensorOps {
public:
    virtual Tensor matmul(const Tensor& a, const Tensor& b) = 0;
    virtual Tensor conv2d(const Tensor& input, const Tensor& kernel, int stride, const std::string& padding) = 0;
    virtual ~TensorOps() = default;
};

class CPUTensorOps : public TensorOps {
public:
    Tensor matmul(const Tensor& a, const Tensor& b) override {
        if (a.shape[1] != b.shape[0]) {
            throw std::runtime_error("Tensor shape mismatch in matmul. Expected (" + 
                std::to_string(a.shape[0]) + "," + std::to_string(a.shape[1]) + ") x (" +
                std::to_string(b.shape[0]) + "," + std::to_string(b.shape[1]) + ")");
        }
        Tensor result;
        result.dense_data = a.dense_data * b.dense_data;
        result.data = torch::matmul(a.data, b.data);
        result.shape = {a.shape[0], b.shape[1]};
        return result;
    }

    Tensor conv2d(const Tensor& input, const Tensor& kernel, int stride, const std::string& padding) override {
        Tensor result;
        result.data = torch::nn::functional::conv2d(
            input.data.view({1, 1, static_cast<int64_t>(input.shape[0]), static_cast<int64_t>(input.shape[1])}),
            kernel.data.view({1, 1, static_cast<int64_t>(kernel.shape[0]), static_cast<int64_t>(kernel.shape[1])}),
            torch::nn::functional::Conv2dFuncOptions().stride(stride).padding(padding == "same" ? 1 : 0)
        );
        result.shape = {static_cast<size_t>(result.data.size(2)), static_cast<size_t>(result.data.size(3))};
        return result;
    }
};

class GPUTensorOps : public TensorOps {
public:
    Tensor matmul(const Tensor& a, const Tensor& b) override {
#ifdef USE_CUDA
        if (a.shape[1] != b.shape[0]) {
            throw std::runtime_error("Tensor shape mismatch in matmul. Expected (" + 
                std::to_string(a.shape[0]) + "," + std::to_string(a.shape[1]) + ") x (" +
                std::to_string(b.shape[0]) + "," + std::to_string(b.shape[1]) + ")");
        }
        Tensor result;
        result.data = torch::matmul(a.data.to(torch::kCUDA), b.data.to(torch::kCUDA));
        result.shape = {a.shape[0], b.shape[1]};
        return result;
#else
        throw std::runtime_error("CUDA not enabled");
#endif
    }

    Tensor conv2d(const Tensor& input, const Tensor& kernel, int stride, const std::string& padding) override {
#ifdef USE_CUDA
        Tensor result;
        result.data = torch::nn::functional::conv2d(
            input.data.to(torch::kCUDA).view({1, 1, static_cast<int64_t>(input.shape[0]), static_cast<int64_t>(input.shape[1])}),
            kernel.data.to(torch::kCUDA).view({1, 1, static_cast<int64_t>(kernel.shape[0]), static_cast<int64_t>(kernel.shape[1])}),
            torch::nn::functional::Conv2dFuncOptions().stride(stride).padding(padding == "same" ? 1 : 0)
        );
        result.shape = {static_cast<size_t>(result.data.size(2)), static_cast<size_t>(result.data.size(3))};
        return result;
#else
        throw std::runtime_error("CUDA not enabled");
#endif
    }
};

} // namespace esx::runtime

#endif // ESX_TENSOR_HPP
