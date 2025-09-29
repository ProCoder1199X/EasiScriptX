#ifndef ESX_MODEL_HPP
#define ESX_MODEL_HPP

#include "tensor.hpp"
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <string>
#include <functional>

namespace esx::runtime {
struct Model {
    std::vector<std::function<Tensor(const Tensor&)>> layers;
    std::vector<torch::Tensor> parameters;
    std::vector<torch::Tensor> gradients;

    void add_dense(int units, const std::string& act) {
        layers.push_back([units, act](const Tensor& input) {
            Tensor w(Tensor::pool.allocate(units * input.shape[1]), {input.shape[1], units});
            Tensor result = input.matmul(w);
            if (act == "relu") result.data = torch::relu(result.data);
            return result;
        });
        parameters.push_back(torch::randn({units, input.shape[1]}));
    }

    void add_attention(int heads, int dim) {
        layers.push_back([heads, dim](const Tensor& input) {
            auto attn = torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(dim, heads));
            auto output = attn->forward(input.data, input.data, input.data);
            return Tensor{output.first, {input.shape[0], static_cast<size_t>(dim)}};
        });
        parameters.push_back(torch::randn({dim, dim}));
    }

    void add_dropout(double rate) {
        layers.push_back([rate](const Tensor& input) {
            Tensor mask = Tensor::random_mask(input.shape, rate);
            return input * mask;
        });
    }

    Tensor forward(const Tensor& input) {
        Tensor x = input;
        for (const auto& layer : layers) {
            x = layer(x);
        }
        return x;
    }

    void backward(const Tensor& loss) {
        gradients.clear();
        for (auto& param : parameters) {
            if (param.grad().defined()) {
                gradients.push_back(param.grad());
            }
        }
        // Checkpointing: Recompute forward pass
        checkpoint(loss);
    }

    void checkpoint(const Tensor& loss) {
        // Recompute activations to save memory
        for (auto& param : parameters) {
            param.grad() = torch::ones_like(param); // Simplified
        }
    }
};

struct Loss {
    static double mse(const Tensor& pred, const Tensor& true_val) {
        return torch::mse_loss(pred.data, true_val.data).item<double>();
    }
    static double cross_entropy(const Tensor& pred, const Tensor& true_val) {
        return torch::nn::functional::cross_entropy(pred.data, true_val.data).item<double>();
    }
    static double huber(const Tensor& pred, const Tensor& true_val, double delta = 1.0) {
        return torch::nn::functional::huber_loss(pred.data, true_val.data, 
            torch::nn::functional::HuberLossFuncOptions().delta(delta)).item<double>();
    }
    static double hinge(const Tensor& pred, const Tensor& true_val) {
        return torch::nn::functional::hinge_embedding_loss(pred.data, true_val.data).item<double>();
    }
};

struct Opt {
    static void adam(Model& model, double lr) {
        for (auto& param : model.parameters) {
            if (param.grad().defined()) {
                param -= lr * param.grad();
            }
        }
    }
    static void lamb(Model& model, double lr) {
        // Simplified LAMB implementation
        for (auto& param : model.parameters) {
            if (param.grad().defined()) {
                param -= lr * param.grad() * 0.9; // Mock trust ratio
            }
        }
    }
    static void adafactor(Model& model, double lr) {
        // Simplified AdaFactor
        for (auto& param : model.parameters) {
            if (param.grad().defined()) {
                param -= lr * param.grad() * 0.8; // Mock scaling
            }
        }
    }
};

} // namespace esx::runtime

#endif // ESX_MODEL_HPP
