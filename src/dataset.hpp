#ifndef ESX_DATASET_HPP
#define ESX_DATASET_HPP

#include "tensor.hpp"
#include <vector>
#include <string>
#include <fstream>

namespace esx::runtime {
struct Dataset {
    std::vector<Tensor> data;
    size_t current_idx = 0;

    void load(const std::string& name, const std::string& preprocess_fn, const std::vector<std::string>& augment_ops) {
        std::ifstream file(name + ".txt");
        if (!file.is_open()) {
            throw std::runtime_error("Failed to load dataset: " + name);
        }

        std::vector<std::vector<double>> sample = {{1, 2}, {3, 4}}; // Mock data
        data.emplace_back(sample);

        if (!preprocess_fn.empty()) {
            for (auto& tensor : data) {
                tensor.data = tensor.data / 255.0; // Normalize
            }
        }

        for (const auto& op : augment_ops) {
            if (op == "rotate") {
                // Rotate the tensor
            } else if (op == "flip") {
                // Flip the tensor
            }
        }
    }

    Tensor next_batch(int size) {
        if (current_idx >= data.size()) {
            current_idx = 0; // Loop
        }
        Tensor batch = data[current_idx];
        current_idx += size;
        return batch;
    }
};

} // namespace esx::runtime

#endif // ESX_DATASET_HPP
