#ifndef ESX_DATASET_HPP
#define ESX_DATASET_HPP

#include "tensor.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>

/**
 * @file dataset.hpp
 * @brief Dataset management for EasiScriptX (ESX) runtime.
 * @details Provides a Dataset class for loading, preprocessing, and streaming datasets,
 * including support for basic tokenization and augmentation.
 */

namespace esx::runtime {
using json = nlohmann::json;

/**
 * @brief Represents a dataset for AI/ML workflows.
 */
struct Dataset {
    std::vector<Tensor> data; ///< Loaded dataset samples.
    size_t current_idx = 0; ///< Current batch index for streaming.

    /**
     * @brief Loads a dataset from a file with preprocessing and augmentation.
     * @param name Dataset name (e.g., "mnist").
     * @param preprocess_fn Preprocessing function (e.g., "normalize").
     * @param augment_ops List of augmentation operations (e.g., "rotate", "flip").
     * @param loc Source location for error reporting.
     * @throws std::runtime_error if loading or preprocessing fails.
     */
    void load(const std::string& name, const std::string& preprocess_fn, 
              const std::vector<std::string>& augment_ops, const ast::Location& loc) {
        std::ifstream file(name + ".txt");
        if (!file.is_open()) {
            throw std::runtime_error("Failed to load dataset '" + name + "' at line " + 
                                     std::to_string(loc.line) + ", column " + std::to_string(loc.column));
        }

        // Mock MNIST-like data: 28x28 images (784 floats)
        std::vector<std::vector<double>> sample(28, std::vector<double>(28, 1.0));
        data.emplace_back(sample);

        if (!preprocess_fn.empty()) {
            Eigen::Map<Eigen::MatrixXd> eigen_data(sample[0].data(), 28, 28);
            if (preprocess_fn == "normalize") {
                eigen_data /= 255.0; // Vectorized normalization
                for (size_t i = 0; i < sample.size(); ++i)
                    for (size_t j = 0; j < sample[i].size(); ++j)
                        sample[i][j] = eigen_data(i, j);
            }
            data[0] = Tensor(sample);
        }

        for (const auto& op : augment_ops) {
            if (op == "rotate") {
                Eigen::Map<Eigen::MatrixXd> eigen_data(sample[0].data(), 28, 28);
                eigen_data = eigen_data.transpose().eval(); // 90-degree rotation (mock)
                for (size_t i = 0; i < sample.size(); ++i)
                    for (size_t j = 0; j < sample[i].size(); ++j)
                        sample[i][j] = eigen_data(i, j);
                data.push_back(Tensor(sample));
            } else if (op == "flip") {
                Eigen::Map<Eigen::MatrixXd> eigen_data(sample[0].data(), 28, 28);
                eigen_data = eigen_data.colwise().reverse().eval(); // Horizontal flip
                for (size_t i = 0; i < sample.size(); ++i)
                    for (size_t j = 0; j < sample[i].size(); ++j)
                        sample[i][j] = eigen_data(i, j);
                data.push_back(Tensor(sample));
            }
        }
    }

    /**
     * @brief Tokenizes text using a JSON vocabulary.
     * @param text Input text to tokenize.
     * @param vocab_path Path to JSON vocabulary file.
     * @param loc Source location for error reporting.
     * @return Tensor of token IDs.
     * @throws std::runtime_error if vocabulary loading fails.
     */
    Tensor tokenize(const std::string& text, const std::string& vocab_path, const ast::Location& loc) {
        std::ifstream vocab_file(vocab_path);
        if (!vocab_file.is_open()) {
            throw std::runtime_error("Failed to load vocab '" + vocab_path + "' at line " + 
                                     std::to_string(loc.line) + ", column " + std::to_string(loc.column));
        }
        json vocab;
        vocab_file >> vocab;
        std::vector<std::vector<double>> tokens;
        for (char c : text) {
            std::string s(1, c);
            if (vocab.contains(s)) {
                tokens.push_back({static_cast<double>(vocab[s].get<int>())});
            } else {
                tokens.push_back({0.0}); // Unknown token
            }
        }
        return Tensor(tokens);
    }

    /**
     * @brief Retrieves the next batch of data.
     * @param size Batch size.
     * @return Tensor containing the batch.
     */
    Tensor next_batch(int size) {
        if (data.empty()) {
            throw std::runtime_error("Dataset is empty");
        }
        std::vector<std::vector<double>> batch_data;
        for (int i = 0; i < size && current_idx < data.size(); ++i) {
            batch_data.insert(batch_data.end(), data[current_idx].data.begin(), data[current_idx].data.end());
            current_idx++;
        }
        if (current_idx >= data.size()) {
            current_idx = 0; // Loop
        }
        return Tensor(batch_data);
    }
};

} // namespace esx::runtime

#endif // ESX_DATASET_HPP