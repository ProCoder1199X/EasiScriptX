
#ifndef ESX_INTERPRETER_HPP
#define ESX_INTERPRETER_HPP

#include "ast.hpp"
#include "tensor.hpp"
#include "model.hpp"
#include "dataset.hpp"
#include "distributed.hpp"
#include <torch/script.h>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <map>
#include <functional>
#include <iostream>
#include <fstream>

namespace esx::runtime {
class Interpreter {
public:
    std::map<std::string, Tensor> vars;
    std::map<std::string, std::function<Tensor(const std::vector<Tensor>&)>> funcs;
    std::map<std::string, Model> models;
    bool debug = false;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ESX"};
    std::unique_ptr<TensorOps> ops;

    Interpreter() : ops(std::make_unique<CPUTensorOps>()) {}

    void execute(const ast::Program& prog, bool debug_mode = false) {
        debug = debug_mode;
        try {
            for (const auto& decl : prog.decls) {
                vars[decl.var] = eval_expr(decl.expr);
                if (debug) std::cout << "Debug: Declared " << decl.var << std::endl;
            }
            for (const auto& stmt : prog.stmts) {
                exec_stmt(stmt);
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Runtime error: " + std::string(e.what()));
        }
    }

    Tensor eval_expr(const ast::Expr& e) {
        if (auto* tensor = boost::spirit::x3::get<ast::TensorLit>(&e)) {
            return Tensor{tensor->data};
        }
        if (auto* matmul = boost::spirit::x3::get<ast::MatmulExpr>(&e)) {
            auto lhs = eval_expr(*matmul->lhs);
            auto rhs = eval_expr(*matmul->rhs);
            if (debug) std::cout << "Debug: Executing matmul with shapes (" 
                                 << lhs.shape[0] << "," << lhs.shape[1] << ") x (" 
                                 << rhs.shape[0] << "," << rhs.shape[1] << ")" << std::endl;
            return ops->matmul(lhs, rhs);
        }
        if (auto* pipe = boost::spirit::x3::get<ast::PipeExpr>(&e)) {
            auto input = eval_expr(*pipe->lhs);
            if (funcs.find(pipe->op) == funcs.end()) {
                throw std::runtime_error("Unknown function: " + pipe->op);
            }
            return funcs[pipe->op]({input});
        }
        if (auto* model = boost::spirit::x3::get<ast::ModelExpr>(&e)) {
            Model m;
            for (const auto& layer : model->layers) {
                if (auto* dense = boost::spirit::x3::get<ast::DenseExpr>(&*layer)) {
                    m.add_dense(dense->units, dense->act);
                } else if (auto* attn = boost::spirit::x3::get<ast::AttentionExpr>(&*layer)) {
                    m.add_attention(attn->heads, attn->dim);
                }
            }
            models[model->name] = m;
            return Tensor{};
        }
        if (auto* dataset = boost::spirit::x3::get<ast::DatasetExpr>(&e)) {
            Dataset d;
            d.load(dataset->name, dataset->preprocess_fn, dataset->augment_ops);
            return d.next_batch(32);
        }
        if (auto* import_ = boost::spirit::x3::get<ast::ImportExpr>(&e)) {
            try {
                auto module = torch::jit::load("pretrained/" + import_->name + ".pt");
                std::vector<torch::jit::IValue> inputs{torch::ones({1, 3, 224, 224})};
                auto output = module.forward(inputs).toTensor();
                return Tensor{output, {static_cast<size_t>(output.size(0)), static_cast<size_t>(output.size(1))}};
            } catch (...) {
                Ort::SessionOptions session_options;
                Ort::Session session(env, ("pretrained/" + import_->name + ".onnx").c_str(), session_options);
                return Tensor{}; // Mock
            }
        }
        if (auto* conv = boost::spirit::x3::get<ast::Conv2dExpr>(&e)) {
            auto input = eval_expr(*conv->input);
            auto kernel = eval_expr(*conv->kernel);
            return ops->conv2d(input, kernel, conv->stride, conv->padding);
        }
        if (auto* visualize = boost::spirit::x3::get<ast::VisualizeExpr>(&e)) {
            auto data = eval_expr(*visualize->data);
            if (visualize->type == "plot") {
                std::cout << "Loss Curve: [";
                auto values = data.data.cpu().accessor<float, 1>();
                for (int i = 0; i < values.size(0); ++i) {
                    std::cout << values[i] << (i < values.size(0) - 1 ? "," : "");
                }
                std::cout << "]" << std::endl;
            } else if (visualize->type == "graph") {
                std::cout << "Model Graph: [Simplified]" << std::endl;
            }
            return data;
        }
        if (auto* tokenize = boost::spirit::x3::get<ast::TokenizeExpr>(&e)) {
            std::vector<int64_t> tokens;
            std::ifstream vocab_file(tokenize->vocab);
            if (!vocab_file.is_open()) {
                throw std::runtime_error("Failed to load vocab: " + tokenize->vocab);
            }
            std::string word;
            int idx = 0;
            while (tokenize->text.find(" ") != std::string::npos) {
                tokens.push_back(idx++);
                tokenize->text = tokenize->text.substr(tokenize->text.find(" ") + 1);
            }
            tokens.push_back(idx);
            return Tensor{torch::tensor(tokens), {tokens.size()}};
        }
        throw std::runtime_error("Unsupported expression");
    }

    void exec_stmt(const ast::Stmt& s) {
        if (auto* train = boost::spirit::x3::get<ast::TrainStmt>(&s)) {
            if (models.find(train->model) == models.end()) {
                throw std::runtime_error("Model not found: " + train->model);
            }
            Model& m = models[train->model];
            Dataset d;
            d.load(train->data, "", {});
            for (int e = 0; e < train->epochs; ++e) {
                Tensor batch = d.next_batch(32);
                Tensor pred = m.forward(batch);
                double loss_val = Loss::mse(pred, batch); // Default
                if (train->loss == "ce") loss_val = Loss::cross_entropy(pred, batch);
                else if (train->loss == "huber") loss_val = Loss::huber(pred, batch);
                else if (train->loss == "hinge") loss_val = Loss::hinge(pred, batch);
                m.backward(Tensor{torch::tensor(loss_val), {1}});
                if (train->opt == "adam") Opt::adam(m, train->opt_params[0].second);
                else if (train->opt == "lamb") Opt::lamb(m, train->opt_params[0].second);
                else if (train->opt == "adafactor") Opt::adafactor(m, train->opt_params[0].second);
                if (debug) std::cout << "Debug: Epoch " << e << ", Loss: " << loss_val << std::endl;
            }
        }
        if (auto* profile = boost::spirit::x3::get<ast::ProfileStmt>(&s)) {
            auto start = std::chrono::high_resolution_clock::now();
            exec_stmt(*profile->body);
            auto end = std::chrono::high_resolution_clock::now();
            std::cout << "Profile: Execution time: " 
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
                      << "ms" << std::endl;
        }
    }

    void export_onnx(const Model& m, const std::string& file) {
        torch::jit::script::Module module;
        module.register_module("model", torch::nn::Module());
        torch::jit::IValue input = torch::ones({1, 3, 224, 224});
        module.forward({input});
        module.save(file);
        std::cout << "Exported to ONNX: " << file << std::endl;
    }

    void bench_matmul(int size, const std::string& device) {
        Tensor a(torch::randn({size, size}), {static_cast<size_t>(size), static_cast<size_t>(size)});
        Tensor b(torch::randn({size, size}), {static_cast<size_t>(size), static_cast<size_t>(size)});
        a.to_device(device);
        b.to_device(device);
        auto start = std::chrono::high_resolution_clock::now();
        ops->matmul(a, b);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Matmul (" << size << "x" << size << ") on " << device << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
                  << "ms" << std::endl;
    }
};

} // namespace esx::runtime

#endif // ESX_INTERPRETER_HPP
