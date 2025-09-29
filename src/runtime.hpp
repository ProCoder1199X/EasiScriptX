#ifndef ESX_RUNTIME_HPP
#define ESX_RUNTIME_HPP

#include "ast.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <onnxruntime_cxx_api.h> // For Hugging Face
#include <mpi.h> // For distributed
#include <vector>
#include <map>
#include <chrono>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <mutex>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nccl.h> // For multi-GPU
#include <cuda_fp16.h> // Mixed prec
#endif

namespace esx::runtime {
// ... (previous memory pool, Tensor with sparse)

struct Model {
    std::vector<std::function<Tensor(const Tensor&)>> layers; // Forward funcs
    void add_dense(int units, std::string act) {
        layers.push_back([units, act](const Tensor& input) {
            Tensor w; // Random weights stub
            return input.matmul(w); // Stub forward
        });
    }
    // Add dropout, etc.
    Tensor forward(const Tensor& input) {
        Tensor x = input;
        for (auto& layer : layers) x = layer(x);
        return x;
    }
};

struct Dataset {
    std::vector<Tensor> data;
    void load(const std::string& name, const std::string& preprocess, const std::vector<std::string>& augment) {
        // Streaming stub: Load chunks
        data.push_back(Tensor()); // Dummy
        std::cout << "Loaded " << name << " with preprocess " << preprocess << std::endl;
    }
    Tensor next_batch(int size) { return data[0]; } // Stub
};

class Interpreter {
public:
    std::map<std::string, Tensor> vars;
    std::map<std::string, std::function<Tensor(const std::vector<Tensor>&)>> funcs;
    std::map<std::string, Model> models;
    bool debug = false;
    Ort::Env env; // ONNX Runtime

    Interpreter() : env(ORT_LOGGING_LEVEL_WARNING, "ESX") {}

    void execute(const ast::Program& prog, bool debug_mode = false) {
        debug = debug_mode;
        try {
            for (const auto& decl : prog.decls) {
                vars[decl.var] = eval_expr(decl.expr);
            }
            for (const auto& stmt : prog.stmts) {
                exec_stmt(stmt);
            }
        } catch (const std::exception& e) {
            std::cerr << "Runtime error: " << e.what() << std::endl;
        }
    }

    Tensor eval_expr(const ast::Expr& e) {
        // ... (previous)
        if (auto* model = boost::spirit::x3::get<ast::ModelExpr>(&e)) {
            Model m;
            for (auto& layer : model->layers) {
                if (auto* dense = boost::spirit::x3::get<ast::DenseExpr>(&*layer)) {
                    m.add_dense(dense->units, dense->act);
                }
                // ... Add dropout, etc.
            }
            models[model->name] = m;
            return Tensor(); // Dummy
        }
        if (auto* dataset = boost::spirit::x3::get<ast::DatasetExpr>(&e)) {
            Dataset d;
            d.load(dataset->name, dataset->preprocess_fn, dataset->augment_ops);
            return d.next_batch(32); // Dummy batch
        }
        if (auto* import_ = boost::spirit::x3::get<ast::ImportExpr>(&e)) {
            // Hugging Face via ONNX
            Ort::SessionOptions session_options;
            Ort::Session session(env, "pretrained/" + import_->name + ".onnx", session_options);
            // Stub input/output tensors
            std::cout << "Loaded pretrained " << import_->name << "\n";
            return Tensor();
        }
        // ... (conv2d, maxpool, etc. with Eigen conv stubs)
        throw std::runtime_error("Unsupported expr");
    }

    void exec_stmt(const ast::Stmt& s) {
        if (auto* train = boost::spirit::x3::get<ast::TrainStmt>(&s)) {
            Model& m = models[train->model];
            Dataset d; // Load from train->data
            for (int e = 0; e < train->epochs; ++e) {
                Tensor batch = d.next_batch(32);
                Tensor pred = m.forward(batch);
                double loss_val = Loss::cross_entropy(pred, batch); // Custom loss
                m.grad(); // Stub backprop with checkpointing
                Opt::adam(m, train->opt_params[0].second); // Custom opt
            }
        }
        // Distributed stub
        if (auto* dist = boost::spirit::x3::get<ast::DistributeStmt>(&s)) {
#ifdef USE_CUDA
            ncclComm_t comms[dist->gpus];
            ncclCommInitAll(comms, dist->gpus, 0); // NCCL multi-GPU
            // Matmul across GPUs
            ncclCommDestroy(comms[0]);
#endif
            MPI_Init(nullptr, nullptr); // MPI for nodes
            // ... Stub
            MPI_Finalize();
        }
        // ... (previous)
    }

    // Gradient checkpointing
    void checkpoint_grad(const Tensor& loss) {
        // Recompute forward from checkpoints (OpenAI style)
        std::cout << "Checkpointing stub: Recomputed grads\n";
    }

    // Custom loss/opt
    void define_custom_loss(const std::string& name, const std::function<double(const Tensor&, const Tensor&)>& fn) {
        // Stub store in map
    }

    // ... (previous)
};

} // namespace

#endif