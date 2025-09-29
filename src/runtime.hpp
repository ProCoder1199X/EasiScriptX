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
            Tensor w(Tensor::pool.allocate(units * input.shape[1]), {input.shape[1], units});
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
        if (auto* tensor = boost::spirit::x3::get<ast::TensorLit>(&e)) {
            return Tensor{tensor->data};
        }

        if (auto* matmul = boost::spirit::x3::get<ast::MatmulExpr>(&e)) {
            auto lhs = eval_expr(*matmul->lhs);
            auto rhs = eval_expr(*matmul->rhs);
            return lhs.matmul(rhs);
        }

        if (auto* conv = boost::spirit::x3::get<ast::Conv2dExpr>(&e)) {
            auto input = eval_expr(*conv->input);
            auto kernel = eval_expr(*conv->kernel);
            return input.conv2d(kernel, conv->stride, conv->padding);
        }

        throw std::runtime_error("Unsupported expression");
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
            ncclCommInitAll(comms, dist->gpus, 0); // Initialize NCCL for multi-GPU
            // Perform distributed matrix multiplication
            ncclCommDestroy(comms[0]);
#endif
            MPI_Init(nullptr, nullptr); // Initialize MPI for multi-node
            // Perform distributed gradient aggregation
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

    void define_custom_optimizer(const std::string& name, const std::function<void(Model&, double)>& fn) {
        // Stub store in map
    }
};


#endif

if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    doxygen_add_docs(docs src ALL)
endif()