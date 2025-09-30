
#ifndef ESX_AST_HPP
#define ESX_AST_HPP

#include <boost/spirit/home/x3/support/ast/variant.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <memory>
#include <string>
#include <vector>

/**
 * @file ast.hpp
 * @brief Abstract Syntax Tree (AST) definitions for EasiScriptX (ESX).
 * @details Defines AST nodes for expressions, statements, and declarations in ESX scripts,
 * used by the parser and interpreter for AI/ML workflows.
 */

namespace esx::ast {
struct Expr;
struct Stmt;
struct TensorLit;
struct PipeExpr;
struct TrainStmt;
struct DistributeStmt;
struct AutonomicStmt;
struct AgentTuneStmt;
struct MatmulExpr;
struct IfStmt;
struct ForStmt;
struct WhileStmt;
struct GradExpr;
struct OdeSolveExpr;
struct QuantizeExpr;
struct PruneExpr;
struct DeployExpr;
struct AttentionExpr;
struct GnnConvExpr;
struct SqlTensorExpr;
struct FactorizeExpr;
struct ScaleExpr;
struct RateReduceExpr;
struct PicoBenchExpr;
struct MobiPruneExpr;
struct ReasoningPassExpr;
struct FnDecl;
struct ModelExpr;
struct DatasetExpr;
struct Conv2dExpr;
struct MaxPoolExpr;
struct BatchNormExpr;
struct LayerNormExpr;
struct DenseExpr;
struct SparseTensorLit;
struct ImportExpr;
struct ProfileStmt;
struct VisualizeExpr;
struct TokenizeExpr;

/**
 * @brief Represents a source code location for error reporting.
 */
struct Location {
    size_t line;
    size_t column;
};

/**
 * @brief Represents a dense tensor literal (e.g., tensor([[1, 2], [3, 4]])).
 */
struct TensorLit {
    std::vector<std::vector<double>> data; ///< Tensor data as 2D matrix.
    std::vector<size_t> shape; ///< Tensor shape (e.g., [2, 2]).
};

/**
 * @brief Represents a pipeline expression (e.g., x |> f).
 */
struct PipeExpr {
    std::shared_ptr<Expr> lhs; ///< Left-hand side expression.
    std::string op; ///< Pipeline operator or function name.
};

/**
 * @brief Represents a training statement (e.g., train(mymodel, mnist, ...)).
 */
struct TrainStmt {
    std::string model; ///< Model name.
    std::string data; ///< Dataset name.
    std::string loss; ///< Loss function (e.g., "ce", "mse").
    std::string opt; ///< Optimizer (e.g., "adam", "lamb", "adafactor").
    std::vector<std::pair<std::string, double>> opt_params; ///< Optimizer parameters (e.g., lr=0.001).
    int epochs; ///< Number of training epochs.
    std::string device; ///< Device (e.g., "cpu", "gpu").
    std::string reserved; ///< Reserved for future extensions (e.g., gradient checkpointing).
};

/**
 * @brief Represents a distributed training block (e.g., distribute(gpus: 2) { ... }).
 */
struct DistributeStmt {
    int gpus; ///< Number of GPUs for distributed training.
    std::shared_ptr<Stmt> body; ///< Statement block to execute.
};

/**
 * @brief Represents an autonomic optimization block (e.g., with autonomic { ... }).
 */
struct AutonomicStmt {
    std::shared_ptr<Stmt> body; ///< Statement block for optimization.
};

/**
 * @brief Represents a multi-agent tuning statement (e.g., multi_agent_tune(...)).
 */
struct AgentTuneStmt {
    std::string fn; ///< Function to tune.
    size_t agents; ///< Number of agents.
    std::string target; ///< Tuning target (e.g., "accuracy").
};

/**
 * @brief Represents a matrix multiplication expression (e.g., a @ b).
 */
struct MatmulExpr {
    std::shared_ptr<Expr> lhs; ///< Left tensor.
    std::shared_ptr<Expr> rhs; ///< Right tensor.
};

/**
 * @brief Represents an if statement (e.g., if cond { ... } else { ... }).
 */
struct IfStmt {
    std::shared_ptr<Expr> cond; ///< Condition expression.
    std::shared_ptr<Stmt> then_body; ///< Then branch.
    std::shared_ptr<Stmt> else_body; ///< Else branch (optional).
};

/**
 * @brief Represents a for loop (e.g., for i in 0..10 { ... }).
 */
struct ForStmt {
    std::string var; ///< Loop variable.
    std::shared_ptr<Expr> start; ///< Start expression.
    std::shared_ptr<Expr> end; ///< End expression.
    std::shared_ptr<Stmt> body; ///< Loop body.
};

/**
 * @brief Represents a while loop (e.g., while cond { ... }).
 */
struct WhileStmt {
    std::shared_ptr<Expr> cond; ///< Condition expression.
    std::shared_ptr<Stmt> body; ///< Loop body.
};

/**
 * @brief Represents a gradient computation (e.g., grad(fn)).
 */
struct GradExpr {
    std::shared_ptr<Expr> fn; ///< Function to compute gradient for.
};

/**
 * @brief Represents an ODE solver (e.g., ode_solve(eq, y0, t)).
 */
struct OdeSolveExpr {
    std::string eq; ///< Differential equation.
    double y0; ///< Initial condition.
    std::vector<double> t; ///< Time points.
};

/**
 * @brief Represents a quantization operation (e.g., quantize(bits=8)).
 */
struct QuantizeExpr {
    int bits; ///< Number of bits for quantization.
    std::string aware; ///< Quantization type (e.g., "post", "aware").
};

/**
 * @brief Represents a pruning operation (e.g., prune(ratio=0.5)).
 */
struct PruneExpr {
    double ratio; ///< Pruning ratio (0.0 to 1.0).
};

/**
 * @brief Represents a deployment operation (e.g., deploy(target: edge)).
 */
struct DeployExpr {
    std::string target; ///< Deployment target (e.g., "edge", "cloud").
    std::string device; ///< Device (e.g., "cpu", "tpu").
};

/**
 * @brief Represents a multi-head attention operation (e.g., attention(q, k, v)).
 */
struct AttentionExpr {
    std::shared_ptr<Expr> q; ///< Query tensor.
    std::shared_ptr<Expr> k; ///< Key tensor.
    std::shared_ptr<Expr> v; ///< Value tensor.
    int heads; ///< Number of attention heads.
    int dim; ///< Dimension per head.
};

/**
 * @brief Represents a graph neural network convolution (e.g., gnn_conv(graph, feats)).
 */
struct GnnConvExpr {
    std::shared_ptr<Expr> graph; ///< Graph structure.
    std::shared_ptr<Expr> feats; ///< Feature tensor.
};

/**
 * @brief Represents a SQL tensor query (e.g., sql_tensor(mat, query)).
 */
struct SqlTensorExpr {
    std::shared_ptr<Expr> mat; ///< Input tensor.
    std::string query; ///< SQL query string.
};

/**
 * @brief Represents a factorization operation (e.g., factorize(layer, mode)).
 */
struct FactorizeExpr {
    std::shared_ptr<Expr> layer; ///< Layer to factorize.
    std::string mode; ///< Factorization mode (e.g., "svd").
};

/**
 * @brief Represents a scaling operation (e.g., scale(model, factor=2.0)).
 */
struct ScaleExpr {
    std::shared_ptr<Expr> model; ///< Model to scale.
    double factor; ///< Scaling factor.
};

/**
 * @brief Represents a rate reduction operation (e.g., rate_reduce(data, freq)).
 */
struct RateReduceExpr {
    std::shared_ptr<Expr> data; ///< Input data.
    double freq; ///< Reduction frequency.
};

/**
 * @brief Represents a benchmarking operation (e.g., pico_bench(model, suite)).
 */
struct PicoBenchExpr {
    std::shared_ptr<Expr> model; ///< Model to benchmark.
    std::string suite; ///< Benchmark suite name.
};

/**
 * @brief Represents a mobile pruning operation (e.g., mobi_prune(battery: true)).
 */
struct MobiPruneExpr {
    bool battery; ///< Battery-aware pruning flag.
};

/**
 * @brief Represents a reasoning pass (e.g., reasoning_pass(fn, target)).
 */
struct ReasoningPassExpr {
    std::shared_ptr<Expr> fn; ///< Function to process.
    std::string target; ///< Reasoning target.
};

/**
 * @brief Represents a function declaration (e.g., fn name(params) -> type { ... }).
 */
struct FnDecl {
    std::string name; ///< Function name.
    std::vector<std::string> params; ///< Parameter names.
    std::string ret_type; ///< Return type.
    std::shared_ptr<Stmt> body; ///< Function body.
};

/**
 * @brief Represents a model definition (e.g., model mymodel { ... }).
 */
struct ModelExpr {
    std::string name; ///< Model name.
    std::vector<std::shared_ptr<Expr>> layers; ///< List of layer expressions.
};

/**
 * @brief Represents a dataset loading operation (e.g., load_dataset(mnist, ...)).
 */
struct DatasetExpr {
    std::string name; ///< Dataset name.
    std::string preprocess_fn; ///< Preprocessing function.
    std::vector<std::string> augment_ops; ///< Augmentation operations.
};

/**
 * @brief Represents a 2D convolution operation (e.g., conv2d(input, kernel, ...)).
 */
struct Conv2dExpr {
    std::shared_ptr<Expr> input; ///< Input tensor.
    std::shared_ptr<Expr> kernel; ///< Kernel tensor.
    int stride; ///< Convolution stride.
    std::string padding; ///< Padding type (e.g., "same", "valid").
};

/**
 * @brief Represents a max pooling operation (e.g., maxpool(input, pool_size)).
 */
struct MaxPoolExpr {
    std::shared_ptr<Expr> input; ///< Input tensor.
    std::vector<int> pool_size; ///< Pooling window size.
};

/**
 * @brief Represents a batch normalization operation (e.g., batchnorm(input)).
 */
struct BatchNormExpr {
    std::shared_ptr<Expr> input; ///< Input tensor.
};

/**
 * @brief Represents a layer normalization operation (e.g., layernorm(input)).
 */
struct LayerNormExpr {
    std::shared_ptr<Expr> input; ///< Input tensor.
};

/**
 * @brief Represents a dense layer operation (e.g., dense(units=128, activation=relu)).
 */
struct DenseExpr {
    std::shared_ptr<Expr> input; ///< Input tensor.
    int units; ///< Number of output units.
    std::string activation; ///< Activation function (e.g., "relu", "softmax").
};

/**
 * @brief Represents a sparse tensor literal (e.g., sparse_tensor([...], shape)).
 */
struct SparseTensorLit {
    std::vector<std::pair<std::vector<size_t>, double>> indices_values; ///< Non-zero indices and values.
    std::vector<size_t> shape; ///< Tensor shape.
};

/**
 * @brief Represents a pretrained model import (e.g., load_pretrained(resnet50)).
 */
struct ImportExpr {
    std::string name; ///< Model name (e.g., "resnet50").
};

/**
 * @brief Represents a profiling block (e.g., profile { ... }).
 */
struct ProfileStmt {
    std::shared_ptr<Stmt> body; ///< Statement block to profile.
};

/**
 * @brief Represents a visualization operation (e.g., visualize(data, type=heatmap)).
 */
struct VisualizeExpr {
    std::shared_ptr<Expr> data; ///< Data to visualize.
    std::string type; ///< Visualization type (e.g., "heatmap", "plot").
};

/**
 * @brief Represents a tokenization operation (e.g., tokenize(text, vocab)).
 */
struct TokenizeExpr {
    std::string text; ///< Input text.
    std::string vocab; ///< Vocabulary file (e.g., "vocab.json").
};

/**
 * @brief Variant type for all expression nodes.
 */
struct Expr : boost::spirit::x3::variant<
    TensorLit, PipeExpr, MatmulExpr, GradExpr, OdeSolveExpr, QuantizeExpr, PruneExpr,
    DeployExpr, AttentionExpr, GnnConvExpr, SqlTensorExpr, FactorizeExpr, ScaleExpr,
    RateReduceExpr, PicoBenchExpr, MobiPruneExpr, ReasoningPassExpr, ModelExpr, DatasetExpr,
    Conv2dExpr, MaxPoolExpr, BatchNormExpr, LayerNormExpr, DenseExpr, SparseTensorLit, ImportExpr,
    VisualizeExpr, TokenizeExpr
> {
    using base_type::base_type;
};

/**
 * @brief Variant type for all statement nodes.
 */
struct Stmt : boost::spirit::x3::variant<
    TrainStmt, DistributeStmt, AutonomicStmt, AgentTuneStmt, IfStmt, ForStmt, WhileStmt, FnDecl, ProfileStmt
> {
    using base_type::base_type;
};

/**
 * @brief Represents a variable declaration (e.g., let x = expr).
 */
struct Decl {
    std::string var; ///< Variable name.
    Expr expr; ///< Assigned expression.
};

/**
 * @brief Represents a complete ESX program.
 */
struct Program {
    std::vector<Decl> decls; ///< List of declarations.
    std::vector<Stmt> stmts; ///< List of statements.
    Location location; ///< Source location for error reporting.
};

/**
 * @brief Validates the AST for semantic correctness.
 * @param prog The program to validate.
 * @throws std::runtime_error if validation fails.
 */
void validate(const Program& prog) {
    // Stub for v1.0: Add checks for epochs > 0, valid tensor shapes, etc. in v1.1
    for (const auto& stmt : prog.stmts) {
        if (auto* train = boost::get<TrainStmt>(&stmt)) {
            if (train->epochs <= 0) {
                throw std::runtime_error("Invalid epochs: must be positive at line " + 
                    std::to_string(prog.location.line));
            }
        }
    }
}

} // namespace esx::ast

BOOST_FUSION_ADAPT_STRUCT(esx::ast::TensorLit, data, shape)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::PipeExpr, lhs, op)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::TrainStmt, model, data, loss, opt, opt_params, epochs, device, reserved)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::DistributeStmt, gpus, body)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::AutonomicStmt, body)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::AgentTuneStmt, fn, agents, target)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::MatmulExpr, lhs, rhs)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::IfStmt, cond, then_body, else_body)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::ForStmt, var, start, end, body)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::WhileStmt, cond, body)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::GradExpr, fn)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::OdeSolveExpr, eq, y0, t)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::QuantizeExpr, bits, aware)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::PruneExpr, ratio)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::DeployExpr, target, device)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::AttentionExpr, q, k, v, heads, dim)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::GnnConvExpr, graph, feats)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::SqlTensorExpr, mat, query)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::FactorizeExpr, layer, mode)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::ScaleExpr, model, factor)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::RateReduceExpr, data, freq)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::PicoBenchExpr, model, suite)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::MobiPruneExpr, battery)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::ReasoningPassExpr, fn, target)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::FnDecl, name, params, ret_type, body)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::ModelExpr, name, layers)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::DatasetExpr, name, preprocess_fn, augment_ops)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::Conv2dExpr, input, kernel, stride, padding)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::MaxPoolExpr, input, pool_size)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::BatchNormExpr, input)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::LayerNormExpr, input)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::DenseExpr, input, units, activation)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::SparseTensorLit, indices_values, shape)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::ImportExpr, name)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::ProfileStmt, body)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::VisualizeExpr, data, type)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::TokenizeExpr, text, vocab)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::Decl, var, expr)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::Program, decls, stmts, location)

#endif // ESX_AST_HPP
