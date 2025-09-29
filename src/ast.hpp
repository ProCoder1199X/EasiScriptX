#ifndef ESX_AST_HPP
#define ESX_AST_HPP

#include <boost/spirit/home/x3/support/ast/variant.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <memory>
#include <string>
#include <vector>

namespace esx::ast {
struct Expr;
struct Stmt;
struct TensorLit;
struct PipeExpr;
struct TrainStmt;
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
struct SparseTensorLit;
struct ImportExpr;
struct ProfileStmt;
struct VisualizeExpr;
struct TokenizeExpr;

struct TensorLit {
    std::vector<std::vector<double>> data;
    std::vector<size_t> shape;
};

struct PipeExpr {
    std::shared_ptr<Expr> lhs;
    std::string op;
};

struct TrainStmt {
    std::string model;
    std::string data;
    std::string loss;
    std::string opt;
    std::vector<std::pair<std::string, double>> opt_params;
    int epochs;
    std::string device;
};

struct AutonomicStmt {
    std::shared_ptr<Stmt> body;
};

struct AgentTuneStmt {
    std::string fn;
    size_t agents;
    std::string target;
};

struct MatmulExpr {
    std::shared_ptr<Expr> lhs;
    std::shared_ptr<Expr> rhs;
};

struct IfStmt {
    std::shared_ptr<Expr> cond;
    std::shared_ptr<Stmt> then_body;
    std::shared_ptr<Stmt> else_body;
};

struct ForStmt {
    std::string var;
    std::shared_ptr<Expr> start;
    std::shared_ptr<Expr> end;
    std::shared_ptr<Stmt> body;
};

struct WhileStmt {
    std::shared_ptr<Expr> cond;
    std::shared_ptr<Stmt> body;
};

struct GradExpr {
    std::shared_ptr<Expr> fn;
};

struct OdeSolveExpr {
    std::string eq;
    double y0;
    std::vector<double> t;
};

struct QuantizeExpr {
    int bits;
    std::string aware;
};

struct PruneExpr {
    double ratio;
};

struct DeployExpr {
    std::string target;
    std::string device;
};

struct AttentionExpr {
    std::shared_ptr<Expr> q;
    std::shared_ptr<Expr> k;
    std::shared_ptr<Expr> v;
    int heads;
    int dim;
};

struct GnnConvExpr {
    std::shared_ptr<Expr> graph;
    std::shared_ptr<Expr> feats;
};

struct SqlTensorExpr {
    std::shared_ptr<Expr> mat;
    std::string query;
};

struct FactorizeExpr {
    std::shared_ptr<Expr> layer;
    std::string mode;
};

struct ScaleExpr {
    std::shared_ptr<Expr> model;
    double factor;
};

struct RateReduceExpr {
    std::shared_ptr<Expr> data;
    double freq;
};

struct PicoBenchExpr {
    std::shared_ptr<Expr> model;
    std::string suite;
};

struct MobiPruneExpr {
    bool battery;
};

struct ReasoningPassExpr {
    std::shared_ptr<Expr> fn;
    std::string target;
};

struct FnDecl {
    std::string name;
    std::vector<std::string> params;
    std::string ret_type;
    std::shared_ptr<Stmt> body;
};

struct ModelExpr {
    std::string name;
    std::vector<std::shared_ptr<Expr>> layers;
};

struct DatasetExpr {
    std::string name;
    std::string preprocess_fn;
    std::vector<std::string> augment_ops;
};

struct Conv2dExpr {
    std::shared_ptr<Expr> input;
    std::shared_ptr<Expr> kernel;
    int stride;
    std::string padding;
};

struct MaxPoolExpr {
    std::shared_ptr<Expr> input;
    std::vector<int> pool_size;
};

struct BatchNormExpr {
    std::shared_ptr<Expr> input;
};

struct LayerNormExpr {
    std::shared_ptr<Expr> input;
};

struct SparseTensorLit {
    std::vector<std::pair<std::vector<size_t>, double>> indices_values;
    std::vector<size_t> shape;
};

struct ImportExpr {
    std::string name;
};

struct ProfileStmt {
    std::shared_ptr<Stmt> body;
};

struct VisualizeExpr {
    std::shared_ptr<Expr> data;
    std::string type;
};

struct TokenizeExpr {
    std::string text;
    std::string vocab;
};

struct Expr : boost::spirit::x3::variant<
    TensorLit, PipeExpr, MatmulExpr, GradExpr, OdeSolveExpr, QuantizeExpr, PruneExpr,
    DeployExpr, AttentionExpr, GnnConvExpr, SqlTensorExpr, FactorizeExpr, ScaleExpr,
    RateReduceExpr, PicoBenchExpr, MobiPruneExpr, ReasoningPassExpr, ModelExpr, DatasetExpr,
    Conv2dExpr, MaxPoolExpr, BatchNormExpr, LayerNormExpr, SparseTensorLit, ImportExpr,
    VisualizeExpr, TokenizeExpr
> {
    using base_type::base_type;
};

struct Stmt : boost::spirit::x3::variant<
    TrainStmt, AutonomicStmt, AgentTuneStmt, IfStmt, ForStmt, WhileStmt, FnDecl, ProfileStmt
> {
    using base_type::base_type;
};

struct Decl {
    std::string var;
    Expr expr;
};

struct Program {
    std::vector<Decl> decls;
    std::vector<Stmt> stmts;
};

} // namespace esx::ast

BOOST_FUSION_ADAPT_STRUCT(esx::ast::TensorLit, data, shape)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::PipeExpr, lhs, op)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::TrainStmt, model, data, loss, opt, opt_params, epochs, device)
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
BOOST_FUSION_ADAPT_STRUCT(esx::ast::SparseTensorLit, indices_values, shape)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::ImportExpr, name)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::ProfileStmt, body)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::VisualizeExpr, data, type)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::TokenizeExpr, text, vocab)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::Decl, var, expr)
BOOST_FUSION_ADAPT_STRUCT(esx::ast::Program, decls, stmts)

#endif // ESX_AST_HPP
