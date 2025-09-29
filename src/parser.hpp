#ifndef ESX_PARSER_HPP
#define ESX_PARSER_HPP

#include "ast.hpp"
#include <boost/spirit/home/x3.hpp>
#include <boost/spirit/home/x3/support/utility/error_reporting.hpp>
#include <boost/spirit/home/x3/support/utility/annotate_on_success.hpp>
#include <string>
#include <sstream>

namespace esx::parser {
namespace x3 = boost::spirit::x3;

struct error_handler {
    template <typename Iterator, typename Exception, typename Context>
    x3::error_handler_result on_error(Iterator& first, Iterator const& last, Exception const& x, Context const& context) {
        auto& error = x3::get<x3::error_handler_tag>(context).get();
        std::stringstream msg;
        msg << "Error: Unexpected token '" << x.which() << "' at line " 
            << (x.where() - first + 1) << ", column " 
            << (x.where() - std::find_if_not(first, x.where(), [](char c) { return c != '\n'; }) + 1)
            << ". Expected: " << expected_tokens(x.which());
        error(first, last, msg.str(), x.where());
        return x3::error_handler_result::fail;
    }

private:
    std::string expected_tokens(const std::string& token) {
        if (token == "train") return "'let', 'model', or 'fn'";
        if (token == "@") return "tensor expression";
        return "valid expression or statement";
    }
};

auto const ident = x3::lexeme[x3::alpha >> *x3::alnum];
auto const number = x3::double_;
auto const skip = x3::space | x3::eol;

auto const tensor_row = '[' >> number % ',' >> ']';
auto const tensor_lit = x3::rule<class tensor_lit, ast::TensorLit>{"tensor_lit"} = 
    x3::lit("tensor") >> '(' >> '[' >> tensor_row % ',' >> ']' >> ')';

auto const matmul_expr = x3::rule<class matmul_expr, ast::MatmulExpr>{"matmul_expr"} = 
    expr >> '@' >> expr;

auto const pipe_expr = x3::rule<class pipe_expr, ast::PipeExpr>{"pipe_expr"} = 
    expr >> *x3::omit[x3::lit("|") >> ">"] >> ident;

auto const param = ident >> '=' >> number;
auto const train_stmt = x3::rule<class train_stmt, ast::TrainStmt>{"train_stmt"} = 
    x3::lit("train") >> '(' >> ident >> ',' >> ident >> ',' >> "loss:" >> ident >> ',' 
    >> "opt:" >> ident >> -('(' >> param % ',' >> ')') >> ',' >> "epochs=" >> x3::int_ 
    >> ',' >> "device:" >> ident >> ')';

auto const autonomic_stmt = x3::rule<class autonomic_stmt, ast::AutonomicStmt>{"autonomic_stmt"} = 
    x3::lit("with") >> "autonomic" >> '{' >> stmt >> '}';

auto const agent_tune_stmt = x3::rule<class agent_tune_stmt, ast::AgentTuneStmt>{"agent_tune_stmt"} = 
    x3::lit("multi_agent_tune") >> '(' >> ident >> ',' >> "agents:" >> x3::uint_ 
    >> ',' >> "target:" >> ident >> ')';

auto const if_stmt = x3::rule<class if_stmt, ast::IfStmt>{"if_stmt"} = 
    x3::lit("if") >> expr >> '{' >> stmt >> '}' >> - (x3::lit("else") >> '{' >> stmt >> '}');

auto const for_stmt = x3::rule<class for_stmt, ast::ForStmt>{"for_stmt"} = 
    x3::lit("for") >> ident >> "in" >> expr >> ".." >> expr >> '{' >> stmt >> '}';

auto const while_stmt = x3::rule<class while_stmt, ast::WhileStmt>{"while_stmt"} = 
    x3::lit("while") >> expr >> '{' >> stmt >> '}';

auto const grad_expr = x3::rule<class grad_expr, ast::GradExpr>{"grad_expr"} = 
    x3::lit("grad") >> '(' >> expr >> ')';

auto const ode_solve_expr = x3::rule<class ode_solve_expr, ast::OdeSolveExpr>{"ode_solve_expr"} = 
    x3::lit("ode_solve") >> '(' >> x3::string >> ',' >> number >> ',' >> '[' >> number % ',' >> ']' >> ')';

auto const quantize_expr = x3::rule<class quantize_expr, ast::QuantizeExpr>{"quantize_expr"} = 
    x3::lit("quantize") >> '(' >> "bits=" >> x3::int_ >> ',' >> "aware:" >> ident >> ')';

auto const prune_expr = x3::rule<class prune_expr, ast::PruneExpr>{"prune_expr"} = 
    x3::lit("prune") >> '(' >> "ratio=" >> number >> ')';

auto const deploy_expr = x3::rule<class deploy_expr, ast::DeployExpr>{"deploy_expr"} = 
    x3::lit("deploy") >> '(' >> "target:" >> ident >> ',' >> "device:" >> ident >> ')';

auto const attention_expr = x3::rule<class attention_expr, ast::AttentionExpr>{"attention_expr"} = 
    x3::lit("attention") >> '(' >> expr >> ',' >> expr >> ',' >> expr >> ',' 
    >> "heads=" >> x3::int_ >> ',' >> "dim=" >> x3::int_ >> ')';

auto const gnn_conv_expr = x3::rule<class gnn_conv_expr, ast::GnnConvExpr>{"gnn_conv_expr"} = 
    x3::lit("gnn_conv") >> '(' >> expr >> ',' >> expr >> ')';

auto const sql_tensor_expr = x3::rule<class sql_tensor_expr, ast::SqlTensorExpr>{"sql_tensor_expr"} = 
    x3::lit("sql_tensor") >> '(' >> expr >> ',' >> x3::string >> ')';

auto const factorize_expr = x3::rule<class factorize_expr, ast::FactorizeExpr>{"factorize_expr"} = 
    x3::lit("factorize") >> '(' >> expr >> ',' >> "mode:" >> ident >> ')';

auto const scale_expr = x3::rule<class scale_expr, ast::ScaleExpr>{"scale_expr"} = 
    x3::lit("scale") >> '(' >> expr >> ',' >> "factor=" >> number >> ')';

auto const rate_reduce_expr = x3::rule<class rate_reduce_expr, ast::RateReduceExpr>{"rate_reduce_expr"} = 
    x3::lit("rate_reduce") >> '(' >> expr >> ',' >> "freq=" >> number >> ')';

auto const pico_bench_expr = x3::rule<class pico_bench_expr, ast::PicoBenchExpr>{"pico_bench_expr"} = 
    x3::lit("pico_bench") >> '(' >> expr >> ',' >> "suite:" >> ident >> ')';

auto const mobi_prune_expr = x3::rule<class mobi_prune_expr, ast::MobiPruneExpr>{"mobi_prune_expr"} = 
    x3::lit("mobi_prune") >> '(' >> "battery:" >> x3::bool_ >> ')';

auto const reasoning_pass_expr = x3::rule<class reasoning_pass_expr, ast::ReasoningPassExpr>{"reasoning_pass_expr"} = 
    x3::lit("reasoning_pass") >> '(' >> expr >> ',' >> "target:" >> ident >> ')';

auto const fn_decl = x3::rule<class fn_decl, ast::FnDecl>{"fn_decl"} = 
    x3::lit("fn") >> ident >> '(' >> (ident % ',') >> ')' >> "->" >> ident >> '{' >> stmt >> '}';

auto const layer_expr = x3::lit("layer") >> ident >> -('(' >> (ident >> '=' >> (number | ident)) % ',' >> ')');
auto const model_expr = x3::rule<class model_expr, ast::ModelExpr>{"model_expr"} = 
    x3::lit("model") >> ident >> '{' >> *layer_expr >> '}';

auto const augment_op = ident;
auto const dataset_expr = x3::rule<class dataset_expr, ast::DatasetExpr>{"dataset_expr"} = 
    x3::lit("load_dataset") >> '(' >> ident >> ',' >> "preprocess:" >> ident >> ',' >> "augment:" >> '[' >> augment_op % ',' >> ']' >> ')';

auto const conv2d_expr = x3::rule<class conv2d_expr, ast::Conv2dExpr>{"conv2d_expr"} = 
    x3::lit("conv2d") >> '(' >> expr >> ',' >> expr >> ',' >> "stride=" >> x3::int_ >> ',' >> "padding=" >> ident >> ')';

auto const pool_size = '(' >> x3::int_ >> ',' >> x3::int_ >> ')';
auto const maxpool_expr = x3::rule<class maxpool_expr, ast::MaxPoolExpr>{"maxpool_expr"} = 
    x3::lit("maxpool") >> '(' >> expr >> ',' >> "pool_size=" >> pool_size >> ')';

auto const batchnorm_expr = x3::rule<class batchnorm_expr, ast::BatchNormExpr>{"batchnorm_expr"} = 
    x3::lit("batchnorm") >> '(' >> expr >> ')';

auto const layernorm_expr = x3::rule<class layernorm_expr, ast::LayerNormExpr>{"layernorm_expr"} = 
    x3::lit("layernorm") >> '(' >> expr >> ')';

auto const index = '[' >> x3::int_ % ',' >> ']';
auto const index_value = index >> ':' >> number;
auto const sparse_tensor_lit = x3::rule<class sparse_tensor_lit, ast::SparseTensorLit>{"sparse_tensor_lit"} = 
    x3::lit("sparse_tensor") >> '(' >> '[' >> index_value % ',' >> ']' >> ',' >> '[' >> x3::int_ % ',' >> ']' >> ')';

auto const import_expr = x3::rule<class import_expr, ast::ImportExpr>{"import_expr"} = 
    x3::lit("load_pretrained") >> '(' >> x3::string >> ')';

auto const profile_stmt = x3::rule<class profile_stmt, ast::ProfileStmt>{"profile_stmt"} = 
    x3::lit("profile") >> '{' >> stmt >> '}';

auto const visualize_expr = x3::rule<class visualize_expr, ast::VisualizeExpr>{"visualize_expr"} = 
    x3::lit("visualize") >> '(' >> expr >> ',' >> "type:" >> ident >> ')';

auto const tokenize_expr = x3::rule<class tokenize_expr, ast::TokenizeExpr>{"tokenize_expr"} = 
    x3::lit("tokenize") >> '(' >> x3::string >> ',' >> "vocab:" >> x3::string >> ')';

auto const expr = x3::rule<class expr, ast::Expr>{"expr"} =
    tensor_lit | matmul_expr | pipe_expr | grad_expr | ode_solve_expr | quantize_expr |
    prune_expr | deploy_expr | attention_expr | gnn_conv_expr | sql_tensor_expr |
    factorize_expr | scale_expr | rate_reduce_expr | pico_bench_expr | mobi_prune_expr |
    reasoning_pass_expr | model_expr | dataset_expr | conv2d_expr | maxpool_expr |
    batchnorm_expr | layernorm_expr | sparse_tensor_lit | import_expr | visualize_expr |
    tokenize_expr;

auto const stmt = x3::rule<class stmt, ast::Stmt>{"stmt"} = 
    train_stmt | autonomic_stmt | agent_tune_stmt | if_stmt | for_stmt | while_stmt | fn_decl | profile_stmt;

auto const decl = x3::rule<class decl, ast::Decl>{"decl"} = 
    x3::lit("let") >> ident >> '=' >> expr;

auto const program = x3::rule<class program, ast::Program>{"program"} = 
    *decl >> *stmt;

} // namespace esx::parser

#endif // ESX_PARSER_HPP
