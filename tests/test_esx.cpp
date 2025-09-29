
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "parser.hpp"
#include "printer.hpp"
#include "tensor.hpp"
#include "interpreter.hpp"
#include "model.hpp"
#include "dataset.hpp"
#include "distributed.hpp"
#include <boost/spirit/home/x3.hpp>

TEST_CASE("Parse tensor literal") {
    std::string input = "let x = tensor([[1,2],[3,4]])";
    esx::ast::Program program;
    REQUIRE(boost::spirit::x3::phrase_parse(input.begin(), input.end(), esx::parser::program, boost::spirit::x3::ascii::space, program));
    REQUIRE(program.decls.size() == 1);
    REQUIRE(program.decls[0].var == "x");
}

TEST_CASE("Execute matmul") {
    esx::runtime::Interpreter interp;
    esx::ast::TensorLit a{{{1,2},{3,4}}, {2,2}};
    esx::ast::TensorLit b{{{5,6},{7,8}}, {2,2}};
    esx::ast::MatmulExpr matmul{std::make_shared<esx::ast::Expr>(a), std::make_shared<esx::ast::Expr>(b)};
    auto result = interp.eval_expr(matmul);
    REQUIRE(result.shape == std::vector<size_t>{2,2});
    REQUIRE(result.dense_data(0,0) == 19.0);
    REQUIRE(result.dense_data(0,1) == 22.0);
    REQUIRE(result.dense_data(1,0) == 43.0);
    REQUIRE(result.dense_data(1,1) == 50.0);
}

TEST_CASE("Invalid tensor shape for matmul") {
    esx::runtime::Interpreter interp;
    esx::ast::TensorLit a{{{1,2,3}}, {1,3}};
    esx::ast::TensorLit b{{{4,5},{6,7},{8,9}}, {3,2}};
    esx::ast::MatmulExpr matmul{std::make_shared<esx::ast::Expr>(a), std::make_shared<esx::ast::Expr>(b)};
    REQUIRE_THROWS_AS(interp.eval_expr(matmul), std::runtime_error);
}

TEST_CASE("Custom loss function") {
    esx::runtime::Interpreter interp;
    esx::ast::TensorLit pred{{{1,2},{3,4}}, {2,2}};
    esx::ast::TensorLit true_val{{{1,2},{3,4}}, {2,2}};
    auto loss = interp.funcs["custom_loss"]({interp.eval_expr(pred), interp.eval_expr(true_val)});
    REQUIRE(loss.dense_data.sum() == 0.0); // Example check
}

TEST_CASE("Distributed training stub") {
    esx::runtime::Interpreter interp;
    esx::ast::TrainStmt train; // Fill with dummy
    interp.exec_stmt(train);
    REQUIRE(true); // No crash
}

TEST_CASE("Performance benchmark matmul CPU") {
    esx::runtime::Interpreter interp;
    interp.bench_matmul(512, "cpu");
    REQUIRE(true); // Manual verify
}

TEST_CASE("Performance benchmark matmul GPU") {
    esx::runtime::Interpreter interp;
    interp.bench_matmul(512, "gpu");
    REQUIRE(true); // Manual verify
}

TEST_CASE("Conv2d operation") {
    esx::runtime::Interpreter interp;
    esx::ast::TensorLit input{{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}}, {4,4}};
    esx::ast::TensorLit kernel{{{1,0},{0,1}}, {2,2}};
    esx::ast::Conv2dExpr conv{std::make_shared<esx::ast::Expr>(input), std::make_shared<esx::ast::Expr>(kernel), 1, "same"};
    auto result = interp.eval_expr(conv);
    REQUIRE(result.shape[0] == 4); // Check dimension
}

TEST_CASE("Load pretrained model") {
    esx::runtime::Interpreter interp;
    esx::ast::ImportExpr import_{"resnet50"};
    auto result = interp.eval_expr(import_);
    REQUIRE(result.shape.empty() == false);
}

TEST_CASE("Invalid syntax in model definition") {
    std::string input = "model mymodel { layer dense(units=128) }"; // Missing ;
    esx::ast::Program program;
    REQUIRE_FALSE(boost::spirit::x3::phrase_parse(input.begin(), input.end(), esx::parser::program, boost::spirit::x3::ascii::space, program));
}

TEST_CASE("Training workflow") {
    esx::runtime::Interpreter interp;
    esx::ast::ModelExpr model; // Dummy
    esx::ast::DatasetExpr data; // Dummy
    esx::ast::TrainStmt train{"mymodel", "mnist", "ce", "adam", {{"lr", 0.001}}, 10, "cpu"};
    interp.eval_expr(model);
    interp.eval_expr(data);
    interp.exec_stmt(train);
    REQUIRE(true); // No crash
}
