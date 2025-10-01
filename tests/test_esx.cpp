
#include "parser.hpp"
#include "interpreter.hpp"
#include <iostream>
#include <cassert>

/**
 * @file test_esx.cpp
 * @brief Unit tests for EasiScriptX (ESX) v1.0.
 * @details Tests parsing and execution of ESX syntax, including new features like
 * LoRA, mixed-precision, pipeline parallelism, and framework interoperability.
 */

void test_parser() {
    std::string source = R"(
        let x = tensor([[1,2],[3,4]])
        let y = x @ x
        let z = conv2d(x, x, stride:1, padding:0)
        let attn = attention(x, x, x, heads:8, dim:64, flash=true)
        let lora_x = lora(x, rank:4)
        let mp_x = mixed_precision(x, bf16)
        train(x, x, loss:ce, opt:lomo(lr=0.001), epochs:10, device:cpu)
        pipeline_parallel(x, stages:2)
        instruction_tune(x, x, "prompt")
        adapt_domain(x, scientific)
        schedule_heterogeneous(cpu:0.7, gpu:0.3)
        energy_aware(x, max_power:100)
        switch_framework(pytorch, model.pt)
        track_experiment(mlflow, run123)
    )";
    try {
        esx::ast::Program program = esx::parser::parse(source);
        assert(program.stmts.size() == 10);
        std::cout << "Parser test passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Parser test failed: " << e.what() << std::endl;
        assert(false);
    }
}

void test_interpreter() {
    std::string source = R"(
        let x = tensor([[1,2],[3,4]])
        let y = x @ x
        train(x, x, loss:ce, opt:adam(lr=0.001), epochs:1, device:cpu)
    )";
    try {
        esx::ast::Program program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        std::cout << "Interpreter test passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Interpreter test failed: " << e.what() << std::endl;
        assert(false);
    }
}

int main() {
    test_parser();
    test_interpreter();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
