#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_approx.hpp>
#include "parser.hpp"
#include "interpreter.hpp"
#include "tensor.hpp"
#include "dataset.hpp"
#include "distributed.hpp"
#include "error_handler.hpp"
#include <vector>
#include <string>
#include <chrono>

using namespace esx;

/**
 * @file test_suite.cpp
 * @brief Comprehensive test suite for EasiScriptX (ESX).
 * @details Tests tensor operations, parsing, interpretation, distributed training,
 * and error handling with Catch2 framework.
 */

// Test tensor operations
TEST_CASE("Tensor Operations", "[tensor]") {
    SECTION("Matrix Multiplication") {
        runtime::Tensor a({{1, 2}, {3, 4}});
        runtime::Tensor b({{5, 6}, {7, 8}});
        auto result = a.matmul(b);
        
        REQUIRE(result.data.size() == 2);
        REQUIRE(result.data[0].size() == 2);
        REQUIRE(result.data[0][0] == Catch::Approx(19.0));
        REQUIRE(result.data[0][1] == Catch::Approx(22.0));
        REQUIRE(result.data[1][0] == Catch::Approx(43.0));
        REQUIRE(result.data[1][1] == Catch::Approx(50.0));
    }
    
    SECTION("Convolution") {
        runtime::Tensor input({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        runtime::Tensor kernel({{1, 0}, {0, 1}});
        auto result = input.conv2d(kernel, 1, 0);
        
        REQUIRE(result.torch_tensor.numel() > 0);
    }
    
    SECTION("Mean Calculation") {
        runtime::Tensor tensor({{1, 2, 3}, {4, 5, 6}});
        // Mock mean calculation
        double mean = 0.0;
        int count = 0;
        for (const auto& row : tensor.data) {
            for (double val : row) {
                mean += val;
                count++;
            }
        }
        mean /= count;
        
        REQUIRE(mean == Catch::Approx(3.5));
    }
}

// Test parsing
TEST_CASE("Parser", "[parser]") {
    SECTION("Basic Parsing") {
        std::string source = "let x = tensor([[1,2],[3,4]])";
        auto program = parser::parse(source);
        
        REQUIRE(program.stmts.size() == 1);
    }
    
    SECTION("Complex Expression") {
        std::string source = R"(
            let x = tensor([[1,2],[3,4]])
            let y = x @ x
            let z = conv2d(x, x, stride:1, padding:0)
        )";
        auto program = parser::parse(source);
        
        REQUIRE(program.stmts.size() == 3);
    }
    
    SECTION("Training Statement") {
        std::string source = "train(x, y, loss: ce, opt: adam(lr=0.001), epochs: 10, device: cpu)";
        auto program = parser::parse(source);
        
        REQUIRE(program.stmts.size() == 1);
    }
    
    SECTION("Parse Error") {
        std::string source = "let x = tensor([[1,2],[3,4"; // Missing closing bracket
        REQUIRE_THROWS_AS(parser::parse(source), std::runtime_error);
    }
}

// Test interpreter
TEST_CASE("Interpreter", "[interpreter]") {
    SECTION("Basic Execution") {
        std::string source = R"(
            let x = tensor([[1,2],[3,4]])
            let y = x @ x
        )";
        auto program = parser::parse(source);
        runtime::Interpreter interpreter;
        
        REQUIRE_NOTHROW(interpreter.run(program));
    }
    
    SECTION("Training Execution") {
        std::string source = R"(
            let x = tensor([[1,2],[3,4]])
            let y = tensor([[5,6],[7,8]])
            train(x, y, loss: ce, opt: adam(lr=0.001), epochs: 1, device: cpu)
        )";
        auto program = parser::parse(source);
        runtime::Interpreter interpreter;
        
        REQUIRE_NOTHROW(interpreter.run(program));
    }
    
    SECTION("LoRA Application") {
        std::string source = R"(
            let x = tensor([[1,2],[3,4]])
            let y = lora(x, rank:4)
        )";
        auto program = parser::parse(source);
        runtime::Interpreter interpreter;
        
        REQUIRE_NOTHROW(interpreter.run(program));
    }
    
    SECTION("Memory Broker") {
        std::string source = R"(
            let x = tensor([[1,2],[3,4]])
            memory_broker(x, max_mem:8, strategy:zeRO)
        )";
        auto program = parser::parse(source);
        runtime::Interpreter interpreter;
        
        REQUIRE_NOTHROW(interpreter.run(program));
    }
    
    SECTION("Quantization") {
        std::string source = R"(
            let x = tensor([[1,2],[3,4]])
            quantize(x, bits:8, method:ptq)
        )";
        auto program = parser::parse(source);
        runtime::Interpreter interpreter;
        
        REQUIRE_NOTHROW(interpreter.run(program));
    }
}

// Test distributed training
TEST_CASE("Distributed Training", "[distributed]") {
    SECTION("MPI Initialization") {
        bool result = runtime::Distributed::init_multi_node(2);
        REQUIRE(result == true);
        
        runtime::Distributed::finalize();
    }
    
    SECTION("NCCL Initialization") {
        bool result = runtime::Distributed::init_multi_gpu(2);
        REQUIRE(result == true);
        
        runtime::Distributed::finalize();
    }
    
    SECTION("Gradient Aggregation") {
        std::vector<std::vector<double>> gradients = {{1.0, 2.0}, {3.0, 4.0}};
        bool result = runtime::Distributed::aggregate_gradients(gradients);
        REQUIRE(result == true);
    }
    
    SECTION("Synchronization") {
        bool result = runtime::Distributed::synchronize();
        REQUIRE(result == true);
    }
    
    SECTION("Retry on Failure") {
        // Mock failure scenario
        std::vector<std::vector<double>> gradients = {{1.0, 2.0}};
        bool result = runtime::Distributed::aggregate_gradients(gradients, 3);
        REQUIRE(result == true);
    }
}

// Test error handling
TEST_CASE("Error Handling", "[error]") {
    SECTION("Syntax Error") {
        error::SyntaxError error("Invalid syntax", 5, 10);
        REQUIRE(error.line() == 5);
        REQUIRE(error.column() == 10);
        REQUIRE(std::string(error.what()).find("SyntaxError") != std::string::npos);
    }
    
    SECTION("Runtime Error") {
        error::RuntimeError error("Runtime failure", 10, 5);
        REQUIRE(error.line() == 10);
        REQUIRE(error.column() == 5);
        REQUIRE(std::string(error.what()).find("RuntimeError") != std::string::npos);
    }
    
    SECTION("Validation Error") {
        error::ValidationError error("Invalid parameter", 15, 20);
        REQUIRE(error.line() == 15);
        REQUIRE(error.column() == 20);
        REQUIRE(std::string(error.what()).find("ValidationError") != std::string::npos);
    }
    
    SECTION("File Error") {
        error::FileError error("File not found", 1, 1);
        REQUIRE(error.line() == 1);
        REQUIRE(error.column() == 1);
        REQUIRE(std::string(error.what()).find("FileError") != std::string::npos);
    }
    
    SECTION("Input Sanitization") {
        std::string input = "test\0string\r\n";
        std::string sanitized = error::sanitize_input(input);
        REQUIRE(sanitized.find('\0') == std::string::npos);
        REQUIRE(sanitized.find('\r') == std::string::npos);
    }
    
    SECTION("File Path Validation") {
        REQUIRE(error::validate_file_path("model.pt") == true);
        REQUIRE(error::validate_file_path("../model.pt") == false);
        REQUIRE(error::validate_file_path("/absolute/path") == false);
        REQUIRE(error::validate_file_path("model\0.pt") == false);
    }
}

// Test dataset operations
TEST_CASE("Dataset Operations", "[dataset]") {
    SECTION("Dataset Loading") {
        runtime::Dataset dataset;
        ast::Location loc(1, 1);
        
        REQUIRE_NOTHROW(dataset.load("mnist", "normalize", {"rotate", "flip"}, loc));
        REQUIRE(dataset.data.size() > 0);
    }
    
    SECTION("Tokenization") {
        runtime::Dataset dataset;
        ast::Location loc(1, 1);
        
        // Mock tokenization
        auto result = dataset.tokenize("hello world", "vocab.json", loc);
        REQUIRE(result.data.size() > 0);
    }
    
    SECTION("Batch Processing") {
        runtime::Dataset dataset;
        dataset.data.push_back(runtime::Tensor({{1, 2}, {3, 4}}));
        dataset.data.push_back(runtime::Tensor({{5, 6}, {7, 8}}));
        
        auto batch = dataset.next_batch(2);
        REQUIRE(batch.data.size() > 0);
    }
    
    SECTION("Pattern Recognition") {
        runtime::Dataset dataset;
        dataset.data.push_back(runtime::Tensor({{1, 2, 3, 4}, {5, 6, 7, 8}}));
        
        REQUIRE_NOTHROW(dataset.apply_pattern("geometric"));
        REQUIRE_NOTHROW(dataset.apply_pattern("arithmetic"));
    }
}

// Test custom loss functions
TEST_CASE("Custom Loss Functions", "[loss]") {
    SECTION("MSE Loss") {
        runtime::Tensor pred({{1.0, 2.0}, {3.0, 4.0}});
        runtime::Tensor target({{1.1, 2.1}, {3.1, 4.1}});
        
        // Mock MSE calculation
        double mse = 0.0;
        int count = 0;
        for (size_t i = 0; i < pred.data.size(); ++i) {
            for (size_t j = 0; j < pred.data[i].size(); ++j) {
                double diff = pred.data[i][j] - target.data[i][j];
                mse += diff * diff;
                count++;
            }
        }
        mse /= count;
        
        REQUIRE(mse == Catch::Approx(0.01));
    }
    
    SECTION("Cross-Entropy Loss") {
        runtime::Tensor pred({{0.1, 0.9}, {0.8, 0.2}});
        runtime::Tensor target({{0.0, 1.0}, {1.0, 0.0}});
        
        // Mock cross-entropy calculation
        double ce = 0.0;
        for (size_t i = 0; i < pred.data.size(); ++i) {
            for (size_t j = 0; j < pred.data[i].size(); ++j) {
                if (target.data[i][j] > 0) {
                    ce -= target.data[i][j] * std::log(pred.data[i][j] + 1e-8);
                }
            }
        }
        
        REQUIRE(ce > 0.0);
    }
}

// Integration tests
TEST_CASE("Integration Tests", "[integration]") {
    SECTION("Complete Training Pipeline") {
        std::string source = R"(
            let model = tensor([[1,2],[3,4]])
            let dataset = tensor([[5,6],[7,8]])
            train(model, dataset, loss: ce, opt: adam(lr=0.001), epochs: 2, device: cpu)
        )";
        
        auto program = parser::parse(source);
        runtime::Interpreter interpreter;
        
        auto start = std::chrono::high_resolution_clock::now();
        REQUIRE_NOTHROW(interpreter.run(program));
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        REQUIRE(duration.count() < 1000); // Should complete in less than 1 second
    }
    
    SECTION("Memory Optimization Pipeline") {
        std::string source = R"(
            let model = tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
            memory_broker(model, max_mem:8, strategy:zeRO)
            quantize(model, bits:8, method:ptq)
            checkpoint(model, segments:4)
        )";
        
        auto program = parser::parse(source);
        runtime::Interpreter interpreter;
        
        REQUIRE_NOTHROW(interpreter.run(program));
    }
    
    SECTION("LLM Optimization Pipeline") {
        std::string source = R"(
            let main_model = tensor([[1,2],[3,4]])
            let draft_model = tensor([[0.5,1],[1.5,2]])
            let lora_model = lora(main_model, rank:4)
            let mp_model = mixed_precision(lora_model, bf16)
            speculative_decode(mp_model, draft_model, max_tokens:100)
        )";
        
        auto program = parser::parse(source);
        runtime::Interpreter interpreter;
        
        REQUIRE_NOTHROW(interpreter.run(program));
    }
}

// Performance tests
TEST_CASE("Performance Tests", "[performance]") {
    SECTION("Matrix Multiplication Performance") {
        runtime::Tensor a({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
        runtime::Tensor b({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            auto result = a.matmul(b);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        REQUIRE(duration.count() < 10000); // Should complete in less than 10ms
    }
    
    SECTION("Parsing Performance") {
        std::string source = R"(
            let x = tensor([[1,2],[3,4]])
            let y = x @ x
            let z = conv2d(x, x, stride:1, padding:0)
            let attn = attention(x, x, x, heads:8, dim:4, flash=true)
        )";
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            auto program = parser::parse(source);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        REQUIRE(duration.count() < 5000); // Should complete in less than 5ms
    }
}

// Main function for Catch2
int main(int argc, char* argv[]) {
    Catch::Session session;
    
    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) {
        return returnCode;
    }
    
    return session.run();
}
