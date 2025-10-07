#include "repl.hpp"
#include "parser.hpp"
#include "interpreter.hpp"
#include "error_handler.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <memory>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#else
#include <readline/readline.h>
#include <readline/history.h>
#endif

namespace esx::repl {

/**
 * @brief Interactive REPL for EasiScriptX.
 */
REPL::REPL() : running_(false), line_count_(0) {
    initialize_readline();
}

REPL::~REPL() {
    cleanup_readline();
}

/**
 * @brief Start the REPL loop.
 */
void REPL::start() {
    std::cout << "EasiScriptX (ESX) v" << ESX_VERSION_STRING << " Interactive REPL\n";
    std::cout << "Type 'help' for commands, 'quit' to exit.\n\n";
    
    running_ = true;
    std::string input;
    
    while (running_) {
        try {
            input = read_line();
            if (input.empty()) continue;
            
            if (input == "quit" || input == "exit") {
                break;
            } else if (input == "help") {
                show_help();
            } else if (input == "clear") {
                clear_screen();
            } else if (input == "reset") {
                reset_interpreter();
            } else if (input == "status") {
                show_status();
            } else {
                execute_input(input);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
    
    std::cout << "Goodbye!\n";
}

/**
 * @brief Read a line of input from the user.
 * @return Input string.
 */
std::string REPL::read_line() {
    std::string line;
    
#ifdef _WIN32
    std::cout << "esx> ";
    std::getline(std::cin, line);
#else
    char* input = readline("esx> ");
    if (input) {
        line = input;
        free(input);
        
        if (!line.empty()) {
            add_history(line.c_str());
        }
    }
#endif
    
    return line;
}

/**
 * @brief Execute input string.
 * @param input Input to execute.
 */
void REPL::execute_input(const std::string& input) {
    try {
        // Parse the input
        auto program = esx::parser::parse(input);
        
        // Execute the program
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        
        line_count_++;
    } catch (const esx::error::ESXError& e) {
        std::cerr << e.format_message() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Execution error: " << e.what() << std::endl;
    }
}

/**
 * @brief Show help information.
 */
void REPL::show_help() {
    std::cout << "\nEasiScriptX REPL Commands:\n";
    std::cout << "  help     - Show this help message\n";
    std::cout << "  quit     - Exit the REPL\n";
    std::cout << "  exit     - Exit the REPL\n";
    std::cout << "  clear    - Clear the screen\n";
    std::cout << "  reset    - Reset the interpreter state\n";
    std::cout << "  status   - Show interpreter status\n";
    std::cout << "\nExample expressions:\n";
    std::cout << "  let x = tensor([[1,2],[3,4]])\n";
    std::cout << "  let y = x @ x\n";
    std::cout << "  train(x, y, loss: ce, opt: adam(lr=0.001), epochs: 1, device: cpu)\n";
    std::cout << "\n";
}

/**
 * @brief Clear the screen.
 */
void REPL::clear_screen() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

/**
 * @brief Reset the interpreter state.
 */
void REPL::reset_interpreter() {
    // In a full implementation, would reset all variables and state
    std::cout << "Interpreter state reset.\n";
    line_count_ = 0;
}

/**
 * @brief Show interpreter status.
 */
void REPL::show_status() {
    std::cout << "\nInterpreter Status:\n";
    std::cout << "  Lines executed: " << line_count_ << "\n";
    std::cout << "  Version: " << ESX_VERSION_STRING << "\n";
    std::cout << "  Platform: ";
    
#if ESX_PLATFORM_WINDOWS
    std::cout << "Windows";
#elif ESX_PLATFORM_MACOS
    std::cout << "macOS";
#else
    std::cout << "Linux";
#endif
    
    std::cout << "\n  Architecture: ";
    
#if ESX_ARCH_ARM64
    std::cout << "ARM64";
#else
    std::cout << "x86_64";
#endif
    
    std::cout << "\n  Features: ";
    
    std::vector<std::string> features;
    if (ESX_HAS_ONNX) features.push_back("ONNX");
    if (ESX_HAS_CUDA) features.push_back("CUDA");
    if (ESX_HAS_MPI) features.push_back("MPI");
    if (ESX_HAS_LORA) features.push_back("LoRA");
    if (ESX_HAS_QUANTIZATION) features.push_back("Quantization");
    
    for (size_t i = 0; i < features.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << features[i];
    }
    
    std::cout << "\n\n";
}

/**
 * @brief Initialize readline support.
 */
void REPL::initialize_readline() {
#ifndef _WIN32
    // Set up readline history
    using_history();
    
    // Set up completion
    rl_attempted_completion_function = completion_function;
#endif
}

/**
 * @brief Cleanup readline support.
 */
void REPL::cleanup_readline() {
#ifndef _WIN32
    // Save history
    write_history(".esx_history");
#endif
}

/**
 * @brief Tab completion function for readline.
 * @param text Text to complete.
 * @param start Start position.
 * @param end End position.
 * @return Array of completion strings.
 */
char** REPL::completion_function(const char* text, int start, int end) {
    ESX_UNUSED(end);
    
    static std::vector<std::string> keywords = {
        "let", "tensor", "train", "model", "distribute", "autonomic",
        "attention", "lora", "quantize", "prune", "rnn", "transformer_block",
        "memory_broker", "speculative_decode", "pattern_recognize", "checkpoint",
        "fused_matmul_relu", "mixed_precision", "conv2d", "pipeline_parallel",
        "instruction_tune", "adapt_domain", "schedule_heterogeneous", "energy_aware",
        "switch_framework", "track_experiment", "help", "quit", "exit", "clear", "reset", "status"
    };
    
    std::vector<std::string> matches;
    std::string text_str(text);
    
    for (const auto& keyword : keywords) {
        if (keyword.find(text_str) == 0) {
            matches.push_back(keyword);
        }
    }
    
    if (matches.empty()) {
        return nullptr;
    }
    
    char** completions = (char**)malloc((matches.size() + 1) * sizeof(char*));
    for (size_t i = 0; i < matches.size(); ++i) {
        completions[i] = strdup(matches[i].c_str());
    }
    completions[matches.size()] = nullptr;
    
    return completions;
}

/**
 * @brief Start the REPL with custom interpreter.
 * @param interpreter Custom interpreter instance.
 */
void REPL::start_with_interpreter(std::shared_ptr<esx::runtime::Interpreter> interpreter) {
    std::cout << "EasiScriptX (ESX) v" << ESX_VERSION_STRING << " Interactive REPL\n";
    std::cout << "Type 'help' for commands, 'quit' to exit.\n\n";
    
    running_ = true;
    std::string input;
    
    while (running_) {
        try {
            input = read_line();
            if (input.empty()) continue;
            
            if (input == "quit" || input == "exit") {
                break;
            } else if (input == "help") {
                show_help();
            } else if (input == "clear") {
                clear_screen();
            } else if (input == "reset") {
                reset_interpreter();
            } else if (input == "status") {
                show_status();
            } else {
                execute_input_with_interpreter(input, interpreter);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
    
    std::cout << "Goodbye!\n";
}

/**
 * @brief Execute input with custom interpreter.
 * @param input Input to execute.
 * @param interpreter Custom interpreter instance.
 */
void REPL::execute_input_with_interpreter(const std::string& input, 
                                         std::shared_ptr<esx::runtime::Interpreter> interpreter) {
    try {
        // Parse the input
        auto program = esx::parser::parse(input);
        
        // Execute the program with custom interpreter
        interpreter->run(program);
        
        line_count_++;
    } catch (const esx::error::ESXError& e) {
        std::cerr << e.format_message() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Execution error: " << e.what() << std::endl;
    }
}

} // namespace esx::repl
