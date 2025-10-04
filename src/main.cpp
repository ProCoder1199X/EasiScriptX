#include "config.hpp"
#include "parser.hpp"
#include "interpreter.hpp"
#include "repl.hpp"
#include "error_handler.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

/**
 * @file main.cpp
 * @brief Main entry point for EasiScriptX (ESX) compiler/interpreter.
 * @details Provides command-line interface for executing ESX files or starting REPL.
 */

void print_version() {
    std::cout << "EasiScriptX (ESX) v" << ESX_VERSION_STRING << std::endl;
    std::cout << "AI/ML Domain-Specific Language" << std::endl;
    std::cout << "Built on " << __DATE__ << " " << __TIME__ << std::endl;
}

void print_help() {
    std::cout << "Usage: esx [OPTIONS] [FILE]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help          Show this help message" << std::endl;
    std::cout << "  -v, --version       Show version information" << std::endl;
    std::cout << "  -i, --interactive   Start interactive REPL" << std::endl;
    std::cout << "  -c, --compile       Compile to intermediate representation" << std::endl;
    std::cout << "  -o, --output FILE   Output file for compilation" << std::endl;
    std::cout << "  -d, --debug         Enable debug output" << std::endl;
    std::cout << "  -p, --profile       Enable performance profiling" << std::endl;
    std::cout << "  --no-optimize       Disable optimizations" << std::endl;
    std::cout << "  --verbose           Verbose output" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  esx train.esx                    # Execute ESX file" << std::endl;
    std::cout << "  esx -i                           # Start interactive REPL" << std::endl;
    std::cout << "  esx -c -o output.ir train.esx    # Compile to IR" << std::endl;
    std::cout << "  esx --profile train.esx          # Profile execution" << std::endl;
}

bool execute_file(const std::string& filename, bool profile = false) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        esx::error::get_error_handler().report_error(
            esx::error::FileError("Cannot open file: " + filename));
        return false;
    }
    
    std::string source((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    file.close();
    
    try {
        // Parse the source code
        auto program = esx::parser::parse(source);
        
        // Execute the program
        esx::runtime::Interpreter interpreter;
        
        if (profile) {
            auto start = std::chrono::high_resolution_clock::now();
            interpreter.run(program);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "Execution time: " << duration.count() << " μs" << std::endl;
        } else {
            interpreter.run(program);
        }
        
        return true;
    } catch (const esx::error::ESXError& e) {
        esx::error::get_error_handler().report_error(e);
        return false;
    } catch (const std::exception& e) {
        esx::error::get_error_handler().report_error(
            esx::error::RuntimeError("Execution failed: " + std::string(e.what())));
        return false;
    }
}

bool compile_file(const std::string& input_file, const std::string& output_file) {
    std::ifstream file(input_file);
    if (!file.is_open()) {
        esx::error::get_error_handler().report_error(
            esx::error::FileError("Cannot open file: " + input_file));
        return false;
    }
    
    std::string source((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    file.close();
    
    try {
        // Parse the source code
        auto program = esx::parser::parse(source);
        
        // In a full implementation, would generate intermediate representation
        std::ofstream out(output_file);
        if (!out.is_open()) {
            esx::error::get_error_handler().report_error(
                esx::error::FileError("Cannot create output file: " + output_file));
            return false;
        }
        
        // Mock IR generation
        out << "// EasiScriptX Intermediate Representation" << std::endl;
        out << "// Generated from: " << input_file << std::endl;
        out << "// Statements: " << program.stmts.size() << std::endl;
        out << std::endl;
        
        for (size_t i = 0; i < program.stmts.size(); ++i) {
            out << "stmt_" << i << ": // Statement " << i << std::endl;
        }
        
        out.close();
        std::cout << "Compiled to: " << output_file << std::endl;
        return true;
    } catch (const esx::error::ESXError& e) {
        esx::error::get_error_handler().report_error(e);
        return false;
    } catch (const std::exception& e) {
        esx::error::get_error_handler().report_error(
            esx::error::RuntimeError("Compilation failed: " + std::string(e.what())));
        return false;
    }
}

int main(int argc, char* argv[]) {
    std::vector<std::string> args(argv + 1, argv + argc);
    
    bool interactive = false;
    bool compile = false;
    bool profile = false;
    bool debug = false;
    bool verbose = false;
    bool optimize = true;
    std::string output_file;
    std::string input_file;
    
    // Parse command line arguments
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        
        if (arg == "-h" || arg == "--help") {
            print_help();
            return 0;
        } else if (arg == "-v" || arg == "--version") {
            print_version();
            return 0;
        } else if (arg == "-i" || arg == "--interactive") {
            interactive = true;
        } else if (arg == "-c" || arg == "--compile") {
            compile = true;
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < args.size()) {
                output_file = args[++i];
            } else {
                std::cerr << "Error: --output requires a filename" << std::endl;
                return 1;
            }
        } else if (arg == "-d" || arg == "--debug") {
            debug = true;
        } else if (arg == "-p" || arg == "--profile") {
            profile = true;
        } else if (arg == "--no-optimize") {
            optimize = false;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg[0] != '-') {
            input_file = arg;
        } else {
            std::cerr << "Error: Unknown option: " << arg << std::endl;
            print_help();
            return 1;
        }
    }
    
    // Set debug mode
    if (debug) {
        // In a full implementation, would set debug flags
        std::cout << "Debug mode enabled" << std::endl;
    }
    
    // Set verbose mode
    if (verbose) {
        std::cout << "Verbose mode enabled" << std::endl;
    }
    
    // Start interactive REPL
    if (interactive) {
        esx::repl::REPL repl;
        repl.start();
        return 0;
    }
    
    // Compile mode
    if (compile) {
        if (input_file.empty()) {
            std::cerr << "Error: No input file specified for compilation" << std::endl;
            return 1;
        }
        
        if (output_file.empty()) {
            output_file = input_file + ".ir";
        }
        
        bool success = compile_file(input_file, output_file);
        return success ? 0 : 1;
    }
    
    // Execute mode
    if (!input_file.empty()) {
        bool success = execute_file(input_file, profile);
        return success ? 0 : 1;
    }
    
    // No input file and not interactive, show help
    print_help();
    return 0;
}