#include "parser.hpp"
#include "interpreter.hpp"
#include "printer.hpp"
#include <fstream>
#include <iostream>
#include <string>

/**
 * @file main.cpp
 * @brief Entry point for EasiScriptX (ESX).
 * @details Reads ESX source code, parses it into an AST, and executes it using the interpreter.
 * Supports debugging, energy-aware scheduling, and MLflow tracking.
 */

int main(int argc, char* argv[]) {
    bool debug = false;
    bool energy_aware = false;
    bool mlflow = false;
    std::string filename;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--debug") {
            debug = true;
        } else if (arg == "--energy-aware") {
            energy_aware = true;
        } else if (arg == "--mlflow") {
            mlflow = true;
        } else {
            filename = arg;
        }
    }

    if (filename.empty()) {
        std::cerr << "Usage: " << argv[0] << " [--debug] [--energy-aware] [--mlflow] <script.esx>" << std::endl;
        return 1;
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file '" << filename << "'. Please check the file path." << std::endl;
        return 1;
    }

    std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    try {
        esx::ast::Program program = esx::parser::parse(source);
        if (debug) {
            esx::printer::Printer printer;
            printer.print(program);
        }
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);

        if (energy_aware) {
            std::cout << "Energy-aware mode enabled. Tracking energy usage during execution." << std::endl;
            std::cout << "Total energy usage: " << interpreter.get_energy_usage() << " J" << std::endl;
        }

        if (mlflow) {
            std::cout << "MLflow tracking enabled. Experiment data will be logged." << std::endl;
            interpreter.track_experiment("mlflow");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << ". Please check the script for syntax or runtime issues." << std::endl;
        return 1;
    }

    return 0;
}