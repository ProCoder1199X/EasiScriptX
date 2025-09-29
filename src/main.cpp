
#include "config.hpp"
#include "interpreter.hpp"
#include "parser.hpp"
#include "printer.hpp"
#include <boost/spirit/home/x3.hpp>
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <string>
#include <curl/curl.h>
#include <filesystem>

namespace esx {
// Callback for curl to write downloaded file
size_t write_data(void* ptr, size_t size, size_t nmemb, FILE* stream) {
    return fwrite(ptr, size, nmemb, stream);
}

// Download pretrained model
bool download_model(const std::string& name, const std::string& dest) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    FILE* fp = fopen(dest.c_str(), "wb");
    if (!fp) {
        curl_easy_cleanup(curl);
        return false;
    }
    // Mock URL (replace with real Hugging Face endpoint)
    std::string url = "https://huggingface.co/models/" + name + ".pt";
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    CURLcode res = curl_easy_perform(curl);
    fclose(fp);
    curl_easy_cleanup(curl);
    return res == CURLE_OK;
}

int main(int argc, char* argv[]) {
    bool debug = false;
    std::string filename;
    std::string log_file = "esx.log";

    // Initialize CUDA if enabled
#ifdef USE_CUDA
    if (torch::cuda::is_available()) {
        torch::cuda::manual_seed_all(42);
    } else if (USE_CUDA) {
        std::cerr << "Warning: CUDA enabled but not available. Falling back to CPU." << std::endl;
    }
#endif

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--debug") {
            debug = true;
        } else if (std::string(argv[i]) == "install" && i + 1 < argc) {
            std::string package = argv[i + 1];
            std::cout << "Installing package: " << package << std::endl;
            std::filesystem::create_directory("pretrained");
            std::string dest = "pretrained/" + package + ".pt";
            if (!download_model(package, dest)) {
                std::cerr << "Error: Failed to download package: " << package << std::endl;
                return 3; // Package install error
            }
            std::cout << "Downloaded " << package << " to " << dest << std::endl;
            return 0;
        } else {
            filename = argv[i];
        }
    }

    if (filename.empty()) {
        std::cerr << "Usage: esx [--debug] <filename.esx> | install <package>" << std::endl;
        return 1; // Invalid usage
    }

    // Open log file for debug mode
    std::ofstream log;
    if (debug) {
        log.open(log_file, std::ios::app);
        if (!log.is_open()) {
            std::cerr << "Warning: Failed to open log file: " << log_file << std::endl;
        }
    }

    // Open the script file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file: " << filename << std::endl;
        if (debug && log.is_open()) log << "Error: Failed to open file: " << filename << std::endl;
        return 2; // File error
    }

    // Read the script content
    std::string input((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    esx::ast::Program program;
    auto iter = input.begin();
    auto end = input.end();

    try {
        // Parse the script
        bool parsed = boost::spirit::x3::phrase_parse(iter, end, esx::parser::program, esx::parser::skip, program);
        if (!parsed || iter != end) {
            size_t line = std::count(input.begin(), iter, '\n') + 1;
            size_t col = iter - std::find_if_not(input.begin(), iter, [](char c) { return c != '\n'; }) + 1;
            std::string token(iter, std::min(iter + 10, end));
            std::cerr << "Parse Error: Unexpected token '" << token << "' at line " << line << ", column " << col
                      << ". Expected valid expression or statement." << std::endl;
            if (debug && log.is_open()) {
                log << "Parse Error: Unexpected token '" << token << "' at line " << line << ", column " << col << std::endl;
            }
            return 2; // Parse error
        }

        // Print the parsed AST (debug mode)
        if (debug) {
            esx::ast::Printer printer;
            std::stringstream ast_out;
            std::streambuf* cout_buf = std::cout.rdbuf();
            std::cout.rdbuf(ast_out.rdbuf());
            printer(program);
            std::cout.rdbuf(cout_buf);
            std::cout << "Debug: AST:\n" << ast_out.str() << std::endl;
            if (log.is_open()) log << "Debug: AST:\n" << ast_out.str() << std::endl;
        }

        // Initialize interpreter
        esx::runtime::Interpreter interpreter;
#ifdef USE_CUDA
        if (torch::cuda::is_available()) {
            interpreter.ops = std::make_unique<esx::runtime::GPUTensorOps>();
        }
#endif
        // Execute the program
        interpreter.execute(program, debug);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (debug && log.is_open()) log << "Error: " << e.what() << std::endl;
        return 3; // Runtime error
    }

    if (log.is_open()) log.close();
    return 0;
}
} // namespace esx
