#ifndef ESX_REPL_HPP
#define ESX_REPL_HPP

#include "config.hpp"
#include <string>
#include <memory>

#ifndef _WIN32
#include <readline/readline.h>
#include <readline/history.h>
#endif

/**
 * @file repl.hpp
 * @brief Interactive REPL (Read-Eval-Print Loop) for EasiScriptX (ESX).
 * @details Provides an interactive command-line interface for executing
 * ESX code with features like command history, tab completion, and help.
 */

namespace esx::runtime {
class Interpreter;
}

namespace esx::repl {

/**
 * @brief Interactive REPL for EasiScriptX.
 */
class REPL {
public:
    REPL();
    ~REPL();

    /**
     * @brief Start the REPL loop.
     */
    void start();

    /**
     * @brief Start the REPL with custom interpreter.
     * @param interpreter Custom interpreter instance.
     */
    void start_with_interpreter(std::shared_ptr<esx::runtime::Interpreter> interpreter);

private:
    bool running_;
    int line_count_;

    /**
     * @brief Read a line of input from the user.
     * @return Input string.
     */
    std::string read_line();

    /**
     * @brief Execute input string.
     * @param input Input to execute.
     */
    void execute_input(const std::string& input);

    /**
     * @brief Execute input with custom interpreter.
     * @param input Input to execute.
     * @param interpreter Custom interpreter instance.
     */
    void execute_input_with_interpreter(const std::string& input, 
                                       std::shared_ptr<esx::runtime::Interpreter> interpreter);

    /**
     * @brief Show help information.
     */
    void show_help();

    /**
     * @brief Clear the screen.
     */
    void clear_screen();

    /**
     * @brief Reset the interpreter state.
     */
    void reset_interpreter();

    /**
     * @brief Show interpreter status.
     */
    void show_status();

    /**
     * @brief Initialize readline support.
     */
    void initialize_readline();

    /**
     * @brief Cleanup readline support.
     */
    void cleanup_readline();

    /**
     * @brief Tab completion function for readline.
     * @param text Text to complete.
     * @param start Start position.
     * @param end End position.
     * @return Array of completion strings.
     */
    static char** completion_function(const char* text, int start, int end);
};

} // namespace esx::repl

#endif // ESX_REPL_HPP
