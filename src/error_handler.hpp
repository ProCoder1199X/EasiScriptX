#ifndef ESX_ERROR_HANDLER_HPP
#define ESX_ERROR_HANDLER_HPP

#include <stdexcept>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>

/**
 * @file error_handler.hpp
 * @brief Error handling and reporting for EasiScriptX (ESX).
 * @details Provides comprehensive error handling with detailed error messages,
 * input sanitization, and security validation.
 */

namespace esx::error {

/**
 * @brief Base error class for EasiScriptX errors.
 */
class ESXError : public std::runtime_error {
public:
    ESXError(const std::string& message, int line = 0, int column = 0);
    
    int line() const;
    int column() const;
    std::string format_message() const;

private:
    int line_;
    int column_;
};

/**
 * @brief Syntax error for parsing issues.
 */
class SyntaxError : public ESXError {
public:
    SyntaxError(const std::string& message, int line = 0, int column = 0);
};

/**
 * @brief Runtime error for execution issues.
 */
class RuntimeError : public ESXError {
public:
    RuntimeError(const std::string& message, int line = 0, int column = 0);
};

/**
 * @brief Type error for type mismatches.
 */
class TypeError : public ESXError {
public:
    TypeError(const std::string& message, int line = 0, int column = 0);
};

/**
 * @brief Validation error for parameter validation.
 */
class ValidationError : public ESXError {
public:
    ValidationError(const std::string& message, int line = 0, int column = 0);
};

/**
 * @brief File error for file I/O issues.
 */
class FileError : public ESXError {
public:
    FileError(const std::string& message, int line = 0, int column = 0);
};

/**
 * @brief Memory error for memory allocation issues.
 */
class MemoryError : public ESXError {
public:
    MemoryError(const std::string& message, int line = 0, int column = 0);
};

/**
 * @brief Distributed training error.
 */
class DistributedError : public ESXError {
public:
    DistributedError(const std::string& message, int line = 0, int column = 0);
};

/**
 * @brief Error handler for managing and reporting errors.
 */
class ErrorHandler {
public:
    ErrorHandler();
    
    void report_error(const ESXError& error);
    void report_warning(const std::string& message, int line = 0, int column = 0);
    void report_info(const std::string& message);
    
    int error_count() const;
    int warning_count() const;
    void reset();

private:
    int error_count_;
    int warning_count_;
    static const int MAX_ERRORS = 100;
};

/**
 * @brief Input sanitization for security.
 * @param input Input string to sanitize.
 * @return Sanitized string.
 */
std::string sanitize_input(const std::string& input);

/**
 * @brief Validate file path for security.
 * @param path File path to validate.
 * @return true if path is safe, false otherwise.
 */
bool validate_file_path(const std::string& path);

/**
 * @brief Create formatted error message with context.
 * @param type Error type.
 * @param message Error message.
 * @param line Line number.
 * @param column Column number.
 * @param context Additional context.
 * @return Formatted error message.
 */
std::string create_error_message(const std::string& type, const std::string& message, 
                                int line, int column, const std::string& context = "");

/**
 * @brief Get global error handler instance.
 * @return Reference to global error handler.
 */
ErrorHandler& get_error_handler();

} // namespace esx::error

#endif // ESX_ERROR_HANDLER_HPP
