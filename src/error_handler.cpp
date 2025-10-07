#include "error_handler.hpp"
#include <sstream>
#include <iomanip>

namespace esx::error {

/**
 * @brief Base error class for EasiScriptX errors.
 */
ESXError::ESXError(const std::string& message, int line, int column)
    : std::runtime_error(message), line_(line), column_(column) {}

int ESXError::line() const { return line_; }
int ESXError::column() const { return column_; }

std::string ESXError::format_message() const {
    std::ostringstream oss;
    oss << "Error at line " << line_ << ", column " << column_ << ": " << what();
    return oss.str();
}

/**
 * @brief Syntax error for parsing issues.
 */
SyntaxError::SyntaxError(const std::string& message, int line, int column)
    : ESXError("SyntaxError: " + message, line, column) {}

/**
 * @brief Runtime error for execution issues.
 */
RuntimeError::RuntimeError(const std::string& message, int line, int column)
    : ESXError("RuntimeError: " + message, line, column) {}

/**
 * @brief Type error for type mismatches.
 */
TypeError::TypeError(const std::string& message, int line, int column)
    : ESXError("TypeError: " + message, line, column) {}

/**
 * @brief Validation error for parameter validation.
 */
ValidationError::ValidationError(const std::string& message, int line, int column)
    : ESXError("ValidationError: " + message, line, column) {}

/**
 * @brief File error for file I/O issues.
 */
FileError::FileError(const std::string& message, int line, int column)
    : ESXError("FileError: " + message, line, column) {}

/**
 * @brief Memory error for memory allocation issues.
 */
MemoryError::MemoryError(const std::string& message, int line, int column)
    : ESXError("MemoryError: " + message, line, column) {}

/**
 * @brief Distributed training error.
 */
DistributedError::DistributedError(const std::string& message, int line, int column)
    : ESXError("DistributedError: " + message, line, column) {}

/**
 * @brief Error handler for managing and reporting errors.
 */
ErrorHandler::ErrorHandler() : error_count_(0), warning_count_(0) {}

void ErrorHandler::report_error(const ESXError& error) {
    error_count_++;
    std::cerr << error.format_message() << std::endl;
    
    if (error_count_ >= MAX_ERRORS) {
        throw std::runtime_error("Too many errors, stopping compilation");
    }
}

void ErrorHandler::report_warning(const std::string& message, int line, int column) {
    warning_count_++;
    std::cerr << "Warning at line " << line << ", column " << column << ": " << message << std::endl;
}

void ErrorHandler::report_info(const std::string& message) {
    std::cout << "Info: " << message << std::endl;
}

int ErrorHandler::error_count() const { return error_count_; }
int ErrorHandler::warning_count() const { return warning_count_; }

void ErrorHandler::reset() {
    error_count_ = 0;
    warning_count_ = 0;
}

/**
 * @brief Input sanitization for security.
 */
std::string sanitize_input(const std::string& input) {
    std::string sanitized = input;
    
    // Remove potentially dangerous characters
    sanitized.erase(std::remove(sanitized.begin(), sanitized.end(), '\0'), sanitized.end());
    sanitized.erase(std::remove(sanitized.begin(), sanitized.end(), '\r'), sanitized.end());
    
    // Escape special characters
    std::string escaped;
    escaped.reserve(sanitized.length() * 2);
    
    for (char c : sanitized) {
        switch (c) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\'': escaped += "\\'"; break;
            case '\n': escaped += "\\n"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += c; break;
        }
    }
    
    return escaped;
}

/**
 * @brief Validate file path for security.
 */
bool validate_file_path(const std::string& path) {
    // Check for directory traversal attempts
    if (path.find("..") != std::string::npos) {
        return false;
    }
    
    // Check for absolute paths (security risk)
    if (!path.empty() && (path[0] == '/' || (path.length() > 1 && path[1] == ':'))) {
        return false;
    }
    
    // Check for null bytes
    if (path.find('\0') != std::string::npos) {
        return false;
    }
    
    return true;
}

/**
 * @brief Create formatted error message with context.
 */
std::string create_error_message(const std::string& type, const std::string& message, 
                                int line, int column, const std::string& context) {
    std::ostringstream oss;
    oss << type << " at line " << line << ", column " << column << ": " << message;
    
    if (!context.empty()) {
        oss << "\nContext: " << context;
    }
    
    return oss.str();
}

/**
 * @brief Global error handler instance.
 */
ErrorHandler& get_error_handler() {
    static ErrorHandler handler;
    return handler;
}

} // namespace esx::error
