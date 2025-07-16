#pragma once

#include <exception>
#include <string>
#include <sstream>
#include <utility>

namespace tiny_llama {

/**
 * @brief Base exception class for Tiny Llama module
 * 
 * This class serves as the base for all exceptions thrown by the Tiny Llama library.
 * It provides context information and a consistent interface for error handling.
 */
class TinyLlamaException : public std::exception {
protected:
    std::string message_;
    std::string context_;
    std::string file_;
    int line_;
    
public:
    /**
     * @brief Construct a new Tiny Llama Exception
     * 
     * @param msg The error message
     * @param context Additional context information (optional)
     * @param file Source file where the exception was thrown (optional)
     * @param line Line number where the exception was thrown (optional)
     */
    explicit TinyLlamaException(
        std::string msg, 
        std::string context = "", 
        std::string file = "", 
        int line = 0
    ) : message_(std::move(msg)), 
        context_(std::move(context)), 
        file_(std::move(file)), 
        line_(line) {}
    
    /**
     * @brief Get the full error message including context if available
     * 
     * @return const char* The formatted error message
     */
    const char* what() const noexcept override {
        static std::string full_message;
        std::ostringstream oss;
        
        oss << message_;
        
        if (!context_.empty()) {
            oss << " [Context: " << context_ << "]";
        }
        
        if (!file_.empty()) {
            oss << " [Location: " << file_;
            if (line_ > 0) {
                oss << ":" << line_;
            }
            oss << "]";
        }
        
        full_message = oss.str();
        return full_message.c_str();
    }
    
    /**
     * @brief Get the raw error message without context
     * 
     * @return const std::string& The raw error message
     */
    const std::string& message() const { return message_; }
    
    /**
     * @brief Get the context information
     * 
     * @return const std::string& The context information
     */
    const std::string& context() const { return context_; }
    
    /**
     * @brief Get the source file where the exception was thrown
     * 
     * @return const std::string& The source file
     */
    const std::string& file() const { return file_; }
    
    /**
     * @brief Get the line number where the exception was thrown
     * 
     * @return int The line number
     */
    int line() const { return line_; }
};

/**
 * @brief Exception thrown by tokenizer operations
 */
class TokenizerException : public TinyLlamaException {
public:
    /**
     * @brief Construct a new Tokenizer Exception
     * 
     * @param msg The error message
     * @param context Additional context information (optional)
     * @param file Source file where the exception was thrown (optional)
     * @param line Line number where the exception was thrown (optional)
     */
    explicit TokenizerException(
        const std::string& msg,
        const std::string& context = "",
        const std::string& file = "",
        int line = 0
    ) : TinyLlamaException("Tokenizer Error: " + msg, context, file, line) {}
};

/**
 * @brief Exception thrown by model operations
 */
class ModelException : public TinyLlamaException {
public:
    /**
     * @brief Construct a new Model Exception
     * 
     * @param msg The error message
     * @param context Additional context information (optional)
     * @param file Source file where the exception was thrown (optional)
     * @param line Line number where the exception was thrown (optional)
     */
    explicit ModelException(
        const std::string& msg,
        const std::string& context = "",
        const std::string& file = "",
        int line = 0
    ) : TinyLlamaException("Model Error: " + msg, context, file, line) {}
};

/**
 * @brief Exception thrown by file I/O operations
 */
class FileIOException : public TinyLlamaException {
public:
    /**
     * @brief Construct a new File IO Exception
     * 
     * @param msg The error message
     * @param filepath The path to the file that caused the error (optional)
     * @param file Source file where the exception was thrown (optional)
     * @param line Line number where the exception was thrown (optional)
     */
    explicit FileIOException(
        const std::string& msg,
        const std::string& filepath = "",
        const std::string& file = "",
        int line = 0
    ) : TinyLlamaException("File I/O Error: " + msg, 
                          filepath.empty() ? "" : "File: " + filepath, 
                          file, line) {}
};

/**
 * @brief Exception thrown when invalid configuration is provided
 */
class ConfigurationException : public TinyLlamaException {
public:
    /**
     * @brief Construct a new Configuration Exception
     * 
     * @param msg The error message
     * @param param The configuration parameter that caused the error (optional)
     * @param file Source file where the exception was thrown (optional)
     * @param line Line number where the exception was thrown (optional)
     */
    explicit ConfigurationException(
        const std::string& msg,
        const std::string& param = "",
        const std::string& file = "",
        int line = 0
    ) : TinyLlamaException("Configuration Error: " + msg, 
                          param.empty() ? "" : "Parameter: " + param, 
                          file, line) {}
};

/**
 * @brief Exception thrown when memory allocation fails or memory limits are exceeded
 */
class MemoryException : public TinyLlamaException {
public:
    /**
     * @brief Construct a new Memory Exception
     * 
     * @param msg The error message
     * @param requested_size The size of memory that was requested (optional)
     * @param file Source file where the exception was thrown (optional)
     * @param line Line number where the exception was thrown (optional)
     */
    explicit MemoryException(
        const std::string& msg,
        size_t requested_size = 0,
        const std::string& file = "",
        int line = 0
    ) : TinyLlamaException("Memory Error: " + msg, 
                          requested_size > 0 ? "Requested size: " + std::to_string(requested_size) + " bytes" : "", 
                          file, line) {}
};

/**
 * @brief Macro to throw exceptions with file and line information
 */
#define TINY_LLAMA_THROW(exception_type, msg, ...) \
    throw exception_type(msg, __VA_ARGS__, __FILE__, __LINE__)

} // namespace tiny_llama