#include <iostream>
#include <cassert>
#include <string>
#include "tiny_llama/exceptions.hpp"

using namespace tiny_llama;

void test_base_exception() {
    std::cout << "Testing base TinyLlamaException..." << std::endl;
    
    // Test basic exception with just a message
    try {
        throw TinyLlamaException("Test error message");
    } catch (const TinyLlamaException& e) {
        std::string what_str = e.what();
        assert(what_str.find("Test error message") != std::string::npos);
        assert(e.message() == "Test error message");
        assert(e.context().empty());
        assert(e.file().empty());
        assert(e.line() == 0);
    }
    
    // Test exception with context
    try {
        throw TinyLlamaException("Test with context", "Context information");
    } catch (const TinyLlamaException& e) {
        std::string what_str = e.what();
        assert(what_str.find("Test with context") != std::string::npos);
        assert(what_str.find("Context: Context information") != std::string::npos);
        assert(e.context() == "Context information");
    }
    
    // Test exception with file and line
    try {
        throw TinyLlamaException("Test with location", "", "test_file.cpp", 42);
    } catch (const TinyLlamaException& e) {
        std::string what_str = e.what();
        assert(what_str.find("Location: test_file.cpp:42") != std::string::npos);
        assert(e.file() == "test_file.cpp");
        assert(e.line() == 42);
    }
    
    std::cout << "Base exception tests passed!" << std::endl;
}

void test_specific_exceptions() {
    std::cout << "Testing specific exception types..." << std::endl;
    
    // Test TokenizerException
    try {
        throw TokenizerException("Invalid token", "Token: <UNK>");
    } catch (const TokenizerException& e) {
        std::string what_str = e.what();
        assert(what_str.find("Tokenizer Error: Invalid token") != std::string::npos);
        assert(what_str.find("Context: Token: <UNK>") != std::string::npos);
    }
    
    // Test ModelException
    try {
        throw ModelException("Dimension mismatch", "Expected: 512, Got: 256");
    } catch (const ModelException& e) {
        std::string what_str = e.what();
        assert(what_str.find("Model Error: Dimension mismatch") != std::string::npos);
        assert(what_str.find("Context: Expected: 512, Got: 256") != std::string::npos);
    }
    
    // Test FileIOException
    try {
        throw FileIOException("File not found", "data/vocab.txt");
    } catch (const FileIOException& e) {
        std::string what_str = e.what();
        assert(what_str.find("File I/O Error: File not found") != std::string::npos);
        assert(what_str.find("Context: File: data/vocab.txt") != std::string::npos);
    }
    
    // Test ConfigurationException
    try {
        throw ConfigurationException("Invalid value", "max_sequence_length");
    } catch (const ConfigurationException& e) {
        std::string what_str = e.what();
        assert(what_str.find("Configuration Error: Invalid value") != std::string::npos);
        assert(what_str.find("Context: Parameter: max_sequence_length") != std::string::npos);
    }
    
    // Test MemoryException
    try {
        throw MemoryException("Allocation failed", 1024*1024*100);
    } catch (const MemoryException& e) {
        std::string what_str = e.what();
        assert(what_str.find("Memory Error: Allocation failed") != std::string::npos);
        assert(what_str.find("Context: Requested size: 104857600 bytes") != std::string::npos);
    }
    
    std::cout << "Specific exception tests passed!" << std::endl;
}

void test_exception_macro() {
    std::cout << "Testing exception macro..." << std::endl;
    
    try {
        // This macro should include file and line information automatically
        TINY_LLAMA_THROW(ModelException, "Test macro", "Macro context");
    } catch (const ModelException& e) {
        std::string what_str = e.what();
        assert(what_str.find("Model Error: Test macro") != std::string::npos);
        assert(what_str.find("Context: Macro context") != std::string::npos);
        assert(what_str.find("Location:") != std::string::npos);
        assert(!e.file().empty());
        assert(e.line() > 0);
    }
    
    std::cout << "Exception macro tests passed!" << std::endl;
}

void test_exception_handling() {
    test_base_exception();
    test_specific_exceptions();
    test_exception_macro();
    
    std::cout << "All exception tests passed!" << std::endl;
}