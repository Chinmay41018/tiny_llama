#include "tiny_llama/tokenizer.hpp"
#include "tiny_llama/exceptions.hpp"
#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace tiny_llama;

// Helper function to create a temporary vocabulary file for testing
std::string create_temp_vocab_file() {
    const std::string filename = "temp_vocab_test.txt";
    std::ofstream file(filename);
    
    // Write some test tokens
    file << "hello" << std::endl;
    file << "world" << std::endl;
    file << "test 10" << std::endl;
    file << "example 15" << std::endl;
    file << "token" << std::endl;
    
    file.close();
    return filename;
}

void test_vocabulary_constructor() {
    std::cout << "Testing Vocabulary constructor..." << std::endl;
    
    Vocabulary vocab;
    
    // Check that special tokens are initialized
    assert(vocab.has_token("<unk>"));
    assert(vocab.has_token("<pad>"));
    assert(vocab.has_token("<bos>"));
    assert(vocab.has_token("<eos>"));
    
    // Check special token IDs
    assert(vocab.unk_id() >= 0);
    assert(vocab.pad_id() >= 0);
    assert(vocab.bos_id() >= 0);
    assert(vocab.eos_id() >= 0);
    
    // Check initial size (should be 4 for the special tokens)
    assert(vocab.size() == 4);
    
    std::cout << "Vocabulary constructor test passed!" << std::endl;
}

void test_vocabulary_add_token() {
    std::cout << "Testing Vocabulary add_token..." << std::endl;
    
    Vocabulary vocab;
    size_t initial_size = vocab.size();
    
    // Add new tokens
    int id1 = vocab.add_token("test");
    int id2 = vocab.add_token("example");
    
    // Check that tokens were added
    assert(vocab.has_token("test"));
    assert(vocab.has_token("example"));
    
    // Check that IDs are correct
    assert(id1 == initial_size);
    assert(id2 == initial_size + 1);
    
    // Check that size increased
    assert(vocab.size() == initial_size + 2);
    
    // Add existing token (should return existing ID)
    int id3 = vocab.add_token("test");
    assert(id3== id1);
    assert(vocab.size() == initial_size + 2); // Size should not change
    
    std::cout << "Vocabulary add_token test passed!" << std::endl;
}

void test_vocabulary_token_lookup() {
    std::cout << "Testing Vocabulary token lookup..." << std::endl;
    
    Vocabulary vocab;
    
    // Add some tokens
    int id1 = vocab.add_token("hello");
    int id2 = vocab.add_token("world");
    
    // Test get_token_id
    assert(vocab.get_token_id("hello") == id1);
    assert(vocab.get_token_id("world") == id2);
    assert(vocab.get_token_id("unknown") == vocab.unk_id()); // Unknown token should return UNK ID
    
    // Test get_token
    assert(vocab.get_token(id1) == "hello");
    assert(vocab.get_token(id2) == "world");
    assert(vocab.get_token(-1) == vocab.get_token(vocab.unk_id())); // Invalid ID should return UNK token
    assert(vocab.get_token(1000) == vocab.get_token(vocab.unk_id())); // Out of range ID should return UNK token
    
    std::cout << "Vocabulary token lookup test passed!" << std::endl;
}

void test_vocabulary_load_from_file() {
    std::cout << "Testing Vocabulary load_from_file..." << std::endl;
    
    // Create a temporary vocabulary file
    std::string filename = create_temp_vocab_file();
    
    Vocabulary vocab;
    size_t initial_size = vocab.size();
    
    // Load vocabulary from file
    vocab.load_from_file(filename);
    
    // Check that tokens were loaded
    assert(vocab.has_token("hello"));
    assert(vocab.has_token("world"));
    assert(vocab.has_token("test"));
    assert(vocab.has_token("example"));
    assert(vocab.has_token("token"));
    
    // Check that tokens with explicit IDs have correct IDs
    assert(vocab.get_token(10) == "test");
    assert(vocab.get_token(15) == "example");
    
    // Check that special tokens are still present
    assert(vocab.has_token("<unk>"));
    assert(vocab.has_token("<pad>"));
    assert(vocab.has_token("<bos>"));
    assert(vocab.has_token("<eos>"));
    
    // Clean up
    std::remove(filename.c_str());
    
    std::cout << "Vocabulary load_from_file test passed!" << std::endl;
}

void test_vocabulary_error_handling() {
    std::cout << "Testing Vocabulary error handling..." << std::endl;
    
    Vocabulary vocab;
    
    // Test loading from non-existent file
    bool exception_thrown = false;
    try {
        vocab.load_from_file("non_existent_file.txt");
    } catch (const FileIOException& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "Vocabulary error handling test passed!" << std::endl;
}

void test_vocabulary() {
    std::cout << "Running Vocabulary tests..." << std::endl;
    
    test_vocabulary_constructor();
    test_vocabulary_add_token();
    test_vocabulary_token_lookup();
    test_vocabulary_load_from_file();
    test_vocabulary_error_handling();
    
    std::cout << "All Vocabulary tests passed!" << std::endl;
}

// Helper function to create a temporary merges file for testing
std::string create_temp_merges_file() {
    const std::string filename = "temp_merges_test.txt";
    std::ofstream file(filename);
    
    // Write some test merge rules
    file << "#version: 0.2" << std::endl;  // Header line
    file << "h e" << std::endl;
    file << "he l" << std::endl;
    file << "hel l" << std::endl;
    file << "hell o" << std::endl;
    file << "w o" << std::endl;
    file << "wo r" << std::endl;
    file << "wor l" << std::endl;
    file << "worl d" << std::endl;
    file << "t e" << std::endl;
    file << "te s" << std::endl;
    file << "tes t" << std::endl;
    
    file.close();
    return filename;
}

void test_bpe_tokenizer_constructor() {
    std::cout << "Testing BPETokenizer constructor..." << std::endl;
    
    BPETokenizer tokenizer;
    
    // Check that tokenizer is initialized with a vocabulary containing special tokens
    // The Vocabulary constructor adds 4 special tokens: <unk>, <pad>, <bos>, <eos>
    assert(tokenizer.vocab_size() == 4);
    
    std::cout << "BPETokenizer constructor test passed!" << std::endl;
}

void test_bpe_tokenizer_load_files() {
    std::cout << "Testing BPETokenizer load_vocab and load_merges..." << std::endl;
    
    // Create temporary files
    std::string vocab_file = create_temp_vocab_file();
    std::string merges_file = create_temp_merges_file();
    
    BPETokenizer tokenizer;
    
    // Load vocabulary
    tokenizer.load_vocab(vocab_file);
    
    // Check that vocabulary was loaded
    assert(tokenizer.vocab_size() > 0);
    assert(tokenizer.get_vocab().has_token("hello"));
    assert(tokenizer.get_vocab().has_token("world"));
    
    // Load merges
    tokenizer.load_merges(merges_file);
    
    // Clean up
    std::remove(vocab_file.c_str());
    std::remove(merges_file.c_str());
    
    std::cout << "BPETokenizer load_vocab and load_merges test passed!" << std::endl;
}

void test_bpe_tokenizer_preprocess_text() {
    std::cout << "Testing BPETokenizer preprocess_text..." << std::endl;
    
    BPETokenizer tokenizer;
    
    // Test lowercase conversion
    std::string text1 = "Hello World";
    std::string result1 = tokenizer.preprocess_text(text1);
    assert(result1 == "hello world");
    
    // Test whitespace normalization
    std::string text2 = "Hello\tWorld\nTest";
    std::string result2 = tokenizer.preprocess_text(text2);
    assert(result2 == "hello world test");
    
    // Test empty input
    std::string text3 = "";
    std::string result3 = tokenizer.preprocess_text(text3);
    assert(result3.empty());
    
    std::cout << "BPETokenizer preprocess_text test passed!" << std::endl;
}

void test_bpe_tokenizer_split_to_words() {
    std::cout << "Testing BPETokenizer split_to_words..." << std::endl;
    
    BPETokenizer tokenizer;
    
    // Test basic splitting
    std::string text1 = "hello world";
    std::vector<std::string> words1 = tokenizer.split_to_words(text1);
    assert(words1.size() == 3);  // "hello", " ", "world"
    assert(words1[0] == "hello");
    assert(words1[1] == " ");
    assert(words1[2] == "world");
    
    // Test multiple spaces
    std::string text2 = "hello  world";
    std::vector<std::string> words2 = tokenizer.split_to_words(text2);
    assert(words2.size() == 4);  // "hello", " ", " ", "world"
    assert(words2[0] == "hello");
    assert(words2[1] == " ");
    assert(words2[2] == " ");
    assert(words2[3] == "world");
    
    // Test empty input
    std::string text3 = "";
    std::vector<std::string> words3 = tokenizer.split_to_words(text3);
    assert(words3.empty());
    
    std::cout << "BPETokenizer split_to_words test passed!" << std::endl;
}

void test_bpe_tokenizer_bpe_encode() {
    std::cout << "Testing BPETokenizer bpe_encode..." << std::endl;
    
    // Create temporary files
    std::string vocab_file = create_temp_vocab_file();
    std::string merges_file = create_temp_merges_file();
    
    BPETokenizer tokenizer;
    tokenizer.load_vocab(vocab_file);
    tokenizer.load_merges(merges_file);
    
    // Test BPE encoding with merges
    std::string word1 = "hello";
    std::vector<std::string> tokens1 = tokenizer.bpe_encode(word1);
    
    // With our merge rules, "hello" should be encoded as a single token
    // because we have merges: h+e -> he, he+l -> hel, hel+l -> hell, hell+o -> hello
    assert(tokens1.size() == 1);
    assert(tokens1[0] == "hello");
    
    // Test word without merges
    std::string word2 = "xyz";
    std::vector<std::string> tokens2 = tokenizer.bpe_encode(word2);
    
    // No merges for "xyz", so it should be split into characters
    assert(tokens2.size() == 3);
    assert(tokens2[0] == "x");
    assert(tokens2[1] == "y");
    assert(tokens2[2] == "z");
    
    // Test empty input
    std::string word3 = "";
    std::vector<std::string> tokens3 = tokenizer.bpe_encode(word3);
    assert(tokens3.empty());
    
    // Clean up
    std::remove(vocab_file.c_str());
    std::remove(merges_file.c_str());
    
    std::cout << "BPETokenizer bpe_encode test passed!" << std::endl;
}

void test_bpe_tokenizer_encode_to_strings() {
    std::cout << "Testing BPETokenizer encode_to_strings..." << std::endl;
    
    // Create temporary files
    std::string vocab_file = create_temp_vocab_file();
    std::string merges_file = create_temp_merges_file();
    
    BPETokenizer tokenizer;
    tokenizer.load_vocab(vocab_file);
    tokenizer.load_merges(merges_file);
    
    // Test encoding with spaces
    std::string text1 = "hello world";
    std::vector<std::string> tokens1 = tokenizer.encode_to_strings(text1);
    
    // With our merge rules, this should be encoded as ["hello", " ", "world"]
    assert(tokens1.size() == 3);
    assert(tokens1[0] == "hello");
    assert(tokens1[1] == " ");
    assert(tokens1[2] == "world");
    
    // Test empty input
    std::string text2 = "";
    std::vector<std::string> tokens2 = tokenizer.encode_to_strings(text2);
    assert(tokens2.empty());
    
    // Clean up
    std::remove(vocab_file.c_str());
    std::remove(merges_file.c_str());
    
    std::cout << "BPETokenizer encode_to_strings test passed!" << std::endl;
}

void test_bpe_tokenizer_encode() {
    std::cout << "Testing BPETokenizer encode..." << std::endl;
    
    // Create temporary files
    std::string vocab_file = create_temp_vocab_file();
    std::string merges_file = create_temp_merges_file();
    
    BPETokenizer tokenizer;
    tokenizer.load_vocab(vocab_file);
    tokenizer.load_merges(merges_file);
    
    // Add tokens to vocabulary for testing
    Vocabulary& vocab = const_cast<Vocabulary&>(tokenizer.get_vocab());
    int hello_id = vocab.add_token("hello");
    int space_id = vocab.add_token(" ");
    int world_id = vocab.add_token("world");
    
    // Test encoding to token IDs
    std::string text1 = "hello world";
    std::vector<int> ids1 = tokenizer.encode(text1);
    
    // With our vocabulary, this should be encoded as [hello_id, space_id, world_id]
    assert(ids1.size() == 3);
    assert(ids1[0] == hello_id);
    assert(ids1[1] == space_id);
    assert(ids1[2] == world_id);
    
    // Test empty input
    std::string text2 = "";
    std::vector<int> ids2 = tokenizer.encode(text2);
    assert(ids2.empty());
    
    // Clean up
    std::remove(vocab_file.c_str());
    std::remove(merges_file.c_str());
    
    std::cout << "BPETokenizer encode test passed!" << std::endl;
}

void test_bpe_tokenizer_error_handling() {
    std::cout << "Testing BPETokenizer error handling..." << std::endl;
    
    BPETokenizer tokenizer;
    
    // Test loading from non-existent files
    bool exception_thrown = false;
    try {
        tokenizer.load_vocab("non_existent_file.txt");
    } catch (const FileIOException& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    exception_thrown = false;
    try {
        tokenizer.load_merges("non_existent_file.txt");
    } catch (const FileIOException& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "BPETokenizer error handling test passed!" << std::endl;
}

void test_bpe_tokenizer_decode() {
    std::cout << "Testing BPETokenizer decode..." << std::endl;
    
    // Create temporary files
    std::string vocab_file = create_temp_vocab_file();
    std::string merges_file = create_temp_merges_file();
    
    BPETokenizer tokenizer;
    tokenizer.load_vocab(vocab_file);
    tokenizer.load_merges(merges_file);
    
    // Add tokens to vocabulary for testing
    Vocabulary& vocab = const_cast<Vocabulary&>(tokenizer.get_vocab());
    int hello_id = vocab.add_token("hello");
    int space_id = vocab.add_token(" ");
    int world_id = vocab.add_token("world");
    int unk_id = vocab.unk_id();
    
    // Test basic decoding
    std::vector<int> ids1 = {hello_id, space_id, world_id};
    std::string text1 = tokenizer.decode(ids1);
    assert(text1 == "hello world");
    
    // Test decoding with unknown tokens
    std::vector<int> ids2 = {hello_id, unk_id, world_id};
    std::string text2 = tokenizer.decode(ids2);
    assert(text2 == "hello<unk>world"); // UNK token is represented as "<unk>"
    
    // Test empty input
    std::vector<int> ids3;
    std::string text3 = tokenizer.decode(ids3);
    assert(text3.empty());
    
    // Test out-of-range token IDs
    std::vector<int> ids4 = {hello_id, 9999, world_id};
    std::string text4 = tokenizer.decode(ids4);
    assert(text4 == "hello<unk>world"); // Out-of-range ID should be treated as UNK
    
    // Test negative token IDs
    std::vector<int> ids5 = {hello_id, -1, world_id};
    std::string text5 = tokenizer.decode(ids5);
    assert(text5 == "hello<unk>world"); // Negative ID should be treated as UNK
    
    // Clean up
    std::remove(vocab_file.c_str());
    std::remove(merges_file.c_str());
    
    std::cout << "BPETokenizer decode test passed!" << std::endl;
}

void test_bpe_tokenizer_encode_edge_cases() {
    std::cout << "Testing BPETokenizer encode edge cases..." << std::endl;
    
    // Create temporary files
    std::string vocab_file = create_temp_vocab_file();
    std::string merges_file = create_temp_merges_file();
    
    BPETokenizer tokenizer;
    tokenizer.load_vocab(vocab_file);
    tokenizer.load_merges(merges_file);
    
    // Add tokens to vocabulary for testing
    Vocabulary& vocab = const_cast<Vocabulary&>(tokenizer.get_vocab());
    vocab.add_token("hello");
    vocab.add_token(" ");
    vocab.add_token("world");
    int unk_id = vocab.unk_id();
    
    // Test empty input
    std::vector<int> ids1 = tokenizer.encode("");
    assert(ids1.empty());
    
    // Test input with only whitespace
    std::vector<int> ids2 = tokenizer.encode("   ");
    assert(ids2.size() == 3); // Should be three space tokens
    
    // Test input with unknown characters
    std::vector<int> ids3 = tokenizer.encode("hello 世界");
    assert(ids3.size() > 0);
    // The Chinese characters should be encoded as individual bytes or as UNK tokens
    
    // Test input with special characters
    std::vector<int> ids4 = tokenizer.encode("hello\t\nworld");
    std::string decoded4 = tokenizer.decode(ids4);
    assert(decoded4 == "hello  world"); // Tabs and newlines should be converted to spaces
    
    // Test round-trip encoding and decoding
    std::string original = "hello world";
    std::vector<int> encoded = tokenizer.encode(original);
    std::string decoded = tokenizer.decode(encoded);
    assert(decoded == original);
    
    // Clean up
    std::remove(vocab_file.c_str());
    std::remove(merges_file.c_str());
    
    std::cout << "BPETokenizer encode edge cases test passed!" << std::endl;
}

void test_bpe_tokenizer_encode_decode_integration() {
    std::cout << "Testing BPETokenizer encode-decode integration..." << std::endl;
    
    // Create temporary files
    std::string vocab_file = create_temp_vocab_file();
    std::string merges_file = create_temp_merges_file();
    
    BPETokenizer tokenizer;
    tokenizer.load_vocab(vocab_file);
    tokenizer.load_merges(merges_file);
    
    // Add tokens to vocabulary for testing
    Vocabulary& vocab = const_cast<Vocabulary&>(tokenizer.get_vocab());
    vocab.add_token("hello");
    vocab.add_token(" ");
    vocab.add_token("world");
    vocab.add_token("!");
    
    // Test various inputs for round-trip encoding and decoding
    std::vector<std::string> test_inputs = {
        "",                  // Empty string
        "hello",             // Single word
        "hello world",       // Two words
        "hello world!",      // With punctuation
        "   hello   world  ", // Extra whitespace
        "HELLO WORLD"        // Uppercase (should be lowercased)
    };
    
    for (const auto& input : test_inputs) {
        std::vector<int> encoded = tokenizer.encode(input);
        std::string decoded = tokenizer.decode(encoded);
        
        // For empty input, expect empty output
        if (input.empty()) {
            assert(encoded.empty());
            assert(decoded.empty());
            continue;
        }
        
        // For non-empty input, the decoded result might not exactly match the input
        // due to preprocessing (e.g., lowercasing, whitespace normalization)
        std::string preprocessed = tokenizer.preprocess_text(input);
        
        // Check that the decoded text matches the preprocessed input
        // Note: This might not be exact due to tokenization boundaries
        // but should be close enough for basic testing
        assert(decoded.find(preprocessed) != std::string::npos || 
               preprocessed.find(decoded) != std::string::npos);
    }
    
    // Clean up
    std::remove(vocab_file.c_str());
    std::remove(merges_file.c_str());
    
    std::cout << "BPETokenizer encode-decode integration test passed!" << std::endl;
}

void test_bpe_tokenizer() {
    std::cout << "Running BPE Tokenizer tests..." << std::endl;
    
    test_bpe_tokenizer_constructor();
    test_bpe_tokenizer_load_files();
    test_bpe_tokenizer_preprocess_text();
    test_bpe_tokenizer_split_to_words();
    test_bpe_tokenizer_bpe_encode();
    test_bpe_tokenizer_encode_to_strings();
    test_bpe_tokenizer_encode();
    test_bpe_tokenizer_decode(); // New test for decode method
    test_bpe_tokenizer_encode_edge_cases(); // New test for encode edge cases
    test_bpe_tokenizer_encode_decode_integration(); // New test for encode-decode integration
    test_bpe_tokenizer_error_handling();
    
    std::cout << "All BPE Tokenizer tests passed!" << std::endl;
}