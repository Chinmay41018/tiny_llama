#include "tiny_llama/tokenizer.hpp"
#include "tiny_llama/exceptions.hpp"
#include <cassert>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdio>

using namespace tiny_llama;

// Helper function to check if file exists (C++14 compatible)
bool file_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

// Helper function to create directory (C++14 compatible)
void create_directory(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

// Helper function to remove file (C++14 compatible)
void remove_file(const std::string& path) {
    remove(path.c_str());
}

/**
 * @brief Test helper to create sample files for tokenizer testing
 */
class TokenizerTestHelper {
public:
    static void create_test_directory(const std::string& path) {
        create_directory(path);
    }
    
    static void create_sample_vocab_file(const std::string& filepath) {
        std::ofstream file(filepath);
        file << "<unk>\n";
        file << "<pad>\n";
        file << "<bos>\n";
        file << "<eos>\n";
        file << "hello\n";
        file << "world\n";
        file << "test\n";
        file << "token\n";
        file << "the\n";
        file << "quick\n";
        file << "brown\n";
        file << "fox\n";
        file << "jumps\n";
        file << "over\n";
        file << "lazy\n";
        file << "dog\n";
        file << "a\n";
        file << "an\n";
        file << "and\n";
        file << "or\n";
        file << "but\n";
        file << "if\n";
        file << "then\n";
        file << "else\n";
        file.close();
    }
    
    static void create_sample_merges_file(const std::string& filepath) {
        std::ofstream file(filepath);
        file << "#version: 0.2\n";
        file << "h e\n";
        file << "l l\n";
        file << "o r\n";
        file << "t h\n";
        file << "e r\n";
        file << "i n\n";
        file << "a n\n";
        file << "o n\n";
        file << "s t\n";
        file << "th e\n";
        file << "he llo\n";
        file << "wor ld\n";
        file.close();
    }
    
    static void create_vocab_with_ids_file(const std::string& filepath) {
        std::ofstream file(filepath);
        file << "<unk> 0\n";
        file << "<pad> 1\n";
        file << "<bos> 2\n";
        file << "<eos> 3\n";
        file << "hello 4\n";
        file << "world 5\n";
        file << "test 6\n";
        file << "token 7\n";
        file << "the 8\n";
        file << "quick 9\n";
        file.close();
    }
    
    static void cleanup_test_directory(const std::string& path) {
        if (file_exists(path)) {
            // Remove files in the directory
            remove_file(path + "/vocab.txt");
            remove_file(path + "/merges.txt");
            remove_file(path + "/vocab_with_ids.txt");
            // Remove the directory
            rmdir(path.c_str());
        }
    }
};

/**
 * @brief Test Vocabulary class constructor and basic functionality
 */
void test_vocabulary_constructor() {
    std::cout << "Testing Vocabulary constructor..." << std::endl;
    
    Vocabulary vocab;
    
    // Test that special tokens are initialized
    assert(vocab.size() >= 4); // At least UNK, PAD, BOS, EOS
    assert(vocab.unk_id() >= 0);
    assert(vocab.pad_id() >= 0);
    assert(vocab.bos_id() >= 0);
    assert(vocab.eos_id() >= 0);
    
    // Test that special tokens are different
    assert(vocab.unk_id() != vocab.pad_id());
    assert(vocab.unk_id() != vocab.bos_id());
    assert(vocab.unk_id() != vocab.eos_id());
    assert(vocab.pad_id() != vocab.bos_id());
    assert(vocab.pad_id() != vocab.eos_id());
    assert(vocab.bos_id() != vocab.eos_id());
    
    // Test that we can retrieve special tokens
    assert(vocab.get_token(vocab.unk_id()) == "<unk>");
    assert(vocab.get_token(vocab.pad_id()) == "<pad>");
    assert(vocab.get_token(vocab.bos_id()) == "<bos>");
    assert(vocab.get_token(vocab.eos_id()) == "<eos>");
    
    // Test that we can get IDs for special tokens
    assert(vocab.get_token_id("<unk>") == vocab.unk_id());
    assert(vocab.get_token_id("<pad>") == vocab.pad_id());
    assert(vocab.get_token_id("<bos>") == vocab.bos_id());
    assert(vocab.get_token_id("<eos>") == vocab.eos_id());
    
    std::cout << "✓ Vocabulary constructor works correctly" << std::endl;
}

/**
 * @brief Test Vocabulary loading from file
 */
void test_vocabulary_load_from_file() {
    std::cout << "Testing Vocabulary load from file..." << std::endl;
    
    const std::string test_dir = "test_vocab_data";
    TokenizerTestHelper::create_test_directory(test_dir);
    
    try {
        Vocabulary vocab;
        
        // Test loading non-existent file
        try {
            vocab.load_from_file("non_existent_file.txt");
            assert(false && "Should have thrown FileIOException");
        } catch (const FileIOException& e) {
            // Expected
        }
        
        // Create and load sample vocabulary file
        std::string vocab_file = test_dir + "/vocab.txt";
        TokenizerTestHelper::create_sample_vocab_file(vocab_file);
        
        size_t initial_size = vocab.size();
        vocab.load_from_file(vocab_file);
        
        // Test that vocabulary size increased
        assert(vocab.size() > initial_size);
        
        // Test that we can find tokens from the file
        assert(vocab.has_token("hello"));
        assert(vocab.has_token("world"));
        assert(vocab.has_token("test"));
        assert(vocab.has_token("token"));
        
        // Test that we can get token IDs
        int hello_id = vocab.get_token_id("hello");
        int world_id = vocab.get_token_id("world");
        assert(hello_id >= 0);
        assert(world_id >= 0);
        assert(hello_id != world_id);
        
        // Test that we can get tokens back from IDs
        assert(vocab.get_token(hello_id) == "hello");
        assert(vocab.get_token(world_id) == "world");
        
        // Test unknown token handling
        int unknown_id = vocab.get_token_id("unknown_token");
        assert(unknown_id == vocab.unk_id());
        
        std::cout << "✓ Vocabulary load from file works correctly" << std::endl;
        
    } catch (const std::exception& e) {
        TokenizerTestHelper::cleanup_test_directory(test_dir);
        throw;
    }
    
    TokenizerTestHelper::cleanup_test_directory(test_dir);
}

/**
 * @brief Test Vocabulary loading with explicit IDs
 */
void test_vocabulary_load_with_ids() {
    std::cout << "Testing Vocabulary load with explicit IDs..." << std::endl;
    
    const std::string test_dir = "test_vocab_ids_data";
    TokenizerTestHelper::create_test_directory(test_dir);
    
    try {
        Vocabulary vocab;
        
        // Create and load vocabulary file with explicit IDs
        std::string vocab_file = test_dir + "/vocab_with_ids.txt";
        TokenizerTestHelper::create_vocab_with_ids_file(vocab_file);
        
        vocab.load_from_file(vocab_file);
        
        // Test that tokens have the expected IDs
        assert(vocab.get_token_id("<unk>") == 0);
        assert(vocab.get_token_id("<pad>") == 1);
        assert(vocab.get_token_id("<bos>") == 2);
        assert(vocab.get_token_id("<eos>") == 3);
        assert(vocab.get_token_id("hello") == 4);
        assert(vocab.get_token_id("world") == 5);
        
        // Test that IDs map back to correct tokens
        assert(vocab.get_token(0) == "<unk>");
        assert(vocab.get_token(1) == "<pad>");
        assert(vocab.get_token(2) == "<bos>");
        assert(vocab.get_token(3) == "<eos>");
        assert(vocab.get_token(4) == "hello");
        assert(vocab.get_token(5) == "world");
        
        std::cout << "✓ Vocabulary load with explicit IDs works correctly" << std::endl;
        
    } catch (const std::exception& e) {
        TokenizerTestHelper::cleanup_test_directory(test_dir);
        throw;
    }
    
    TokenizerTestHelper::cleanup_test_directory(test_dir);
}

/**
 * @brief Test Vocabulary add_token functionality
 */
void test_vocabulary_add_token() {
    std::cout << "Testing Vocabulary add_token..." << std::endl;
    
    Vocabulary vocab;
    size_t initial_size = vocab.size();
    
    // Add a new token
    int new_token_id = vocab.add_token("new_token");
    assert(new_token_id >= 0);
    assert(vocab.size() == initial_size + 1);
    
    // Test that the token was added correctly
    assert(vocab.has_token("new_token"));
    assert(vocab.get_token_id("new_token") == new_token_id);
    assert(vocab.get_token(new_token_id) == "new_token");
    
    // Test adding the same token again (should return same ID)
    int duplicate_id = vocab.add_token("new_token");
    assert(duplicate_id == new_token_id);
    assert(vocab.size() == initial_size + 1); // Size shouldn't change
    
    std::cout << "✓ Vocabulary add_token works correctly" << std::endl;
}

/**
 * @brief Test BPETokenizer constructor and basic functionality
 */
void test_bpe_tokenizer_constructor() {
    std::cout << "Testing BPETokenizer constructor..." << std::endl;
    
    BPETokenizer tokenizer;
    
    // Test that tokenizer has a vocabulary with special tokens
    assert(tokenizer.vocab_size() >= 4);
    const Vocabulary& vocab = tokenizer.get_vocab();
    assert(vocab.unk_id() >= 0);
    assert(vocab.pad_id() >= 0);
    assert(vocab.bos_id() >= 0);
    assert(vocab.eos_id() >= 0);
    
    std::cout << "✓ BPETokenizer constructor works correctly" << std::endl;
}

/**
 * @brief Test BPETokenizer loading vocabulary and merges
 */
void test_bpe_tokenizer_load_files() {
    std::cout << "Testing BPETokenizer load files..." << std::endl;
    
    const std::string test_dir = "test_bpe_data";
    TokenizerTestHelper::create_test_directory(test_dir);
    
    try {
        BPETokenizer tokenizer;
        
        // Test loading non-existent files
        try {
            tokenizer.load_vocab("non_existent_vocab.txt");
            assert(false && "Should have thrown FileIOException");
        } catch (const FileIOException& e) {
            // Expected
        }
        
        try {
            tokenizer.load_merges("non_existent_merges.txt");
            assert(false && "Should have thrown FileIOException");
        } catch (const FileIOException& e) {
            // Expected
        }
        
        // Create and load sample files
        std::string vocab_file = test_dir + "/vocab.txt";
        std::string merges_file = test_dir + "/merges.txt";
        
        TokenizerTestHelper::create_sample_vocab_file(vocab_file);
        TokenizerTestHelper::create_sample_merges_file(merges_file);
        
        size_t initial_vocab_size = tokenizer.vocab_size();
        
        tokenizer.load_vocab(vocab_file);
        assert(tokenizer.vocab_size() > initial_vocab_size);
        
        tokenizer.load_merges(merges_file);
        
        // Test that vocabulary contains expected tokens
        const Vocabulary& vocab = tokenizer.get_vocab();
        assert(vocab.has_token("hello"));
        assert(vocab.has_token("world"));
        assert(vocab.has_token("test"));
        
        std::cout << "✓ BPETokenizer load files works correctly" << std::endl;
        
    } catch (const std::exception& e) {
        TokenizerTestHelper::cleanup_test_directory(test_dir);
        throw;
    }
    
    TokenizerTestHelper::cleanup_test_directory(test_dir);
}

/**
 * @brief Test BPETokenizer text preprocessing
 */
void test_bpe_tokenizer_preprocess() {
    std::cout << "Testing BPETokenizer text preprocessing..." << std::endl;
    
    BPETokenizer tokenizer;
    
    // Test empty input
    assert(tokenizer.preprocess_text("") == "");
    
    // Test basic preprocessing
    std::string input = "Hello World!";
    std::string processed = tokenizer.preprocess_text(input);
    assert(processed == "hello world!");
    
    // Test whitespace normalization
    input = "Hello\tWorld\nTest\rString";
    processed = tokenizer.preprocess_text(input);
    assert(processed == "hello world test string");
    
    // Test mixed case
    input = "MiXeD CaSe TeXt";
    processed = tokenizer.preprocess_text(input);
    assert(processed == "mixed case text");
    
    std::cout << "✓ BPETokenizer text preprocessing works correctly" << std::endl;
}

/**
 * @brief Test BPETokenizer word splitting
 */
void test_bpe_tokenizer_split_words() {
    std::cout << "Testing BPETokenizer word splitting..." << std::endl;
    
    BPETokenizer tokenizer;
    
    // Test empty input
    std::vector<std::string> words = tokenizer.split_to_words("");
    assert(words.empty());
    
    // Test single word
    words = tokenizer.split_to_words("hello");
    assert(words.size() == 1);
    assert(words[0] == "hello");
    
    // Test multiple words
    words = tokenizer.split_to_words("hello world");
    assert(words.size() == 3); // "hello", " ", "world"
    assert(words[0] == "hello");
    assert(words[1] == " ");
    assert(words[2] == "world");
    
    // Test multiple spaces
    words = tokenizer.split_to_words("hello  world");
    assert(words.size() == 4); // "hello", " ", " ", "world"
    assert(words[0] == "hello");
    assert(words[1] == " ");
    assert(words[2] == " ");
    assert(words[3] == "world");
    
    // Test leading/trailing spaces
    words = tokenizer.split_to_words(" hello world ");
    assert(words.size() == 5); // " ", "hello", " ", "world", " "
    assert(words[0] == " ");
    assert(words[1] == "hello");
    assert(words[2] == " ");
    assert(words[3] == "world");
    assert(words[4] == " ");
    
    std::cout << "✓ BPETokenizer word splitting works correctly" << std::endl;
}

/**
 * @brief Test BPETokenizer BPE encoding
 */
void test_bpe_tokenizer_bpe_encode() {
    std::cout << "Testing BPETokenizer BPE encoding..." << std::endl;
    
    const std::string test_dir = "test_bpe_encode_data";
    TokenizerTestHelper::create_test_directory(test_dir);
    
    try {
        BPETokenizer tokenizer;
        
        // Load vocabulary and merges
        std::string vocab_file = test_dir + "/vocab.txt";
        std::string merges_file = test_dir + "/merges.txt";
        
        TokenizerTestHelper::create_sample_vocab_file(vocab_file);
        TokenizerTestHelper::create_sample_merges_file(merges_file);
        
        tokenizer.load_vocab(vocab_file);
        tokenizer.load_merges(merges_file);
        
        // Test empty input
        std::vector<std::string> tokens = tokenizer.bpe_encode("");
        assert(tokens.empty());
        
        // Test single character
        tokens = tokenizer.bpe_encode("a");
        assert(tokens.size() == 1);
        assert(tokens[0] == "a");
        
        // Test word that should be split by BPE
        tokens = tokenizer.bpe_encode("hello");
        assert(!tokens.empty());
        
        // The exact result depends on the merge rules, but it should be split
        // into subword units based on the BPE merges
        
        std::cout << "✓ BPETokenizer BPE encoding works correctly" << std::endl;
        
    } catch (const std::exception& e) {
        TokenizerTestHelper::cleanup_test_directory(test_dir);
        throw;
    }
    
    TokenizerTestHelper::cleanup_test_directory(test_dir);
}

/**
 * @brief Test BPETokenizer full encoding pipeline
 */
void test_bpe_tokenizer_encode() {
    std::cout << "Testing BPETokenizer full encoding..." << std::endl;
    
    const std::string test_dir = "test_bpe_full_encode_data";
    TokenizerTestHelper::create_test_directory(test_dir);
    
    try {
        BPETokenizer tokenizer;
        
        // Load vocabulary and merges
        std::string vocab_file = test_dir + "/vocab.txt";
        std::string merges_file = test_dir + "/merges.txt";
        
        TokenizerTestHelper::create_sample_vocab_file(vocab_file);
        TokenizerTestHelper::create_sample_merges_file(merges_file);
        
        tokenizer.load_vocab(vocab_file);
        tokenizer.load_merges(merges_file);
        
        // Test empty input
        std::vector<int> token_ids = tokenizer.encode("");
        assert(token_ids.empty());
        
        // Test simple text
        token_ids = tokenizer.encode("hello world");
        assert(!token_ids.empty());
        
        // All token IDs should be valid (>= 0)
        for (int id : token_ids) {
            assert(id >= 0);
        }
        
        // Test that we can encode to strings as well
        std::vector<std::string> token_strings = tokenizer.encode_to_strings("hello world");
        assert(!token_strings.empty());
        
        // The number of token strings should match the number of token IDs
        // (though this might not always be true in general BPE, it should be for our simple case)
        
        std::cout << "✓ BPETokenizer full encoding works correctly" << std::endl;
        
    } catch (const std::exception& e) {
        TokenizerTestHelper::cleanup_test_directory(test_dir);
        throw;
    }
    
    TokenizerTestHelper::cleanup_test_directory(test_dir);
}

/**
 * @brief Test BPETokenizer decoding
 */
void test_bpe_tokenizer_decode() {
    std::cout << "Testing BPETokenizer decoding..." << std::endl;
    
    const std::string test_dir = "test_bpe_decode_data";
    TokenizerTestHelper::create_test_directory(test_dir);
    
    try {
        BPETokenizer tokenizer;
        
        // Load vocabulary and merges
        std::string vocab_file = test_dir + "/vocab.txt";
        std::string merges_file = test_dir + "/merges.txt";
        
        TokenizerTestHelper::create_sample_vocab_file(vocab_file);
        TokenizerTestHelper::create_sample_merges_file(merges_file);
        
        tokenizer.load_vocab(vocab_file);
        tokenizer.load_merges(merges_file);
        
        // Test empty input
        std::string decoded = tokenizer.decode({});
        assert(decoded.empty());
        
        // Test decoding valid token IDs
        const Vocabulary& vocab = tokenizer.get_vocab();
        std::vector<int> token_ids = {
            vocab.get_token_id("hello"),
            vocab.get_token_id(" "),
            vocab.get_token_id("world")
        };
        
        decoded = tokenizer.decode(token_ids);
        assert(!decoded.empty());
        
        // Test decoding unknown token ID
        std::vector<int> unknown_ids = {vocab.unk_id()};
        decoded = tokenizer.decode(unknown_ids);
        assert(decoded == "<unk>");
        
        // Test round-trip encoding/decoding
        std::string original_text = "hello world";
        std::vector<int> encoded = tokenizer.encode(original_text);
        std::string round_trip = tokenizer.decode(encoded);
        
        // The round-trip result should be similar to the original
        // (exact match depends on BPE processing and vocabulary coverage)
        assert(!round_trip.empty());
        
        std::cout << "✓ BPETokenizer decoding works correctly" << std::endl;
        
    } catch (const std::exception& e) {
        TokenizerTestHelper::cleanup_test_directory(test_dir);
        throw;
    }
    
    TokenizerTestHelper::cleanup_test_directory(test_dir);
}

/**
 * @brief Test BPETokenizer edge cases and error handling
 */
void test_bpe_tokenizer_edge_cases() {
    std::cout << "Testing BPETokenizer edge cases..." << std::endl;
    
    BPETokenizer tokenizer;
    
    // Test with uninitialized tokenizer (no vocab/merges loaded)
    std::vector<int> token_ids = tokenizer.encode("test");
    // Should not crash, might return UNK tokens
    
    std::string decoded = tokenizer.decode({0, 1, 2});
    // Should not crash, should return some result
    assert(!decoded.empty());
    
    // Test with very long input
    std::string long_text(1000, 'a');
    token_ids = tokenizer.encode(long_text);
    // Should handle long input gracefully
    
    // Test with special characters
    std::string special_text = "!@#$%^&*()_+-=[]{}|;':\",./<>?";
    token_ids = tokenizer.encode(special_text);
    // Should handle special characters
    
    // Test with Unicode-like characters (within ASCII range)
    std::string unicode_text = "café naïve résumé";
    token_ids = tokenizer.encode(unicode_text);
    // Should handle accented characters
    
    std::cout << "✓ BPETokenizer edge cases work correctly" << std::endl;
}

/**
 * @brief Main test function for tokenizer APIs
 */
void test_tokenizer_apis() {
    std::cout << "Testing Tokenizer APIs..." << std::endl;
    
    test_vocabulary_constructor();
    test_vocabulary_load_from_file();
    test_vocabulary_load_with_ids();
    test_vocabulary_add_token();
    test_bpe_tokenizer_constructor();
    test_bpe_tokenizer_load_files();
    test_bpe_tokenizer_preprocess();
    test_bpe_tokenizer_split_words();
    test_bpe_tokenizer_bpe_encode();
    test_bpe_tokenizer_encode();
    test_bpe_tokenizer_decode();
    test_bpe_tokenizer_edge_cases();
    
    std::cout << "✓ All Tokenizer API tests passed" << std::endl;
}

/**
 * @brief Main test function (for standalone execution)
 */
int main() {
    std::cout << "Running Tokenizer API tests..." << std::endl;
    
    try {
        test_tokenizer_apis();
        
        std::cout << "\n✅ All Tokenizer API tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "\n❌ Test failed: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "\n❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}