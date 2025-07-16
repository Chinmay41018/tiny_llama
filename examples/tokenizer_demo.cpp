#include "tiny_llama/tiny_llama.hpp"
#include "tiny_llama/tokenizer.hpp"
#include "tiny_llama/exceptions.hpp"
#include <iostream>
#include <fstream>

void create_demo_files() {
    // Create a simple vocabulary file
    std::ofstream vocab_file("demo_vocab.txt");
    vocab_file << "<unk>\n<pad>\n<bos>\n<eos>\n";
    vocab_file << "hello\nworld\nthis\nis\na\ntest\nof\nthe\ntokenizer\n";
    vocab_file << "system\nworking\ncorrectly\nwith\nsome\nexample\ntext\n";
    vocab_file.close();
    
    // Create a simple merges file
    std::ofstream merges_file("demo_merges.txt");
    merges_file << "#version: 0.2\n";
    merges_file << "h e\n";
    merges_file << "l l\n";
    merges_file << "t h\n";
    merges_file << "i s\n";
    merges_file << "o f\n";
    merges_file << "th e\n";
    merges_file << "is a\n";
    merges_file.close();
}

void cleanup_demo_files() {
    std::remove("demo_vocab.txt");
    std::remove("demo_merges.txt");
}

int main() {
    try {
        std::cout << "=== Tokenizer API Demo ===" << std::endl;
        
        // Create demo files
        create_demo_files();
        
        // Test standalone BPE tokenizer
        std::cout << "\n1. Testing standalone BPE tokenizer:" << std::endl;
        tiny_llama::BPETokenizer tokenizer;
        
        // Load vocabulary and merges
        tokenizer.load_vocab("demo_vocab.txt");
        tokenizer.load_merges("demo_merges.txt");
        
        std::cout << "   Vocabulary size: " << tokenizer.vocab_size() << std::endl;
        
        // Test text preprocessing
        std::string original_text = "Hello World! This is a TEST.";
        std::string preprocessed = tokenizer.preprocess_text(original_text);
        std::cout << "   Original text: \"" << original_text << "\"" << std::endl;
        std::cout << "   Preprocessed: \"" << preprocessed << "\"" << std::endl;
        
        // Test tokenization to strings
        std::vector<std::string> token_strings = tokenizer.encode_to_strings(original_text);
        std::cout << "   Token strings: ";
        for (size_t i = 0; i < token_strings.size(); ++i) {
            std::cout << "\"" << token_strings[i] << "\"";
            if (i < token_strings.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        
        // Test tokenization to IDs
        std::vector<int> token_ids = tokenizer.encode(original_text);
        std::cout << "   Token IDs: ";
        for (size_t i = 0; i < token_ids.size(); ++i) {
            std::cout << token_ids[i];
            if (i < token_ids.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        
        // Test detokenization
        std::string decoded = tokenizer.decode(token_ids);
        std::cout << "   Decoded text: \"" << decoded << "\"" << std::endl;
        
        // Test vocabulary access
        const tiny_llama::Vocabulary& vocab = tokenizer.get_vocab();
        std::cout << "   Special tokens - UNK: " << vocab.unk_id() 
                  << ", PAD: " << vocab.pad_id() 
                  << ", BOS: " << vocab.bos_id() 
                  << ", EOS: " << vocab.eos_id() << std::endl;
        
        // Test edge cases
        std::cout << "\n2. Testing edge cases:" << std::endl;
        
        // Empty string
        std::vector<int> empty_tokens = tokenizer.encode("");
        std::cout << "   Empty string tokens: " << empty_tokens.size() << " tokens" << std::endl;
        
        // Single character
        std::vector<int> single_char = tokenizer.encode("a");
        std::cout << "   Single char 'a' tokens: " << single_char.size() << " tokens" << std::endl;
        
        // Unknown words
        std::vector<int> unknown_tokens = tokenizer.encode("unknownword");
        std::cout << "   Unknown word tokens: " << unknown_tokens.size() << " tokens" << std::endl;
        
        std::cout << "\n3. Testing vocabulary operations:" << std::endl;
        
        // Test vocabulary methods directly
        tiny_llama::Vocabulary test_vocab;
        std::cout << "   Initial vocab size: " << test_vocab.size() << std::endl;
        
        int new_token_id = test_vocab.add_token("new_token");
        std::cout << "   Added 'new_token' with ID: " << new_token_id << std::endl;
        std::cout << "   New vocab size: " << test_vocab.size() << std::endl;
        
        // Test token lookup
        std::cout << "   Token 'new_token' has ID: " << test_vocab.get_token_id("new_token") << std::endl;
        std::cout << "   ID " << new_token_id << " maps to token: \"" << test_vocab.get_token(new_token_id) << "\"" << std::endl;
        
        // Test unknown token
        int unknown_id = test_vocab.get_token_id("definitely_unknown");
        std::cout << "   Unknown token ID: " << unknown_id << " (should be UNK ID: " << test_vocab.unk_id() << ")" << std::endl;
        
        std::cout << "\n✅ Tokenizer API demo completed successfully!" << std::endl;
        
        // Cleanup
        cleanup_demo_files();
        
    } catch (const tiny_llama::TinyLlamaException& e) {
        std::cerr << "TinyLlama error: " << e.what() << std::endl;
        cleanup_demo_files();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        cleanup_demo_files();
        return 1;
    }
    
    return 0;
}