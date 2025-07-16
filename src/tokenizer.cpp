#include "tiny_llama/tokenizer.hpp"
#include "tiny_llama/exceptions.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

namespace tiny_llama {

// Special token strings
const std::string UNK_TOKEN = "<unk>";
const std::string PAD_TOKEN = "<pad>";
const std::string BOS_TOKEN = "<bos>";
const std::string EOS_TOKEN = "<eos>";

// Vocabulary implementation
Vocabulary::Vocabulary() : unk_token_id_(-1), pad_token_id_(-1), bos_token_id_(-1), eos_token_id_(-1) {
    // Initialize with special tokens
    unk_token_id_ = add_token(UNK_TOKEN);
    pad_token_id_ = add_token(PAD_TOKEN);
    bos_token_id_ = add_token(BOS_TOKEN);
    eos_token_id_ = add_token(EOS_TOKEN);
}

void Vocabulary::load_from_file(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        TINY_LLAMA_THROW(FileIOException, "Failed to open vocabulary file", vocab_file);
    }

    // Clear existing vocabulary but keep special tokens
    int unk_id = unk_token_id_;
    int pad_id = pad_token_id_;
    int bos_id = bos_token_id_;
    int eos_id = eos_token_id_;
    
    token_to_id_.clear();
    id_to_token_.clear();
    
    // Re-add special tokens
    token_to_id_[UNK_TOKEN] = unk_id;
    token_to_id_[PAD_TOKEN] = pad_id;
    token_to_id_[BOS_TOKEN] = bos_id;
    token_to_id_[EOS_TOKEN] = eos_id;
    
    // Ensure id_to_token_ has enough space
    id_to_token_.resize(std::max({unk_id, pad_id, bos_id, eos_id}) + 1);
    id_to_token_[unk_id] = UNK_TOKEN;
    id_to_token_[pad_id] = PAD_TOKEN;
    id_to_token_[bos_id] = BOS_TOKEN;
    id_to_token_[eos_id] = EOS_TOKEN;
    
    // Read tokens from file
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty()) {
            continue;
        }
        
        // Parse token and ID if format is "token id"
        std::istringstream iss(line);
        std::string token;
        int id = -1;
        
        if (iss >> token >> id) {
            // If ID is provided in the file, use it
            if (id >= 0) {
                if (id >= static_cast<int>(id_to_token_.size())) {
                    id_to_token_.resize(id + 1);
                }
                id_to_token_[id] = token;
                token_to_id_[token] = id;
            } else {
                // Invalid ID, add as new token
                add_token(token);
            }
        } else {
            // If only token is provided, add it
            add_token(token);
        }
    }
    
    // Update special token IDs in case they were in the file
    unk_token_id_ = token_to_id_[UNK_TOKEN];
    pad_token_id_ = token_to_id_[PAD_TOKEN];
    bos_token_id_ = token_to_id_[BOS_TOKEN];
    eos_token_id_ = token_to_id_[EOS_TOKEN];
    
    file.close();
}

int Vocabulary::get_token_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        return it->second;
    }
    return unk_token_id_;
}

std::string Vocabulary::get_token(int id) const {
    if (id < 0 || id >= static_cast<int>(id_to_token_.size())) {
        return id_to_token_[unk_token_id_];
    }
    return id_to_token_[id];
}

int Vocabulary::add_token(const std::string& token) {
    // Check if token already exists
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        return it->second;
    }
    
    // Add new token
    int id = static_cast<int>(id_to_token_.size());
    token_to_id_[token] = id;
    id_to_token_.push_back(token);
    return id;
}

bool Vocabulary::has_token(const std::string& token) const {
    return token_to_id_.find(token) != token_to_id_.end();
}

// BPETokenizer implementation
BPETokenizer::BPETokenizer() {
    // Initialize with empty vocabulary and merges
    // Note: The vocab_ member is initialized with its own constructor
    // which adds special tokens, so it's not actually empty
}

void BPETokenizer::load_vocab(const std::string& vocab_file) {
    vocab_.load_from_file(vocab_file);
}

void BPETokenizer::load_merges(const std::string& merges_file) {
    std::ifstream file(merges_file);
    if (!file.is_open()) {
        TINY_LLAMA_THROW(FileIOException, "Failed to open merges file", merges_file);
    }

    // Clear existing merges
    bpe_merges_.clear();
    bpe_ranks_.clear();

    // Read merges from file
    std::string line;
    int rank = 0;
    
    // Skip header line if present (e.g., "#version: 0.2")
    if (std::getline(file, line) && line.find("#version") != std::string::npos) {
        // This is a header line, continue to next line
    } else {
        // Not a header, process this line
        file.seekg(0); // Reset to beginning of file
    }
    
    // Process merge rules
    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty()) {
            continue;
        }
        
        // Parse merge rule (format: "token1 token2")
        std::istringstream iss(line);
        std::string token1, token2;
        
        if (iss >> token1 >> token2) {
            // Add merge rule
            bpe_merges_.emplace_back(token1, token2);
            
            // Add to ranks map (used for fast lookup during encoding)
            std::string pair_key = token1 + " " + token2;
            bpe_ranks_[pair_key] = rank;
            rank++;
        }
    }
    
    file.close();
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    // Handle empty input case
    if (text.empty()) {
        return {};
    }
    
    try {
        // Preprocess text and split into words
        std::string preprocessed = preprocess_text(text);
        std::vector<std::string> words = split_to_words(preprocessed);
        
        // Encode each word using BPE and convert to token IDs
        std::vector<int> token_ids;
        token_ids.reserve(words.size()); // Reserve space for efficiency
        
        for (const auto& word : words) {
            // Skip empty words
            if (word.empty()) {
                continue;
            }
            
            std::vector<std::string> subwords = bpe_encode(word);
            for (const auto& subword : subwords) {
                // Get token ID, will return UNK token ID if not in vocabulary
                int token_id = vocab_.get_token_id(subword);
                token_ids.push_back(token_id);
            }
        }
        
        return token_ids;
    } catch (const std::exception& e) {
        // Log the error (in a real implementation, you might want to use a proper logging system)
        std::cerr << "Error during tokenization: " << e.what() << std::endl;
        
        // Return UNK token for the entire text in case of error
        return {vocab_.unk_id()};
    }
}

std::string BPETokenizer::decode(const std::vector<int>& tokens) const {
    // Handle empty input case
    if (tokens.empty()) {
        return "";
    }
    
    std::string result;
    
    // Convert each token ID to its string representation and concatenate
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::string token_str = vocab_.get_token(tokens[i]);
        
        // If this is an unknown token, use the UNK token representation
        if (tokens[i] == vocab_.unk_id()) {
            // We could add special handling for unknown tokens here if needed
        }
        
        // Append the token string to the result
        result += token_str;
    }
    
    return result;
}

std::vector<std::string> BPETokenizer::encode_to_strings(const std::string& text) const {
    // Preprocess text and split into words
    std::string preprocessed = preprocess_text(text);
    std::vector<std::string> words = split_to_words(preprocessed);
    
    // Encode each word using BPE
    std::vector<std::string> all_tokens;
    for (const auto& word : words) {
        std::vector<std::string> subwords = bpe_encode(word);
        all_tokens.insert(all_tokens.end(), subwords.begin(), subwords.end());
    }
    
    return all_tokens;
}

std::vector<std::string> BPETokenizer::bpe_encode(const std::string& word) const {
    if (word.empty()) {
        return {};
    }
    
    // Start by splitting the word into individual characters
    std::vector<std::string> chars;
    for (char c : word) {
        chars.push_back(std::string(1, c));
    }
    
    // If only one character, return it directly
    if (chars.size() == 1) {
        return chars;
    }
    
    // Build pairs for initial word
    std::vector<std::pair<std::string, std::string>> pairs;
    for (size_t i = 0; i < chars.size() - 1; i++) {
        pairs.emplace_back(chars[i], chars[i + 1]);
    }
    
    // Main BPE algorithm loop
    while (!pairs.empty()) {
        // Find the pair with the lowest rank (highest priority)
        int best_rank = std::numeric_limits<int>::max();
        size_t best_idx = pairs.size();
        
        for (size_t i = 0; i < pairs.size(); i++) {
            std::string pair_key = pairs[i].first + " " + pairs[i].second;
            auto it = bpe_ranks_.find(pair_key);
            
            if (it != bpe_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }
        
        // If no mergeable pair found, break
        if (best_idx == pairs.size()) {
            break;
        }
        
        // Merge the best pair
        std::string merged = pairs[best_idx].first + pairs[best_idx].second;
        std::vector<std::string> new_chars;
        
        for (size_t i = 0; i < chars.size(); i++) {
            if (i < chars.size() - 1 && 
                chars[i] == pairs[best_idx].first && 
                chars[i + 1] == pairs[best_idx].second) {
                new_chars.push_back(merged);
                i++; // Skip the next character as it's part of the merge
            } else {
                new_chars.push_back(chars[i]);
            }
        }
        
        // Update chars with merged result
        chars = std::move(new_chars);
        
        // Rebuild pairs
        pairs.clear();
        for (size_t i = 0; i < chars.size() - 1; i++) {
            pairs.emplace_back(chars[i], chars[i + 1]);
        }
    }
    
    return chars;
}

std::string BPETokenizer::preprocess_text(const std::string& text) const {
    if (text.empty()) {
        return "";
    }
    
    std::string result;
    result.reserve(text.size());
    
    // Basic preprocessing: lowercase, handle whitespace
    for (char c : text) {
        // Convert to lowercase (simple ASCII conversion)
        if (c >= 'A' && c <= 'Z') {
            c = c - 'A' + 'a';
        }
        
        // Normalize whitespace (convert tabs, newlines to spaces)
        if (c == '\t' || c == '\n' || c == '\r') {
            c = ' ';
        }
        
        result.push_back(c);
    }
    
    return result;
}

std::vector<std::string> BPETokenizer::split_to_words(const std::string& text) const {
    if (text.empty()) {
        return {};
    }
    
    std::vector<std::string> words;
    std::string current_word;
    
    for (char c : text) {
        if (std::isspace(c)) {
            // End of word
            if (!current_word.empty()) {
                words.push_back(current_word);
                current_word.clear();
            }
            // Add space as a separate token
            words.push_back(" ");
        } else {
            current_word.push_back(c);
        }
    }
    
    // Add the last word if not empty
    if (!current_word.empty()) {
        words.push_back(current_word);
    }
    
    return words;
}

} // namespace tiny_llama