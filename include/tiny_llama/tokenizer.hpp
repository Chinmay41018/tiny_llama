#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace tiny_llama {

/**
 * @brief Vocabulary management class
 */
class Vocabulary {
private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    int unk_token_id_;
    int pad_token_id_;
    int bos_token_id_;
    int eos_token_id_;
    
public:
    /**
     * @brief Default constructor
     */
    Vocabulary();
    
    /**
     * @brief Load vocabulary from file
     * @param vocab_file Path to vocabulary file
     */
    void load_from_file(const std::string& vocab_file);
    
    /**
     * @brief Get token ID for given token string
     * @param token Token string
     * @return Token ID (or UNK ID if not found)
     */
    int get_token_id(const std::string& token) const;
    
    /**
     * @brief Get token string for given ID
     * @param id Token ID
     * @return Token string
     */
    std::string get_token(int id) const;
    
    /**
     * @brief Get vocabulary size
     * @return Number of tokens in vocabulary
     */
    size_t size() const { return id_to_token_.size(); }
    
    /**
     * @brief Get unknown token ID
     * @return UNK token ID
     */
    int unk_id() const { return unk_token_id_; }
    
    /**
     * @brief Get padding token ID
     * @return PAD token ID
     */
    int pad_id() const { return pad_token_id_; }
    
    /**
     * @brief Get beginning-of-sequence token ID
     * @return BOS token ID
     */
    int bos_id() const { return bos_token_id_; }
    
    /**
     * @brief Get end-of-sequence token ID
     * @return EOS token ID
     */
    int eos_id() const { return eos_token_id_; }
    
    /**
     * @brief Add token to vocabulary
     * @param token Token string
     * @return Token ID
     */
    int add_token(const std::string& token);
    
    /**
     * @brief Check if token exists in vocabulary
     * @param token Token string
     * @return True if token exists
     */
    bool has_token(const std::string& token) const;
};

/**
 * @brief Byte Pair Encoding (BPE) tokenizer
 */
class BPETokenizer {
private:
    Vocabulary vocab_;
    std::vector<std::pair<std::string, std::string>> bpe_merges_;
    std::unordered_map<std::string, int> bpe_ranks_;
    
public:
    /**
     * @brief Apply BPE encoding to a word
     * @param word Input word
     * @return Vector of subword tokens
     */
    std::vector<std::string> bpe_encode(const std::string& word) const;
    
    /**
     * @brief Preprocess text before tokenization
     * @param text Input text
     * @return Preprocessed text
     */
    std::string preprocess_text(const std::string& text) const;
    
    /**
     * @brief Split text into words
     * @param text Input text
     * @return Vector of words
     */
    std::vector<std::string> split_to_words(const std::string& text) const;
    /**
     * @brief Default constructor
     */
    BPETokenizer();
    
    /**
     * @brief Load vocabulary from file
     * @param vocab_file Path to vocabulary file
     */
    void load_vocab(const std::string& vocab_file);
    
    /**
     * @brief Load BPE merges from file
     * @param merges_file Path to merges file
     */
    void load_merges(const std::string& merges_file);
    
    /**
     * @brief Encode text to token IDs
     * @param text Input text
     * @return Vector of token IDs
     */
    std::vector<int> encode(const std::string& text) const;
    
    /**
     * @brief Decode token IDs to text
     * @param tokens Vector of token IDs
     * @return Decoded text
     */
    std::string decode(const std::vector<int>& tokens) const;
    
    /**
     * @brief Encode text to token strings
     * @param text Input text
     * @return Vector of token strings
     */
    std::vector<std::string> encode_to_strings(const std::string& text) const;
    
    /**
     * @brief Get vocabulary size
     * @return Size of vocabulary
     */
    size_t vocab_size() const { return vocab_.size(); }
    
    /**
     * @brief Get vocabulary reference
     * @return Reference to vocabulary
     */
    const Vocabulary& get_vocab() const { return vocab_; }
};

} // namespace tiny_llama