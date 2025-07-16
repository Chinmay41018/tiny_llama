# Tiny Llama Weight File Format

This document describes the binary file format used for storing Tiny Llama model weights.

## File Structure

The weight file is a binary file with the following structure:

### Header (24 bytes)
- **Magic Number** (4 bytes): `0x544C4C4D` ("TLLM" in ASCII)
- **Version** (4 bytes): Format version number (currently 1)
- **Model Configuration** (28 bytes):
  - Model dimension (4 bytes, int)
  - Number of layers (4 bytes, int)
  - Number of heads (4 bytes, int)
  - FFN hidden dimension (4 bytes, int)
  - Max sequence length (4 bytes, int)
  - Vocabulary size (4 bytes, int)
  - Dropout rate (4 bytes, float)

### Embedding Weights
- **Dimensions** (16 bytes): rows (8 bytes, size_t), cols (8 bytes, size_t)
- **Data**: rows × cols × 4 bytes (float array)

### Position Embeddings
- **Dimensions** (16 bytes): rows (8 bytes, size_t), cols (8 bytes, size_t)
- **Data**: rows × cols × 4 bytes (float array)

### Transformer Blocks (repeated for each layer)
For each layer, the following components are stored:

#### Attention Weights
- **Query Weights**: dimensions (16 bytes) + data (model_dim × model_dim × 4 bytes)
- **Key Weights**: dimensions (16 bytes) + data (model_dim × model_dim × 4 bytes)
- **Value Weights**: dimensions (16 bytes) + data (model_dim × model_dim × 4 bytes)
- **Output Weights**: dimensions (16 bytes) + data (model_dim × model_dim × 4 bytes)

#### Feed-Forward Network Weights
- **Linear1 Weights**: dimensions (16 bytes) + data (model_dim × ffn_hidden_dim × 4 bytes)
- **Linear1 Bias**: size (8 bytes) + data (ffn_hidden_dim × 4 bytes)
- **Linear2 Weights**: dimensions (16 bytes) + data (ffn_hidden_dim × model_dim × 4 bytes)
- **Linear2 Bias**: size (8 bytes) + data (model_dim × 4 bytes)

#### Layer Normalization Weights
- **Layer Norm 1 Weights**: size (8 bytes) + data (model_dim × 4 bytes)
- **Layer Norm 1 Bias**: size (8 bytes) + data (model_dim × 4 bytes)
- **Layer Norm 2 Weights**: size (8 bytes) + data (model_dim × 4 bytes)
- **Layer Norm 2 Bias**: size (8 bytes) + data (model_dim × 4 bytes)

### Output Projection
- **Dimensions** (16 bytes): rows (8 bytes, size_t), cols (8 bytes, size_t)
- **Data**: rows × cols × 4 bytes (float array)

## Data Types

- **int**: 32-bit signed integer
- **float**: 32-bit IEEE 754 floating point
- **size_t**: 64-bit unsigned integer (on 64-bit systems)

## Validation

The loader performs the following validations:

1. **Magic Number**: Must match `0x544C4C4D`
2. **Version**: Must be supported (currently only version 1)
3. **Configuration**: Must match the model's expected configuration
4. **Dimensions**: All weight matrices must have correct dimensions
5. **File Integrity**: File must not contain extra data at the end

## Error Handling

The loader throws `FileIOException` for:
- File not found or cannot be opened
- Invalid magic number or version
- Configuration mismatch
- Dimension mismatch
- Corrupted or truncated files
- I/O errors during reading

## Example Usage

```cpp
#include "tiny_llama/model.hpp"

// Create and save model weights
TinyLlamaModel model;
model.save_model_weights("model_weights.bin");

// Load weights into a new model
TinyLlamaModel loaded_model;
loaded_model.load_model_weights("model_weights.bin");
```

## File Size Calculation

For a model with default configuration (512 dim, 6 layers, 8 heads, 32K vocab):
- Header: ~52 bytes
- Embeddings: 32000 × 512 × 4 = ~65MB
- Position embeddings: 1024 × 512 × 4 = ~2MB
- Per layer: ~4MB (6 layers = ~24MB)
- Output projection: 512 × 32000 × 4 = ~65MB
- **Total**: ~156MB

The actual file size may vary slightly due to padding and alignment.