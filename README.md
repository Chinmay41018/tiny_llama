# Tiny Llama C++

A lightweight C++ implementation of a transformer-based language model with integrated tokenizer.

## Features

- **Lightweight Design**: Minimal dependencies, uses only standard C++ libraries
- **GCC Compatible**: Compiles cleanly with GCC 7+ using C++14 standard
- **Modular Architecture**: Clean separation between tokenizer, model, and API layers
- **Memory Efficient**: RAII principles and smart pointer usage for automatic resource management
- **Exception Safe**: Comprehensive error handling with descriptive exception types

## Requirements

- GCC 7.0 or later
- C++14 standard support
- CMake 3.10+ (optional, for automated builds)

## Project Structure

```
tiny_llama_cpp/
├── include/tiny_llama/     # Public header files
│   ├── tiny_llama.hpp      # Main public API
│   ├── tokenizer.hpp       # Tokenizer interface
│   ├── model.hpp           # Model components
│   ├── matrix.hpp          # Matrix operations
│   └── exceptions.hpp      # Exception definitions
├── src/                    # Source implementation files
├── tests/                  # Unit and integration tests
├── examples/               # Usage examples
├── data/                   # Sample data files
├── build/                  # Build output directory
├── CMakeLists.txt          # CMake build configuration
└── README.md              # This file
```

## Building

### Using GCC directly

```bash
# Create build directory
mkdir -p build

# Compile source files
g++ -std=c++14 -Wall -Wextra -Wpedantic -Iinclude -c src/*.cpp
mv *.o build/

# Create static library
ar rcs build/libtiny_llama_cpp.a build/*.o

# Compile examples
g++ -std=c++14 -Iinclude examples/basic_usage.cpp -Lbuild -ltiny_llama_cpp -o build/basic_usage
g++ -std=c++14 -Iinclude examples/advanced_usage.cpp -Lbuild -ltiny_llama_cpp -o build/advanced_usage

# Compile tests
g++ -std=c++14 -Iinclude tests/*.cpp -Lbuild -ltiny_llama_cpp -o build/run_tests
```

### Using CMake (when available)

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

### Basic Usage

```cpp
#include "tiny_llama/tiny_llama.hpp"

int main() {
    try {
        tiny_llama::TinyLlama llama;
        
        // Initialize with model files (implementation in progress)
        // llama.initialize("path/to/model");
        
        // Generate text (implementation in progress)
        // std::string result = llama.generate("Hello world", 50);
        
    } catch (const tiny_llama::TinyLlamaException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}
```

### Advanced Configuration

```cpp
#include "tiny_llama/tiny_llama.hpp"
#include "tiny_llama/model.hpp"

int main() {
    try {
        tiny_llama::TinyLlama llama;
        
        // Custom initialization (implementation in progress)
        // llama.initialize_with_config("vocab.txt", "merges.txt", "weights.bin");
        
        // Configure generation parameters
        // llama.set_temperature(0.8f);
        // llama.set_max_sequence_length(1024);
        
    } catch (const tiny_llama::TinyLlamaException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}
```

## Running Examples

```bash
# Basic usage example
./build/basic_usage

# Advanced usage example  
./build/advanced_usage

# Run tests
./build/run_tests
```

## Development Status

This project is currently under development. The following components are implemented:

- ✅ Project structure and build system
- ✅ Header files with class declarations
- ✅ Basic compilation and linking
- ⏳ Matrix operations (in progress)
- ⏳ Tokenizer implementation (in progress)
- ⏳ Model components (in progress)
- ⏳ Text generation (in progress)

## Architecture

The module consists of four main layers:

1. **Public API Layer**: Simple interface for end users
2. **Model Layer**: Transformer architecture components
3. **Tokenizer Layer**: BPE tokenization and vocabulary management
4. **Utilities Layer**: Matrix operations, file I/O, and error handling

## Contributing

This is an educational implementation focusing on clarity and simplicity. The code follows modern C++ best practices and emphasizes:

- Clear, readable code structure
- Comprehensive error handling
- Memory safety through RAII
- Minimal external dependencies

## License

[License information to be added]