#include "core/gguf_parser.h"

#include <iostream>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: dump_gguf <path.gguf>\n";
        return 1;
    }

    try {
        llm::GGUFFile file = llm::GGUFParser::parse(argv[1]);
        std::cout << "version: " << file.version << "\n";
        std::cout << "n_tensors: " << file.n_tensors << "\n";
        std::cout << "n_kv: " << file.n_kv << "\n\n";

        std::cout << "Metadata (first 10 keys):\n";
        size_t shown = 0;
        for (const auto& [key, value] : file.metadata) {
            if (shown >= 10) {
                break;
            }
            std::cout << "  " << key << " = " << llm::gguf_value_to_string(value) << "\n";
            ++shown;
        }

        std::cout << "\nTensors (first 5):\n";
        const size_t n = std::min<size_t>(5, file.tensors.size());
        for (size_t i = 0; i < n; ++i) {
            const auto& t = file.tensors[i];
            std::cout << "  " << t.name << " type=" << llm::ggml_type_name(t.type) << " shape=[";
            for (size_t d = 0; d < t.shape.size(); ++d) {
                if (d > 0) {
                    std::cout << ", ";
                }
                std::cout << t.shape[d];
            }
            std::cout << "] offset=" << t.offset << " nbytes=" << t.n_bytes << "\n";
        }

        std::cout << "\nArchitecture:\n";
        std::cout << "  layers=" << file.arch_layers() << "\n";
        std::cout << "  heads=" << file.n_heads() << " kv_heads=" << file.n_kv_heads() << "\n";
        std::cout << "  embed_dim=" << file.embed_dim() << " ff_dim=" << file.ff_dim() << "\n";
        std::cout << "  vocab_size=" << file.vocab_size() << " context_len=" << file.context_len()
                  << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
