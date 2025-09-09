#include "libtorchtcl.h"

// Parameter structure for embedding command
struct EmbeddingArgs {
    std::string input;
    int num_embeddings = 0;
    int embedding_dim = 0;
    int padding_idx = -1;
    
    bool IsValid() const {
        return !input.empty() && num_embeddings > 0 && embedding_dim > 0;
    }
};

// Parse dual syntax for embedding
EmbeddingArgs ParseEmbeddingArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    EmbeddingArgs args;
    
    if (objc < 4) {
        throw std::runtime_error("Usage: torch::embedding input num_embeddings embedding_dim [padding_idx] | torch::embedding -input tensor -num_embeddings int -embedding_dim int [-padding_idx int]");
    }
    
    if (objc >= 4 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.num_embeddings) != TCL_OK) {
            throw std::runtime_error("Invalid num_embeddings value");
        }
        
        if (Tcl_GetIntFromObj(interp, objv[3], &args.embedding_dim) != TCL_OK) {
            throw std::runtime_error("Invalid embedding_dim value");
        }
        
        if (objc >= 5) {
            if (Tcl_GetIntFromObj(interp, objv[4], &args.padding_idx) != TCL_OK) {
                throw std::runtime_error("Invalid padding_idx value");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-num_embeddings") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.num_embeddings) != TCL_OK) {
                    throw std::runtime_error("Invalid num_embeddings value");
                }
            } else if (param == "-embedding_dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.embedding_dim) != TCL_OK) {
                    throw std::runtime_error("Invalid embedding_dim value");
                }
            } else if (param == "-padding_idx") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.padding_idx) != TCL_OK) {
                    throw std::runtime_error("Invalid padding_idx value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor, -num_embeddings, -embedding_dim, -padding_idx");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor, num_embeddings > 0, and embedding_dim > 0 required");
    }
    
    if (args.num_embeddings <= 0) {
        throw std::runtime_error("num_embeddings must be positive");
    }
    
    if (args.embedding_dim <= 0) {
        throw std::runtime_error("embedding_dim must be positive");
    }
    
    return args;
}

// Embedding Layer
int Embedding_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        EmbeddingArgs args = ParseEmbeddingArgs(interp, objc, objv);
        
        // Look up tensor from storage
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        torch::Tensor input = tensor_storage[args.input];

        // Create embedding weight matrix
        torch::Tensor weight = torch::randn({args.num_embeddings, args.embedding_dim});
        
        // Zero out padding index
        if (args.padding_idx >= 0 && args.padding_idx < args.num_embeddings) {
            weight.index_put_({args.padding_idx}, torch::zeros({args.embedding_dim}));
        }

        // Use correct API signature: embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)
        torch::Tensor result = torch::embedding(weight, input, args.padding_idx, false, false);
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for embedding_bag command
struct EmbeddingBagArgs {
    std::string input;
    std::string weight;
    std::string offsets;
    int mode = 0;  // 0=sum, 1=mean, 2=max
    std::string per_sample_weights;  // Optional
    
    bool IsValid() const {
        return !input.empty() && !weight.empty() && !offsets.empty();
    }
};

// Parse dual syntax for embedding_bag
EmbeddingBagArgs ParseEmbeddingBagArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    EmbeddingBagArgs args;
    
    if (objc < 5) {
        throw std::runtime_error("Usage: torch::embedding_bag input weight offsets mode [per_sample_weights] | torch::embedding_bag -input tensor -weight tensor -offsets tensor -mode int [-per_sample_weights tensor]");
    }
    
    if (objc >= 5 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        args.input = Tcl_GetString(objv[1]);
        args.weight = Tcl_GetString(objv[2]);
        args.offsets = Tcl_GetString(objv[3]);
        
        if (Tcl_GetIntFromObj(interp, objv[4], &args.mode) != TCL_OK) {
            throw std::runtime_error("Invalid mode value");
        }
        
        if (objc >= 6) {
            args.per_sample_weights = Tcl_GetString(objv[5]);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-weight") {
                args.weight = Tcl_GetString(objv[i + 1]);
            } else if (param == "-offsets") {
                args.offsets = Tcl_GetString(objv[i + 1]);
            } else if (param == "-mode") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.mode) != TCL_OK) {
                    throw std::runtime_error("Invalid mode value");
                }
            } else if (param == "-per_sample_weights") {
                args.per_sample_weights = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -weight, -offsets, -mode, -per_sample_weights");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input, weight, and offsets tensors required");
    }
    
    if (args.mode < 0 || args.mode > 2) {
        throw std::runtime_error("Mode must be 0 (sum), 1 (mean), or 2 (max)");
    }
    
    return args;
}

// Embedding Bag
int EmbeddingBag_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        EmbeddingBagArgs args = ParseEmbeddingBagArgs(interp, objc, objv);
        
        // Look up tensors from storage
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        torch::Tensor input = tensor_storage[args.input];
        
        if (tensor_storage.find(args.weight) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid weight tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        torch::Tensor weight = tensor_storage[args.weight];
        
        if (tensor_storage.find(args.offsets) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid offsets tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        torch::Tensor offsets = tensor_storage[args.offsets];
        
        torch::Tensor per_sample_weights;
        if (!args.per_sample_weights.empty()) {
            if (tensor_storage.find(args.per_sample_weights) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>("Invalid per_sample_weights tensor"), TCL_VOLATILE);
                return TCL_ERROR;
            }
            per_sample_weights = tensor_storage[args.per_sample_weights];
        }

        // Use correct API signature 
        auto result_tuple = torch::embedding_bag(weight, input, offsets, false, args.mode, false, 
                                                per_sample_weights, false);
        torch::Tensor result = std::get<0>(result_tuple);
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for sparse_embedding command
struct SparseEmbeddingArgs {
    std::string input;
    int num_embeddings = 0;
    int embedding_dim = 0;
    int padding_idx = -1;
    
    bool IsValid() const {
        return !input.empty() && num_embeddings > 0 && embedding_dim > 0;
    }
};

// Parse dual syntax for sparse_embedding
SparseEmbeddingArgs ParseSparseEmbeddingArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SparseEmbeddingArgs args;
    
    // Check for named parameter syntax
    bool use_named_params = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');
    
    if (!use_named_params) {
        // Positional syntax (backward compatibility)
        if (objc != 5) {
            Tcl_WrongNumArgs(interp, 1, objv, "input num_embeddings embedding_dim padding_idx");
            throw std::runtime_error("");  // Empty message since Tcl_WrongNumArgs already set the error
        }
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.num_embeddings) != TCL_OK) {
            throw std::runtime_error("Invalid num_embeddings value");
        }
        
        if (Tcl_GetIntFromObj(interp, objv[3], &args.embedding_dim) != TCL_OK) {
            throw std::runtime_error("Invalid embedding_dim value");
        }
        
        if (Tcl_GetIntFromObj(interp, objv[4], &args.padding_idx) != TCL_OK) {
            throw std::runtime_error("Invalid padding_idx value");
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-num_embeddings") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.num_embeddings) != TCL_OK) {
                    throw std::runtime_error("Invalid num_embeddings value");
                }
            } else if (param == "-embedding_dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.embedding_dim) != TCL_OK) {
                    throw std::runtime_error("Invalid embedding_dim value");
                }
            } else if (param == "-padding_idx") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.padding_idx) != TCL_OK) {
                    throw std::runtime_error("Invalid padding_idx value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input, num_embeddings, embedding_dim");
    }
    
    return args;
}

// Sparse Embedding
int SparseEmbedding_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        SparseEmbeddingArgs args = ParseSparseEmbeddingArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        
        // Create embedding weight matrix
        torch::Tensor weight = torch::randn({args.num_embeddings, args.embedding_dim});
        
        // Zero out padding index
        if (args.padding_idx >= 0 && args.padding_idx < args.num_embeddings) {
            weight.index_put_({args.padding_idx}, torch::zeros({args.embedding_dim}));
        }

        // Use sparse=true for embedding gradients
        torch::Tensor result = torch::embedding(weight, input, args.padding_idx, false, true);
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 