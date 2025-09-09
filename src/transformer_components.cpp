#include "libtorchtcl.h"
#include <cmath>

// Parameter structure for multihead_attention
struct MultiHeadAttentionArgs {
    torch::Tensor query;
    torch::Tensor key;
    torch::Tensor value;
    int embed_dim;
    int num_heads;
    
    bool IsValid() const {
        return query.defined() && key.defined() && value.defined() && 
               embed_dim > 0 && num_heads > 0;
    }
};

// Dual syntax parser for multihead_attention
MultiHeadAttentionArgs ParseMultiHeadAttentionArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MultiHeadAttentionArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 6) {
            throw std::runtime_error("Usage: torch::multihead_attention query key value embed_dim num_heads");
        }
        args.query = GetTensorFromObj(interp, objv[1]);
        args.key = GetTensorFromObj(interp, objv[2]);
        args.value = GetTensorFromObj(interp, objv[3]);
        args.embed_dim = GetIntFromObj(interp, objv[4]);
        args.num_heads = GetIntFromObj(interp, objv[5]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-query") {
                args.query = GetTensorFromObj(interp, objv[i + 1]);
            } else if (param == "-key") {
                args.key = GetTensorFromObj(interp, objv[i + 1]);
            } else if (param == "-value") {
                args.value = GetTensorFromObj(interp, objv[i + 1]);
            } else if (param == "-embedDim") {
                args.embed_dim = GetIntFromObj(interp, objv[i + 1]);
            } else if (param == "-numHeads") {
                args.num_heads = GetIntFromObj(interp, objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: query, key, value, embedDim, numHeads");
    }
    
    return args;
}

// Multi-Head Attention
int MultiHeadAttention_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    if (objc < 6) {
        Tcl_SetResult(interp, (char*)"Usage: torch::multihead_attention query key value embed_dim num_heads\n"
                      "   or: torch::multihead_attention -query TENSOR -key TENSOR -value TENSOR -embedDim INT -numHeads INT", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        MultiHeadAttentionArgs args = ParseMultiHeadAttentionArgs(interp, objc, objv);
        
        torch::Tensor query = args.query;
        torch::Tensor key = args.key;
        torch::Tensor value = args.value;
        int embed_dim = args.embed_dim;
        int num_heads = args.num_heads;

        int head_dim = embed_dim / num_heads;
        double scale = 1.0 / std::sqrt(head_dim);

        // Reshape for multi-head attention
        auto sizes = query.sizes();
        int seq_len = sizes[0];
        int batch_size = sizes[1];
        
        query = query.view({seq_len, batch_size, num_heads, head_dim}).transpose(1, 2);
        key = key.view({seq_len, batch_size, num_heads, head_dim}).transpose(1, 2);
        value = value.view({seq_len, batch_size, num_heads, head_dim}).transpose(1, 2);

        // Scaled dot-product attention
        torch::Tensor scores = torch::matmul(query, key.transpose(-2, -1)) * scale;
        torch::Tensor attn_weights = torch::softmax(scores, -1);
        torch::Tensor attn_output = torch::matmul(attn_weights, value);

        // Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view({seq_len, batch_size, embed_dim});
        
        return SetTensorResult(interp, attn_output);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for scaled_dot_product_attention
struct ScaledDotProductAttentionArgs {
    torch::Tensor query;
    torch::Tensor key;
    torch::Tensor value;
    
    bool IsValid() const {
        return query.defined() && key.defined() && value.defined();
    }
};

// Parse dual syntax for scaled_dot_product_attention
ScaledDotProductAttentionArgs ParseScaledDotProductAttentionArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ScaledDotProductAttentionArgs args;
    
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "query key value");
        throw std::runtime_error("");  // Empty error message since we use Tcl_WrongNumArgs
    }
    
    if (Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "query key value");
            throw std::runtime_error("");  // Empty error message since we use Tcl_WrongNumArgs
        }
        args.query = GetTensorFromObj(interp, objv[1]);
        args.key = GetTensorFromObj(interp, objv[2]);
        args.value = GetTensorFromObj(interp, objv[3]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-query") {
                args.query = GetTensorFromObj(interp, objv[i + 1]);
            } else if (param == "-key") {
                args.key = GetTensorFromObj(interp, objv[i + 1]);
            } else if (param == "-value") {
                args.value = GetTensorFromObj(interp, objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: query, key, and value");
    }
    
    return args;
}

// Scaled Dot-Product Attention
int ScaledDotProductAttention_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        ScaledDotProductAttentionArgs args = ParseScaledDotProductAttentionArgs(interp, objc, objv);
        
        double scale = 1.0 / std::sqrt(args.query.size(-1));
        
        torch::Tensor scores = torch::matmul(args.query, args.key.transpose(-2, -1)) * scale;
        torch::Tensor attn_weights = torch::softmax(scores, -1);
        torch::Tensor result = torch::matmul(attn_weights, args.value);
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        if (std::string(e.what()).empty()) {
            return TCL_ERROR;  // Error message already set by Tcl_WrongNumArgs
        }
        Tcl_SetResult(interp, const_cast<char*>(("Error in scaled_dot_product_attention: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for positional_encoding command
struct PositionalEncodingArgs {
    int seqLen;
    int dModel;
    double dropout;
    
    bool IsValid() const {
        return seqLen > 0 && dModel > 0 && dropout >= 0.0 && dropout <= 1.0;
    }
};

// Parse dual syntax for positional_encoding command
PositionalEncodingArgs ParsePositionalEncodingArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    PositionalEncodingArgs args;
    args.seqLen = 0;  // Initialize with invalid values
    args.dModel = 0;
    args.dropout = -1.0;
    
    if (objc == 1) {
        throw std::runtime_error("Usage: torch::positional_encoding seq_len d_model dropout");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 4) {
            throw std::runtime_error("Usage: torch::positional_encoding seq_len d_model dropout");
        }
        
        if (Tcl_GetIntFromObj(interp, objv[1], &args.seqLen) != TCL_OK) {
            throw std::runtime_error("Invalid seq_len value");
        }
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dModel) != TCL_OK) {
            throw std::runtime_error("Invalid d_model value");
        }
        if (Tcl_GetDoubleFromObj(interp, objv[3], &args.dropout) != TCL_OK) {
            throw std::runtime_error("Invalid dropout value");
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string flag = Tcl_GetString(objv[i]);
            if (flag == "-seqLen") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.seqLen) != TCL_OK) {
                    throw std::runtime_error("Invalid seqLen value");
                }
            } else if (flag == "-dModel") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dModel) != TCL_OK) {
                    throw std::runtime_error("Invalid dModel value");
                }
            } else if (flag == "-dropout") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.dropout) != TCL_OK) {
                    throw std::runtime_error("Invalid dropout value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + flag);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Invalid parameters: seq_len must be positive, d_model must be positive, dropout must be in range [0,1]");
    }
    
    return args;
}

// Positional Encoding
int PositionalEncoding_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        PositionalEncodingArgs args = ParsePositionalEncodingArgs(interp, objc, objv);
        
        torch::Tensor pe = torch::zeros({args.seqLen, args.dModel});
        torch::Tensor position = torch::arange(0, args.seqLen).unsqueeze(1).to(torch::kFloat);
        
        torch::Tensor div_term = torch::exp(torch::arange(0, args.dModel, 2).to(torch::kFloat) * 
                                          (-std::log(10000.0) / args.dModel));
        
        pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)}, 
                     torch::sin(position * div_term));
        pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)}, 
                     torch::cos(position * div_term));
        
        torch::Tensor result = torch::dropout(pe, args.dropout, true);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Transformer Encoder Layer
struct TransformerEncoderLayerArgs {
    torch::Tensor src;
    int dModel;
    int nhead;
    int dimFeedforward;
    double dropout;
    
    bool IsValid() const {
        return src.defined() && dModel > 0 && nhead > 0 && dimFeedforward > 0 && dropout >= 0.0 && dropout <= 1.0;
    }
};

TransformerEncoderLayerArgs ParseTransformerEncoderLayerArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TransformerEncoderLayerArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 6) {
            throw std::runtime_error("Usage: torch::transformer_encoder_layer src d_model nhead dim_feedforward dropout");
        }
        args.src = GetTensorFromObj(interp, objv[1]);
        args.dModel = GetIntFromObj(interp, objv[2]);
        args.nhead = GetIntFromObj(interp, objv[3]);
        args.dimFeedforward = GetIntFromObj(interp, objv[4]);
        args.dropout = GetDoubleFromObj(interp, objv[5]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string flag = Tcl_GetString(objv[i]);
            if (flag == "-src") {
                args.src = GetTensorFromObj(interp, objv[i + 1]);
            } else if (flag == "-dModel") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dModel) != TCL_OK) {
                    throw std::runtime_error("Invalid dModel value");
                }
            } else if (flag == "-nhead") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.nhead) != TCL_OK) {
                    throw std::runtime_error("Invalid nhead value");
                }
            } else if (flag == "-dimFeedforward") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dimFeedforward) != TCL_OK) {
                    throw std::runtime_error("Invalid dimFeedforward value");
                }
            } else if (flag == "-dropout") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.dropout) != TCL_OK) {
                    throw std::runtime_error("Invalid dropout value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + flag);
            }
        }
    }
    if (!args.IsValid()) {
        throw std::runtime_error("Invalid parameters: src tensor must be defined, dModel, nhead, dimFeedforward, and dropout must be valid");
    }
    return args;
}

int TransformerEncoderLayer_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TransformerEncoderLayerArgs args = ParseTransformerEncoderLayerArgs(interp, objc, objv);
        // Self-attention (simplified - just pass through)
        torch::Tensor attn_output = args.src;
        torch::Tensor norm1 = torch::layer_norm(args.src + attn_output, {args.dModel});
        // Feed forward (simplified - just add some transformation)
        torch::Tensor ff_output = torch::relu(norm1);
        torch::Tensor identity = torch::eye(args.dModel, torch::TensorOptions().dtype(args.src.dtype()));
        ff_output = torch::linear(ff_output, identity);
        ff_output = torch::dropout(ff_output, args.dropout, true);
        // Add & Norm
        torch::Tensor result = torch::layer_norm(norm1 + ff_output, {args.dModel});
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Transformer Decoder Layer
struct TransformerDecoderLayerArgs {
    torch::Tensor tgt;
    torch::Tensor memory;
    int dModel;
    int nhead;
    int dimFeedforward;
    double dropout;
    
    bool IsValid() const {
        return tgt.defined() && memory.defined() && dModel > 0 && 
               nhead > 0 && dimFeedforward > 0 && dropout >= 0.0 && dropout <= 1.0;
    }
};

TransformerDecoderLayerArgs ParseTransformerDecoderLayerArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TransformerDecoderLayerArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 7) {
            throw std::runtime_error("Usage: torch::transformer_decoder_layer tgt memory d_model nhead dim_feedforward dropout");
        }
        
        args.tgt = GetTensorFromObj(interp, objv[1]);
        args.memory = GetTensorFromObj(interp, objv[2]);
        args.dModel = GetIntFromObj(interp, objv[3]);
        args.nhead = GetIntFromObj(interp, objv[4]);
        args.dimFeedforward = GetIntFromObj(interp, objv[5]);
        args.dropout = GetDoubleFromObj(interp, objv[6]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string flag = Tcl_GetString(objv[i]);
            
            if (flag == "-tgt") {
                args.tgt = GetTensorFromObj(interp, objv[i + 1]);
            } else if (flag == "-memory") {
                args.memory = GetTensorFromObj(interp, objv[i + 1]);
            } else if (flag == "-dModel") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dModel) != TCL_OK) {
                    throw std::runtime_error("Invalid dModel value");
                }
            } else if (flag == "-nhead") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.nhead) != TCL_OK) {
                    throw std::runtime_error("Invalid nhead value");
                }
            } else if (flag == "-dimFeedforward") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dimFeedforward) != TCL_OK) {
                    throw std::runtime_error("Invalid dimFeedforward value");
                }
            } else if (flag == "-dropout") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.dropout) != TCL_OK) {
                    throw std::runtime_error("Invalid dropout value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + flag);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Invalid parameters: all tensors must be defined, dModel, nhead, dimFeedforward must be positive, dropout must be in range [0,1]");
    }
    
    return args;
}

int TransformerDecoderLayer_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TransformerDecoderLayerArgs args = ParseTransformerDecoderLayerArgs(interp, objc, objv);
        
        // Ensure input tensors have the correct shape for the dModel
        torch::Tensor tgt = args.tgt;
        if (tgt.size(-1) != args.dModel) {
            if (tgt.size(-1) < args.dModel) {
                torch::Tensor padding = torch::zeros({tgt.size(0), args.dModel - tgt.size(-1)});
                tgt = torch::cat({tgt, padding}, -1);
            } else {
                tgt = tgt.narrow(-1, 0, args.dModel);
            }
        }
        
        torch::Tensor memory = args.memory;
        if (memory.size(-1) != args.dModel) {
            if (memory.size(-1) < args.dModel) {
                torch::Tensor padding = torch::zeros({memory.size(0), args.dModel - memory.size(-1)});
                memory = torch::cat({memory, padding}, -1);
            } else {
                memory = memory.narrow(-1, 0, args.dModel);
            }
        }

        // Self-attention (simplified)
        torch::Tensor self_attn_output = tgt;
        torch::Tensor norm1 = torch::layer_norm(tgt + self_attn_output, {args.dModel});
        
        // Cross-attention (simplified)
        torch::Tensor cross_attn_output = memory;
        torch::Tensor norm2 = torch::layer_norm(norm1 + cross_attn_output, {args.dModel});
        
        // Feed forward (simplified)
        torch::Tensor ff_output = torch::relu(norm2);
        ff_output = torch::linear(ff_output, torch::eye(args.dModel));
        ff_output = torch::dropout(ff_output, args.dropout, true);
        
        // Add & Norm
        torch::Tensor result = torch::layer_norm(norm2 + ff_output, {args.dModel});
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Transformer Encoder
struct TransformerEncoderArgs {
    torch::Tensor src;
    int dModel;
    int nhead;
    int numLayers;
    int dimFeedforward;
    
    bool IsValid() const {
        return src.defined() && dModel > 0 && nhead > 0 && 
               numLayers > 0 && dimFeedforward > 0;
    }
};

TransformerEncoderArgs ParseTransformerEncoderArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TransformerEncoderArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 6) {
            throw std::runtime_error("Usage: torch::transformer_encoder src d_model nhead num_layers dim_feedforward");
        }
        
        args.src = GetTensorFromObj(interp, objv[1]);
        args.dModel = GetIntFromObj(interp, objv[2]);
        args.nhead = GetIntFromObj(interp, objv[3]);
        args.numLayers = GetIntFromObj(interp, objv[4]);
        args.dimFeedforward = GetIntFromObj(interp, objv[5]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string flag = Tcl_GetString(objv[i]);
            
            if (flag == "-src") {
                args.src = GetTensorFromObj(interp, objv[i + 1]);
            } else if (flag == "-dModel") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dModel) != TCL_OK) {
                    throw std::runtime_error("Invalid dModel value");
                }
            } else if (flag == "-nhead") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.nhead) != TCL_OK) {
                    throw std::runtime_error("Invalid nhead value");
                }
            } else if (flag == "-numLayers") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.numLayers) != TCL_OK) {
                    throw std::runtime_error("Invalid numLayers value");
                }
            } else if (flag == "-dimFeedforward") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dimFeedforward) != TCL_OK) {
                    throw std::runtime_error("Invalid dimFeedforward value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + flag);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Invalid parameters: src tensor must be defined, dModel, nhead, numLayers, and dimFeedforward must be positive");
    }
    
    return args;
}

int TransformerEncoder_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TransformerEncoderArgs args = ParseTransformerEncoderArgs(interp, objc, objv);

        torch::Tensor output = args.src;
        
        for (int i = 0; i < args.numLayers; i++) {
            // Self-attention (simplified - just pass through)
            torch::Tensor attn_output = output;
            torch::Tensor norm1 = torch::layer_norm(output + attn_output, {args.dModel});
            
            // Feed forward (simplified - just add some transformation)
            torch::Tensor ff_output = torch::relu(norm1);
            // Create identity matrix with same dtype as input
            torch::Tensor identity = torch::eye(args.dModel, torch::TensorOptions().dtype(output.dtype()));
            ff_output = torch::linear(ff_output, identity);
            
            // Add & Norm
            output = torch::layer_norm(norm1 + ff_output, {args.dModel});
        }
        
        return SetTensorResult(interp, output);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Transformer Decoder
struct TransformerDecoderArgs {
    torch::Tensor tgt;
    torch::Tensor memory;
    int dModel;
    int nhead;
    int numLayers;
    int dimFeedforward;
    
    bool IsValid() const {
        return tgt.defined() && memory.defined() && dModel > 0 && 
               nhead > 0 && numLayers > 0 && dimFeedforward > 0;
    }
};

TransformerDecoderArgs ParseTransformerDecoderArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TransformerDecoderArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 7) {
            throw std::runtime_error("Usage: torch::transformer_decoder tgt memory d_model nhead num_layers dim_feedforward");
        }
        
        args.tgt = GetTensorFromObj(interp, objv[1]);
        args.memory = GetTensorFromObj(interp, objv[2]);
        args.dModel = GetIntFromObj(interp, objv[3]);
        args.nhead = GetIntFromObj(interp, objv[4]);
        args.numLayers = GetIntFromObj(interp, objv[5]);
        args.dimFeedforward = GetIntFromObj(interp, objv[6]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string flag = Tcl_GetString(objv[i]);
            
            if (flag == "-tgt") {
                args.tgt = GetTensorFromObj(interp, objv[i + 1]);
            } else if (flag == "-memory") {
                args.memory = GetTensorFromObj(interp, objv[i + 1]);
            } else if (flag == "-dModel") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dModel) != TCL_OK) {
                    throw std::runtime_error("Invalid dModel value");
                }
            } else if (flag == "-nhead") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.nhead) != TCL_OK) {
                    throw std::runtime_error("Invalid nhead value");
                }
            } else if (flag == "-numLayers") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.numLayers) != TCL_OK) {
                    throw std::runtime_error("Invalid numLayers value");
                }
            } else if (flag == "-dimFeedforward") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dimFeedforward) != TCL_OK) {
                    throw std::runtime_error("Invalid dimFeedforward value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + flag);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Invalid parameters: all tensors must be defined, dModel, nhead, numLayers, and dimFeedforward must be positive");
    }
    
    return args;
}

int TransformerDecoder_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TransformerDecoderArgs args = ParseTransformerDecoderArgs(interp, objc, objv);
        
        // Ensure input tensors have the correct shape for the dModel
        torch::Tensor output = args.tgt;
        if (output.size(-1) != args.dModel) {
            // Reshape or pad to match dModel
            if (output.size(-1) < args.dModel) {
                torch::Tensor padding = torch::zeros({output.size(0), args.dModel - output.size(-1)});
                output = torch::cat({output, padding}, -1);
            } else {
                output = output.narrow(-1, 0, args.dModel);
            }
        }
        
        torch::Tensor memory = args.memory;
        if (memory.size(-1) != args.dModel) {
            if (memory.size(-1) < args.dModel) {
                torch::Tensor padding = torch::zeros({memory.size(0), args.dModel - memory.size(-1)});
                memory = torch::cat({memory, padding}, -1);
            } else {
                memory = memory.narrow(-1, 0, args.dModel);
            }
        }
        
        for (int i = 0; i < args.numLayers; i++) {
            // Self-attention (simplified - just pass through)
            torch::Tensor self_attn_output = output;
            torch::Tensor norm1 = torch::layer_norm(output + self_attn_output, {args.dModel});
            
            // Cross-attention (simplified - just pass through)
            torch::Tensor cross_attn_output = memory;
            torch::Tensor norm2 = torch::layer_norm(norm1 + cross_attn_output, {args.dModel});
            
            // Feed forward (simplified - just add some transformation)
            torch::Tensor ff_output = torch::relu(norm2);
            ff_output = torch::linear(ff_output, torch::eye(args.dModel));
            
            // Add & Norm
            output = torch::layer_norm(norm2 + ff_output, {args.dModel});
        }
        
        return SetTensorResult(interp, output);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 