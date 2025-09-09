#include "libtorchtcl.h"

// Concrete LSTM implementation
class ConcreteLSTM : public torch::nn::Module {
public:
    torch::nn::LSTM lstm{nullptr};
    
    ConcreteLSTM(int64_t input_size, int64_t hidden_size, int64_t num_layers = 1, 
                 bool bias = true, bool batch_first = false, double dropout = 0.0, 
                 bool bidirectional = false) {
        lstm = register_module("lstm", torch::nn::LSTM(
            torch::nn::LSTMOptions(input_size, hidden_size)
                .num_layers(num_layers)
                .bias(bias)
                .batch_first(batch_first)
                .dropout(dropout)
                .bidirectional(bidirectional)
        ));
    }
    
    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> forward(
        torch::Tensor input, 
        c10::optional<std::tuple<torch::Tensor, torch::Tensor>> hx = c10::nullopt) {
        return lstm->forward(input, hx);
    }
};

// Concrete GRU implementation
class ConcreteGRU : public torch::nn::Module {
public:
    torch::nn::GRU gru{nullptr};
    
    ConcreteGRU(int64_t input_size, int64_t hidden_size, int64_t num_layers = 1, 
                bool bias = true, bool batch_first = false, double dropout = 0.0, 
                bool bidirectional = false) {
        gru = register_module("gru", torch::nn::GRU(
            torch::nn::GRUOptions(input_size, hidden_size)
                .num_layers(num_layers)
                .bias(bias)
                .batch_first(batch_first)
                .dropout(dropout)
                .bidirectional(bidirectional)
        ));
    }
    
    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor input, 
        c10::optional<torch::Tensor> hx = c10::nullopt) {
        if (hx.has_value()) {
            return gru->forward(input, hx.value());
        } else {
            return gru->forward(input);
        }
    }
};

// Concrete RNN implementation
class ConcreteRNN : public torch::nn::Module {
public:
    torch::nn::RNN rnn{nullptr};
    
    ConcreteRNN(int64_t input_size, int64_t hidden_size, int64_t num_layers = 1, 
                const std::string& nonlinearity = "tanh", bool bias = true, 
                bool batch_first = false, double dropout = 0.0, bool bidirectional = false) {
        auto rnn_options = torch::nn::RNNOptions(input_size, hidden_size)
                .num_layers(num_layers)
                .bias(bias)
                .batch_first(batch_first)
                .dropout(dropout)
                .bidirectional(bidirectional);
        
        if (nonlinearity == "relu") {
            rnn_options = rnn_options.nonlinearity(torch::kReLU);
        } else {
            rnn_options = rnn_options.nonlinearity(torch::kTanh);
        }
        
        rnn = register_module("rnn", torch::nn::RNN(rnn_options));
    }
    
    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor input, 
        c10::optional<torch::Tensor> hx = c10::nullopt) {
        if (hx.has_value()) {
            return rnn->forward(input, hx.value());
        } else {
            return rnn->forward(input);
        }
    }
};

// Parameter structure for LSTM
struct LSTMArgs {
    int input_size = 0;      // Initialize to 0 for proper validation
    int hidden_size = 0;     // Initialize to 0 for proper validation
    int num_layers = 1;
    bool bias = true;
    bool batch_first = false;
    double dropout = 0.0;
    bool bidirectional = false;
    
    bool IsValid() const {
        return input_size > 0 && hidden_size > 0 && num_layers > 0 && dropout >= 0.0;
    }
};

// Dual syntax parser for LSTM
LSTMArgs ParseLSTMArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LSTMArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 8) {
            throw std::runtime_error("Usage: torch::lstm input_size hidden_size ?num_layers? ?bias? ?batch_first? ?dropout? ?bidirectional?");
        }
        
        // Parse required parameters
        if (Tcl_GetIntFromObj(interp, objv[1], &args.input_size) != TCL_OK) {
            throw std::runtime_error("Invalid input_size value");
        }
        if (Tcl_GetIntFromObj(interp, objv[2], &args.hidden_size) != TCL_OK) {
            throw std::runtime_error("Invalid hidden_size value");
        }
        
        // Parse optional parameters
        if (objc >= 4) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.num_layers) != TCL_OK) {
                throw std::runtime_error("Invalid num_layers value");
            }
        }
        if (objc >= 5) {
            int bias_int;
            if (Tcl_GetBooleanFromObj(interp, objv[4], &bias_int) != TCL_OK) {
                throw std::runtime_error("Invalid bias value");
            }
            args.bias = bias_int != 0;
        }
        if (objc >= 6) {
            int batch_first_int;
            if (Tcl_GetBooleanFromObj(interp, objv[5], &batch_first_int) != TCL_OK) {
                throw std::runtime_error("Invalid batch_first value");
            }
            args.batch_first = batch_first_int != 0;
        }
        if (objc >= 7) {
            if (Tcl_GetDoubleFromObj(interp, objv[6], &args.dropout) != TCL_OK) {
                throw std::runtime_error("Invalid dropout value");
            }
        }
        if (objc >= 8) {
            int bidirectional_int;
            if (Tcl_GetBooleanFromObj(interp, objv[7], &bidirectional_int) != TCL_OK) {
                throw std::runtime_error("Invalid bidirectional value");
            }
            args.bidirectional = bidirectional_int != 0;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* valueObj = objv[i + 1];
            
            if (param == "-input_size" || param == "-inputSize") {
                if (Tcl_GetIntFromObj(interp, valueObj, &args.input_size) != TCL_OK) {
                    throw std::runtime_error("Invalid input_size value");
                }
            } else if (param == "-hidden_size" || param == "-hiddenSize") {
                if (Tcl_GetIntFromObj(interp, valueObj, &args.hidden_size) != TCL_OK) {
                    throw std::runtime_error("Invalid hidden_size value");
                }
            } else if (param == "-num_layers" || param == "-numLayers") {
                if (Tcl_GetIntFromObj(interp, valueObj, &args.num_layers) != TCL_OK) {
                    throw std::runtime_error("Invalid num_layers value");
                }
            } else if (param == "-bias") {
                int bias_int;
                if (Tcl_GetBooleanFromObj(interp, valueObj, &bias_int) != TCL_OK) {
                    throw std::runtime_error("Invalid bias value");
                }
                args.bias = bias_int != 0;
            } else if (param == "-batch_first" || param == "-batchFirst") {
                int batch_first_int;
                if (Tcl_GetBooleanFromObj(interp, valueObj, &batch_first_int) != TCL_OK) {
                    throw std::runtime_error("Invalid batch_first value");
                }
                args.batch_first = batch_first_int != 0;
            } else if (param == "-dropout") {
                if (Tcl_GetDoubleFromObj(interp, valueObj, &args.dropout) != TCL_OK) {
                    throw std::runtime_error("Invalid dropout value");
                }
            } else if (param == "-bidirectional") {
                int bidirectional_int;
                if (Tcl_GetBooleanFromObj(interp, valueObj, &bidirectional_int) != TCL_OK) {
                    throw std::runtime_error("Invalid bidirectional value");
                }
                args.bidirectional = bidirectional_int != 0;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Error: input_size, hidden_size, and num_layers must be > 0, dropout must be >= 0.0");
    }
    
    return args;
}

// torch::lstm with dual syntax support
int LSTM_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        LSTMArgs args = ParseLSTMArgs(interp, objc, objv);
        
        // Create LSTM layer
        auto lstm = std::make_shared<ConcreteLSTM>(args.input_size, args.hidden_size, args.num_layers, 
                                                   args.bias, args.batch_first, args.dropout, args.bidirectional);
        
        // Store and return handle
        std::string handle = StoreModule("lstm", lstm);
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for GRU
struct GRUArgs {
    int input_size = 0;      // Initialize to 0 for proper validation
    int hidden_size = 0;     // Initialize to 0 for proper validation
    int num_layers = 1;
    bool bias = true;
    bool batch_first = false;
    double dropout = 0.0;
    bool bidirectional = false;
    
    bool IsValid() const {
        return input_size > 0 && hidden_size > 0 && num_layers > 0 && dropout >= 0.0;
    }
};

// Dual syntax parser for GRU
GRUArgs ParseGRUArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    GRUArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 8) {
            throw std::runtime_error("Usage: torch::gru input_size hidden_size ?num_layers? ?bias? ?batch_first? ?dropout? ?bidirectional?");
        }
        
        // Parse required parameters
        if (Tcl_GetIntFromObj(interp, objv[1], &args.input_size) != TCL_OK) {
            throw std::runtime_error("Invalid input_size value");
        }
        if (Tcl_GetIntFromObj(interp, objv[2], &args.hidden_size) != TCL_OK) {
            throw std::runtime_error("Invalid hidden_size value");
        }
        
        // Parse optional parameters
        if (objc >= 4) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.num_layers) != TCL_OK) {
                throw std::runtime_error("Invalid num_layers value");
            }
        }
        if (objc >= 5) {
            int bias_int;
            if (Tcl_GetBooleanFromObj(interp, objv[4], &bias_int) != TCL_OK) {
                throw std::runtime_error("Invalid bias value");
            }
            args.bias = bias_int != 0;
        }
        if (objc >= 6) {
            int batch_first_int;
            if (Tcl_GetBooleanFromObj(interp, objv[5], &batch_first_int) != TCL_OK) {
                throw std::runtime_error("Invalid batch_first value");
            }
            args.batch_first = batch_first_int != 0;
        }
        if (objc >= 7) {
            if (Tcl_GetDoubleFromObj(interp, objv[6], &args.dropout) != TCL_OK) {
                throw std::runtime_error("Invalid dropout value");
            }
        }
        if (objc >= 8) {
            int bidirectional_int;
            if (Tcl_GetBooleanFromObj(interp, objv[7], &bidirectional_int) != TCL_OK) {
                throw std::runtime_error("Invalid bidirectional value");
            }
            args.bidirectional = bidirectional_int != 0;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-inputSize" || param == "-input_size") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.input_size) != TCL_OK) {
                    throw std::runtime_error("Invalid inputSize value");
                }
            } else if (param == "-hiddenSize" || param == "-hidden_size") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.hidden_size) != TCL_OK) {
                    throw std::runtime_error("Invalid hiddenSize value");
                }
            } else if (param == "-numLayers" || param == "-num_layers") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.num_layers) != TCL_OK) {
                    throw std::runtime_error("Invalid numLayers value");
                }
            } else if (param == "-bias") {
                int bias_int;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &bias_int) != TCL_OK) {
                    throw std::runtime_error("Invalid bias value");
                }
                args.bias = bias_int != 0;
            } else if (param == "-batchFirst" || param == "-batch_first") {
                int batch_first_int;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &batch_first_int) != TCL_OK) {
                    throw std::runtime_error("Invalid batchFirst value");
                }
                args.batch_first = batch_first_int != 0;
            } else if (param == "-dropout") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.dropout) != TCL_OK) {
                    throw std::runtime_error("Invalid dropout value");
                }
            } else if (param == "-bidirectional") {
                int bidirectional_int;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &bidirectional_int) != TCL_OK) {
                    throw std::runtime_error("Invalid bidirectional value");
                }
                args.bidirectional = bidirectional_int != 0;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid");
    }
    
    return args;
}

// torch::gru with dual syntax support
int GRU_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        GRUArgs args = ParseGRUArgs(interp, objc, objv);
        
        // Create GRU layer
        auto gru = std::make_shared<ConcreteGRU>(args.input_size, args.hidden_size, args.num_layers, 
                                                 args.bias, args.batch_first, args.dropout, args.bidirectional);
        
        // Store and return handle
        std::string handle = StoreModule("gru", gru);
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for rnn_tanh
struct RNNTanhArgs {
    int input_size = 0;      // Initialize to 0 for proper validation
    int hidden_size = 0;     // Initialize to 0 for proper validation
    int num_layers = 1;
    bool bias = true;
    bool batch_first = false;
    double dropout = 0.0;
    bool bidirectional = false;
    
    bool IsValid() const {
        return input_size > 0 && hidden_size > 0 && num_layers > 0 && dropout >= 0.0;
    }
};

// Dual syntax parser for rnn_tanh
RNNTanhArgs ParseRNNTanhArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    RNNTanhArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 8) {
            throw std::runtime_error("Usage: torch::rnn_tanh input_size hidden_size ?num_layers? ?bias? ?batch_first? ?dropout? ?bidirectional?");
        }
        
        // Parse required parameters
        if (Tcl_GetIntFromObj(interp, objv[1], &args.input_size) != TCL_OK) {
            throw std::runtime_error("Invalid input_size value");
        }
        if (Tcl_GetIntFromObj(interp, objv[2], &args.hidden_size) != TCL_OK) {
            throw std::runtime_error("Invalid hidden_size value");
        }
        
        // Parse optional parameters
        if (objc >= 4) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.num_layers) != TCL_OK) {
                throw std::runtime_error("Invalid num_layers value");
            }
        }
        if (objc >= 5) {
            int bias_int;
            if (Tcl_GetBooleanFromObj(interp, objv[4], &bias_int) != TCL_OK) {
                throw std::runtime_error("Invalid bias value");
            }
            args.bias = bias_int != 0;
        }
        if (objc >= 6) {
            int batch_first_int;
            if (Tcl_GetBooleanFromObj(interp, objv[5], &batch_first_int) != TCL_OK) {
                throw std::runtime_error("Invalid batch_first value");
            }
            args.batch_first = batch_first_int != 0;
        }
        if (objc >= 7) {
            if (Tcl_GetDoubleFromObj(interp, objv[6], &args.dropout) != TCL_OK) {
                throw std::runtime_error("Invalid dropout value");
            }
        }
        if (objc >= 8) {
            int bidirectional_int;
            if (Tcl_GetBooleanFromObj(interp, objv[7], &bidirectional_int) != TCL_OK) {
                throw std::runtime_error("Invalid bidirectional value");
            }
            args.bidirectional = bidirectional_int != 0;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-inputSize") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.input_size) != TCL_OK) {
                    throw std::runtime_error("Invalid inputSize value");
                }
            } else if (param == "-hiddenSize") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.hidden_size) != TCL_OK) {
                    throw std::runtime_error("Invalid hiddenSize value");
                }
            } else if (param == "-numLayers") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.num_layers) != TCL_OK) {
                    throw std::runtime_error("Invalid numLayers value");
                }
            } else if (param == "-bias") {
                int bias_int;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &bias_int) != TCL_OK) {
                    throw std::runtime_error("Invalid bias value");
                }
                args.bias = bias_int != 0;
            } else if (param == "-batchFirst") {
                int batch_first_int;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &batch_first_int) != TCL_OK) {
                    throw std::runtime_error("Invalid batchFirst value");
                }
                args.batch_first = batch_first_int != 0;
            } else if (param == "-dropout") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.dropout) != TCL_OK) {
                    throw std::runtime_error("Invalid dropout value");
                }
            } else if (param == "-bidirectional") {
                int bidirectional_int;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &bidirectional_int) != TCL_OK) {
                    throw std::runtime_error("Invalid bidirectional value");
                }
                args.bidirectional = bidirectional_int != 0;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid");
    }
    
    return args;
}

// torch::rnn_tanh(input_size, hidden_size, num_layers?) - RNN with tanh activation
int RNNTanh_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax
        RNNTanhArgs args = ParseRNNTanhArgs(interp, objc, objv);
        
        // Create RNN layer with tanh activation
        auto rnn = std::make_shared<ConcreteRNN>(args.input_size, args.hidden_size, args.num_layers, 
                                                 "tanh", args.bias, args.batch_first, args.dropout, args.bidirectional);
        
        // Store and return handle
        std::string handle = StoreModule("rnn", rnn);
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for rnn_relu
struct RNNReluArgs {
    int input_size = 0;      // Initialize to 0 for proper validation
    int hidden_size = 0;     // Initialize to 0 for proper validation
    int num_layers = 1;
    bool bias = true;
    bool batch_first = false;
    double dropout = 0.0;
    bool bidirectional = false;
    
    bool IsValid() const {
        return input_size > 0 && hidden_size > 0 && num_layers > 0 && dropout >= 0.0;
    }
};

// Dual syntax parser for rnn_relu
RNNReluArgs ParseRNNReluArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    RNNReluArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 8) {
            throw std::runtime_error("Usage: torch::rnn_relu input_size hidden_size ?num_layers? ?bias? ?batch_first? ?dropout? ?bidirectional?");
        }
        
        // Parse required parameters
        if (Tcl_GetIntFromObj(interp, objv[1], &args.input_size) != TCL_OK) {
            throw std::runtime_error("Invalid input_size value");
        }
        if (Tcl_GetIntFromObj(interp, objv[2], &args.hidden_size) != TCL_OK) {
            throw std::runtime_error("Invalid hidden_size value");
        }
        
        // Parse optional parameters
        if (objc >= 4) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.num_layers) != TCL_OK) {
                throw std::runtime_error("Invalid num_layers value");
            }
        }
        if (objc >= 5) {
            int bias_int;
            if (Tcl_GetBooleanFromObj(interp, objv[4], &bias_int) != TCL_OK) {
                throw std::runtime_error("Invalid bias value");
            }
            args.bias = bias_int != 0;
        }
        if (objc >= 6) {
            int batch_first_int;
            if (Tcl_GetBooleanFromObj(interp, objv[5], &batch_first_int) != TCL_OK) {
                throw std::runtime_error("Invalid batch_first value");
            }
            args.batch_first = batch_first_int != 0;
        }
        if (objc >= 7) {
            if (Tcl_GetDoubleFromObj(interp, objv[6], &args.dropout) != TCL_OK) {
                throw std::runtime_error("Invalid dropout value");
            }
        }
        if (objc >= 8) {
            int bidirectional_int;
            if (Tcl_GetBooleanFromObj(interp, objv[7], &bidirectional_int) != TCL_OK) {
                throw std::runtime_error("Invalid bidirectional value");
            }
            args.bidirectional = bidirectional_int != 0;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-inputSize") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.input_size) != TCL_OK) {
                    throw std::runtime_error("Invalid inputSize value");
                }
            } else if (param == "-hiddenSize") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.hidden_size) != TCL_OK) {
                    throw std::runtime_error("Invalid hiddenSize value");
                }
            } else if (param == "-numLayers") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.num_layers) != TCL_OK) {
                    throw std::runtime_error("Invalid numLayers value");
                }
            } else if (param == "-bias") {
                int bias_int;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &bias_int) != TCL_OK) {
                    throw std::runtime_error("Invalid bias value");
                }
                args.bias = bias_int != 0;
            } else if (param == "-batchFirst") {
                int batch_first_int;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &batch_first_int) != TCL_OK) {
                    throw std::runtime_error("Invalid batchFirst value");
                }
                args.batch_first = batch_first_int != 0;
            } else if (param == "-dropout") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.dropout) != TCL_OK) {
                    throw std::runtime_error("Invalid dropout value");
                }
            } else if (param == "-bidirectional") {
                int bidirectional_int;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &bidirectional_int) != TCL_OK) {
                    throw std::runtime_error("Invalid bidirectional value");
                }
                args.bidirectional = bidirectional_int != 0;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid: inputSize, hiddenSize must be positive");
    }
    
    return args;
}

// torch::rnn_relu(input_size, hidden_size, num_layers?) - RNN with ReLU activation
int RNNRelu_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    // Determine if the call is positional or named syntax. We only apply the
    // upfront argument-count check for positional syntax; for named syntax we
    // delegate to the parser so it can emit detailed "Missing value" errors
    // that some tests expect.
    bool positionalSyntax = (objc >= 2 && Tcl_GetString(objv[1])[0] != '-');

    if (positionalSyntax && objc < 3) {
        Tcl_SetResult(interp, (char*)"Usage: torch::rnn_relu input_size hidden_size ?num_layers? ?bias? ?batch_first? ?dropout? ?bidirectional?\n"
                      "   or: torch::rnn_relu -inputSize INT -hiddenSize INT [-numLayers INT] [-bias BOOL] [-batchFirst BOOL] [-dropout DOUBLE] [-bidirectional BOOL]", TCL_STATIC);
        return TCL_ERROR;
    }
    
    try {
        RNNReluArgs args = ParseRNNReluArgs(interp, objc, objv);
        
        // Create RNN layer with ReLU activation
        auto rnn = std::make_shared<ConcreteRNN>(args.input_size, args.hidden_size, args.num_layers, 
                                                 "relu", args.bias, args.batch_first, args.dropout, args.bidirectional);
        
        // Store and return handle
        std::string handle = StoreModule("rnn", rnn);
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 