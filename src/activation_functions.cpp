#include "libtorchtcl.h"

// Helper function for unary activation operations
static int ActivationUnaryOp(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[], const char* op) {
    // ------------------------------------------------------------------
    // Dual syntax parsing for unary activation operations
    // Supports both:
    //   Positional syntax:  <command> tensor
    //   Named syntax:       <command> -input tensor
    // ------------------------------------------------------------------

    std::string name;

    // Detect positional vs named based on first argument
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax should have exactly 1 argument after the command
        if (objc != 2) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor");
            return TCL_ERROR;
        }
        name = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax expects an odd number of arguments (command + pairs)
        if (objc < 3 || (objc % 2) == 0) {
            Tcl_WrongNumArgs(interp, 1, objv, "-input tensor");
            return TCL_ERROR;
        }

        for (int i = 1; i < objc; i += 2) {
            std::string key = Tcl_GetString(objv[i]);

            if (i + 1 >= objc) {
                Tcl_SetResult(interp, const_cast<char*>("Missing value for option"), TCL_VOLATILE);
                return TCL_ERROR;
            }

            std::string value = Tcl_GetString(objv[i + 1]);

            if (key == "-input" || key == "-tensor") {
                name = value;
            } else {
                std::string msg = "Unknown parameter: " + key;
                Tcl_SetResult(interp, const_cast<char*>(msg.c_str()), TCL_VOLATILE);
                return TCL_ERROR;
            }
        }

        if (name.empty()) {
            Tcl_SetResult(interp, const_cast<char*>("Required parameter -input missing"), TCL_VOLATILE);
            return TCL_ERROR;
        }
    }

    try {
        if (tensor_storage.find(name) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[name];
        torch::Tensor result;
        
        if (strcmp(op, "gelu") == 0) {
            result = torch::gelu(tensor);
        } else if (strcmp(op, "selu") == 0) {
            result = torch::selu(tensor);
        } else if (strcmp(op, "elu") == 0) {
            result = torch::elu(tensor);
        } else if (strcmp(op, "relu6") == 0) {
            result = torch::relu6(tensor);
        } else if (strcmp(op, "hardtanh") == 0) {
            result = torch::hardtanh(tensor);
        } else if (strcmp(op, "hardswish") == 0) {
            result = torch::hardswish(tensor);
        } else if (strcmp(op, "hardsigmoid") == 0) {
            result = torch::hardsigmoid(tensor);
        } else if (strcmp(op, "silu") == 0) {
            result = torch::silu(tensor);
        } else if (strcmp(op, "mish") == 0) {
            result = torch::mish(tensor);
        } else if (strcmp(op, "softplus") == 0) {
            result = torch::softplus(tensor);
        } else if (strcmp(op, "softsign") == 0) {
            result = torch::nn::functional::softsign(tensor);
        } else if (strcmp(op, "tanhshrink") == 0) {
            result = torch::nn::functional::tanhshrink(tensor);
        } else if (strcmp(op, "celu") == 0) {
            result = torch::celu(tensor);
        } else if (strcmp(op, "glu") == 0) {
            result = torch::glu(tensor);
        } else {
            Tcl_SetResult(interp, const_cast<char*>("Unknown activation function"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::gelu - GELU activation
int TensorGelu_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    return ActivationUnaryOp(cd, interp, objc, objv, "gelu");
}

// torch::selu - Scaled Exponential Linear Unit activation function
// Mathematical formula: selu(x) = scale * (max(0, x) + min(0, α * (exp(x) - 1)))
// With α ≈ 1.6733 and scale ≈ 1.0507
// Supports both positional and named parameter syntax

struct SeluArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

SeluArgs ParseSeluArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SeluArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::selu tensor
        if (objc != 2) {
            throw std::runtime_error("torch::selu: wrong # args: should be \"torch::selu tensor\"");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax: torch::selu -input tensor
        if (objc < 3 || objc % 2 == 0) {
            throw std::runtime_error("torch::selu: wrong # args: should be \"torch::selu -input tensor\"");
        }
        
        for (int i = 1; i < objc; i += 2) {
            std::string key = Tcl_GetString(objv[i]);
            if (i + 1 >= objc) {
                throw std::runtime_error("torch::selu: missing value for option " + key);
            }
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (key == "-input") {
                args.input = value;
            } else {
                throw std::runtime_error("torch::selu: unknown option " + key);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("torch::selu: required parameter -input missing");
    }
    
    return args;
}

int TensorSelu_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        SeluArgs args = ParseSeluArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::selu(tensor);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::elu - Exponential Linear Unit activation function
// Mathematical formula: elu(x) = max(0, x) + min(0, α * (exp(x) - 1))
// Default α = 1.0
// Supports both positional and named parameter syntax

struct EluArgs {
    std::string input;
    double alpha = 1.0;
    
    bool IsValid() const {
        return !input.empty() && alpha > 0.0;
    }
};

EluArgs ParseEluArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    EluArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::elu tensor ?alpha?
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("torch::elu: wrong # args: should be \"torch::elu tensor ?alpha?\"");
        }
        args.input = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.alpha) != TCL_OK) {
                throw std::runtime_error("torch::elu: invalid alpha value");
            }
        }
    } else {
        // Named parameter syntax: torch::elu -input tensor ?-alpha value?
        if (objc < 3 || objc % 2 == 0) {
            throw std::runtime_error("torch::elu: wrong # args: should be \"torch::elu -input tensor ?-alpha value?\"");
        }
        
        for (int i = 1; i < objc; i += 2) {
            std::string key = Tcl_GetString(objv[i]);
            if (i + 1 >= objc) {
                throw std::runtime_error("torch::elu: missing value for option " + key);
            }
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (key == "-input" || key == "-tensor") {
                args.input = value;
            } else if (key == "-alpha") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.alpha) != TCL_OK) {
                    throw std::runtime_error("torch::elu: invalid alpha value");
                }
            } else {
                throw std::runtime_error("torch::elu: unknown option " + key);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("torch::elu: required parameter -input missing or alpha must be > 0");
    }
    
    return args;
}

int TensorElu_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        EluArgs args = ParseEluArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::elu(tensor, args.alpha);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for leaky_relu command
struct LeakyReluArgs {
    std::string input;
    double negative_slope = 0.01;
    
    bool IsValid() const {
        return !input.empty() && negative_slope >= 0.0;
    }
};

// Parse dual syntax for leaky_relu
LeakyReluArgs ParseLeakyReluArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LeakyReluArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("Usage: torch::leaky_relu tensor ?negative_slope?");
        }
        args.input = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.negative_slope) != TCL_OK) {
                throw std::runtime_error("Invalid negative_slope");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input") {
                args.input = value;
            } else if (param == "-negativeSlope" || param == "-negative_slope" || param == "-slope") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.negative_slope) != TCL_OK) {
                    throw std::runtime_error("Invalid negative_slope");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Input tensor is required and negative_slope must be >= 0");
    }
    
    return args;
}

// torch::leaky_relu - Leaky ReLU activation  
int TensorLeakyRelu_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        LeakyReluArgs args = ParseLeakyReluArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::leaky_relu(tensor, args.negative_slope);
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameters for torch::prelu
struct PreluArgs {
    std::string input;
    std::string weight;
    
    bool IsValid() const {
        return !input.empty() && !weight.empty();
    }
};

// Parse arguments for torch::prelu (dual syntax support)
PreluArgs ParsePreluArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    PreluArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::prelu tensor weight");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.weight = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must be in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-weight") {
                args.weight = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing (input and weight tensors required)");
    }
    
    return args;
}

// torch::prelu - Parametric ReLU activation with dual syntax support
int TensorPrelu_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        auto args = ParsePreluArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.weight) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid weight tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input_tensor = tensor_storage[args.input];
        auto& weight_tensor = tensor_storage[args.weight];
        torch::Tensor result = torch::prelu(input_tensor, weight_tensor);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for relu6 command
struct Relu6Args {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for relu6
Relu6Args ParseRelu6Args(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    Relu6Args args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::relu6 tensor");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input") {
                args.input = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Input tensor is required");
    }
    
    return args;
}

// torch::relu6 - ReLU6 activation
int TensorRelu6_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        Relu6Args args = ParseRelu6Args(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::relu6(tensor);
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for hardtanh command
struct HardtanhArgs {
    std::string input;
    double min_val = -1.0;
    double max_val = 1.0;
    
    bool IsValid() const {
        return !input.empty() && min_val <= max_val;
    }
};

// Parse dual syntax for hardtanh
HardtanhArgs ParseHardtanhArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    HardtanhArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 4) {
            throw std::runtime_error("Usage: torch::hardtanh tensor ?min_val? ?max_val?");
        }
        args.input = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.min_val) != TCL_OK) {
                throw std::runtime_error("Invalid min_val");
            }
        }
        if (objc > 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.max_val) != TCL_OK) {
                throw std::runtime_error("Invalid max_val");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input") {
                args.input = value;
            } else if (param == "-min" || param == "-minVal") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.min_val) != TCL_OK) {
                    throw std::runtime_error("Invalid min_val");
                }
            } else if (param == "-max" || param == "-maxVal") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.max_val) != TCL_OK) {
                    throw std::runtime_error("Invalid max_val");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Input tensor is required and min_val must be <= max_val");
    }
    
    return args;
}

// torch::hardtanh - Hard Tanh activation
int TensorHardtanh_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        HardtanhArgs args = ParseHardtanhArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::hardtanh(tensor, args.min_val, args.max_val);
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::hardswish - Hard Swish activation
int TensorHardswish_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    return ActivationUnaryOp(cd, interp, objc, objv, "hardswish");
}

// Parameter structure for hardsigmoid command
struct HardsigmoidArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for hardsigmoid
HardsigmoidArgs ParseHardsigmoidArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    HardsigmoidArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::hardsigmoid tensor");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input") {
                args.input = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Input tensor is required");
    }
    
    return args;
}

// torch::hardsigmoid - Hard Sigmoid activation
int TensorHardsigmoid_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        HardsigmoidArgs args = ParseHardsigmoidArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::hardsigmoid(tensor);
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::silu - SiLU/Swish activation function
// Mathematical formula: silu(x) = x * sigmoid(x)
// Supports both positional and named parameter syntax

struct SiluArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

SiluArgs ParseSiluArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SiluArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::silu tensor
        if (objc != 2) {
            throw std::runtime_error("torch::silu: wrong # args: should be \"torch::silu tensor\"");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax: torch::silu -input tensor
        if (objc == 1) {
            throw std::runtime_error("torch::silu: wrong # args: should be \"torch::silu tensor\"");
        }
        
        for (int i = 1; i < objc; i += 2) {
            std::string key = Tcl_GetString(objv[i]);
            if (i + 1 >= objc) {
                throw std::runtime_error("torch::silu: missing value for option " + key);
            }
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (key == "-input") {
                args.input = value;
            } else {
                throw std::runtime_error("torch::silu: unknown option " + key);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("torch::silu: required parameter -input missing");
    }
    
    return args;
}

// torch::silu - SiLU/Swish activation
int TensorSilu_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        SiluArgs args = ParseSiluArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::silu(tensor);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::mish - Mish activation function
// Mathematical formula: mish(x) = x * tanh(softplus(x))
// Supports both positional and named parameter syntax

struct MishArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

MishArgs ParseMishArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MishArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::mish tensor
        if (objc != 2) {
            throw std::runtime_error("torch::mish: wrong # args: should be \"torch::mish tensor\"");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax: torch::mish -input tensor
        if (objc < 3 || objc % 2 == 0) {
            throw std::runtime_error("torch::mish: wrong # args: should be \"torch::mish -input tensor\"");
        }
        
        for (int i = 1; i < objc; i += 2) {
            std::string key = Tcl_GetString(objv[i]);
            if (i + 1 >= objc) {
                throw std::runtime_error("torch::mish: missing value for option " + key);
            }
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (key == "-input") {
                args.input = value;
            } else {
                throw std::runtime_error("torch::mish: unknown option " + key);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("torch::mish: required parameter -input missing");
    }
    
    return args;
}

int TensorMish_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        MishArgs args = ParseMishArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::mish(tensor);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::softplus - Softplus activation
int TensorSoftplus_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    return ActivationUnaryOp(cd, interp, objc, objv, "softplus");
}

// torch::softsign - Softsign activation
struct SoftsignArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

SoftsignArgs ParseSoftsignArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SoftsignArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::softsign tensor
        if (objc != 2) {
            throw std::runtime_error("wrong # args: should be \"torch::softsign tensor | -input tensor\"");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax: torch::softsign -input tensor
        if (objc < 3 || objc % 2 == 0) {
            throw std::runtime_error("wrong # args: should be \"torch::softsign tensor | -input tensor\"");
        }
        
        for (int i = 1; i < objc; i += 2) {
            std::string key = Tcl_GetString(objv[i]);
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (key == "-input" || key == "-tensor") {
                args.input = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + key);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter -input missing");
    }
    
    return args;
}

int TensorSoftsign_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        SoftsignArgs args = ParseSoftsignArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::nn::functional::softsign(tensor);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::tanhshrink - Tanh shrink activation function
// Mathematical formula: tanhshrink(x) = x - tanh(x)
// Supports both positional and named parameter syntax

struct TanhshrinkArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

TanhshrinkArgs ParseTanhshrinkArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TanhshrinkArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::tanhshrink tensor
        if (objc != 2) {
            throw std::runtime_error("torch::tanhshrink: wrong # args: should be \"torch::tanhshrink tensor\"");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax: torch::tanhshrink -input tensor
        if (objc < 3 || objc % 2 == 0) {
            throw std::runtime_error("torch::tanhshrink: wrong # args: should be \"torch::tanhshrink -input tensor\"");
        }
        
        for (int i = 1; i < objc; i += 2) {
            std::string key = Tcl_GetString(objv[i]);
            if (i + 1 >= objc) {
                throw std::runtime_error("torch::tanhshrink: missing value for option " + key);
            }
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (key == "-input") {
                args.input = value;
            } else {
                throw std::runtime_error("torch::tanhshrink: unknown option " + key);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("torch::tanhshrink: required parameter -input missing");
    }
    
    return args;
}

int TensorTanhshrink_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TanhshrinkArgs args = ParseTanhshrinkArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::nn::functional::tanhshrink(tensor);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for threshold command
struct ThresholdArgs {
    std::string input;
    double threshold;
    double value;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for threshold
static ThresholdArgs ParseThresholdArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ThresholdArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor threshold value
        if (objc != 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor threshold value");
            throw std::runtime_error("Usage: torch::threshold tensor threshold value");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.threshold) != TCL_OK) {
            throw std::runtime_error("Invalid threshold value");
        }
        
        if (Tcl_GetDoubleFromObj(interp, objv[3], &args.value) != TCL_OK) {
            throw std::runtime_error("Invalid value");
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
            } else if (param == "-threshold") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.threshold) != TCL_OK) {
                    throw std::runtime_error("Invalid threshold value");
                }
            } else if (param == "-value") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.value) != TCL_OK) {
                    throw std::runtime_error("Invalid value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -threshold, -value");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor required");
    }
    
    return args;
}

// torch::threshold - Threshold activation
int TensorThreshold_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        ThresholdArgs args = ParseThresholdArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::threshold(tensor, args.threshold, args.value);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for rrelu
struct RreluArgs {
    std::string input;
    double lower = 1.0 / 8.0;
    double upper = 1.0 / 3.0;
    
    bool IsValid() const {
        return !input.empty() && lower >= 0.0 && upper >= lower;
    }
};

// Dual syntax parser for rrelu
RreluArgs ParseRreluArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    RreluArgs args;
    
    // Provide immediate feedback if no additional arguments were supplied.
    // Tests expect the error message to contain the word "Usage" when the
    // command is invoked without the required tensor argument.
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::rrelu tensor ?lower? ?upper? | torch::rrelu -input tensor ?-lower value? ?-upper value?");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 4) {
            throw std::runtime_error("Usage: torch::rrelu tensor ?lower? ?upper?");
        }
        
        // Parse required parameter
        args.input = Tcl_GetString(objv[1]);
        
        // Parse optional parameters
        if (objc > 2) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.lower) != TCL_OK) {
                throw std::runtime_error("Invalid lower value");
            }
        }
        if (objc > 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.upper) != TCL_OK) {
                throw std::runtime_error("Invalid upper value");
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
            } else if (param == "-lower") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.lower) != TCL_OK) {
                    throw std::runtime_error("Invalid lower value");
                }
            } else if (param == "-upper") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.upper) != TCL_OK) {
                    throw std::runtime_error("Invalid upper value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing or invalid: tensor required, lower >= 0, upper >= lower");
    }
    
    return args;
}

// torch::rrelu - Randomized ReLU activation
int TensorRrelu_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        RreluArgs args = ParseRreluArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::rrelu(tensor, args.lower, args.upper);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for celu command
struct CeluArgs {
    std::string input;
    double alpha = 1.0;  // Default alpha value for CELU
    
    bool IsValid() const {
        return !input.empty() && alpha > 0.0;
    }
};

// Parse dual syntax for celu
CeluArgs ParseCeluArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    CeluArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("Usage: torch::celu tensor ?alpha?");
        }
        args.input = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.alpha) != TCL_OK) {
                throw std::runtime_error("Invalid alpha parameter");
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
            } else if (param == "-alpha") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.alpha) != TCL_OK) {
                    throw std::runtime_error("Invalid alpha parameter");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor required, alpha must be > 0");
    }
    
    return args;
}

// torch::celu - CELU activation
int TensorCelu_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        CeluArgs args = ParseCeluArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::celu(tensor, args.alpha);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for softmin
struct SoftminArgs {
    std::string input;
    int dim = -1;  // Default to last dimension
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Dual syntax parser for softmin
SoftminArgs ParseSoftminArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SoftminArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::softmin tensor ?dim? | torch::softmin -input tensor ?-dim dimension?");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("Usage: torch::softmin tensor ?dim?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
                throw std::runtime_error("Invalid dimension parameter");
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
            } else if (param == "-dim" || param == "-dimension") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dimension value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: tensor required");
    }
    
    return args;
}

// torch::softmin - Softmin activation
int TensorSoftmin_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        SoftminArgs args = ParseSoftminArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::nn::functional::softmin(tensor, torch::nn::functional::SoftminFuncOptions(args.dim));
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for softmax2d
struct Softmax2dArgs {
    std::string input;
    int dim = 1;  // Default to channel dimension for 2D softmax
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Dual syntax parser for softmax2d
Softmax2dArgs ParseSoftmax2dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    Softmax2dArgs args;
    
    // Provide immediate feedback if no additional arguments were supplied.
    // Tests expect the error message to contain the word "Usage" when the
    // command is invoked without the required tensor argument.
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::softmax2d tensor ?dim? | torch::softmax2d -input tensor ?-dim dimension?");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("Usage: torch::softmax2d tensor ?dim?");
        }
        
        // Parse required parameter
        args.input = Tcl_GetString(objv[1]);
        
        // Parse optional dimension parameter
        if (objc > 2) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
                throw std::runtime_error("Invalid dimension parameter");
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
            } else if (param == "-dim" || param == "-dimension") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dimension value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: tensor required");
    }
    
    return args;
}

// torch::softmax2d - 2D Softmax activation
int TensorSoftmax2d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        Softmax2dArgs args = ParseSoftmax2dArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::softmax(tensor, args.dim); // Apply along specified dimension
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct TensorLogsoftmaxArgs {
    std::string input;
    int dim = -1;  // Default to last dimension
    
    bool IsValid() const {
        return !input.empty();
    }
};

TensorLogsoftmaxArgs ParseTensorLogsoftmaxArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorLogsoftmaxArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        args.input = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            int dim;
            if (Tcl_GetIntFromObj(interp, objv[2], &dim) == TCL_OK) {
                args.dim = dim;
            } else {
                throw std::runtime_error("Invalid dimension parameter");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dim" || param == "-dimension") {
                int dim;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dimension value: " + std::string(Tcl_GetString(objv[i + 1])));
                }
                args.dim = dim;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: -input or -tensor");
    }
    
    return args;
}

// torch::logsoftmax - Log Softmax activation
int TensorLogsoftmax_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "tensor ?dim? | -input tensor ?-dim dimension?");
        return TCL_ERROR;
    }

    try {
        TensorLogsoftmaxArgs args = ParseTensorLogsoftmaxArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::log_softmax(tensor, args.dim);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::glu - Gated Linear Unit
int TensorGlu_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    return ActivationUnaryOp(cd, interp, objc, objv, "glu");
} 