#include "libtorchtcl.h"

// Parameter structure for tensor_var command
struct TensorVarArgs {
    std::string input;
    int dim = -1;  // -1 means no dimension specified
    bool has_dim = false;
    bool unbiased = true;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_var
TensorVarArgs ParseTensorVarArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorVarArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::tensor_var tensor ?dim? ?unbiased? | torch::tensor_var -input tensor ?-dim int? ?-unbiased bool?");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc > 4) {
            throw std::runtime_error("Usage: torch::tensor_var tensor ?dim? ?unbiased?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        // Parse optional dim or unbiased
        if (objc >= 3) {
            // Try to parse as dimension first (integer)
            int temp_dim;
            if (Tcl_GetIntFromObj(interp, objv[2], &temp_dim) == TCL_OK) {
                // Successfully parsed as integer - it's a dimension
                args.dim = temp_dim;
                args.has_dim = true;
                
                // Parse optional unbiased if there's a 4th argument
                if (objc >= 4) {
                    int unbiased_int;
                    if (Tcl_GetBooleanFromObj(interp, objv[3], &unbiased_int) != TCL_OK) {
                        throw std::runtime_error("Invalid unbiased parameter");
                    }
                    args.unbiased = unbiased_int != 0;
                }
            } else {
                // Failed to parse as integer - try as boolean (unbiased)
                int unbiased_int;
                if (Tcl_GetBooleanFromObj(interp, objv[2], &unbiased_int) != TCL_OK) {
                    throw std::runtime_error("Third parameter must be either dimension (integer) or unbiased (boolean)");
                }
                args.unbiased = unbiased_int != 0;
                
                // No 4th argument should exist in this case
                if (objc >= 4) {
                    throw std::runtime_error("Too many arguments when third parameter is unbiased");
                }
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
                    throw std::runtime_error("Invalid dim parameter");
                }
                args.has_dim = true;
            } else if (param == "-unbiased") {
                int unbiased_int;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &unbiased_int) != TCL_OK) {
                    throw std::runtime_error("Invalid unbiased parameter");
                }
                args.unbiased = unbiased_int != 0;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: -input");
    }
    
    return args;
}

// torch::tensor_var(tensor, dim?, unbiased?) - Variance with dual syntax support
int TensorVar_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        TensorVarArgs args = ParseTensorVarArgs(interp, objc, objv);
        
        // Check if tensor exists
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        // Compute variance
        torch::Tensor result;
        if (args.has_dim) {
            result = torch::var(tensor, args.dim, args.unbiased);
        } else {
            result = torch::var(tensor, args.unbiased);
        }
        
        // Store and return handle
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_std command
struct TensorStdArgs {
    std::string input;
    int dim = -1;  // -1 means no dimension specified
    bool has_dim = false;
    bool unbiased = true;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_std
TensorStdArgs ParseTensorStdArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorStdArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor ?dim? ?unbiased?
        if (objc < 2 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor ?dim? ?unbiased?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc >= 3) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
                throw std::runtime_error("Invalid dimension value");
            }
            args.has_dim = true;
        }
        
        if (objc >= 4) {
            int unbiased_int;
            if (Tcl_GetBooleanFromObj(interp, objv[3], &unbiased_int) != TCL_OK) {
                throw std::runtime_error("Invalid unbiased value");
            }
            args.unbiased = unbiased_int != 0;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = value;
            } else if (param == "-dim" || param == "-dimension") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dimension value");
                }
                args.has_dim = true;
            } else if (param == "-unbiased") {
                int unbiased_int;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &unbiased_int) != TCL_OK) {
                    throw std::runtime_error("Invalid unbiased value");
                }
                args.unbiased = unbiased_int != 0;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required input parameter missing");
    }
    
    return args;
}

// torch::tensor_std(tensor, dim?, unbiased?) - Standard deviation
int TensorStd_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        TensorStdArgs args = ParseTensorStdArgs(interp, objc, objv);
        
        // Check if tensor exists
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        // Calculate standard deviation
        torch::Tensor result;
        if (args.has_dim) {
            result = torch::std(tensor, args.dim, args.unbiased);
        } else {
            result = torch::std(tensor, args.unbiased);
        }
        
        // Store and return handle
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_is_cuda command
struct TensorIsCudaArgs {
    std::string tensor;
    
    bool IsValid() const {
        return !tensor.empty();
    }
};

// Parse dual syntax for tensor_is_cuda
TensorIsCudaArgs ParseTensorIsCudaArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorIsCudaArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor
        if (objc != 2) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.tensor = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-tensor" || param == "-input") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required tensor parameter missing");
    }
    
    return args;
}

// torch::tensor_is_cuda(tensor) - Check if tensor is on CUDA
int TensorIsCuda_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        TensorIsCudaArgs args = ParseTensorIsCudaArgs(interp, objc, objv);
        
        // Check if tensor exists
        if (tensor_storage.find(args.tensor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.tensor];
        bool is_cuda = tensor.is_cuda();
        
        Tcl_SetResult(interp, is_cuda ? const_cast<char*>("1") : const_cast<char*>("0"), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_is_contiguous command
struct TensorIsContiguousArgs {
    std::string tensor;
    
    bool IsValid() const {
        return !tensor.empty();
    }
};

// Parse dual syntax for tensor_is_contiguous
TensorIsContiguousArgs ParseTensorIsContiguousArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorIsContiguousArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor
        if (objc != 2) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.tensor = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-tensor" || param == "-input") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required tensor parameter missing");
    }
    
    return args;
}

// torch::tensor_is_contiguous(tensor) - Check memory layout
int TensorIsContiguous_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        TensorIsContiguousArgs args = ParseTensorIsContiguousArgs(interp, objc, objv);
        
        // Check if tensor exists
        if (tensor_storage.find(args.tensor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.tensor];
        bool is_contiguous = tensor.is_contiguous();
        
        Tcl_SetResult(interp, is_contiguous ? const_cast<char*>("1") : const_cast<char*>("0"), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_contiguous command
struct TensorContiguousArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_contiguous
TensorContiguousArgs ParseTensorContiguousArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorContiguousArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor
        if (objc != 2) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: -input");
    }
    
    return args;
}

// torch::tensor_contiguous(tensor) - Make tensor contiguous
int TensorContiguous_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        TensorContiguousArgs args = ParseTensorContiguousArgs(interp, objc, objv);
        
        // Check if tensor exists
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.contiguous();
        
        // Store and return handle
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_where command
typedef struct {
    std::string condition;
    std::string x;
    std::string y;
    bool IsValid() const {
        return !condition.empty() && !x.empty() && !y.empty();
    }
} TensorWhereArgs;

// Parse dual syntax for tensor_where
static TensorWhereArgs ParseTensorWhereArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorWhereArgs args;
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: condition x y
        if (objc != 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "condition x y");
            throw std::runtime_error("Usage: torch::tensor_where condition x y");
        }
        args.condition = Tcl_GetString(objv[1]);
        args.x = Tcl_GetString(objv[2]);
        args.y = Tcl_GetString(objv[3]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-condition") {
                args.condition = Tcl_GetString(objv[i + 1]);
            } else if (param == "-x") {
                args.x = Tcl_GetString(objv[i + 1]);
            } else if (param == "-y") {
                args.y = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: condition, x, y");
    }
    return args;
}

// torch::tensor_where(condition, x, y) - Conditional selection
int TensorWhere_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        TensorWhereArgs args = ParseTensorWhereArgs(interp, objc, objv);
        // Check if tensors exist
        if (tensor_storage.find(args.condition) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid condition tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.x) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid x tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.y) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid y tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        auto& condition = tensor_storage[args.condition];
        auto& x = tensor_storage[args.x];
        auto& y = tensor_storage[args.y];
        torch::Tensor result = torch::where(condition, x, y);
        // Store and return handle
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_expand command
struct TensorExpandArgs {
    std::string input;
    std::vector<int64_t> sizes;
    
    bool IsValid() const {
        return !input.empty() && !sizes.empty();
    }
};

// Parse dual syntax for tensor_expand
TensorExpandArgs ParseTensorExpandArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorExpandArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor sizes
        if (objc != 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor sizes");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.sizes = TclListToShape(interp, objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-sizes" || param == "-shape") {
                args.sizes = TclListToShape(interp, objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -input and -sizes");
    }
    
    return args;
}

// torch::tensor_expand(tensor, sizes) - Expand tensor (broadcasting)
int TensorExpand_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        TensorExpandArgs args = ParseTensorExpandArgs(interp, objc, objv);
        
        // Check if tensor exists
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        torch::Tensor result = tensor.expand(args.sizes);
        
        // Store and return handle
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_repeat command
struct TensorRepeatArgs {
    std::string input;
    std::vector<int64_t> repeats;
    
    bool IsValid() const {
        return !input.empty() && !repeats.empty();
    }
};

// Parse dual syntax for tensor_repeat
TensorRepeatArgs ParseTensorRepeatArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorRepeatArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor repeats");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.repeats = TclListToShape(interp, objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = value;
            } else if (param == "-repeats") {
                args.repeats = TclListToShape(interp, objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and repeats are required");
    }
    
    return args;
}

// torch::tensor_repeat(tensor, repeats) - Repeat tensor
int TensorRepeat_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        TensorRepeatArgs args = ParseTensorRepeatArgs(interp, objc, objv);
        
        // Check if tensor exists
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        torch::Tensor result = tensor.repeat(args.repeats);
        
        // Store and return handle
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct TensorIndexSelectArgs {
    std::string input;
    int dim;
    std::string indices;
    
    bool IsValid() const {
        return !input.empty() && !indices.empty();
    }
};

TensorIndexSelectArgs ParseTensorIndexSelectArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorIndexSelectArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor dim indices");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
            throw std::runtime_error("Invalid dimension value");
        }
        args.indices = Tcl_GetString(objv[3]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = value;
            } else if (param == "-dim" || param == "-dimension") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dimension value");
                }
            } else if (param == "-indices") {
                args.indices = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and indices tensor are required");
    }
    
    return args;
}

// torch::tensor_index_select(tensor, dim, indices) - Select by indices
int TensorIndexSelect_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        TensorIndexSelectArgs args = ParseTensorIndexSelectArgs(interp, objc, objv);
        
        // Check if tensors exist
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.indices) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid indices tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        auto& indices = tensor_storage[args.indices];
        
        torch::Tensor result = torch::index_select(tensor, args.dim, indices);
        
        // Store and return handle
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_median command
struct TensorMedianArgs {
    std::string input;
    int dim = -1;  // -1 means no dimension specified
    bool has_dim = false;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_median
TensorMedianArgs ParseTensorMedianArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorMedianArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor ?dim?
        if (objc < 2 || objc > 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor ?dim?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc == 3) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
                throw std::runtime_error("Invalid dimension value");
            }
            args.has_dim = true;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = value;
            } else if (param == "-dim" || param == "-dimension") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dimension value");
                }
                args.has_dim = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required input parameter missing");
    }
    
    return args;
}

// torch::tensor_median(tensor, dim?) - Median
int TensorMedian_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        TensorMedianArgs args = ParseTensorMedianArgs(interp, objc, objv);
        
        // Check if tensor exists
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        torch::Tensor result;
        if (args.has_dim) {
            // With dimension
            auto median_result = torch::median(tensor, args.dim);
            result = std::get<0>(median_result);  // values
        } else {
            // Without dimension
            result = torch::median(tensor);
        }
        
        // Store and return handle
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_quantile command
struct TensorQuantileArgs {
    std::string input;
    double q;
    int dim = -1;  // -1 means no dimension specified
    bool has_dim = false;
    
    bool IsValid() const {
        return !input.empty() && q >= 0.0 && q <= 1.0;
    }
};

// Parse dual syntax for tensor_quantile
TensorQuantileArgs ParseTensorQuantileArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorQuantileArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor q ?dim?
        if (objc < 3 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor q ?dim?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.q) != TCL_OK) {
            throw std::runtime_error("Invalid quantile value");
        }
        
        if (objc == 4) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.dim) != TCL_OK) {
                throw std::runtime_error("Invalid dimension value");
            }
            args.has_dim = true;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = value;
            } else if (param == "-q" || param == "-quantile") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.q) != TCL_OK) {
                    throw std::runtime_error("Invalid quantile value");
                }
            } else if (param == "-dim" || param == "-dimension") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dimension value");
                }
                args.has_dim = true;
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

// torch::tensor_quantile(tensor, q, dim?) - Quantiles
int TensorQuantile_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        TensorQuantileArgs args = ParseTensorQuantileArgs(interp, objc, objv);
        
        // Check if tensor exists
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        torch::Tensor result;
        if (args.has_dim) {
            // With dimension
            result = torch::quantile(tensor, args.q, args.dim);
        } else {
            // Without dimension
            result = torch::quantile(tensor, args.q);
        }
        
        // Store and return handle
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_mode command
struct TensorModeArgs {
    std::string input;
    int dim = -1;  // -1 means no dimension specified
    bool has_dim = false;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_mode
TensorModeArgs ParseTensorModeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorModeArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor ?dim?
        if (objc < 2 || objc > 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor ?dim?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc == 3) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
                throw std::runtime_error("Invalid dimension value");
            }
            args.has_dim = true;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = value;
            } else if (param == "-dim" || param == "-dimension") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dimension value");
                }
                args.has_dim = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required input parameter missing");
    }
    
    return args;
}

// torch::tensor_mode(tensor, dim?) - Mode (most frequent value)
int TensorMode_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        TensorModeArgs args = ParseTensorModeArgs(interp, objc, objv);
        
        // Check if tensor exists
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        torch::Tensor result;
        if (args.has_dim) {
            // With dimension
            auto mode_result = torch::mode(tensor, args.dim);
            result = std::get<0>(mode_result);  // values
        } else {
            // Without dimension - flatten and find mode
            auto flattened = tensor.flatten();
            auto mode_result = torch::mode(flattened, 0);
            result = std::get<0>(mode_result);
        }
        
        // Store and return handle
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 