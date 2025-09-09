#include "libtorchtcl.h"

// Parameter structure for sin command
struct TensorSinArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for sin
TensorSinArgs ParseTensorSinArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorSinArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::sin tensor | torch::sin -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::sin tensor");
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
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor required");
    }
    
    return args;
}

// Helper function for unary tensor operations
static int TensorUnaryOp(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[], const char* op) {
    if (objc != 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "tensor");
        return TCL_ERROR;
    }

    try {
        std::string name = Tcl_GetString(objv[1]);
        if (tensor_storage.find(name) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[name];
        torch::Tensor result;
        
        if (strcmp(op, "sin") == 0) {
            result = tensor.sin();
        } else if (strcmp(op, "cos") == 0) {
            result = tensor.cos();
        } else if (strcmp(op, "tan") == 0) {
            result = tensor.tan();
        } else if (strcmp(op, "asin") == 0) {
            result = tensor.asin();
        } else if (strcmp(op, "acos") == 0) {
            result = tensor.acos();
        } else if (strcmp(op, "atan") == 0) {
            result = tensor.atan();
        } else if (strcmp(op, "sinh") == 0) {
            result = tensor.sinh();
        } else if (strcmp(op, "cosh") == 0) {
            result = tensor.cosh();
        } else if (strcmp(op, "asinh") == 0) {
            result = tensor.asinh();
        } else if (strcmp(op, "acosh") == 0) {
            result = tensor.acosh();
        } else if (strcmp(op, "atanh") == 0) {
            result = tensor.atanh();
        } else if (strcmp(op, "deg2rad") == 0) {
            result = tensor.deg2rad();
        } else if (strcmp(op, "rad2deg") == 0) {
            result = tensor.rad2deg();
        } else if (strcmp(op, "exp2") == 0) {
            result = tensor.exp2();
        } else if (strcmp(op, "expm1") == 0) {
            result = tensor.expm1();
        } else if (strcmp(op, "log2") == 0) {
            result = tensor.log2();
        } else if (strcmp(op, "log10") == 0) {
            result = tensor.log10();
        } else if (strcmp(op, "log1p") == 0) {
            result = tensor.log1p();
        } else if (strcmp(op, "rsqrt") == 0) {
            result = tensor.rsqrt();
        } else if (strcmp(op, "square") == 0) {
            result = tensor.square();
        } else if (strcmp(op, "floor") == 0) {
            result = tensor.floor();
        } else if (strcmp(op, "ceil") == 0) {
            result = tensor.ceil();
        } else if (strcmp(op, "trunc") == 0) {
            result = tensor.trunc();
        } else if (strcmp(op, "frac") == 0) {
            result = tensor.frac();
        } else if (strcmp(op, "isnan") == 0) {
            result = tensor.isnan();
        } else if (strcmp(op, "isinf") == 0) {
            result = tensor.isinf();
        } else if (strcmp(op, "isfinite") == 0) {
            result = tensor.isfinite();
        } else if (strcmp(op, "logical_not") == 0) {
            result = tensor.logical_not();
        } else if (strcmp(op, "bitwise_not") == 0) {
            result = tensor.bitwise_not();
        } else {
            Tcl_SetResult(interp, const_cast<char*>("Unknown operation"), TCL_VOLATILE);
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

// Helper function for binary tensor operations
static int TensorBinaryOp(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[], const char* op) {
    if (objc != 3) {
        Tcl_WrongNumArgs(interp, 1, objv, "tensor1 tensor2");
        return TCL_ERROR;
    }

    try {
        std::string name1 = Tcl_GetString(objv[1]);
        std::string name2 = Tcl_GetString(objv[2]);
        
        if (tensor_storage.find(name1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid first tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(name2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid second tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor1 = tensor_storage[name1];
        auto& tensor2 = tensor_storage[name2];
        torch::Tensor result;
        
        if (strcmp(op, "atan2") == 0) {
            result = tensor1.atan2(tensor2);
        } else if (strcmp(op, "pow") == 0) {
            result = tensor1.pow(tensor2);
        } else if (strcmp(op, "eq") == 0) {
            result = tensor1.eq(tensor2);
        } else if (strcmp(op, "ne") == 0) {
            result = tensor1.ne(tensor2);
        } else if (strcmp(op, "lt") == 0) {
            result = tensor1.lt(tensor2);
        } else if (strcmp(op, "le") == 0) {
            result = tensor1.le(tensor2);
        } else if (strcmp(op, "gt") == 0) {
            result = tensor1.gt(tensor2);
        } else if (strcmp(op, "ge") == 0) {
            result = tensor1.ge(tensor2);
        } else if (strcmp(op, "logical_and") == 0) {
            result = tensor1.logical_and(tensor2);
        } else if (strcmp(op, "logical_or") == 0) {
            result = tensor1.logical_or(tensor2);
        } else if (strcmp(op, "logical_xor") == 0) {
            result = tensor1.logical_xor(tensor2);
        } else if (strcmp(op, "bitwise_and") == 0) {
            result = tensor1.bitwise_and(tensor2);
        } else if (strcmp(op, "bitwise_or") == 0) {
            result = tensor1.bitwise_or(tensor2);
        } else if (strcmp(op, "bitwise_xor") == 0) {
            result = tensor1.bitwise_xor(tensor2);
        } else if (strcmp(op, "bitwise_left_shift") == 0) {
            result = tensor1.bitwise_left_shift(tensor2);
        } else if (strcmp(op, "bitwise_right_shift") == 0) {
            result = tensor1.bitwise_right_shift(tensor2);
        } else if (strcmp(op, "isclose") == 0) {
            result = tensor1.isclose(tensor2);
        } else {
            Tcl_SetResult(interp, const_cast<char*>("Unknown operation"), TCL_VOLATILE);
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

// Trigonometric functions
int TensorSin_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorSinArgs args = ParseTensorSinArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.sin();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for cos command
struct TensorCosArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for cos
TensorCosArgs ParseTensorCosArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorCosArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::cos tensor | torch::cos -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::cos tensor");
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
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor required");
    }
    
    return args;
}

int TensorCos_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorCosArgs args = ParseTensorCosArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.cos();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}
  
// Parameter structure for tan command
struct TensorTanArgs {
    std::string input;
    bool IsValid() const { return !input.empty(); }
};

// Parse dual syntax for tan
TensorTanArgs ParseTensorTanArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorTanArgs args;
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::tan tensor | torch::tan -input tensor");
    }
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::tan tensor");
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
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor");
            }
        }
    }
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor required");
    }
    return args;
}

int TensorTan_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd;
    try {
        TensorTanArgs args = ParseTensorTanArgs(interp, objc, objv);
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.tan();
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for asin command
struct TensorAsinArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for asin
TensorAsinArgs ParseTensorAsinArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorAsinArgs args;
    
    // Provide immediate feedback if no additional arguments were supplied.
    // Tests expect the error message to contain the word "Usage" when the
    // command is invoked without the required tensor argument.
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::asin tensor | torch::asin -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::asin tensor");
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
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor required");
    }
    
    return args;
}

int TensorAsin_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorAsinArgs args = ParseTensorAsinArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.asin();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for acos command
struct TensorAcosArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for acos
TensorAcosArgs ParseTensorAcosArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorAcosArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::acos tensor");
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
        throw std::runtime_error("Required parameter missing: input");
    }
    
    return args;
}

int TensorAcos_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorAcosArgs args = ParseTensorAcosArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.acos();
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for atan command
struct TensorAtanArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for atan
TensorAtanArgs ParseTensorAtanArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorAtanArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::atan tensor");
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
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor required");
    }
    
    return args;
}

int TensorAtan_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorAtanArgs args = ParseTensorAtanArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.atan();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for atan2 command
struct TensorAtan2Args {
    std::string input1;
    std::string input2;
    
    bool IsValid() const {
        return !input1.empty() && !input2.empty();
    }
};

// Parse dual syntax for atan2
TensorAtan2Args ParseTensorAtan2Args(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorAtan2Args args;
    
    // Provide immediate feedback if insufficient arguments were supplied.
    // Tests expect the error message to contain the word "Usage" when the
    // command is invoked without the required tensor arguments.
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::atan2 y x | torch::atan2 -y y -x x");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::atan2 y x");
        }
        args.input1 = Tcl_GetString(objv[1]);
        args.input2 = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-y" || param == "-input1") {
                args.input1 = Tcl_GetString(objv[i + 1]);
            } else if (param == "-x" || param == "-input2") {
                args.input2 = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -y, -x, -input1, -input2");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: y and x tensors required");
    }
    
    return args;
}

int TensorAtan2_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorAtan2Args args = ParseTensorAtan2Args(interp, objc, objv);
        
        if (tensor_storage.find(args.input1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid y tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.input2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid x tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& y_tensor = tensor_storage[args.input1];
        auto& x_tensor = tensor_storage[args.input2];
        torch::Tensor result = torch::atan2(y_tensor, x_tensor);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for sinh command
struct TensorSinhArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for sinh
TensorSinhArgs ParseTensorSinhArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorSinhArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::sinh tensor | torch::sinh -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::sinh tensor");
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
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor required");
    }
    
    return args;
}

int TensorSinh_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorSinhArgs args = ParseTensorSinhArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.sinh();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for cosh command
struct TensorCoshArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for cosh
TensorCoshArgs ParseTensorCoshArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorCoshArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::cosh tensor | torch::cosh -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::cosh tensor");
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
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor required");
    }
    
    return args;
}

int TensorCosh_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorCoshArgs args = ParseTensorCoshArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.cosh();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for asinh command
struct TensorAsinhArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for asinh
TensorAsinhArgs ParseTensorAsinhArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorAsinhArgs args;
    
    // Provide immediate feedback if no additional arguments were supplied.
    // Tests expect the error message to contain the word "Usage" when the
    // command is invoked without the required tensor argument.
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::asinh tensor | torch::asinh -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::asinh tensor");
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
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor required");
    }
    
    return args;
}

int TensorAsinh_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorAsinhArgs args = ParseTensorAsinhArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.asinh();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for acosh command
struct TensorAcoshArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for acosh
TensorAcoshArgs ParseTensorAcoshArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorAcoshArgs args;
    
    // Provide immediate feedback if no additional arguments were supplied.
    // Tests expect the error message to contain the word "Usage" when the
    // command is invoked without the required tensor argument.
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::acosh tensor | torch::acosh -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::acosh tensor");
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
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor required");
    }
    
    return args;
}

int TensorAcosh_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorAcoshArgs args = ParseTensorAcoshArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.acosh();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_atanh command
struct TensorAtanhArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_atanh
TensorAtanhArgs ParseTensorAtanhArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorAtanhArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::atanh tensor");
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
        throw std::runtime_error("Required parameter missing: input");
    }
    
    return args;
}

int TensorAtanh_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorAtanhArgs args = ParseTensorAtanhArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.atanh();
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(result_handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for deg2rad command
struct TensorDeg2radArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for deg2rad
TensorDeg2radArgs ParseTensorDeg2radArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorDeg2radArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::deg2rad tensor | torch::deg2rad -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::deg2rad tensor");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter -input missing");
    }
    
    return args;
}

int TensorDeg2rad_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorDeg2radArgs args = ParseTensorDeg2radArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.deg2rad();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for rad2deg command
struct TensorRad2degArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for rad2deg
TensorRad2degArgs ParseTensorRad2degArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorRad2degArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::rad2deg tensor | torch::rad2deg -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::rad2deg tensor");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter -input missing");
    }
    
    return args;
}

int TensorRad2deg_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorRad2degArgs args = ParseTensorRad2degArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.rad2deg();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for exp2 command
struct TensorExp2Args {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for exp2
TensorExp2Args ParseTensorExp2Args(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorExp2Args args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::exp2 tensor");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter -input missing");
    }
    
    return args;
}

// Exponential and Logarithmic functions
int TensorExp2_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorExp2Args args = ParseTensorExp2Args(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::exp2(tensor);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for exp10 command
struct TensorExp10Args {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for exp10
TensorExp10Args ParseTensorExp10Args(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorExp10Args args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::exp10 tensor");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter -input missing");
    }
    
    return args;
}

int TensorExp10_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorExp10Args args = ParseTensorExp10Args(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::pow(10.0, tensor);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Dual syntax support for torch::expm1
struct TensorExpm1Args {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

TensorExpm1Args ParseTensorExpm1Args(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorExpm1Args args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::expm1 tensor OR torch::expm1 -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameter requires a value");
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
        throw std::runtime_error("Required parameter missing: input tensor");
    }
    
    return args;
}

int TensorExpm1_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorExpm1Args args = ParseTensorExpm1Args(interp, objc, objv);
        
        // Validate tensor exists
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Perform expm1 operation
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::expm1(tensor);
        
        // Store result
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for log2
struct TensorLog2Args {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Dual syntax parser for log2
TensorLog2Args ParseTensorLog2Args(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorLog2Args args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::log2 tensor");
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
        throw std::runtime_error("Required parameters missing: input tensor");
    }
    
    return args;
}

int TensorLog2_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    if (objc < 2) {
        Tcl_SetResult(interp, (char*)"Usage: torch::log2 tensor\n"
                      "   or: torch::log2 -input TENSOR", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        TensorLog2Args args = ParseTensorLog2Args(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.log2();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for log10
struct TensorLog10Args {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Dual syntax parser for log10
TensorLog10Args ParseTensorLog10Args(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorLog10Args args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::log10 tensor");
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
        throw std::runtime_error("Required parameters missing: input tensor");
    }
    
    return args;
}

int TensorLog10_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    if (objc < 2) {
        Tcl_SetResult(interp, (char*)"Usage: torch::log10 tensor\n"
                      "   or: torch::log10 -input TENSOR", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        TensorLog10Args args = ParseTensorLog10Args(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.log10();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Tensor log1p (log(1+x)) Args structure
struct TensorLog1pArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Dual syntax parser for log1p
TensorLog1pArgs ParseTensorLog1pArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorLog1pArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::log1p tensor");
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
        throw std::runtime_error("Required parameters missing: input tensor");
    }
    
    return args;
}

int TensorLog1p_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    if (objc < 2) {
        Tcl_SetResult(interp, (char*)"Usage: torch::log1p tensor\n"
                      "   or: torch::log1p -input TENSOR", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        TensorLog1pArgs args = ParseTensorLog1pArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.log1p();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for pow command
struct TensorPowArgs {
    std::string input1;
    std::string input2;
    
    bool IsValid() const {
        return !input1.empty() && !input2.empty();
    }
};

// Dual syntax parser for pow
TensorPowArgs ParseTensorPowArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorPowArgs args;
    
    // Provide immediate feedback if insufficient arguments were supplied.
    // Tests expect the error message to contain the word "Usage" when the
    // command is invoked without the required tensor arguments.
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::pow base exponent | torch::pow -base base -exponent exponent");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::pow base exponent");
        }
        args.input1 = Tcl_GetString(objv[1]);
        args.input2 = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-base" || param == "-input1") {
                args.input1 = Tcl_GetString(objv[i + 1]);
            } else if (param == "-exponent" || param == "-input2") {
                args.input2 = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -base, -exponent");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: base and exponent tensors required");
    }
    
    return args;
}

int TensorPow_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorPowArgs args = ParseTensorPowArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid base tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.input2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid exponent tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& base_tensor = tensor_storage[args.input1];
        auto& exponent_tensor = tensor_storage[args.input2];
        torch::Tensor result = base_tensor.pow(exponent_tensor);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for rsqrt
struct RsqrtArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Dual syntax parser for rsqrt
RsqrtArgs ParseRsqrtArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    RsqrtArgs args;
    
    // Provide immediate feedback if no additional arguments were supplied.
    // Tests expect the error message to contain the word "Usage" when the
    // command is invoked without the required tensor argument.
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::rsqrt tensor | torch::rsqrt -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::rsqrt tensor");
        }
        
        // Parse required parameter
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
        throw std::runtime_error("Required parameter missing: tensor required");
    }
    
    return args;
}

// torch::rsqrt - Reciprocal square root
int TensorRsqrt_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        RsqrtArgs args = ParseRsqrtArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.rsqrt();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for square command
typedef struct {
    std::string input;
    bool IsValid() const { return !input.empty(); }
} TensorSquareArgs;

// Parse dual syntax for square
static TensorSquareArgs ParseTensorSquareArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorSquareArgs args;
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::square tensor | torch::square -input tensor");
    }
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Wrong number of positional arguments. Expected: torch::square tensor");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameter requires a value");
            }
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            if (param == "-input" || param == "-tensor") {
                args.input = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor");
    }
    return args;
}

int TensorSquare_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd;
    try {
        TensorSquareArgs args = ParseTensorSquareArgs(interp, objc, objv);
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.square();
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Rounding and comparison functions
// Parameter structure for floor command
struct TensorFloorArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for floor
TensorFloorArgs ParseTensorFloorArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorFloorArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::floor tensor | torch::floor -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Wrong number of positional arguments. Expected: torch::floor tensor");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameter requires a value");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor");
    }
    
    return args;
}

int TensorFloor_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    
    try {
        TensorFloorArgs args = ParseTensorFloorArgs(interp, objc, objv);
        
        if (args.input.empty() || tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.floor();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for ceil command
struct TensorCeilArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for ceil
TensorCeilArgs ParseTensorCeilArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorCeilArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::ceil tensor | torch::ceil -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Wrong number of positional arguments. Expected: torch::ceil tensor");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameter requires a value");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor");
    }
    
    return args;
}

int TensorCeil_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    
    try {
        TensorCeilArgs args = ParseTensorCeilArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.ceil();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for trunc command
struct TensorTruncArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for trunc
TensorTruncArgs ParseTensorTruncArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorTruncArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::trunc tensor | torch::trunc -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::trunc tensor");
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
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor");
    }
    
    return args;
}

int TensorTrunc_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorTruncArgs args = ParseTensorTruncArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.trunc();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// TensorFrac_Cmd - Implementation moved to comprehensive dual syntax version below



// Parameter structure for ne command
struct TensorNeArgs {
    std::string input1;
    std::string input2;
    
    bool IsValid() const {
        return !input1.empty() && !input2.empty();
    }
};

// Parse dual syntax for ne
TensorNeArgs ParseTensorNeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorNeArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Error in ne: Usage: torch::ne tensor1 tensor2 | torch::ne -input1 tensor1 -input2 tensor2");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Error in ne: Usage: torch::ne tensor1 tensor2");
        }
        args.input1 = Tcl_GetString(objv[1]);
        args.input2 = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Error in ne: Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input1" || param == "-tensor1") {
                args.input1 = Tcl_GetString(objv[i + 1]);
            } else if (param == "-input2" || param == "-tensor2") {
                args.input2 = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Error in ne: Unknown parameter: " + param + ". Valid parameters are: -input1, -tensor1, -input2, -tensor2");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Error in ne: Required parameters missing: input1 and input2 tensors");
    }
    
    return args;
}

int TensorNe_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax parser
        TensorNeArgs args = ParseTensorNeArgs(interp, objc, objv);
        
        // Validate tensors exist
        if (tensor_storage.find(args.input1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Error in ne: Invalid tensor name for input1"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.input2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Error in ne: Invalid tensor name for input2"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Perform operation
        auto& tensor1 = tensor_storage[args.input1];
        auto& tensor2 = tensor_storage[args.input2];
        torch::Tensor result = tensor1.ne(tensor2);
        
        // Store result and return
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for lt command
struct TensorLtArgs {
    std::string input1;
    std::string input2;
    
    bool IsValid() const {
        return !input1.empty() && !input2.empty();
    }
};

// Parse dual syntax for lt
TensorLtArgs ParseTensorLtArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorLtArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::lt tensor1 tensor2 | torch::lt -input1 tensor1 -input2 tensor2");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::lt tensor1 tensor2");
        }
        args.input1 = Tcl_GetString(objv[1]);
        args.input2 = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input1" || param == "-tensor1") {
                args.input1 = Tcl_GetString(objv[i + 1]);
            } else if (param == "-input2" || param == "-tensor2") {
                args.input2 = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input1, -tensor1, -input2, -tensor2");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input1 and input2 tensors");
    }
    
    return args;
}

int TensorLt_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorLtArgs args = ParseTensorLtArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name for input1"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.input2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name for input2"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor1 = tensor_storage[args.input1];
        auto& tensor2 = tensor_storage[args.input2];
        torch::Tensor result = tensor1.lt(tensor2);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for le command
struct TensorLeArgs {
    std::string input1;
    std::string input2;
    
    bool IsValid() const {
        return !input1.empty() && !input2.empty();
    }
};

// Parse dual syntax for le
TensorLeArgs ParseTensorLeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorLeArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::le tensor1 tensor2 | torch::le -input1 tensor1 -input2 tensor2");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::le tensor1 tensor2");
        }
        args.input1 = Tcl_GetString(objv[1]);
        args.input2 = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input1" || param == "-tensor1") {
                args.input1 = Tcl_GetString(objv[i + 1]);
            } else if (param == "-input2" || param == "-tensor2") {
                args.input2 = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input1, -tensor1, -input2, -tensor2");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input1 and input2 tensors");
    }
    
    return args;
}

int TensorLe_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorLeArgs args = ParseTensorLeArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name for input1"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.input2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name for input2"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor1 = tensor_storage[args.input1];
        auto& tensor2 = tensor_storage[args.input2];
        torch::Tensor result = tensor1.le(tensor2);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for gt command
struct TensorGtArgs {
    std::string input1;
    std::string input2;
    
    bool IsValid() const {
        return !input1.empty() && !input2.empty();
    }
};

// Parse dual syntax for gt
TensorGtArgs ParseTensorGtArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorGtArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::gt tensor1 tensor2 | torch::gt -input1 tensor1 -input2 tensor2");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::gt tensor1 tensor2");
        }
        args.input1 = Tcl_GetString(objv[1]);
        args.input2 = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input1" || param == "-tensor1") {
                args.input1 = Tcl_GetString(objv[i + 1]);
            } else if (param == "-input2" || param == "-tensor2") {
                args.input2 = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input1, -tensor1, -input2, -tensor2");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input1 and input2 tensors");
    }
    
    return args;
}

int TensorGt_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorGtArgs args = ParseTensorGtArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name for input1"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.input2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name for input2"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor1 = tensor_storage[args.input1];
        auto& tensor2 = tensor_storage[args.input2];
        torch::Tensor result = tensor1.gt(tensor2);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

#if 0 // Disable old TensorGe_Cmd
int TensorGe_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    return TensorBinaryOp(cd, interp, objc, objv, "ge");
}
#endif

// Parameter structure for isnan command
struct TensorIsnanArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for isnan
TensorIsnanArgs ParseTensorIsnanArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorIsnanArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::isnan tensor | torch::isnan -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::isnan tensor");
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
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor");
    }
    
    return args;
}

int TensorIsnan_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    
    if (objc < 2) {
        Tcl_SetResult(interp, (char*)"Usage: torch::isnan tensor\n"
                      "   or: torch::isnan -input TENSOR", TCL_STATIC);
        return TCL_ERROR;
    }
    
    try {
        // Parse arguments using dual syntax parser
        TensorIsnanArgs args = ParseTensorIsnanArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.isnan();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for isinf command
struct TensorIsinfArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for isinf
TensorIsinfArgs ParseTensorIsinfArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorIsinfArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::isinf tensor | torch::isinf -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::isinf tensor");
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
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor");
    }
    
    return args;
}

int TensorIsinf_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    
    if (objc < 2) {
        Tcl_SetResult(interp, (char*)"Usage: torch::isinf tensor\n"
                      "   or: torch::isinf -input TENSOR", TCL_STATIC);
        return TCL_ERROR;
    }
    
    try {
        // Parse arguments using dual syntax parser
        TensorIsinfArgs args = ParseTensorIsinfArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.isinf();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for isfinite command
struct TensorIsfiniteArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for isfinite
TensorIsfiniteArgs ParseTensorIsfiniteArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorIsfiniteArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::isfinite tensor | torch::isfinite -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::isfinite tensor");
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
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor");
    }
    
    return args;
}

int TensorIsfinite_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    
    if (objc < 2) {
        Tcl_SetResult(interp, (char*)"Usage: torch::isfinite tensor\n"
                      "   or: torch::isfinite -input TENSOR", TCL_STATIC);
        return TCL_ERROR;
    }
    
    try {
        // Parse arguments using dual syntax parser
        TensorIsfiniteArgs args = ParseTensorIsfiniteArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.isfinite();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for torch::isclose
// ============================================================================
struct TensorIscloseArgs {
    std::string input;          // first tensor (required)
    std::string other;          // second tensor (required)
    double rtol = 1e-05;        // relative tolerance (default: 1e-05)
    double atol = 1e-08;        // absolute tolerance (default: 1e-08)
    bool equal_nan = false;     // whether to consider NaN values as equal (default: false)
    
    bool IsValid() const {
        return !input.empty() && !other.empty() && rtol >= 0.0 && atol >= 0.0;
    }
};

TensorIscloseArgs ParseTensorIscloseArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorIscloseArgs args;
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::isclose input other ?rtol? ?atol? ?equal_nan?
        if (objc < 3 || objc > 6) {
            throw std::runtime_error("Usage: torch::isclose input other ?rtol? ?atol? ?equal_nan?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.other = Tcl_GetString(objv[2]);
        
        if (objc >= 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.rtol) != TCL_OK) {
                throw std::runtime_error("Invalid rtol: must be positive number");
            }
            if (args.rtol < 0.0) {
                throw std::runtime_error("Invalid rtol: must be positive number");
            }
        }
        
        if (objc >= 5) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &args.atol) != TCL_OK) {
                throw std::runtime_error("Invalid atol: must be positive number");
            }
            if (args.atol < 0.0) {
                throw std::runtime_error("Invalid atol: must be positive number");
            }
        }
        
        if (objc >= 6) {
            int equal_nan_val;
            if (Tcl_GetIntFromObj(interp, objv[5], &equal_nan_val) != TCL_OK) {
                throw std::runtime_error("Invalid equal_nan: must be 0 or 1");
            }
            args.equal_nan = (equal_nan_val != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor1") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-other" || param == "-tensor2") {
                args.other = Tcl_GetString(objv[i + 1]);
            } else if (param == "-rtol" || param == "-relativeTolerance") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.rtol) != TCL_OK) {
                    throw std::runtime_error("Invalid rtol: must be positive number");
                }
                if (args.rtol < 0.0) {
                    throw std::runtime_error("Invalid rtol: must be positive number");
                }
            } else if (param == "-atol" || param == "-absoluteTolerance") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.atol) != TCL_OK) {
                    throw std::runtime_error("Invalid atol: must be positive number");
                }
                if (args.atol < 0.0) {
                    throw std::runtime_error("Invalid atol: must be positive number");
                }
            } else if (param == "-equal_nan" || param == "-equalNan") {
                int equal_nan_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &equal_nan_val) != TCL_OK) {
                    throw std::runtime_error("Invalid equal_nan: must be 0 or 1");
                }
                args.equal_nan = (equal_nan_val != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor1, -other/-tensor2, -rtol/-relativeTolerance, -atol/-absoluteTolerance, -equal_nan/-equalNan");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and other tensors required, tolerances must be non-negative");
    }
    
    return args;
}

// torch::isclose - Element-wise comparison with tolerances using dual syntax support
int TensorIsclose_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    
    if (objc < 3) {
        Tcl_SetResult(interp, (char*)"Usage: torch::isclose input other ?rtol? ?atol? ?equal_nan?\n"
                      "   or: torch::isclose -input TENSOR1 -other TENSOR2 ?-rtol DOUBLE? ?-atol DOUBLE? ?-equal_nan BOOL?", TCL_STATIC);
        return TCL_ERROR;
    }
    
    try {
        // Parse arguments using dual syntax parser
        TensorIscloseArgs args = ParseTensorIscloseArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name for input"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.other) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name for other"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input_tensor = tensor_storage[args.input];
        auto& other_tensor = tensor_storage[args.other];
        
        // Use PyTorch's isclose with tolerance parameters
        torch::Tensor result = torch::isclose(input_tensor, other_tensor, args.rtol, args.atol, args.equal_nan);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Logical operations
// Parameter structure for logical_and
struct TensorLogicalAndArgs {
    std::string input1;
    std::string input2;
    
    bool IsValid() const {
        return !input1.empty() && !input2.empty();
    }
};

// Dual syntax parser for logical_and
TensorLogicalAndArgs ParseTensorLogicalAndArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorLogicalAndArgs args;
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::logical_and tensor1 tensor2");
        }
        args.input1 = Tcl_GetString(objv[1]);
        args.input2 = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input1" || param == "-tensor1") {
                args.input1 = Tcl_GetString(objv[i + 1]);
            } else if (param == "-input2" || param == "-tensor2") {
                args.input2 = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input1 and input2 tensors");
    }
    
    return args;
}

int TensorLogicalAnd_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    if (objc < 3) {
        Tcl_SetResult(interp, (char*)"Usage: torch::logical_and tensor1 tensor2\n"
                      "   or: torch::logical_and -input1 TENSOR1 -input2 TENSOR2", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        TensorLogicalAndArgs args = ParseTensorLogicalAndArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name for input1"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.input2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name for input2"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor1 = tensor_storage[args.input1];
        auto& tensor2 = tensor_storage[args.input2];
        torch::Tensor result = tensor1.logical_and(tensor2);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct TensorLogicalOrArgs {
    std::string input1;
    std::string input2;
    
    bool IsValid() const {
        return !input1.empty() && !input2.empty();
    }
};

TensorLogicalOrArgs ParseTensorLogicalOrArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorLogicalOrArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: torch::logical_or input1 input2
        if (objc != 3) {
            throw std::runtime_error("Wrong number of positional arguments. Expected: torch::logical_or input1 input2");
        }
        args.input1 = Tcl_GetString(objv[1]);
        args.input2 = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameter requires a value");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input1" || param == "-tensor1") {
                args.input1 = value;
            } else if (param == "-input2" || param == "-tensor2") {
                args.input2 = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input1 and input2 tensors");
    }
    
    return args;
}

int TensorLogicalOr_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        if (objc < 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "input1 input2 OR -input1 tensor1 -input2 tensor2");
            return TCL_ERROR;
        }

        TensorLogicalOrArgs args = ParseTensorLogicalOrArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor handle for input1"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.input2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor handle for input2"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto& input1_tensor = tensor_storage[args.input1];
        auto& input2_tensor = tensor_storage[args.input2];
        torch::Tensor result = torch::logical_or(input1_tensor, input2_tensor);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in logical_or: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct TensorLogicalNotArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

TensorLogicalNotArgs ParseTensorLogicalNotArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorLogicalNotArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: torch::logical_not input
        if (objc != 2) {
            throw std::runtime_error("Wrong number of positional arguments. Expected: torch::logical_not input");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameter requires a value");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor");
    }
    
    return args;
}

int TensorLogicalNot_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        if (objc < 2) {
            Tcl_WrongNumArgs(interp, 1, objv, "input OR -input tensor");
            return TCL_ERROR;
        }

        TensorLogicalNotArgs args = ParseTensorLogicalNotArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor handle"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto& input_tensor = tensor_storage[args.input];
        torch::Tensor result = torch::logical_not(input_tensor);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in logical_not: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct TensorLogicalXorArgs {
    std::string input1;
    std::string input2;
    
    bool IsValid() const {
        return !input1.empty() && !input2.empty();
    }
};

TensorLogicalXorArgs ParseTensorLogicalXorArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorLogicalXorArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: torch::logical_xor input1 input2
        if (objc != 3) {
            throw std::runtime_error("Wrong number of positional arguments. Expected: torch::logical_xor input1 input2");
        }
        args.input1 = Tcl_GetString(objv[1]);
        args.input2 = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameter requires a value");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input1" || param == "-tensor1") {
                args.input1 = value;
            } else if (param == "-input2" || param == "-tensor2") {
                args.input2 = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input1 and input2 tensors");
    }
    
    return args;
}

int TensorLogicalXor_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        if (objc < 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "input1 input2 OR -input1 tensor1 -input2 tensor2");
            return TCL_ERROR;
        }

        TensorLogicalXorArgs args = ParseTensorLogicalXorArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor handle for input1"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.input2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor handle for input2"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto& input1_tensor = tensor_storage[args.input1];
        auto& input2_tensor = tensor_storage[args.input2];
        torch::Tensor result = torch::logical_xor(input1_tensor, input2_tensor);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in logical_xor: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for bitwise_and command
struct TensorBitwiseAndArgs {
    std::string input1;
    std::string input2;
    
    bool IsValid() const {
        return !input1.empty() && !input2.empty();
    }
};

// Parse dual syntax for bitwise_and
TensorBitwiseAndArgs ParseTensorBitwiseAndArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorBitwiseAndArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::bitwise_and tensor1 tensor2 | torch::bitwise_and -input tensor1 -other tensor2");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::bitwise_and tensor1 tensor2");
        }
        args.input1 = Tcl_GetString(objv[1]);
        args.input2 = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input" || param == "-tensor1") {
                args.input1 = value;
            } else if (param == "-other" || param == "-tensor2") {
                args.input2 = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -other, -tensor1, -tensor2");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and other tensors required");
    }
    
    return args;
}

int TensorBitwiseAnd_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorBitwiseAndArgs args = ParseTensorBitwiseAndArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid first tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.input2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid second tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor1 = tensor_storage[args.input1];
        auto& tensor2 = tensor_storage[args.input2];
        torch::Tensor result = tensor1.bitwise_and(tensor2);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for bitwise_or command
struct TensorBitwiseOrArgs {
    std::string input1;
    std::string input2;
    
    bool IsValid() const {
        return !input1.empty() && !input2.empty();
    }
};

// Parse dual syntax for bitwise_or
TensorBitwiseOrArgs ParseTensorBitwiseOrArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorBitwiseOrArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::bitwise_or tensor1 tensor2 | torch::bitwise_or -input tensor1 -other tensor2");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::bitwise_or tensor1 tensor2");
        }
        args.input1 = Tcl_GetString(objv[1]);
        args.input2 = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input" || param == "-tensor1") {
                args.input1 = value;
            } else if (param == "-other" || param == "-tensor2") {
                args.input2 = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -other, -tensor1, -tensor2");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and other tensors required");
    }
    
    return args;
}

int TensorBitwiseOr_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorBitwiseOrArgs args = ParseTensorBitwiseOrArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid first tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.input2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid second tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor1 = tensor_storage[args.input1];
        auto& tensor2 = tensor_storage[args.input2];
        torch::Tensor result = tensor1.bitwise_or(tensor2);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for bitwise_not command
struct TensorBitwiseNotArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for bitwise_not
TensorBitwiseNotArgs ParseTensorBitwiseNotArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorBitwiseNotArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::bitwise_not tensor | torch::bitwise_not -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::bitwise_not tensor");
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
            
            if (param == "-input" || param == "-tensor") {
                args.input = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor required");
    }
    
    return args;
}

int TensorBitwiseNot_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorBitwiseNotArgs args = ParseTensorBitwiseNotArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.bitwise_not();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for bitwise_xor command
struct TensorBitwiseXorArgs {
    std::string input1;
    std::string input2;
    
    bool IsValid() const {
        return !input1.empty() && !input2.empty();
    }
};

// Parse dual syntax for bitwise_xor
TensorBitwiseXorArgs ParseTensorBitwiseXorArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorBitwiseXorArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::bitwise_xor tensor1 tensor2 | torch::bitwise_xor -input tensor1 -other tensor2");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::bitwise_xor tensor1 tensor2");
        }
        args.input1 = Tcl_GetString(objv[1]);
        args.input2 = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input" || param == "-tensor1") {
                args.input1 = value;
            } else if (param == "-other" || param == "-tensor2") {
                args.input2 = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -other, -tensor1, -tensor2");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and other tensors required");
    }
    
    return args;
}

int TensorBitwiseXor_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorBitwiseXorArgs args = ParseTensorBitwiseXorArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid first tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.input2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid second tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor1 = tensor_storage[args.input1];
        auto& tensor2 = tensor_storage[args.input2];
        torch::Tensor result = tensor1.bitwise_xor(tensor2);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for bitwise_left_shift command
struct TensorBitwiseLeftShiftArgs {
    std::string input1;
    std::string input2;
    
    bool IsValid() const {
        return !input1.empty() && !input2.empty();
    }
};

// Parse dual syntax for bitwise_left_shift
TensorBitwiseLeftShiftArgs ParseTensorBitwiseLeftShiftArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorBitwiseLeftShiftArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::bitwise_left_shift tensor1 tensor2 | torch::bitwise_left_shift -input tensor1 -other tensor2");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::bitwise_left_shift tensor1 tensor2");
        }
        args.input1 = Tcl_GetString(objv[1]);
        args.input2 = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input" || param == "-tensor1") {
                args.input1 = value;
            } else if (param == "-other" || param == "-tensor2") {
                args.input2 = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -other, -tensor1, -tensor2");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and other tensors required");
    }
    
    return args;
}

int TensorBitwiseLeftShift_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorBitwiseLeftShiftArgs args = ParseTensorBitwiseLeftShiftArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid first tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.input2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid second tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor1 = tensor_storage[args.input1];
        auto& tensor2 = tensor_storage[args.input2];
        torch::Tensor result = tensor1.bitwise_left_shift(tensor2);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for bitwise_right_shift command
struct TensorBitwiseRightShiftArgs {
    std::string input1;
    std::string input2;
    
    bool IsValid() const {
        return !input1.empty() && !input2.empty();
    }
};

// Parse dual syntax for bitwise_right_shift
TensorBitwiseRightShiftArgs ParseTensorBitwiseRightShiftArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorBitwiseRightShiftArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::bitwise_right_shift tensor1 tensor2 | torch::bitwise_right_shift -input tensor1 -other tensor2");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::bitwise_right_shift tensor1 tensor2");
        }
        args.input1 = Tcl_GetString(objv[1]);
        args.input2 = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input" || param == "-tensor1") {
                args.input1 = value;
            } else if (param == "-other" || param == "-tensor2") {
                args.input2 = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -other, -tensor1, -tensor2");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and other tensors required");
    }
    
    return args;
}

int TensorBitwiseRightShift_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        TensorBitwiseRightShiftArgs args = ParseTensorBitwiseRightShiftArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid first tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.input2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid second tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor1 = tensor_storage[args.input1];
        auto& tensor2 = tensor_storage[args.input2];
        torch::Tensor result = tensor1.bitwise_right_shift(tensor2);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Reduction operations
// =====================
// Dual Syntax Support for torch::mean_dim / torch::meanDim
// =====================
struct TensorMeanDimArgs {
    std::string input;
    int dim = 0;
    bool keepdim = false;
    
    bool IsValid() const {
        return !input.empty();
    }
};

TensorMeanDimArgs ParseTensorMeanDimArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorMeanDimArgs args;
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "tensor dim ?keepdim?");
        throw std::runtime_error("Wrong number of arguments: tensor dim ?keepdim? required");
    }
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor dim ?keepdim?");
            throw std::runtime_error("Invalid number of arguments");
        }
        args.input = Tcl_GetString(objv[1]);
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
            throw std::runtime_error("Invalid dim value");
        }
        if (objc > 3) {
            int keep;
            if (Tcl_GetBooleanFromObj(interp, objv[3], &keep) != TCL_OK) {
                throw std::runtime_error("Invalid keepdim value");
            }
            args.keepdim = keep != 0;
        }
    } else {
        // Named parameter syntax
        if (objc < 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "-input tensor -dim dim ?-keepdim bool?");
            throw std::runtime_error("Missing required named parameters: -input and -dim");
        }
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dim value");
                }
            } else if (param == "-keepdim") {
                int keep;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &keep) != TCL_OK) {
                    throw std::runtime_error("Invalid keepdim value");
                }
                args.keepdim = keep != 0;
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

int TensorMeanDim_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorMeanDimArgs args = ParseTensorMeanDimArgs(interp, objc, objv);
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.mean(args.dim, args.keepdim);
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    } catch (...) {
        Tcl_SetResult(interp, (char*)"Unknown error in mean_dim", TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for std_dim command
struct TensorStdDimArgs {
    std::string input;
    int dim = 0;
    bool unbiased = true;
    bool keepdim = false;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for std_dim
TensorStdDimArgs ParseTensorStdDimArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorStdDimArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::std_dim tensor dim ?unbiased? ?keepdim? | torch::std_dim -input tensor -dim dim ?-unbiased bool? ?-keepdim bool?");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 5) {
            throw std::runtime_error("Usage: torch::std_dim tensor dim ?unbiased? ?keepdim?");
        }
        args.input = Tcl_GetString(objv[1]);
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
            throw std::runtime_error("Invalid dim value");
        }
        if (objc > 3) {
            int unbias;
            if (Tcl_GetBooleanFromObj(interp, objv[3], &unbias) != TCL_OK) {
                throw std::runtime_error("Invalid unbiased value");
            }
            args.unbiased = unbias != 0;
        }
        if (objc > 4) {
            int keep;
            if (Tcl_GetBooleanFromObj(interp, objv[4], &keep) != TCL_OK) {
                throw std::runtime_error("Invalid keepdim value");
            }
            args.keepdim = keep != 0;
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
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dim value");
                }
            } else if (param == "-unbiased") {
                int unbias;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &unbias) != TCL_OK) {
                    throw std::runtime_error("Invalid unbiased value");
                }
                args.unbiased = unbias != 0;
            } else if (param == "-keepdim") {
                int keep;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &keep) != TCL_OK) {
                    throw std::runtime_error("Invalid keepdim value");
                }
                args.keepdim = keep != 0;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor");
    }
    
    return args;
}

int TensorStdDim_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorStdDimArgs args = ParseTensorStdDimArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.std(args.dim, args.unbiased, args.keepdim);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}



// =====================
// Dual Syntax Support for torch::median_dim / torch::medianDim
// =====================
struct TensorMedianDimArgs {
    std::string input;
    int dim = 0;
    bool keepdim = false;
    
    bool IsValid() const {
        return !input.empty();
    }
};

TensorMedianDimArgs ParseTensorMedianDimArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorMedianDimArgs args;
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "tensor dim ?keepdim?");
        throw std::runtime_error("Wrong number of arguments: tensor dim ?keepdim? required");
    }
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor dim ?keepdim?");
            throw std::runtime_error("Invalid number of arguments");
        }
        args.input = Tcl_GetString(objv[1]);
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
            throw std::runtime_error("Invalid dim value");
        }
        if (objc > 3) {
            int keep;
            if (Tcl_GetBooleanFromObj(interp, objv[3], &keep) != TCL_OK) {
                throw std::runtime_error("Invalid keepdim value");
            }
            args.keepdim = keep != 0;
        }
    } else {
        // Named parameter syntax
        if (objc < 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "-input tensor -dim dim ?-keepdim bool?");
            throw std::runtime_error("Missing required named parameters: -input and -dim");
        }
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dim value");
                }
            } else if (param == "-keepdim") {
                int keep;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &keep) != TCL_OK) {
                    throw std::runtime_error("Invalid keepdim value");
                }
                args.keepdim = keep != 0;
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

int TensorMedianDim_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorMedianDimArgs args = ParseTensorMedianDimArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        auto result_tuple = tensor.median(args.dim, args.keepdim);
        torch::Tensor result = std::get<0>(result_tuple);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// KTHVALUE Command - Dual Syntax Support
// ============================================================================

struct TensorKthvalueArgs {
    std::string input;
    int k;
    int dim;
    bool keepdim = false;
    
    bool IsValid() const {
        return !input.empty();
    }
};

TensorKthvalueArgs ParseTensorKthvalueArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorKthvalueArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::kthvalue tensor k dim ?keepdim?
        if (objc < 4 || objc > 5) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::kthvalue tensor k dim ?keepdim?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.k) != TCL_OK) {
            throw std::runtime_error("Invalid k value. Expected integer.");
        }
        
        if (Tcl_GetIntFromObj(interp, objv[3], &args.dim) != TCL_OK) {
            throw std::runtime_error("Invalid dim value. Expected integer.");
        }
        
        if (objc > 4) {
            int keep;
            if (Tcl_GetBooleanFromObj(interp, objv[4], &keep) != TCL_OK) {
                throw std::runtime_error("Invalid keepdim value. Expected boolean.");
            }
            args.keepdim = keep != 0;
        }
    } else {
        // Named parameter syntax: torch::kthvalue -input tensor -k k -dim dim -keepdim keepdim
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-k") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.k) != TCL_OK) {
                    throw std::runtime_error("Invalid k value. Expected integer.");
                }
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dim value. Expected integer.");
                }
            } else if (param == "-keepdim") {
                int keep;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &keep) != TCL_OK) {
                    throw std::runtime_error("Invalid keepdim value. Expected boolean.");
                }
                args.keepdim = keep != 0;
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

int TensorKthvalue_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorKthvalueArgs args = ParseTensorKthvalueArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        auto result_tuple = tensor.kthvalue(args.k, args.dim, args.keepdim);
        torch::Tensor result = std::get<0>(result_tuple);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// CUMSUM Command - Dual Syntax Support
// ============================================================================

struct TensorCumsumArgs {
    std::string input;
    int dim = 0;
    
    bool IsValid() const {
        return !input.empty();
    }
};

TensorCumsumArgs ParseTensorCumsumArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorCumsumArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::cumsum tensor dim
        if (objc != 3) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::cumsum tensor dim");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
            throw std::runtime_error("Invalid dim value. Expected integer.");
        }
    } else {
        // Named parameter syntax: torch::cumsum -input tensor -dim dim
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dim value. Expected integer.");
                }
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

int TensorCumsum_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorCumsumArgs args = ParseTensorCumsumArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.cumsum(args.dim);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// CUMPROD Command - Dual Syntax Support
// ============================================================================

struct TensorCumprodArgs {
    std::string input;
    int dim = 0;
    
    bool IsValid() const {
        return !input.empty();
    }
};

TensorCumprodArgs ParseTensorCumprodArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorCumprodArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::cumprod tensor dim
        if (objc != 3) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::cumprod tensor dim");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
            throw std::runtime_error("Invalid dim value. Expected integer.");
        }
    } else {
        // Named parameter syntax: torch::cumprod -input tensor -dim dim
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dim value. Expected integer.");
                }
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

int TensorCumprod_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorCumprodArgs args = ParseTensorCumprodArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.cumprod(args.dim);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// CUMMAX Command - Dual Syntax Support
// ============================================================================

struct TensorCummaxArgs {
    std::string input;
    int dim = 0;
    
    bool IsValid() const {
        return !input.empty();
    }
};

TensorCummaxArgs ParseTensorCummaxArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorCummaxArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::cummax tensor dim
        if (objc != 3) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::cummax tensor dim");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
            throw std::runtime_error("Invalid dim value. Expected integer.");
        }
    } else {
        // Named parameter syntax: torch::cummax -input tensor -dim dim
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dim value. Expected integer.");
                }
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

int TensorCummax_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorCummaxArgs args = ParseTensorCummaxArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        auto result_tuple = tensor.cummax(args.dim);
        torch::Tensor result = std::get<0>(result_tuple);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// CUMMIN Command - Dual Syntax Support
// ============================================================================

struct TensorCumminArgs {
    std::string input;
    int dim = 0;
    
    bool IsValid() const {
        return !input.empty();
    }
};

TensorCumminArgs ParseTensorCumminArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorCumminArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::cummin tensor dim
        if (objc != 3) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::cummin tensor dim");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
            throw std::runtime_error("Invalid dim value. Expected integer.");
        }
    } else {
        // Named parameter syntax: torch::cummin -input tensor -dim dim
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dim value. Expected integer.");
                }
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

int TensorCummin_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorCumminArgs args = ParseTensorCumminArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        auto result_tuple = tensor.cummin(args.dim);
        torch::Tensor result = std::get<0>(result_tuple);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// DIFF Command - Dual Syntax Support
// ============================================================================

struct TensorDiffArgs {
    std::string input;
    int n = 1;      // number of times to apply diff
    int dim = -1;   // dimension along which to compute diff
    
    bool IsValid() const {
        return !input.empty();
    }
};

TensorDiffArgs ParseTensorDiffArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorDiffArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::diff tensor ?n? ?dim?
        if (objc < 2 || objc > 4) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::diff tensor ?n? ?dim?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.n) != TCL_OK) {
                throw std::runtime_error("Invalid n value. Expected integer.");
            }
        }
        
        if (objc > 3) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.dim) != TCL_OK) {
                throw std::runtime_error("Invalid dim value. Expected integer.");
            }
        }
    } else {
        // Named parameter syntax: torch::diff -input tensor ?-n n? ?-dim dim?
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-n") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.n) != TCL_OK) {
                    throw std::runtime_error("Invalid n value. Expected integer.");
                }
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dim value. Expected integer.");
                }
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

int TensorDiff_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorDiffArgs args = ParseTensorDiffArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::diff(tensor, args.n, args.dim);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// Parameter structure for gradient command
struct TensorGradientArgs {
    std::string input;
    std::vector<double> spacing;  // Optional spacing parameter
    int dim = -1;                 // Dimension along which to compute gradient (-1 means infer)
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for gradient command
TensorGradientArgs ParseTensorGradientArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorGradientArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor ?spacing? ?dim?
        if (objc < 2 || objc > 4) {
            throw std::runtime_error("Usage: torch::gradient tensor ?spacing? ?dim?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        // Parse optional spacing (currently not fully implemented in LibTorch, skip for now)
        if (objc > 2) {
            // For backward compatibility, accept but ignore spacing parameter
        }
        
        // Parse optional dim
        if (objc > 3) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.dim) != TCL_OK) {
                throw std::runtime_error("Invalid dim parameter: must be integer");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dim" || param == "-dimension") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dim parameter: must be integer");
                }
            } else if (param == "-spacing") {
                // For now, accept but don't process spacing parameter
                // Could be implemented later with proper LibTorch gradient support
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor, -dim/-dimension, -spacing");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor");
    }
    
    return args;
}

int TensorGradient_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorGradientArgs args = ParseTensorGradientArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        // Implement basic gradient using diff (approximation)
        // For true gradient computation, would need proper LibTorch gradient support
        torch::Tensor result = torch::diff(tensor, 1, args.dim);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for eq command
struct TensorEqArgs {
    std::string input1;
    std::string input2;
    
    bool IsValid() const {
        return !input1.empty() && !input2.empty();
    }
};

// Parse dual syntax for eq
TensorEqArgs ParseTensorEqArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorEqArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::eq tensor1 tensor2 | torch::eq -input1 tensor1 -input2 tensor2");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::eq tensor1 tensor2");
        }
        args.input1 = Tcl_GetString(objv[1]);
        args.input2 = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input1" || param == "-tensor1") {
                args.input1 = Tcl_GetString(objv[i + 1]);
            } else if (param == "-input2" || param == "-tensor2") {
                args.input2 = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input1/-tensor1, -input2/-tensor2");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input1 and input2 tensors required");
    }
    
    return args;
}

int TensorEq_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        TensorEqArgs args = ParseTensorEqArgs(interp, objc, objv);
        
        // Look up tensors from storage
        if (tensor_storage.find(args.input1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name for input1"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.input2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name for input2"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor1 = tensor_storage[args.input1];
        auto& tensor2 = tensor_storage[args.input2];
        torch::Tensor result = tensor1.eq(tensor2);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for frac
struct TensorFracArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Dual syntax parser for frac
TensorFracArgs ParseTensorFracArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorFracArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::frac input_tensor");
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
            
            if (param == "-input" || param == "-tensor") {
                args.input = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor");
    }
    
    return args;
}

int TensorFrac_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    
    if (objc < 2) {
        Tcl_SetResult(interp, (char*)"Usage: torch::frac input_tensor\n"
                      "   or: torch::frac -input TENSOR", TCL_STATIC);
        return TCL_ERROR;
    }
    
    try {
        TensorFracArgs args = ParseTensorFracArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.frac();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ================== Dual syntax support for ge (greater_equal) ===============
struct TensorGeArgs {
    std::string input1;
    std::string input2;

    bool IsValid() const {
        return !input1.empty() && !input2.empty();
    }
};

TensorGeArgs ParseTensorGeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorGeArgs args;

    if (objc < 3) {
        throw std::runtime_error("Usage: torch::ge tensor1 tensor2 | torch::ge -input1 tensor1 -input2 tensor2");
    }

    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::ge tensor1 tensor2");
        }
        args.input1 = Tcl_GetString(objv[1]);
        args.input2 = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);

            if (param == "-input1" || param == "-tensor1") {
                args.input1 = Tcl_GetString(objv[i + 1]);
            } else if (param == "-input2" || param == "-tensor2") {
                args.input2 = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input1/-tensor1, -input2/-tensor2");
            }
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input1 and input2 tensors required");
    }

    return args;
}

int TensorGe_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused param
    try {
        TensorGeArgs args = ParseTensorGeArgs(interp, objc, objv);

        if (tensor_storage.find(args.input1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name for input1"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.input2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name for input2"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto& tensor1 = tensor_storage[args.input1];
        auto& tensor2 = tensor_storage[args.input2];
        torch::Tensor result = tensor1.ge(tensor2);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for round command
struct TensorRoundArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for round
TensorRoundArgs ParseTensorRoundArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorRoundArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::round tensor | torch::round -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::round tensor");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        bool has_input = false;
        
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
                has_input = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
        
        if (!has_input) {
            throw std::runtime_error("Input tensor is required");
        }
    }
    
    return args;
}

// torch::round - Round tensor values to nearest integer
int TensorRound_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorRoundArgs args = ParseTensorRoundArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto output = torch::round(input);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for torch::allclose
// ============================================================================
struct TensorAllcloseArgs {
    std::string input;          // first tensor (required)
    std::string other;          // second tensor (required)
    double rtol = 1e-05;        // relative tolerance (default: 1e-05)
    double atol = 1e-08;        // absolute tolerance (default: 1e-08)
    bool equal_nan = false;     // whether to consider NaN values as equal (default: false)
    
    bool IsValid() const {
        return !input.empty() && !other.empty() && rtol >= 0.0 && atol >= 0.0;
    }
};

TensorAllcloseArgs ParseTensorAllcloseArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorAllcloseArgs args;
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::allclose input other ?rtol? ?atol? ?equal_nan?
        if (objc < 3 || objc > 6) {
            throw std::runtime_error("Usage: torch::allclose input other ?rtol? ?atol? ?equal_nan?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.other = Tcl_GetString(objv[2]);
        
        if (objc >= 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.rtol) != TCL_OK) {
                throw std::runtime_error("Invalid rtol: must be positive number");
            }
            if (args.rtol < 0.0) {
                throw std::runtime_error("Invalid rtol: must be positive number");
            }
        }
        
        if (objc >= 5) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &args.atol) != TCL_OK) {
                throw std::runtime_error("Invalid atol: must be positive number");
            }
            if (args.atol < 0.0) {
                throw std::runtime_error("Invalid atol: must be positive number");
            }
        }
        
        if (objc >= 6) {
            int equal_nan_val;
            if (Tcl_GetIntFromObj(interp, objv[5], &equal_nan_val) != TCL_OK) {
                throw std::runtime_error("Invalid equal_nan: must be 0 or 1");
            }
            args.equal_nan = (equal_nan_val != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            std::string param = Tcl_GetString(objv[i]);
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            if (param == "-input" || param == "-tensor1") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-other" || param == "-tensor2") {
                args.other = Tcl_GetString(objv[i + 1]);
            } else if (param == "-rtol" || param == "-relativeTolerance") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.rtol) != TCL_OK) {
                    throw std::runtime_error("Invalid rtol: must be positive number");
                }
                if (args.rtol < 0.0) {
                    throw std::runtime_error("Invalid rtol: must be positive number");
                }
            } else if (param == "-atol" || param == "-absoluteTolerance") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.atol) != TCL_OK) {
                    throw std::runtime_error("Invalid atol: must be positive number");
                }
                if (args.atol < 0.0) {
                    throw std::runtime_error("Invalid atol: must be positive number");
                }
            } else if (param == "-equal_nan" || param == "-equalNan") {
                int equal_nan_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &equal_nan_val) != TCL_OK) {
                    throw std::runtime_error("Invalid equal_nan: must be 0 or 1");
                }
                args.equal_nan = (equal_nan_val != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and other tensors required, tolerances must be non-negative");
    }
    
    return args;
}

// torch::allclose - Check if all elements in two tensors are close within tolerances
int TensorAllclose_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    
    if (objc < 3) {
        Tcl_SetResult(interp, (char*)"Usage: torch::allclose input other ?rtol? ?atol? ?equal_nan?\n"
                      "   or: torch::allclose -input TENSOR1 -other TENSOR2 ?-rtol DOUBLE? ?-atol DOUBLE? ?-equal_nan BOOL?", TCL_STATIC);
        return TCL_ERROR;
    }
    
    try {
        // Parse arguments using dual syntax parser
        TensorAllcloseArgs args = ParseTensorAllcloseArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name for input"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.other) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name for other"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input_tensor = tensor_storage[args.input];
        auto& other_tensor = tensor_storage[args.other];
        
        // Use PyTorch's allclose with tolerance parameters
        bool result = torch::allclose(input_tensor, other_tensor, args.rtol, args.atol, args.equal_nan);
        
        Tcl_SetObjResult(interp, Tcl_NewBooleanObj(result));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// =====================
// Dual Syntax Support for torch::var_dim / torch::varDim
// =====================
struct TensorVarDimArgs {
    std::string input;
    int dim = 0;
    bool unbiased = true;
    bool keepdim = false;
    
    bool IsValid() const {
        return !input.empty();
    }
};

TensorVarDimArgs ParseTensorVarDimArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorVarDimArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::var_dim tensor dim ?unbiased? ?keepdim?
        if (objc < 3 || objc > 5) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor dim ?unbiased? ?keepdim?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
            throw std::runtime_error("Invalid dim value");
        }
        
        if (objc > 3) {
            int unbias;
            if (Tcl_GetBooleanFromObj(interp, objv[3], &unbias) != TCL_OK) {
                throw std::runtime_error("Invalid unbiased value");
            }
            args.unbiased = unbias != 0;
        }
        
        if (objc > 4) {
            int keep;
            if (Tcl_GetBooleanFromObj(interp, objv[4], &keep) != TCL_OK) {
                throw std::runtime_error("Invalid keepdim value");
            }
            args.keepdim = keep != 0;
        }
    } else {
        // Named parameter syntax: torch::var_dim -input tensor -dim dim ?-unbiased bool? ?-keepdim bool?
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dim value");
                }
            } else if (param == "-unbiased") {
                int unbias;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &unbias) != TCL_OK) {
                    throw std::runtime_error("Invalid unbiased value");
                }
                args.unbiased = unbias != 0;
            } else if (param == "-keepdim") {
                int keep;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &keep) != TCL_OK) {
                    throw std::runtime_error("Invalid keepdim value");
                }
                args.keepdim = keep != 0;
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

int TensorVarDim_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorVarDimArgs args = ParseTensorVarDimArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = tensor.var(args.dim, args.unbiased, args.keepdim);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}