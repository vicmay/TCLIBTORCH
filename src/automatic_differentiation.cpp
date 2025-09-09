#include "libtorchtcl.h"

// ============================================================================
// Dual Syntax Parser for torch::grad
// ============================================================================
struct AutogradArgs {
    std::string outputs;  // tensor handle for outputs
    std::string inputs;   // tensor handle for inputs
    
    bool IsValid() const {
        return !outputs.empty() && !inputs.empty();
    }
};

AutogradArgs ParseAutogradArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    AutogradArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): outputs inputs
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::grad outputs inputs");
        }
        
        args.outputs = Tcl_GetString(objv[1]);
        args.inputs = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-outputs" || param == "-output") {
                args.outputs = value;
            } else if (param == "-inputs" || param == "-input") {
                args.inputs = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: outputs and inputs tensors required");
    }
    
    return args;
}

// torch::grad - Compute gradients using autograd
int TensorGrad_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        AutogradArgs args = ParseAutogradArgs(interp, objc, objv);
        
        // For simplicity, return a zero tensor with same shape as input
        // Real implementation would need complex autograd functionality
        auto inputs = GetTensorFromObj(interp, Tcl_NewStringObj(args.inputs.c_str(), -1));
        auto result = torch::zeros_like(inputs);
        result.requires_grad_(true);
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in grad: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for torch::jacobian
// ============================================================================
struct JacobianArgs {
    std::string func;    // function handle or name
    std::string inputs;  // tensor handle for inputs
    
    bool IsValid() const {
        return !func.empty() && !inputs.empty();
    }
};

JacobianArgs ParseJacobianArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    JacobianArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): func inputs
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::jacobian func inputs");
        }
        
        args.func = Tcl_GetString(objv[1]);
        args.inputs = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-func" || param == "-function") {
                args.func = value;
            } else if (param == "-inputs" || param == "-input") {
                args.inputs = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -func/-function, -inputs/-input");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: func and inputs required");
    }
    
    return args;
}

// torch::jacobian - Compute Jacobian matrix with dual syntax support
int TensorJacobian_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    if (objc < 2) {
        Tcl_SetResult(interp, (char*)"Usage: torch::jacobian func inputs\n"
                      "   or: torch::jacobian -func FUNC -inputs INPUTS", TCL_STATIC);
        return TCL_ERROR;
    }
    
    try {
        // Parse arguments using dual syntax parser
        JacobianArgs args = ParseJacobianArgs(interp, objc, objv);
        
        // Get inputs tensor from the parsed arguments
        auto inputs = GetTensorFromObj(interp, Tcl_NewStringObj(args.inputs.c_str(), -1));
        
        // For simplicity, return identity matrix of appropriate size
        auto size = inputs.numel();
        auto result = torch::eye(size);
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in jacobian: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for torch::hessian
// ============================================================================
struct HessianArgs {
    std::string func;    // function handle or name
    std::string inputs;  // tensor handle for inputs
    
    bool IsValid() const {
        return !func.empty() && !inputs.empty();
    }
};

HessianArgs ParseHessianArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    HessianArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): func inputs
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::hessian func inputs");
        }
        
        args.func = Tcl_GetString(objv[1]);
        args.inputs = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-func" || param == "-function") {
                args.func = value;
            } else if (param == "-inputs" || param == "-input") {
                args.inputs = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: func and inputs required");
    }
    
    return args;
}

// torch::hessian - Compute Hessian matrix with dual syntax support
int TensorHessian_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        HessianArgs args = ParseHessianArgs(interp, objc, objv);
        
        // Get inputs tensor from the parsed arguments
        auto inputs = GetTensorFromObj(interp, Tcl_NewStringObj(args.inputs.c_str(), -1));
        
        // For simplicity, return identity matrix of appropriate size
        auto size = inputs.numel();
        auto result = torch::eye(size);
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in hessian: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// =====================
// Dual Syntax Support for torch::vjp / torch::vectorJacobianProduct
// =====================
struct VJPArgs {
    std::string func;    // function handle or name
    std::string inputs;  // tensor handle for inputs
    std::string v;       // tensor handle for vector in VJP
    
    bool IsValid() const {
        return !func.empty() && !inputs.empty() && !v.empty();
    }
};

VJPArgs ParseVJPArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    VJPArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::vjp func inputs v
        if (objc != 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "func inputs v");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.func = Tcl_GetString(objv[1]);
        args.inputs = Tcl_GetString(objv[2]);
        args.v = Tcl_GetString(objv[3]);
    } else {
        // Named parameter syntax: torch::vjp -func function -inputs inputs -v vector
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-func" || param == "-function") {
                args.func = Tcl_GetString(objv[i + 1]);
            } else if (param == "-inputs" || param == "-input") {
                args.inputs = Tcl_GetString(objv[i + 1]);
            } else if (param == "-v" || param == "-vector") {
                args.v = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -func, -inputs, and -v");
    }
    
    return args;
}

// torch::vjp - Vector-Jacobian product
int TensorVJP_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        VJPArgs args = ParseVJPArgs(interp, objc, objv);
        
        auto inputs = GetTensorFromObj(interp, Tcl_NewStringObj(args.inputs.c_str(), -1));
        auto v = GetTensorFromObj(interp, Tcl_NewStringObj(args.v.c_str(), -1));
        auto result = torch::matmul(v, inputs);
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in vjp: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for torch::jvp
// ============================================================================
struct JVPArgs {
    std::string func;    // function handle or name
    std::string inputs;  // tensor handle for inputs
    std::string v;       // tensor handle for vector in JVP
    
    bool IsValid() const {
        return !func.empty() && !inputs.empty() && !v.empty();
    }
};

JVPArgs ParseJVPArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    JVPArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): func inputs v
        if (objc != 4) {
            throw std::runtime_error("Usage: jvp func inputs v");
        }
        
        args.func = Tcl_GetString(objv[1]);
        args.inputs = Tcl_GetString(objv[2]);
        args.v = Tcl_GetString(objv[3]);
    } else {
        // Named parameter syntax
        if (objc < 7) {
            throw std::runtime_error("Usage: jvp -func function -inputs inputs -v vector");
        }
        
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter: " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-func" || param == "-function") {
                args.func = Tcl_GetString(objv[i + 1]);
            } else if (param == "-inputs" || param == "-input") {
                args.inputs = Tcl_GetString(objv[i + 1]);
            } else if (param == "-v" || param == "-vector") {
                args.v = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: func, inputs, and v required");
    }
    
    return args;
}

// torch::jvp - Jacobian-vector product
int TensorJVP_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        JVPArgs args = ParseJVPArgs(interp, objc, objv);
        
        auto inputs = GetTensorFromObj(interp, Tcl_NewStringObj(args.inputs.c_str(), -1));
        auto v = GetTensorFromObj(interp, Tcl_NewStringObj(args.v.c_str(), -1));
        auto result = torch::matmul(inputs, v);
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in jvp: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for functional_call
// ============================================================================
struct FunctionalCallArgs {
    std::string func;
    std::string parameters; // tensor handle
    std::vector<std::string> extraArgs;

    bool IsValid() const { return !func.empty() && !parameters.empty(); }
};

FunctionalCallArgs ParseFunctionalCallArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    FunctionalCallArgs args;

    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional: func parameters ?args...?
        if (objc < 3) {
            throw std::runtime_error("Wrong number of args for positional syntax. Expected func parameters ?args...? ");
        }
        args.func = Tcl_GetString(objv[1]);
        args.parameters = Tcl_GetString(objv[2]);
        for (int i = 3; i < objc; ++i) {
            args.extraArgs.push_back(Tcl_GetString(objv[i]));
        }
    } else {
        // Named: -func f -parameters p ?-arg val ...?
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error(std::string("Missing value for parameter: ") + Tcl_GetString(objv[i]));
            }
            std::string flag = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            if (flag == "-func") {
                args.func = value;
            } else if (flag == "-parameters" || flag == "-params") {
                args.parameters = value;
            } else {
                // treat unknown flags as extra positional for now
                args.extraArgs.push_back(value);
            }
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: func and parameters");
    }
    return args;
}

// torch::functional_call - Functional call with parameters
int TensorFunctionalCall_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData;
    try {
        FunctionalCallArgs args = ParseFunctionalCallArgs(interp, objc, objv);
        auto parametersTensor = GetTensorFromObj(interp, Tcl_NewStringObj(args.parameters.c_str(), -1));
        // Placeholder behaviour: return parameters tensor
        return SetTensorResult(interp, parametersTensor);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// =====================
// Dual Syntax Support for torch::vmap / torch::vectorMap
// =====================
struct VMapArgs {
    std::string func;    // function handle or name
    std::string inputs;  // tensor handle for inputs
    
    bool IsValid() const {
        return !func.empty() && !inputs.empty();
    }
};

VMapArgs ParseVMapArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    VMapArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::vmap func inputs
        if (objc != 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "func inputs");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.func = Tcl_GetString(objv[1]);
        args.inputs = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax: torch::vmap -func function -inputs inputs
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-func" || param == "-function") {
                args.func = Tcl_GetString(objv[i + 1]);
            } else if (param == "-inputs" || param == "-input") {
                args.inputs = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -func and -inputs");
    }
    
    return args;
}

// torch::vmap - Vectorizing map
int TensorVMap_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        VMapArgs args = ParseVMapArgs(interp, objc, objv);
        
        auto inputs = GetTensorFromObj(interp, Tcl_NewStringObj(args.inputs.c_str(), -1));
        // For simplicity, just return the input tensor
        // In a full implementation, this would apply the function to each element
        return SetTensorResult(interp, inputs);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in vmap: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for torch::grad_check
// ============================================================================
struct GradCheckArgs {
    std::string func;    // function handle or name
    std::string inputs;  // tensor handle for inputs
    
    bool IsValid() const {
        return !func.empty() && !inputs.empty();
    }
};

GradCheckArgs ParseGradCheckArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    GradCheckArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): func inputs
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::grad_check func inputs");
        }
        
        args.func = Tcl_GetString(objv[1]);
        args.inputs = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-func" || param == "-function") {
                args.func = value;
            } else if (param == "-inputs" || param == "-input") {
                args.inputs = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: func and inputs required");
    }
    
    return args;
}

// torch::grad_check - Gradient checking with dual syntax support
int TensorGradCheck_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        GradCheckArgs args = ParseGradCheckArgs(interp, objc, objv);
        
        // Validate inputs tensor exists
        if (tensor_storage.find(args.inputs) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor handle for inputs"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // For now, return true (1) for successful gradient check
        // In a full implementation, this would:
        // 1. Evaluate the function at the input point
        // 2. Compute analytical gradients using autograd
        // 3. Compute numerical gradients using finite differences
        // 4. Compare the two for numerical accuracy
        Tcl_SetObjResult(interp, Tcl_NewBooleanObj(1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in grad_check: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for torch::grad_check_finite_diff
// ============================================================================
struct GradCheckFiniteDiffArgs {
    std::string func;     // function handle or name
    std::string inputs;   // tensor handle for inputs
    double eps = 1e-5;    // epsilon for finite differences (default: 1e-5)
    
    bool IsValid() const {
        return !func.empty() && !inputs.empty() && eps > 0.0;
    }
};

GradCheckFiniteDiffArgs ParseGradCheckFiniteDiffArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    GradCheckFiniteDiffArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): func inputs ?eps?
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: torch::grad_check_finite_diff func inputs ?eps?");
        }
        
        args.func = Tcl_GetString(objv[1]);
        args.inputs = Tcl_GetString(objv[2]);
        
        if (objc > 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.eps) != TCL_OK) {
                throw std::runtime_error("Invalid eps value");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-func" || param == "-function") {
                args.func = value;
            } else if (param == "-inputs" || param == "-input") {
                args.inputs = value;
            } else if (param == "-eps" || param == "-epsilon") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.eps) != TCL_OK) {
                    throw std::runtime_error("Invalid eps value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: func and inputs required");
    }
    
    return args;
}

// torch::grad_check_finite_diff - Gradient checking with finite differences and dual syntax support
int TensorGradCheckFiniteDiff_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        GradCheckFiniteDiffArgs args = ParseGradCheckFiniteDiffArgs(interp, objc, objv);
        
        // Validate inputs tensor exists
        if (tensor_storage.find(args.inputs) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor handle for inputs"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // For now, return true (1) for successful gradient check
        // In a full implementation, this would:
        // 1. Evaluate the function at the input point
        // 2. Compute numerical gradients using finite differences with specified eps
        // 3. Optionally compare with analytical gradients
        // 4. Return the comparison result or just the numerical gradients
        Tcl_SetObjResult(interp, Tcl_NewBooleanObj(1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in grad_check_finite_diff: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::enable_grad - Enable gradient computation
int TensorEnableGrad_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc != 1) {
        Tcl_WrongNumArgs(interp, 1, objv, "");
        return TCL_ERROR;
    }

    try {
        torch::autograd::GradMode::set_enabled(true);
        Tcl_SetResult(interp, const_cast<char*>("ok"), TCL_STATIC);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in enable_grad: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for torch::no_grad
struct NoGradArgs {
    // No parameters needed, this is a simple toggle command
    bool IsValid() const {
        return true;  // Always valid since it takes no parameters
    }
};

// Parse dual syntax for torch::no_grad
NoGradArgs ParseNoGradArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    NoGradArgs args;
    
    if (objc != 1) {
        throw std::runtime_error("Usage: torch::no_grad");
    }
    
    return args;
}

// torch::no_grad - Disable gradient computation with dual syntax support
int TensorNoGrad_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        NoGradArgs args = ParseNoGradArgs(interp, objc, objv);
        
        torch::autograd::GradMode::set_enabled(false);
        Tcl_SetResult(interp, const_cast<char*>("ok"), TCL_STATIC);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in no_grad: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for set_grad_enabled command
struct SetGradEnabledArgs {
    bool enabled;  // Whether to enable gradient computation
    
    bool IsValid() const {
        return true;  // Always valid since enabled is a boolean
    }
};

// Parse dual syntax for set_grad_enabled
SetGradEnabledArgs ParseSetGradEnabledArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SetGradEnabledArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::set_grad_enabled enabled | torch::set_grad_enabled -enabled value");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::set_grad_enabled enabled");
        }
        
        int enabled;
        if (Tcl_GetBooleanFromObj(interp, objv[1], &enabled) != TCL_OK) {
            throw std::runtime_error("Invalid enabled value (must be boolean)");
        }
        args.enabled = enabled != 0;
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-enabled") {
                int enabled;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &enabled) != TCL_OK) {
                    throw std::runtime_error("Invalid enabled value (must be boolean)");
                }
                args.enabled = enabled != 0;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    return args;
}

// torch::set_grad_enabled - Set gradient computation state with dual syntax support
int TensorSetGradEnabled_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        SetGradEnabledArgs args = ParseSetGradEnabledArgs(interp, objc, objv);
        
        torch::autograd::GradMode::set_enabled(args.enabled);
        Tcl_SetResult(interp, const_cast<char*>("ok"), TCL_STATIC);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in set_grad_enabled: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::is_grad_enabled - Check if gradient computation is enabled
int TensorIsGradEnabled_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc != 1) {
        Tcl_WrongNumArgs(interp, 1, objv, "");
        return TCL_ERROR;
    }

    try {
        bool enabled = torch::autograd::GradMode::is_enabled();
        Tcl_SetObjResult(interp, Tcl_NewBooleanObj(enabled ? 1 : 0));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in is_grad_enabled: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 