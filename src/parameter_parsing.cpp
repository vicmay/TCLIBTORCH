#include "parameter_parsing.h"
#include "libtorchtcl.h"
#include <torch/torch.h>

// Tensor creation arguments implementation
bool TensorCreationArgs::IsValid() const {
    return !shape.empty() && !dtype.empty() && !device.empty();
}

TensorCreationArgs TensorCreationArgs::Parse(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Check if any argument starts with '-' (named parameter)
    bool hasNamedParams = false;
    for (int i = 1; i < objc; i++) {
        if (Tcl_GetString(objv[i])[0] == '-') {
            hasNamedParams = true;
            break;
        }
    }
    
    if (!hasNamedParams) {
        // Pure positional syntax (backward compatibility)
        return ParsePositionalArgs(interp, objc, objv);
    } else if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Mixed syntax: starts with positional, then named parameters
        return ParseMixedArgs(interp, objc, objv);
    } else {
        // Pure named parameter syntax
        return ParseNamedArgs(interp, objc, objv);
    }
}

TensorCreationArgs TensorCreationArgs::ParsePositionalArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorCreationArgs args;
    
    if (objc < 2 || objc > 5) {
        Tcl_SetResult(interp, (char*)"Wrong number of arguments for positional syntax", TCL_STATIC);
        return args;
    }
    
    // Parse shape (required)
    args.shape = TclListToShape(interp, objv[1]);
    
    // Parse optional parameters
    if (objc > 2) {
        args.dtype = Tcl_GetString(objv[2]);
    }
    if (objc > 3) {
        args.device = Tcl_GetString(objv[3]);
    }
    if (objc > 4) {
        int grad;
        if (Tcl_GetBooleanFromObj(interp, objv[4], &grad) != TCL_OK) {
            return args;
        }
        args.requires_grad = grad != 0;
    }
    
    return args;
}

TensorCreationArgs TensorCreationArgs::ParseMixedArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorCreationArgs args;
    
    // First argument should be shape (positional)
    if (objc < 2) {
        Tcl_SetResult(interp, (char*)"Missing required shape argument", TCL_STATIC);
        return args;
    }
    
    args.shape = TclListToShape(interp, objv[1]);
    
    // Parse remaining arguments as named parameters
    for (int i = 2; i < objc; i += 2) {
        if (i + 1 >= objc) {
            Tcl_SetResult(interp, (char*)"Missing value for parameter", TCL_STATIC);
            return args;
        }
        
        std::string param = Tcl_GetString(objv[i]);
        Tcl_Obj* value = objv[i + 1];
        
        if (param == "-dtype") {
            args.dtype = Tcl_GetString(value);
        } else if (param == "-device") {
            args.device = Tcl_GetString(value);
        } else if (param == "-requiresGrad") {
            int grad;
            if (Tcl_GetBooleanFromObj(interp, value, &grad) != TCL_OK) {
                return args;
            }
            args.requires_grad = grad != 0;
        } else {
            Tcl_SetResult(interp, (char*)("Unknown parameter: " + param).c_str(), TCL_VOLATILE);
            return args;
        }
    }
    
    return args;
}

TensorCreationArgs TensorCreationArgs::ParseNamedArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorCreationArgs args;
    
    // Parse named parameters
    for (int i = 1; i < objc; i += 2) {
        if (i + 1 >= objc) {
            Tcl_SetResult(interp, (char*)"Missing value for parameter", TCL_STATIC);
            return args;
        }
        
        std::string param = Tcl_GetString(objv[i]);
        Tcl_Obj* value = objv[i + 1];
        
        if (param == "-shape") {
            args.shape = TclListToShape(interp, value);
        } else if (param == "-dtype") {
            args.dtype = Tcl_GetString(value);
        } else if (param == "-device") {
            args.device = Tcl_GetString(value);
        } else if (param == "-requiresGrad") {
            int grad;
            if (Tcl_GetBooleanFromObj(interp, value, &grad) != TCL_OK) {
                return args;
            }
            args.requires_grad = grad != 0;
        } else {
            Tcl_SetResult(interp, (char*)("Unknown parameter: " + param).c_str(), TCL_VOLATILE);
            return args;
        }
    }
    
    // Validate required parameters
    if (args.shape.empty()) {
        Tcl_SetResult(interp, (char*)"Missing required parameter: -shape", TCL_STATIC);
        return args;
    }
    
    return args;
}

// Helper functions implementation (using existing ones from helpers.cpp)
c10::ScalarType GetScalarType(const std::string& type_str) {
    if (type_str == "float32" || type_str == "Float32" || type_str == "float") return torch::kFloat32;
    if (type_str == "float64" || type_str == "Float64" || type_str == "double") return torch::kFloat64;
    if (type_str == "int32" || type_str == "Int32" || type_str == "int") return torch::kInt32;
    if (type_str == "int64" || type_str == "Int64" || type_str == "long") return torch::kInt64;
    if (type_str == "bool" || type_str == "Bool") return torch::kBool;
    throw std::runtime_error("Unknown scalar type: " + type_str);
}

torch::Device GetDevice(const std::string& device_str) {
    if (device_str == "cpu") return torch::kCPU;
    if (device_str == "cuda") {
        // Try to use CUDA, but fall back to CPU if it fails
        try {
            if (torch::cuda::is_available()) {
                return torch::kCUDA;
            }
        } catch (const std::exception& e) {
            // CUDA not available or error, fall back to CPU
        }
        return torch::kCPU; // Fall back to CPU
    }
    throw std::runtime_error("Invalid device string: " + device_str);
} 