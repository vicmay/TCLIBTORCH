#include "libtorchtcl.h"

// Parameter structure for tensor_to_list command
struct TensorToListArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_to_list
TensorToListArgs ParseTensorToListArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorToListArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::tensor_to_list tensor | torch::tensor_to_list -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::tensor_to_list tensor");
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

int TensorToList_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax parser
        TensorToListArgs args = ParseTensorToListArgs(interp, objc, objv);
        
        // Validate tensor exists
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get tensor
        auto& tensor = tensor_storage[args.input];
        
        // Convert tensor to 1D if needed
        auto flattened = tensor.flatten().contiguous();
        
        // Create TCL list
        Tcl_Obj* result_list = Tcl_NewListObj(0, nullptr);
        
        // Convert tensor values to list elements based on dtype
        if (tensor.dtype() == torch::kFloat32) {
            auto accessor = flattened.accessor<float,1>();
            for (int64_t i = 0; i < flattened.numel(); i++) {
                Tcl_ListObjAppendElement(interp, result_list, Tcl_NewDoubleObj(accessor[i]));
            }
        } else if (tensor.dtype() == torch::kFloat64) {
            auto accessor = flattened.accessor<double,1>();
            for (int64_t i = 0; i < flattened.numel(); i++) {
                Tcl_ListObjAppendElement(interp, result_list, Tcl_NewDoubleObj(accessor[i]));
            }
        } else if (tensor.dtype() == torch::kInt32) {
            auto accessor = flattened.accessor<int32_t,1>();
            for (int64_t i = 0; i < flattened.numel(); i++) {
                Tcl_ListObjAppendElement(interp, result_list, Tcl_NewLongObj(accessor[i]));
            }
        } else if (tensor.dtype() == torch::kInt64) {
            auto accessor = flattened.accessor<int64_t,1>();
            for (int64_t i = 0; i < flattened.numel(); i++) {
                Tcl_ListObjAppendElement(interp, result_list, Tcl_NewLongObj(accessor[i]));
            }
        } else if (tensor.dtype() == torch::kBool) {
            // For boolean tensors (like from comparison ops), convert to integers 0/1
            auto accessor = flattened.accessor<bool,1>();
            for (int64_t i = 0; i < flattened.numel(); i++) {
                Tcl_ListObjAppendElement(interp, result_list, Tcl_NewIntObj(accessor[i] ? 1 : 0));
            }
        } else {
            // For other types, convert to double via float
            auto float_tensor = flattened.to(torch::kFloat32);
            auto accessor = float_tensor.accessor<float,1>();
            for (int64_t i = 0; i < flattened.numel(); i++) {
                Tcl_ListObjAppendElement(interp, result_list, Tcl_NewDoubleObj(accessor[i]));
            }
        }
        
        Tcl_SetObjResult(interp, result_list);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 