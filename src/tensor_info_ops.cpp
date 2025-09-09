#include "libtorchtcl.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>

extern std::unordered_map<std::string, torch::Tensor> tensor_storage;

// Parameter structure for tensor_size
struct TensorSizeArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Dual syntax parser for tensor_size
TensorSizeArgs ParseTensorSizeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorSizeArgs args;
    
    // Check for named parameter syntax
    bool use_named_params = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');
    
    if (!use_named_params) {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::tensor_size tensor");
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
        throw std::runtime_error("Required parameters missing: input");
    }
    
    return args;
}

// torch::tensor_size - Get tensor size
int TensorSize_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 2) {
        Tcl_SetResult(interp, (char*)"Usage: torch::tensor_size tensor\n"
                      "   or: torch::tensor_size -input TENSOR", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        TensorSizeArgs args = ParseTensorSizeArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto sizes = input.sizes();
        
        // Convert sizes to Tcl list
        Tcl_Obj* result_list = Tcl_NewListObj(0, NULL);
        for (const auto& size : sizes) {
            Tcl_ListObjAppendElement(interp, result_list, Tcl_NewLongObj(size));
        }
        
        Tcl_SetObjResult(interp, result_list);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 