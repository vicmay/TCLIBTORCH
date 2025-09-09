#include "libtorchtcl.h"

// Parameter structure for tensor_randn command
struct TensorRandnArgs {
    std::vector<int64_t> shape;
    std::string device = "cpu";
    std::string dtype = "float32";

    bool IsValid() const {
        return true; // Allow empty shape for scalar tensors
    }
};

// Parse dual syntax for tensor_randn
TensorRandnArgs ParseTensorRandnArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorRandnArgs args;
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: shape ?device? ?dtype?
        if (objc < 2 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "shape ?device? ?dtype?");
            throw std::runtime_error("Invalid number of arguments");
        }
        args.shape = TclListToShape(interp, objv[1]);
        if (objc >= 3) {
            args.device = Tcl_GetString(objv[2]);
        }
        if (objc >= 4) {
            args.dtype = Tcl_GetString(objv[3]);
        }
    } else {
        // Named parameter syntax
        bool has_shape = false;
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            if (param == "-shape") {
                args.shape = TclListToShape(interp, objv[i + 1]);
                has_shape = true;
            } else if (param == "-device") {
                args.device = value;
            } else if (param == "-dtype") {
                args.dtype = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
        if (!has_shape) {
            throw std::runtime_error("Required parameter missing: shape");
        }
    }
    if (!args.IsValid()) {
        throw std::runtime_error("Invalid shape parameter");
    }
    return args;
}

// torch::tensor_randn(shape, device?, dtype?) - Normal distribution tensors
int TensorRandn_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        TensorRandnArgs args = ParseTensorRandnArgs(interp, objc, objv);
        // Create tensor options
        auto options = torch::TensorOptions()
            .dtype(GetScalarType(args.dtype.c_str()))
            .device(GetDevice(args.device.c_str()));
        // Create tensor with normal distribution
        torch::Tensor tensor = torch::randn(args.shape, options);
        // Store and return handle
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = tensor;
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_rand command
struct TensorRandArgs {
    std::vector<int64_t> shape;
    std::string device = "cpu";
    std::string dtype = "float32";

    bool IsValid() const {
        return true; // Allow empty shape for scalar tensors
    }
};

// Parse dual syntax for tensor_rand
TensorRandArgs ParseTensorRandArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorRandArgs args;
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: shape ?device? ?dtype?
        if (objc < 2 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "shape ?device? ?dtype?");
            throw std::runtime_error("Invalid number of arguments");
        }
        args.shape = TclListToShape(interp, objv[1]);
        if (objc >= 3) {
            args.device = Tcl_GetString(objv[2]);
        }
        if (objc >= 4) {
            args.dtype = Tcl_GetString(objv[3]);
        }
    } else {
        // Named parameter syntax
        bool has_shape = false;
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            if (param == "-shape") {
                args.shape = TclListToShape(interp, objv[i + 1]);
                has_shape = true;
            } else if (param == "-device") {
                args.device = value;
            } else if (param == "-dtype") {
                args.dtype = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
        if (!has_shape) {
            throw std::runtime_error("Required parameter missing: shape");
        }
    }
    if (!args.IsValid()) {
        throw std::runtime_error("Invalid shape parameter");
    }
    return args;
}

// torch::tensor_rand(shape, device?, dtype?) - Uniform distribution tensors
int TensorRand_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        TensorRandArgs args = ParseTensorRandArgs(interp, objc, objv);
        // Create tensor options
        auto options = torch::TensorOptions()
            .dtype(GetScalarType(args.dtype.c_str()))
            .device(GetDevice(args.device.c_str()));
        // Create tensor with uniform distribution [0, 1)
        torch::Tensor tensor = torch::rand(args.shape, options);
        // Store and return handle
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = tensor;
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct TensorItemArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

TensorItemArgs ParseTensorItemArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorItemArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::tensor_item tensor");
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
            
            if (param == "-tensor" || param == "-input") {
                args.input = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: tensor");
    }
    
    return args;
}

// torch::tensor_item(tensor) - Extract scalar value from single-element tensor
int TensorItem_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        TensorItemArgs args = ParseTensorItemArgs(interp, objc, objv);
        
        // Check if tensor exists
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>(("Invalid tensor name: " + args.input).c_str()), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        // Check if tensor has exactly one element
        if (tensor.numel() != 1) {
            Tcl_SetResult(interp, const_cast<char*>("Tensor must have exactly one element"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Extract the scalar value based on dtype
        std::string result;
        if (tensor.dtype() == torch::kFloat32 || tensor.dtype() == torch::kFloat64) {
            double value = tensor.item<double>();
            result = std::to_string(value);
        } else if (tensor.dtype() == torch::kInt32 || tensor.dtype() == torch::kInt64) {
            int64_t value = tensor.item<int64_t>();
            result = std::to_string(value);
        } else if (tensor.dtype() == torch::kBool) {
            bool value = tensor.item<bool>();
            result = value ? "1" : "0";
        } else {
            // For other types, convert to double
            double value = tensor.item<double>();
            result = std::to_string(value);
        }
        
        Tcl_SetResult(interp, const_cast<char*>(result.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct TensorNumelArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

TensorNumelArgs ParseTensorNumelArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorNumelArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::tensor_numel tensor");
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
            
            if (param == "-tensor" || param == "-input") {
                args.input = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: tensor");
    }
    
    return args;
}

// torch::tensor_numel(tensor) - Get total number of elements
int TensorNumel_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        TensorNumelArgs args = ParseTensorNumelArgs(interp, objc, objv);
        
        // Check if tensor exists
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>(("Invalid tensor name: " + args.input).c_str()), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        // Get number of elements
        int64_t num_elements = tensor.numel();
        std::string result = std::to_string(num_elements);
        
        Tcl_SetResult(interp, const_cast<char*>(result.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 