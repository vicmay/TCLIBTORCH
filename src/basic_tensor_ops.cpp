#include "libtorchtcl.h"

// -----------------------------------------------------------------------------
// Parameter structure and dual-syntax parser for tensor_create (TensorCreate)
// -----------------------------------------------------------------------------
struct TensorCreateArgs {
    Tcl_Obj* dataObj = nullptr;                 // list of values or tensor handle of source data
    std::vector<int64_t> shape;                 // optional shape to reshape into
    std::string dtype = "float32";               // data type
    std::string device = "cpu";                  // device string
    bool requiresGrad = false;                  // whether requires_grad

    bool IsValid() const { return dataObj != nullptr; }
};

// Parse both positional and named syntaxes. Throws std::runtime_error on error so
// callers can translate to TCL error properly.
static TensorCreateArgs ParseTensorCreateArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorCreateArgs args;

    // Decide syntax flavour by inspecting second arg (index 1)
    bool namedSyntax = false;
    if (objc >= 2) {
        std::string firstArg = Tcl_GetString(objv[1]);
        namedSyntax = (firstArg.size() > 0 && firstArg[0] == '-');
    }

    if (!namedSyntax) {
        // The first argument is data. Inspect whether the *next* argument starts with '-' to
        // decide if the caller switched to named-parameter style after the data argument.
        bool remainingNamedSyntax = false;
        if (objc > 2) {
            std::string nextArg = Tcl_GetString(objv[2]);
            remainingNamedSyntax = (!nextArg.empty() && nextArg[0] == '-');
        }

        if (!remainingNamedSyntax) {
            // ------------------------------------------------------------------
            // Pure positional syntax: values ?shape? ?dtype? ?device? ?requires_grad?
            // ------------------------------------------------------------------
            if (objc < 2 || objc > 6) {
                Tcl_WrongNumArgs(interp, 1, objv, "values ?shape? ?dtype? ?device? ?requires_grad?");
                throw std::runtime_error("Incorrect number of arguments");
            }

            args.dataObj = objv[1];

            // Parse remaining positional arguments
            int arg_idx = 2;

            // Check if argument 2 is a shape (list of integers)
            bool has_shape = false;
            if (objc > 2) {
                int list_len;
                if (Tcl_ListObjLength(interp, objv[2], &list_len) == TCL_OK && list_len > 0) {
                    bool all_integers = true;
                    for (int i = 0; i < list_len; i++) {
                        Tcl_Obj* element;
                        Tcl_ListObjIndex(interp, objv[2], i, &element);
                        int dummy;
                        if (Tcl_GetIntFromObj(interp, element, &dummy) != TCL_OK) {
                            all_integers = false;
                            break;
                        }
                    }
                    if (all_integers) {
                        has_shape = true;
                        args.shape = TclListToShape(interp, objv[2]);
                        arg_idx = 3;
                    }
                }
            }

            // Parse dtype if provided
            if (objc > arg_idx) {
                args.dtype = Tcl_GetString(objv[arg_idx]);
                arg_idx++;
            }

            // Parse device if provided
            if (objc > arg_idx) {
                args.device = Tcl_GetString(objv[arg_idx]);
                arg_idx++;
            }

            // Parse requires_grad if provided
            if (objc > arg_idx) {
                int grad;
                if (Tcl_GetBooleanFromObj(interp, objv[arg_idx], &grad) != TCL_OK) {
                    throw std::runtime_error("Invalid requires_grad boolean");
                }
                args.requiresGrad = (grad != 0);
            }
        } else {
            // ------------------------------------------------------------------
            // Hybrid syntax: first positional data argument followed by named parameters
            // Example: tensor_create DATA -dtype int64 -device cpu
            // ------------------------------------------------------------------
            args.dataObj = objv[1];
            for (int i = 2; i < objc; i += 2) {
                if (i + 1 >= objc) {
                    throw std::runtime_error("Missing value for parameter");
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
                        throw std::runtime_error("Invalid boolean for -requiresGrad");
                    }
                    args.requiresGrad = (grad != 0);
                } else {
                    throw std::runtime_error("Unknown parameter: " + param);
                }
            }
        }
    } else {
        // Named syntax: expect flag/value pairs
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* value = objv[i + 1];

            if (param == "-data") {
                args.dataObj = value;
            } else if (param == "-shape") {
                args.shape = TclListToShape(interp, value);
            } else if (param == "-dtype") {
                args.dtype = Tcl_GetString(value);
            } else if (param == "-device") {
                args.device = Tcl_GetString(value);
            } else if (param == "-requiresGrad") {
                int grad;
                if (Tcl_GetBooleanFromObj(interp, value, &grad) != TCL_OK) {
                    throw std::runtime_error("Invalid boolean for -requiresGrad");
                }
                args.requiresGrad = (grad != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("Missing required parameter: -data");
    }

    // Validate dtype
    if (args.dtype != "float32" && args.dtype != "float64" && args.dtype != "int32" && 
        args.dtype != "int64" && args.dtype != "bool" && args.dtype != "float" && 
        args.dtype != "double" && args.dtype != "int" && args.dtype != "long") {
        throw std::runtime_error("Invalid dtype: " + args.dtype);
    }

    return args;
}


// Create a new tensor
int TensorCreate_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorCreateArgs args = ParseTensorCreateArgs(interp, objc, objv);
        
        // Create tensor using the parsed arguments
        torch::Tensor tensor = TclListToTensor(interp, args.dataObj, args.dtype.c_str(), args.device.c_str(), args.requiresGrad);
        
        // Apply shape if provided
        if (!args.shape.empty()) {
            tensor = tensor.reshape(args.shape);
        }
        
        // Apply requires_grad if specified
        if (args.requiresGrad) {
            tensor.set_requires_grad(true);
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = tensor;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Get tensor properties
static int TensorProperty(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[], const char* property) {
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
        if (strcmp(property, "dtype") == 0) {
            std::string dtype = torch::toString(tensor.scalar_type());
            Tcl_SetResult(interp, const_cast<char*>(dtype.c_str()), TCL_VOLATILE);
        } else if (strcmp(property, "device") == 0) {
            std::string device = torch::toString(tensor.device());
            Tcl_SetResult(interp, const_cast<char*>(device.c_str()), TCL_VOLATILE);
        } else if (strcmp(property, "requires_grad") == 0) {
            Tcl_SetResult(interp, const_cast<char*>(tensor.requires_grad() ? "1" : "0"), TCL_VOLATILE);
        } else if (strcmp(property, "grad") == 0) {
            if (tensor.grad().defined()) {
                std::string grad_handle = GetNextHandle("tensor");
                tensor_storage[grad_handle] = tensor.grad();
                Tcl_SetResult(interp, const_cast<char*>(grad_handle.c_str()), TCL_VOLATILE);
            } else {
                Tcl_SetResult(interp, const_cast<char*>(""), TCL_VOLATILE);
            }
        }
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_dtype command
struct TensorDtypeArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_dtype
TensorDtypeArgs ParseTensorDtypeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorDtypeArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor
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

int TensorGetDtype_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorDtypeArgs args = ParseTensorDtypeArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        std::string dtype;
        
        // Convert scalar type to expected format for tests
        if (tensor.dtype() == torch::kFloat32) {
            dtype = "Float32";
        } else if (tensor.dtype() == torch::kFloat64) {
            dtype = "Float64";
        } else if (tensor.dtype() == torch::kInt32) {
            dtype = "Int32";
        } else if (tensor.dtype() == torch::kInt64) {
            dtype = "Int64";
        } else if (tensor.dtype() == torch::kBool) {
            dtype = "Bool";
        } else {
            dtype = "Unknown";
        }
        
        Tcl_SetResult(interp, const_cast<char*>(dtype.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_device command
struct TensorDeviceArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_device
TensorDeviceArgs ParseTensorDeviceArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorDeviceArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor
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

int TensorGetDevice_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorDeviceArgs args = ParseTensorDeviceArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        std::string device = torch::toString(tensor.device());
        
        Tcl_SetResult(interp, const_cast<char*>(device.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_requires_grad command
struct TensorRequiresGradArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_requires_grad
TensorRequiresGradArgs ParseTensorRequiresGradArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorRequiresGradArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor
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

int TensorRequiresGrad_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorRequiresGradArgs args = ParseTensorRequiresGradArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        Tcl_SetResult(interp, const_cast<char*>(tensor.requires_grad() ? "1" : "0"), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_grad command
struct TensorGradArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_grad
TensorGradArgs ParseTensorGradArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorGradArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor
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

int TensorGetGrad_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorGradArgs args = ParseTensorGradArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        if (!tensor.requires_grad()) {
            Tcl_SetResult(interp, const_cast<char*>("Tensor does not require gradients"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (!tensor.grad().defined()) {
            Tcl_SetResult(interp, const_cast<char*>("No gradient computed yet"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        std::string grad_handle = GetNextHandle("tensor");
        tensor_storage[grad_handle] = tensor.grad();
        
        Tcl_SetResult(interp, const_cast<char*>(grad_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_print command
struct TensorPrintArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_print
TensorPrintArgs ParseTensorPrintArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorPrintArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor
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

int TensorPrint_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorPrintArgs args = ParseTensorPrintArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        std::ostringstream oss;
        
        // Get tensor data
        auto tensor_data = tensor.to(torch::kCPU).contiguous();
        auto sizes = tensor.sizes();
        
        // Print tensor data recursively
        std::function<void(const torch::Tensor&, int)> print_tensor = [&](const torch::Tensor& t, int depth) {
            if (t.dim() == 0) {
                // Print scalar value with decimal point for floating point types
                float val = t.item().to<float>();
                if (std::floor(val) == val) {
                    oss << val << ".";  // Add decimal point for whole numbers
                } else {
                    oss << val;  // Already has decimal point
                }
                return;
            }
            
            // Special case for single-element tensors
            if (t.numel() == 1) {
                oss << "{";
                auto flat = t.view({1});  // Flatten to 1D
                print_tensor(flat[0], depth);
                oss << "}";
                return;
            }
            
            oss << "{";
            int64_t size = t.size(0);
            for (int64_t i = 0; i < size; ++i) {
                if (i > 0) {
                    if (t.dim() == 2) {
                        oss << " ";
                    } else {
                        oss << " ";
                        if (t.dim() == 3) {
                            oss << "\n";
                            oss << std::string(11, ' ');  // Base indentation for 5D tensor
                        } else {
                            oss << std::string(depth + 1, ' ');
                        }
                    }
                }
                print_tensor(t[i], depth + 1);
            }
            oss << "}";
        };
        
        print_tensor(tensor_data, 1);  // Start at depth 1 for consistent indentation
        
        Tcl_SetResult(interp, const_cast<char*>(oss.str().c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_backward command
struct TensorBackwardArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_backward
TensorBackwardArgs ParseTensorBackwardArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorBackwardArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor
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

int TensorBackward_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorBackwardArgs args = ParseTensorBackwardArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        if (!tensor.requires_grad()) {
            Tcl_SetResult(interp, const_cast<char*>("Tensor does not require gradients"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        tensor.backward();
        Tcl_SetResult(interp, const_cast<char*>("OK"), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Advanced tensor operations helper
static int TensorAdvancedOp(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[], const char* op) {
    if (objc < 2 || objc > 3) {
        Tcl_WrongNumArgs(interp, 1, objv, "tensor ?dim?");
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
        
        if (objc == 3) {
            int dim;
            if (Tcl_GetIntFromObj(interp, objv[2], &dim) != TCL_OK) {
                return TCL_ERROR;
            }
            
            if (strcmp(op, "sum") == 0) {
                result = tensor.sum(dim);
            } else if (strcmp(op, "mean") == 0) {
                result = tensor.mean(dim);
            } else if (strcmp(op, "max") == 0) {
                result = std::get<0>(tensor.max(dim));
            } else if (strcmp(op, "min") == 0) {
                result = std::get<0>(tensor.min(dim));
            }
        } else {
            if (strcmp(op, "abs") == 0) {
                result = tensor.abs();
            } else if (strcmp(op, "exp") == 0) {
                result = tensor.exp();
            } else if (strcmp(op, "log") == 0) {
                result = tensor.log();
            } else if (strcmp(op, "sqrt") == 0) {
                result = tensor.sqrt();
            } else if (strcmp(op, "sum") == 0) {
                result = tensor.sum();
            } else if (strcmp(op, "mean") == 0) {
                result = tensor.mean();
            } else if (strcmp(op, "max") == 0) {
                result = tensor.max();
            } else if (strcmp(op, "min") == 0) {
                result = tensor.min();
            } else if (strcmp(op, "sigmoid") == 0) {
                result = tensor.sigmoid();
            } else if (strcmp(op, "relu") == 0) {
                result = tensor.relu();
            } else if (strcmp(op, "tanh") == 0) {
                result = tensor.tanh();
            }
        }
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_abs command
struct TensorAbsArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_abs
TensorAbsArgs ParseTensorAbsArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorAbsArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor
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

// Individual operation commands
int TensorAbs_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorAbsArgs args = ParseTensorAbsArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        // Apply abs operation while preserving all tensor options
        torch::Tensor result = tensor.abs().to(tensor.options());
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct TensorExpArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

TensorExpArgs ParseTensorExpArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorExpArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::tensor_exp tensor");
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

int TensorExp_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorExpArgs args = ParseTensorExpArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        // Apply exp operation while preserving all tensor options
        torch::Tensor result = torch::exp(tensor).to(tensor.options());
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct TensorLogArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

TensorLogArgs ParseTensorLogArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorLogArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::tensor_log tensor");
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

int TensorLog_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorLogArgs args = ParseTensorLogArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        // Apply log operation while preserving all tensor options
        torch::Tensor result = torch::log(tensor).to(tensor.options());
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_sqrt command
struct TensorSqrtArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_sqrt
TensorSqrtArgs ParseTensorSqrtArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorSqrtArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::tensor_sqrt tensor");
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

int TensorSqrt_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorSqrtArgs args = ParseTensorSqrtArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        // Apply sqrt operation while preserving all tensor options
        torch::Tensor result = torch::sqrt(tensor).to(tensor.options());
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_sum command
struct TensorSumArgs {
    std::string input;
    int dim = -1;  // -1 means no dimension specified (reduce all)
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_sum
TensorSumArgs ParseTensorSumArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorSumArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("Usage: torch::tensor_sum tensor ?dim?");
        }
        args.input = Tcl_GetString(objv[1]);
        if (objc == 3) {
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
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input") {
                args.input = value;
            } else if (param == "-dim") {
                args.dim = std::stoi(value);
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

int TensorSum_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorSumArgs args = ParseTensorSumArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result;
        
        if (args.dim >= 0) {
            result = tensor.sum(args.dim);
        } else {
            result = tensor.sum();
        }
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_mean command
struct TensorMeanArgs {
    std::string input;
    int dim = -1;  // -1 means no dimension specified (reduce all)
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_mean
TensorMeanArgs ParseTensorMeanArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorMeanArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor ?dim?
        if (objc < 2 || objc > 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor ?dim?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc == 3) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
                throw std::runtime_error("Invalid dimension value");
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
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dimension value: " + value);
                }
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

int TensorMean_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorMeanArgs args = ParseTensorMeanArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result;
        
        if (args.dim >= 0) {
            result = tensor.mean(args.dim);
        } else {
            result = tensor.mean();
        }
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_max command
struct TensorMaxArgs {
    std::string input;
    int dim = -1;  // -1 means no dimension specified (reduce all)
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_max
TensorMaxArgs ParseTensorMaxArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorMaxArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor ?dim?
        if (objc < 2 || objc > 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor ?dim?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc == 3) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
                throw std::runtime_error("Invalid dimension value");
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
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dimension value: " + value);
                }
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

int TensorMax_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorMaxArgs args = ParseTensorMaxArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result;
        
        if (args.dim >= 0) {
            // Max along specified dimension
            result = std::get<0>(tensor.max(args.dim));
        } else {
            // Max of entire tensor
            result = tensor.max();
        }
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_min command
struct TensorMinArgs {
    std::string input;
    int dim = -1;  // -1 means no dimension specified (reduce all)
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_min
TensorMinArgs ParseTensorMinArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorMinArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("Usage: torch::tensor_min tensor ?dim?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc == 3) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
                throw std::runtime_error("Invalid dim parameter");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid value for -dim parameter");
                }
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

int TensorMin_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorMinArgs args = ParseTensorMinArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result;
        
        if (args.dim == -1) {
            // No dimension specified - reduce all dimensions
            result = tensor.min();
        } else {
            // Specific dimension
            result = std::get<0>(tensor.min(args.dim));
        }
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_sigmoid command
struct TensorSigmoidArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_sigmoid
TensorSigmoidArgs ParseTensorSigmoidArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorSigmoidArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::tensor_sigmoid tensor");
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

int TensorSigmoid_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorSigmoidArgs args = ParseTensorSigmoidArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        // Apply sigmoid operation while preserving all tensor options
        torch::Tensor result = torch::sigmoid(tensor).to(tensor.options());
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_relu command
struct TensorReluArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_relu
TensorReluArgs ParseTensorReluArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorReluArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::tensor_relu tensor");
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

int TensorRelu_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorReluArgs args = ParseTensorReluArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        // Apply ReLU operation while preserving all tensor options
        torch::Tensor result = torch::relu(tensor).to(tensor.options());
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_tanh command
struct TensorTanhArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_tanh
TensorTanhArgs ParseTensorTanhArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorTanhArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::tensor_tanh tensor");
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

int TensorTanh_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorTanhArgs args = ParseTensorTanhArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        // Ensure we preserve the data type by using the same options
        torch::Tensor result = tensor.tanh().to(tensor.options());
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_add command
struct TensorAddArgs {
    std::string input1;
    std::string input2;
    double alpha = 1.0;
    
    bool IsValid() const {
        return !input1.empty() && !input2.empty();
    }
};

// Parse dual syntax for tensor_add
TensorAddArgs ParseTensorAddArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorAddArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor1 tensor2 ?alpha?
        if (objc < 3 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor1 tensor2 ?alpha?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input1 = Tcl_GetString(objv[1]);
        args.input2 = Tcl_GetString(objv[2]);
        
        if (objc > 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.alpha) != TCL_OK) {
                throw std::runtime_error("Invalid alpha value");
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
            
            if (param == "-input1" || param == "-input") {
                args.input1 = value;
            } else if (param == "-input2" || param == "-other") {
                args.input2 = value;
            } else if (param == "-alpha") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.alpha) != TCL_OK) {
                    throw std::runtime_error("Invalid alpha value: " + value);
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Both input tensors are required");
    }
    
    return args;
}

int TensorAdd_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorAddArgs args = ParseTensorAddArgs(interp, objc, objv);
        
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
        // Perform addition while preserving tensor options from the first tensor
        torch::Tensor result = (tensor1 + args.alpha * tensor2).to(tensor1.options());
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_sub command
struct TensorSubArgs {
    std::string input;
    std::string other;
    double alpha = 1.0;
    
    bool IsValid() const {
        return !input.empty() && !other.empty();
    }
};

// Parse dual syntax for tensor_sub
TensorSubArgs ParseTensorSubArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorSubArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::tensor_sub tensor1 tensor2");
        }
        args.input = Tcl_GetString(objv[1]);
        args.other = Tcl_GetString(objv[2]);
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
            } else if (param == "-other") {
                args.other = value;
            } else if (param == "-alpha") {
                args.alpha = std::stod(value);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -input and -other");
    }
    
    return args;
}

int TensorSub_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorSubArgs args = ParseTensorSubArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid first tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.other) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid second tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor1 = tensor_storage[args.input];
        auto& tensor2 = tensor_storage[args.other];
        // Perform subtraction while preserving tensor options from the first tensor
        torch::Tensor result = (tensor1 - (args.alpha * tensor2)).to(tensor1.options());
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct TensorMulArgs {
    std::string input;
    std::string other;
    double scalar = 0.0;  // For scalar multiplication
    bool is_scalar = false;  // Flag to indicate scalar multiplication
    
    bool IsValid() const {
        return !input.empty() && (!other.empty() || is_scalar);
    }
};

TensorMulArgs ParseTensorMulArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorMulArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::tensor_mul tensor1 tensor2|scalar");
        }
        args.input = Tcl_GetString(objv[1]);
        
        // Try to parse second argument as scalar
        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.scalar) == TCL_OK) {
            args.is_scalar = true;
        } else {
            // If not a scalar, treat as tensor handle
            Tcl_ResetResult(interp);  // Clear error from failed double conversion
            args.other = Tcl_GetString(objv[2]);
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
            } else if (param == "-other") {
                // Try to parse as scalar first
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.scalar) == TCL_OK) {
                    args.is_scalar = true;
                } else {
                    // If not a scalar, treat as tensor handle
                    Tcl_ResetResult(interp);  // Clear error from failed double conversion
                    args.other = value;
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -input and -other");
    }
    
    return args;
}

int TensorMul_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorMulArgs args = ParseTensorMulArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid first tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor1 = tensor_storage[args.input];
        torch::Tensor result;
        
        if (args.is_scalar) {
            result = tensor1 * args.scalar;
        } else {
            if (tensor_storage.find(args.other) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>("Invalid second tensor name"), TCL_VOLATILE);
                return TCL_ERROR;
            }
            auto& tensor2 = tensor_storage[args.other];
            result = tensor1 * tensor2;
        }
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_div command
struct TensorDivArgs {
    std::string input;
    std::string other;
    
    bool IsValid() const {
        return !input.empty() && !other.empty();
    }
};

// Parse dual syntax for tensor_div
TensorDivArgs ParseTensorDivArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorDivArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::tensor_div tensor1 tensor2");
        }
        args.input = Tcl_GetString(objv[1]);
        args.other = Tcl_GetString(objv[2]);
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
            } else if (param == "-other") {
                args.other = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -input and -other");
    }
    
    return args;
}

int TensorDiv_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorDivArgs args = ParseTensorDivArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid first tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.other) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid second tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor1 = tensor_storage[args.input];
        auto& tensor2 = tensor_storage[args.other];
        // Perform division while preserving tensor options from the first tensor
        torch::Tensor result = (tensor1 / tensor2).to(tensor1.options());
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_matmul command
struct TensorMatmulArgs {
    std::string input;
    std::string other;
    
    bool IsValid() const {
        return !input.empty() && !other.empty();
    }
};

// Parse dual syntax for tensor_matmul
TensorMatmulArgs ParseTensorMatmulArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorMatmulArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::tensor_matmul tensor1 tensor2");
        }
        args.input = Tcl_GetString(objv[1]);
        args.other = Tcl_GetString(objv[2]);
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
            } else if (param == "-other") {
                args.other = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -input and -other");
    }
    
    return args;
}

int TensorMatmul_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorMatmulArgs args = ParseTensorMatmulArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid first tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.other) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid second tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor1 = tensor_storage[args.input];
        auto& tensor2 = tensor_storage[args.other];
        // Perform matrix multiplication while preserving tensor options from the first tensor
        torch::Tensor result = tensor1.matmul(tensor2).to(tensor1.options());
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct TensorBmmArgs {
    std::string input;
    std::string other;
    
    bool IsValid() const {
        return !input.empty() && !other.empty();
    }
};

TensorBmmArgs ParseTensorBmmArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorBmmArgs args;
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        args.input = Tcl_GetString(objv[1]);
        args.other = Tcl_GetString(objv[2]);
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
            } else if (param == "-other") {
                args.other = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and other");
    }
    
    return args;
}

int TensorBmm_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorBmmArgs args = ParseTensorBmmArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.other) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid other tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input_tensor = tensor_storage[args.input];
        auto& other_tensor = tensor_storage[args.other];
        
        // Perform batch matrix multiplication while preserving tensor options from the input tensor
        torch::Tensor result = torch::bmm(input_tensor, other_tensor).to(input_tensor.options());
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_to command
struct TensorToArgs {
    std::string input;
    std::string device;
    std::string dtype = "";  // Optional
    
    bool IsValid() const {
        return !input.empty() && !device.empty();
    }
};

// Parse dual syntax for tensor_to
TensorToArgs ParseTensorToArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorToArgs args;
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor device ?dtype?
        if (objc < 3 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor device ?dtype?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.device = Tcl_GetString(objv[2]);
        
        if (objc == 4) {
            args.dtype = Tcl_GetString(objv[3]);
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
            } else if (param == "-device") {
                args.device = value;
            } else if (param == "-dtype") {
                args.dtype = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and device");
    }
    
    return args;
}

int TensorTo_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorToArgs args = ParseTensorToArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Device device = GetDevice(args.device.c_str());
        torch::Tensor result = tensor.to(device);
        
        if (!args.dtype.empty()) {
            c10::ScalarType dtype = GetScalarType(args.dtype.c_str());
            result = result.to(dtype);
        }
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Tensor manipulation operations

struct TensorReshapeArgs {
    std::string input;
    std::vector<int64_t> shape;
    
    bool IsValid() const {
        return !input.empty() && !shape.empty();
    }
};

TensorReshapeArgs ParseTensorReshapeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorReshapeArgs args;
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::tensor_reshape tensor shape");
        }
        args.input = Tcl_GetString(objv[1]);
        args.shape = TclListToShape(interp, objv[2]);
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
            } else if (param == "-shape") {
                Tcl_Obj* shape_obj = Tcl_NewStringObj(value.c_str(), -1);
                args.shape = TclListToShape(interp, shape_obj);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and shape");
    }
    
    return args;
}

int TensorReshape_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorReshapeArgs args = ParseTensorReshapeArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        // Perform reshape while preserving all tensor options
        torch::Tensor result = tensor.reshape(args.shape).to(tensor.options());
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct TensorPermuteArgs {
    std::string input;
    std::vector<int64_t> dims;
    
    bool IsValid() const {
        return !input.empty() && !dims.empty();
    }
};

TensorPermuteArgs ParseTensorPermuteArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorPermuteArgs args;
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::tensor_permute tensor dims");
        }
        args.input = Tcl_GetString(objv[1]);
        args.dims = TclListToShape(interp, objv[2]);
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
            } else if (param == "-dims") {
                Tcl_Obj* dims_obj = Tcl_NewStringObj(value.c_str(), -1);
                args.dims = TclListToShape(interp, dims_obj);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and dims");
    }
    
    return args;
}

int TensorPermute_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorPermuteArgs args = ParseTensorPermuteArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        // Perform permute while preserving all tensor options
        torch::Tensor result = tensor.permute(args.dims).to(tensor.options());
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct TensorCatArgs {
    std::vector<std::string> tensors;
    int dim = 0;
    
    bool IsValid() const {
        return !tensors.empty() && tensors.size() >= 2;
    }
};

TensorCatArgs ParseTensorCatArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorCatArgs args;
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        // Parse tensor list
        int tensor_count;
        Tcl_ListObjLength(interp, objv[1], &tensor_count);
        
        for (int i = 0; i < tensor_count; i++) {
            Tcl_Obj* tensor_obj;
            Tcl_ListObjIndex(interp, objv[1], i, &tensor_obj);
            args.tensors.push_back(Tcl_GetString(tensor_obj));
        }
        
        // Parse dimension
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
            throw std::runtime_error("Invalid dimension parameter");
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-tensors") {
                // Parse tensor list
                Tcl_Obj* list_obj = Tcl_NewStringObj(value.c_str(), -1);
                int tensor_count;
                Tcl_ListObjLength(interp, list_obj, &tensor_count);
                
                for (int j = 0; j < tensor_count; j++) {
                    Tcl_Obj* tensor_obj;
                    Tcl_ListObjIndex(interp, list_obj, j, &tensor_obj);
                    args.tensors.push_back(Tcl_GetString(tensor_obj));
                }
            } else if (param == "-dim") {
                args.dim = std::stoi(value);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: at least 2 tensors and dimension");
    }
    
    return args;
}

int TensorCat_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorCatArgs args = ParseTensorCatArgs(interp, objc, objv);
        
        std::vector<torch::Tensor> tensors;
        for (const auto& tensor_name : args.tensors) {
            if (tensor_storage.find(tensor_name) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>(("Invalid tensor name: " + tensor_name).c_str()), TCL_VOLATILE);
                return TCL_ERROR;
            }
            tensors.push_back(tensor_storage[tensor_name]);
        }
        
        // Preserve tensor options from the first tensor in the list
        torch::Tensor result = torch::cat(tensors, args.dim);
        if (!tensors.empty()) {
            result = result.to(tensors[0].options());
        }
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct TensorStackArgs {
    std::vector<std::string> tensors;
    int dim = 0;
    
    bool IsValid() const {
        return !tensors.empty();
    }
};

TensorStackArgs ParseTensorStackArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorStackArgs args;
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::tensor_stack tensors dim");
        }
        int tensor_count;
        Tcl_ListObjLength(interp, objv[1], &tensor_count);
        for (int i = 0; i < tensor_count; i++) {
            Tcl_Obj* tensor_obj;
            Tcl_ListObjIndex(interp, objv[1], i, &tensor_obj);
            args.tensors.push_back(Tcl_GetString(tensor_obj));
        }
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
            throw std::runtime_error("Invalid dim parameter");
        }
    } else if (objc == 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax with missing dim parameter
        throw std::runtime_error("Usage: torch::tensor_stack tensors dim");
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            if (param == "-tensors") {
                Tcl_Obj* list_obj = Tcl_NewStringObj(value.c_str(), -1);
                int tensor_count;
                Tcl_ListObjLength(interp, list_obj, &tensor_count);
                for (int j = 0; j < tensor_count; j++) {
                    Tcl_Obj* tensor_obj;
                    Tcl_ListObjIndex(interp, list_obj, j, &tensor_obj);
                    args.tensors.push_back(Tcl_GetString(tensor_obj));
                }
            } else if (param == "-dim") {
                args.dim = std::stoi(value);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: tensors");
    }
    return args;
}

int TensorStack_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorStackArgs args = ParseTensorStackArgs(interp, objc, objv);
        std::vector<torch::Tensor> tensors;
        for (const auto& tensor_name : args.tensors) {
            if (tensor_storage.find(tensor_name) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>(("Invalid tensor name: " + tensor_name).c_str()), TCL_VOLATILE);
                return TCL_ERROR;
            }
            tensors.push_back(tensor_storage[tensor_name]);
        }
        // Preserve tensor options from the first tensor in the list
        torch::Tensor result = torch::stack(tensors, args.dim);
        if (!tensors.empty()) {
            result = result.to(tensors[0].options());
        }
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct TensorShapeArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

TensorShapeArgs ParseTensorShapeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorShapeArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::tensor_shape tensor");
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

int TensorShape_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorShapeArgs args = ParseTensorShapeArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>(("Invalid tensor name: " + args.input).c_str()), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        auto sizes = tensor.sizes();
        
        Tcl_Obj* shape_list = Tcl_NewListObj(0, NULL);
        for (int64_t size : sizes) {
            Tcl_ListObjAppendElement(interp, shape_list, Tcl_NewLongObj(size));
        }
        
        Tcl_SetObjResult(interp, shape_list);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 