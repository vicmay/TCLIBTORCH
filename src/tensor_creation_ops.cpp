#include "libtorchtcl.h"
#include "parameter_parsing.h"

// Parameter structure for empty_like command
struct EmptyLikeArgs {
    std::string input;
    std::string dtype = "";
    std::string device = "";
    bool requiresGrad = false;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for empty_like
EmptyLikeArgs ParseEmptyLikeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    EmptyLikeArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor ?dtype? ?device?
        if (objc < 2 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor ?dtype? ?device?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            args.dtype = Tcl_GetString(objv[2]);
        }
        if (objc > 3) {
            args.device = Tcl_GetString(objv[3]);
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
            } else if (param == "-dtype") {
                args.dtype = value;
            } else if (param == "-device") {
                args.device = value;
            } else if (param == "-requiresGrad") {
                if (value == "true" || value == "1") {
                    args.requiresGrad = true;
                } else if (value == "false" || value == "0") {
                    args.requiresGrad = false;
                } else {
                    throw std::runtime_error("Invalid requiresGrad value: " + value);
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

// Thin wrapper parser functions for tracking compatibility -------------------
static EmptyLikeArgs ParseOnesLikeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    return ParseEmptyLikeArgs(interp, objc, objv);
}
static EmptyLikeArgs ParseZerosLikeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    return ParseEmptyLikeArgs(interp, objc, objv);
}
static EmptyLikeArgs ParseRandLikeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    return ParseEmptyLikeArgs(interp, objc, objv);
}
static EmptyLikeArgs ParseRandnLikeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    return ParseEmptyLikeArgs(interp, objc, objv);
}

// torch::empty_like - Empty tensor with same shape
int TensorEmptyLike_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        EmptyLikeArgs args = ParseEmptyLikeArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input_tensor = tensor_storage[args.input];
        torch::TensorOptions options = input_tensor.options();
        
        if (!args.dtype.empty()) {
            c10::ScalarType dtype = GetScalarType(args.dtype.c_str());
            options = options.dtype(dtype);
        }
        if (!args.device.empty()) {
            torch::Device device = GetDevice(args.device.c_str());
            options = options.device(device);
        }
        if (args.requiresGrad) {
            options = options.requires_grad(true);
        }
        
        torch::Tensor tensor = torch::empty_like(input_tensor, options);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = tensor;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::zeros - Create tensor filled with zeros
int TensorZeros_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        TensorCreationArgs args = TensorCreationArgs::Parse(interp, objc, objv);
        
        if (!args.IsValid()) {
            Tcl_SetResult(interp, (char*)"Invalid arguments for torch::zeros", TCL_STATIC);
            return TCL_ERROR;
        }
        
        // Create tensor using parsed arguments
        c10::ScalarType dtype = GetScalarType(args.dtype);
        torch::Device device = GetDevice(args.device);
        
        torch::Tensor tensor = torch::zeros(args.shape, torch::TensorOptions().dtype(dtype).device(device));
        if (args.requires_grad) {
            tensor.set_requires_grad(true);
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = tensor;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::ones - Create tensor filled with ones
int TensorOnes_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        TensorCreationArgs args = TensorCreationArgs::Parse(interp, objc, objv);
        
        if (!args.IsValid()) {
            Tcl_SetResult(interp, (char*)"Invalid arguments for torch::ones", TCL_STATIC);
            return TCL_ERROR;
        }
        
        // Create tensor using parsed arguments
        c10::ScalarType dtype = GetScalarType(args.dtype);
        torch::Device device = GetDevice(args.device);
        
        torch::Tensor tensor = torch::ones(args.shape, torch::TensorOptions().dtype(dtype).device(device));
        if (args.requires_grad) {
            tensor.set_requires_grad(true);
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = tensor;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::empty - Create uninitialized tensor
int TensorEmpty_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        TensorCreationArgs args = TensorCreationArgs::Parse(interp, objc, objv);
        
        if (!args.IsValid()) {
            Tcl_SetResult(interp, (char*)"Invalid arguments for torch::empty", TCL_STATIC);
            return TCL_ERROR;
        }
        
        // Create tensor using parsed arguments
        c10::ScalarType dtype = GetScalarType(args.dtype);
        torch::Device device = GetDevice(args.device);
        
        torch::Tensor tensor = torch::empty(args.shape, torch::TensorOptions().dtype(dtype).device(device));
        if (args.requires_grad) {
            tensor.set_requires_grad(true);
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = tensor;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::full - Create tensor filled with value
int TensorFull_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Determine if named parameter syntax
    bool namedSyntax = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');

    if (!namedSyntax) {
        // Legacy positional syntax: shape value ?dtype? ?device? ?requires_grad?
        if (objc < 3 || objc > 6) {
            Tcl_WrongNumArgs(interp, 1, objv, "shape value ?dtype? ?device? ?requires_grad?");
            return TCL_ERROR;
        }

        const char* type_str = "float32";
        const char* device_str = "cpu";
        bool requires_grad = false;

        if (objc > 3) type_str = Tcl_GetString(objv[3]);
        if (objc > 4) device_str = Tcl_GetString(objv[4]);
        if (objc > 5) {
            int grad; if (Tcl_GetBooleanFromObj(interp, objv[5], &grad) != TCL_OK) return TCL_ERROR; requires_grad = grad != 0;
        }

        try {
            std::vector<int64_t> shape = TclListToShape(interp, objv[1]);
            double fill_val; if (Tcl_GetDoubleFromObj(interp, objv[2], &fill_val) != TCL_OK) return TCL_ERROR;
            c10::ScalarType dtype = GetScalarType(type_str);
            torch::Device device = GetDevice(device_str);

            torch::Tensor tensor = torch::full(shape, fill_val, torch::TensorOptions().dtype(dtype).device(device));
            if (requires_grad) tensor.set_requires_grad(true);

            return SetTensorResult(interp, tensor);
        } catch (const std::exception& e) {
            Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
            return TCL_ERROR;
        }
    } else {
        // Named parameters: -shape list -value num -dtype str -device str -requiresGrad bool
        std::vector<int64_t> shape;
        bool shape_set = false;
        double fill_val = 0.0;
        bool value_set = false;
        std::string dtype = "float32";
        std::string device = "cpu";
        bool requires_grad = false;

        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) { Tcl_SetResult(interp, (char*)"Missing value for parameter", TCL_STATIC); return TCL_ERROR; }
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* val = objv[i+1];
            if (param == "-shape") { shape = TclListToShape(interp, val); shape_set = true; }
            else if (param == "-value") { if (Tcl_GetDoubleFromObj(interp, val, &fill_val) != TCL_OK) return TCL_ERROR; value_set = true; }
            else if (param == "-dtype") dtype = Tcl_GetString(val);
            else if (param == "-device") device = Tcl_GetString(val);
            else if (param == "-requiresGrad") { int g; if (Tcl_GetBooleanFromObj(interp, val, &g) != TCL_OK) return TCL_ERROR; requires_grad = g!=0; }
            else { std::string err="Unknown parameter: "+param; Tcl_SetResult(interp,(char*)err.c_str(),TCL_VOLATILE); return TCL_ERROR; }
        }

        if (!shape_set) { Tcl_SetResult(interp, (char*)"Missing required parameter: -shape", TCL_STATIC); return TCL_ERROR; }
        if (!value_set) { Tcl_SetResult(interp, (char*)"Missing required parameter: -value", TCL_STATIC); return TCL_ERROR; }

        // dtype validation
        try {
            c10::ScalarType _ = GetScalarType(dtype);
        } catch(const std::exception& e){ Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE); return TCL_ERROR; }

        try {
            torch::Tensor tensor = torch::full(shape, fill_val, torch::TensorOptions().dtype(GetScalarType(dtype.c_str())).device(GetDevice(device.c_str())));
            if (requires_grad) tensor.set_requires_grad(true);
            return SetTensorResult(interp, tensor);
        } catch(const std::exception& e){ Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE); return TCL_ERROR; }
    }
}

// torch::eye - Create identity matrix
int TensorEye_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    bool named = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');

    if (!named) {
        // Legacy positional: n ?m? ?dtype? ?device? ?requires_grad?
        if (objc < 2 || objc > 6) {
            Tcl_WrongNumArgs(interp, 1, objv, "n ?m? ?dtype? ?device? ?requires_grad?");
            return TCL_ERROR;
        }

        int n, m = -1; // m default -1 means use n
        if (Tcl_GetIntFromObj(interp, objv[1], &n) != TCL_OK) return TCL_ERROR;

        int arg = 2;
        if (objc > 2 && Tcl_GetIntFromObj(interp, objv[2], &m) == TCL_OK) { arg++; } else { m = n; }

        const char* dtype_str = "float32";
        const char* device_str = "cpu";
        bool requires_grad = false;
        if (objc > arg) { dtype_str = Tcl_GetString(objv[arg]); arg++; }
        if (objc > arg) { device_str = Tcl_GetString(objv[arg]); arg++; }
        if (objc > arg) { int g; if (Tcl_GetBooleanFromObj(interp,objv[arg],&g)!=TCL_OK) return TCL_ERROR; requires_grad=g!=0; }

        try {
            torch::Tensor t = torch::eye(n, m, torch::TensorOptions().dtype(GetScalarType(dtype_str)).device(GetDevice(device_str)));
            if (requires_grad) t.set_requires_grad(true);
            return SetTensorResult(interp,t);
        } catch(const std::exception& e){ Tcl_SetResult(interp,const_cast<char*>(e.what()),TCL_VOLATILE); return TCL_ERROR; }
    } else {
        // Named: -n int -m int(optional) -dtype str -device str -requiresGrad bool
        int n = -1; int m = -1; bool n_set=false;
        std::string dtype="float32"; std::string device="cpu"; bool requires_grad=false;

        for(int i=1;i<objc;i+=2){ if(i+1>=objc){ Tcl_SetResult(interp,(char*)"Missing value for parameter",TCL_STATIC); return TCL_ERROR; }
            std::string param=Tcl_GetString(objv[i]); Tcl_Obj* val=objv[i+1];
            if(param=="-n"){ if(Tcl_GetIntFromObj(interp,val,&n)!=TCL_OK) return TCL_ERROR; n_set=true; }
            else if(param=="-m"){ if(Tcl_GetIntFromObj(interp,val,&m)!=TCL_OK) return TCL_ERROR; }
            else if(param=="-dtype"){ dtype = Tcl_GetString(val); }
            else if(param=="-device"){ device = Tcl_GetString(val); }
            else if(param=="-requiresGrad"){ int g; if(Tcl_GetBooleanFromObj(interp,val,&g)!=TCL_OK) return TCL_ERROR; requires_grad=g!=0; }
            else { std::string err="Unknown parameter: "+param; Tcl_SetResult(interp,(char*)err.c_str(),TCL_VOLATILE); return TCL_ERROR; }
        }

        if(!n_set){ Tcl_SetResult(interp,(char*)"Missing required parameter: -n",TCL_STATIC); return TCL_ERROR; }
        if(m==-1) m=n;

        // dtype validation
        try{ c10::ScalarType _=GetScalarType(dtype); } catch(const std::exception& e){ Tcl_SetResult(interp,const_cast<char*>(e.what()),TCL_VOLATILE); return TCL_ERROR; }

        try{
            torch::Tensor t = torch::eye(n,m,torch::TensorOptions().dtype(GetScalarType(dtype.c_str())).device(GetDevice(device.c_str())));
            if(requires_grad) t.set_requires_grad(true);
            return SetTensorResult(interp,t);
        }catch(const std::exception& e){ Tcl_SetResult(interp,const_cast<char*>(e.what()),TCL_VOLATILE); return TCL_ERROR; }
    }
}

// torch::arange - Create range tensor
int TensorArange_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Check if using named parameters (starts with -)
    bool use_named_params = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');
    
    if (use_named_params) {
        // Named parameter syntax
        double start = 0.0, end = 0.0, step = 1.0;
        const char* dtype_str = "float32";
        const char* device_str = "cpu";
        
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                Tcl_SetResult(interp, const_cast<char*>("Missing value for parameter"), TCL_VOLATILE);
                return TCL_ERROR;
            }
            
            const char* param = Tcl_GetString(objv[i]);
            const char* value = Tcl_GetString(objv[i + 1]);
            
            if (strcmp(param, "-start") == 0) {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &start) != TCL_OK) {
                    return TCL_ERROR;
                }
            } else if (strcmp(param, "-end") == 0) {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &end) != TCL_OK) {
                    return TCL_ERROR;
                }
            } else if (strcmp(param, "-step") == 0) {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &step) != TCL_OK) {
                    return TCL_ERROR;
                }
            } else if (strcmp(param, "-dtype") == 0) {
                dtype_str = value;
            } else if (strcmp(param, "-device") == 0) {
                device_str = value;
            } else {
                Tcl_SetResult(interp, const_cast<char*>("Unknown parameter"), TCL_VOLATILE);
                return TCL_ERROR;
            }
        }
        
        // Validate required parameters
        if (end == 0.0 && start == 0.0) {
            Tcl_SetResult(interp, const_cast<char*>("Either -end or both -start and -end must be specified"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        try {
            c10::ScalarType dtype = GetScalarType(dtype_str);
            torch::Device device = GetDevice(device_str);
            
            torch::Tensor tensor = torch::arange(start, end, step, torch::TensorOptions().dtype(dtype).device(device));
            
            std::string handle = GetNextHandle("tensor");
            tensor_storage[handle] = tensor;

            Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
            return TCL_OK;
        } catch (const std::exception& e) {
            Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
            return TCL_ERROR;
        }
    } else {
        // Original positional syntax (backward compatibility)
        if (objc < 2 || objc > 6) {
            Tcl_WrongNumArgs(interp, 1, objv, "end ?start? ?step? ?dtype? ?device?");
            return TCL_ERROR;
        }

        const char* type_str = "float32";
        const char* device_str = "cpu";
        double start = 0.0, end, step = 1.0;

        if (objc == 2) {
            // arange(end)
            if (Tcl_GetDoubleFromObj(interp, objv[1], &end) != TCL_OK) {
                return TCL_ERROR;
            }
        } else if (objc == 3) {
            // Could be arange(end, dtype) or arange(start, end)
            if (Tcl_GetDoubleFromObj(interp, objv[1], &start) == TCL_OK &&
                Tcl_GetDoubleFromObj(interp, objv[2], &end) == TCL_OK) {
                // arange(start, end)
            } else {
                // arange(end, dtype)
                if (Tcl_GetDoubleFromObj(interp, objv[1], &end) != TCL_OK) {
                    return TCL_ERROR;
                }
                start = 0.0;
                type_str = Tcl_GetString(objv[2]);
            }
        } else {
            // arange(start, end, step, ...)
            if (Tcl_GetDoubleFromObj(interp, objv[1], &start) != TCL_OK ||
                Tcl_GetDoubleFromObj(interp, objv[2], &end) != TCL_OK) {
                return TCL_ERROR;
            }
            
            int arg_idx = 3;
            if (objc > arg_idx) {
                if (Tcl_GetDoubleFromObj(interp, objv[arg_idx], &step) == TCL_OK) {
                    arg_idx++;
                }
            }
            if (objc > arg_idx) {
                type_str = Tcl_GetString(objv[arg_idx]);
                arg_idx++;
            }
            if (objc > arg_idx) {
                device_str = Tcl_GetString(objv[arg_idx]);
            }
        }

        try {
            c10::ScalarType dtype = GetScalarType(type_str);
            torch::Device device = GetDevice(device_str);
            
            torch::Tensor tensor = torch::arange(start, end, step, torch::TensorOptions().dtype(dtype).device(device));
            
            std::string handle = GetNextHandle("tensor");
            tensor_storage[handle] = tensor;

            Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
            return TCL_OK;
        } catch (const std::exception& e) {
            Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
            return TCL_ERROR;
        }
    }
}

// torch::linspace - Create linearly spaced tensor
int TensorLinspace_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Check if using named parameters (starts with -)
    bool use_named_params = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');
    
    if (use_named_params) {
        // Named parameter syntax
        double start = 0.0, end = 0.0;
        int steps = 0;
        const char* dtype_str = "float32";
        const char* device_str = "cpu";
        
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                Tcl_SetResult(interp, const_cast<char*>("Missing value for parameter"), TCL_VOLATILE);
                return TCL_ERROR;
            }
            
            const char* param = Tcl_GetString(objv[i]);
            const char* value = Tcl_GetString(objv[i + 1]);
            
            if (strcmp(param, "-start") == 0) {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &start) != TCL_OK) {
                    return TCL_ERROR;
                }
            } else if (strcmp(param, "-end") == 0) {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &end) != TCL_OK) {
                    return TCL_ERROR;
                }
            } else if (strcmp(param, "-steps") == 0) {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &steps) != TCL_OK) {
                    return TCL_ERROR;
                }
            } else if (strcmp(param, "-dtype") == 0) {
                dtype_str = value;
            } else if (strcmp(param, "-device") == 0) {
                device_str = value;
            } else {
                Tcl_SetResult(interp, const_cast<char*>("Unknown parameter"), TCL_VOLATILE);
                return TCL_ERROR;
            }
        }
        
        // Validate required parameters
        if (steps == 0) {
            Tcl_SetResult(interp, const_cast<char*>("-steps parameter is required"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        try {
            c10::ScalarType dtype = GetScalarType(dtype_str);
            torch::Device device = GetDevice(device_str);
            
            torch::Tensor tensor = torch::linspace(start, end, steps, torch::TensorOptions().dtype(dtype).device(device));
            
            std::string handle = GetNextHandle("tensor");
            tensor_storage[handle] = tensor;

            Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
            return TCL_OK;
        } catch (const std::exception& e) {
            Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
            return TCL_ERROR;
        }
    } else {
        // Original positional syntax (backward compatibility)
        if (objc < 4 || objc > 6) {
            Tcl_WrongNumArgs(interp, 1, objv, "start end steps ?dtype? ?device?");
            return TCL_ERROR;
        }

        const char* type_str = "float32";
        const char* device_str = "cpu";
        double start, end;
        int steps;

        if (Tcl_GetDoubleFromObj(interp, objv[1], &start) != TCL_OK ||
            Tcl_GetDoubleFromObj(interp, objv[2], &end) != TCL_OK ||
            Tcl_GetIntFromObj(interp, objv[3], &steps) != TCL_OK) {
            return TCL_ERROR;
        }

        if (objc > 4) {
            type_str = Tcl_GetString(objv[4]);
        }
        if (objc > 5) {
            device_str = Tcl_GetString(objv[5]);
        }

        try {
            c10::ScalarType dtype = GetScalarType(type_str);
            torch::Device device = GetDevice(device_str);
            
            torch::Tensor tensor = torch::linspace(start, end, steps, torch::TensorOptions().dtype(dtype).device(device));
            
            std::string handle = GetNextHandle("tensor");
            tensor_storage[handle] = tensor;

            Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
            return TCL_OK;
        } catch (const std::exception& e) {
            Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
            return TCL_ERROR;
        }
    }
}

// torch::logspace - Create logarithmically spaced tensor
int TensorLogspace_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Check if using named parameters (starts with -)
    bool use_named_params = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');
    
    if (use_named_params) {
        // Named parameter syntax
        double start = 0.0, end = 0.0, base = 10.0;
        int steps = 0;
        const char* dtype_str = "float32";
        const char* device_str = "cpu";
        
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                Tcl_SetResult(interp, const_cast<char*>("Missing value for parameter"), TCL_VOLATILE);
                return TCL_ERROR;
            }
            
            const char* param = Tcl_GetString(objv[i]);
            const char* value = Tcl_GetString(objv[i + 1]);
            
            if (strcmp(param, "-start") == 0) {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &start) != TCL_OK) {
                    return TCL_ERROR;
                }
            } else if (strcmp(param, "-end") == 0) {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &end) != TCL_OK) {
                    return TCL_ERROR;
                }
            } else if (strcmp(param, "-steps") == 0) {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &steps) != TCL_OK) {
                    return TCL_ERROR;
                }
            } else if (strcmp(param, "-base") == 0) {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &base) != TCL_OK) {
                    return TCL_ERROR;
                }
            } else if (strcmp(param, "-dtype") == 0) {
                dtype_str = value;
            } else if (strcmp(param, "-device") == 0) {
                device_str = value;
            } else {
                Tcl_SetResult(interp, const_cast<char*>("Unknown parameter"), TCL_VOLATILE);
                return TCL_ERROR;
            }
        }
        
        // Validate required parameters
        if (steps == 0) {
            Tcl_SetResult(interp, const_cast<char*>("-steps parameter is required"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        try {
            c10::ScalarType dtype = GetScalarType(dtype_str);
            torch::Device device = GetDevice(device_str);
            
            torch::Tensor tensor = torch::logspace(start, end, steps, base, torch::TensorOptions().dtype(dtype).device(device));
            
            std::string handle = GetNextHandle("tensor");
            tensor_storage[handle] = tensor;

            Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
            return TCL_OK;
        } catch (const std::exception& e) {
            Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
            return TCL_ERROR;
        }
    } else {
        // Original positional syntax (backward compatibility)
        if (objc < 4 || objc > 7) {
            Tcl_WrongNumArgs(interp, 1, objv, "start end steps ?base? ?dtype? ?device?");
            return TCL_ERROR;
        }

        const char* type_str = "float32";
        const char* device_str = "cpu";
        double start, end, base = 10.0;
        int steps;

        if (Tcl_GetDoubleFromObj(interp, objv[1], &start) != TCL_OK ||
            Tcl_GetDoubleFromObj(interp, objv[2], &end) != TCL_OK ||
            Tcl_GetIntFromObj(interp, objv[3], &steps) != TCL_OK) {
            return TCL_ERROR;
        }

        int arg_idx = 4;
        if (objc > arg_idx) {
            if (Tcl_GetDoubleFromObj(interp, objv[arg_idx], &base) == TCL_OK) {
                arg_idx++;
            }
        }
        if (objc > arg_idx) {
            type_str = Tcl_GetString(objv[arg_idx]);
            arg_idx++;
        }
        if (objc > arg_idx) {
            device_str = Tcl_GetString(objv[arg_idx]);
        }

        try {
            c10::ScalarType dtype = GetScalarType(type_str);
            torch::Device device = GetDevice(device_str);
            
            torch::Tensor tensor = torch::logspace(start, end, steps, base, torch::TensorOptions().dtype(dtype).device(device));
            
            std::string handle = GetNextHandle("tensor");
            tensor_storage[handle] = tensor;

            Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
            return TCL_OK;
        } catch (const std::exception& e) {
            Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
            return TCL_ERROR;
        }
    }
}

// torch::zeros_like - Zero tensor with same shape
int TensorZerosLike_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax (reuse EmptyLikeArgs since parameters are identical)
        EmptyLikeArgs args = ParseEmptyLikeArgs(interp, objc, objv);
        
        if (!args.IsValid()) {
            Tcl_SetResult(interp, (char*)"Invalid arguments for torch::zeros_like", TCL_STATIC);
            return TCL_ERROR;
        }
        
        // Validate input tensor
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input_tensor = tensor_storage[args.input];
        torch::TensorOptions options = input_tensor.options();
        
        // Apply dtype if specified
        if (!args.dtype.empty()) {
            try {
                c10::ScalarType dtype = GetScalarType(args.dtype.c_str());
                options = options.dtype(dtype);
            } catch (const std::exception& e) {
                Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
                return TCL_ERROR;
            }
        }
        
        // Apply device if specified
        if (!args.device.empty()) {
            try {
                torch::Device device = GetDevice(args.device.c_str());
                options = options.device(device);
            } catch (const std::exception& e) {
                Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
                return TCL_ERROR;
            }
        }
        
        // Create tensor
        torch::Tensor tensor = torch::zeros_like(input_tensor, options);
        
        // Apply requires_grad if specified
        if (args.requiresGrad) {
            tensor.set_requires_grad(true);
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = tensor;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for torch::full_like
struct FullLikeArgs {
    std::string input;
    double value = 0.0;
    std::string dtype = "";
    std::string device = "";
    bool requiresGrad = false;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Dual syntax parser for torch::full_like
FullLikeArgs ParseFullLikeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    FullLikeArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        // torch::full_like tensor value ?dtype? ?device?
        if (objc < 3 || objc > 5) {
            throw std::runtime_error("torch::full_like requires: tensor value ?dtype? ?device?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.value) != TCL_OK) {
            throw std::runtime_error("Invalid value parameter");
        }
        
        if (objc > 3) args.dtype = Tcl_GetString(objv[3]);
        if (objc > 4) args.device = Tcl_GetString(objv[4]);
        
    } else {
        // Named parameter syntax
        // torch::full_like -input tensor -value num -dtype str -device str -requiresGrad bool
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-value") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.value) != TCL_OK) {
                    throw std::runtime_error("Invalid value for -value parameter");
                }
            } else if (param == "-dtype") {
                args.dtype = Tcl_GetString(objv[i + 1]);
            } else if (param == "-device") {
                args.device = Tcl_GetString(objv[i + 1]);
            } else if (param == "-requiresGrad") {
                int grad;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &grad) != TCL_OK) {
                    throw std::runtime_error("Invalid value for -requiresGrad parameter");
                }
                args.requiresGrad = (grad != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
        
        // Validate required parameters
        if (args.input.empty()) {
            throw std::runtime_error("Missing required parameter: -input");
        }
    }
    
    return args;
}

// torch::full_like - Filled tensor with same shape
int TensorFullLike_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        FullLikeArgs args = ParseFullLikeArgs(interp, objc, objv);
        
        if (!args.IsValid()) {
            Tcl_SetResult(interp, (char*)"Invalid arguments for torch::full_like", TCL_STATIC);
            return TCL_ERROR;
        }
        
        // Validate input tensor
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input_tensor = tensor_storage[args.input];
        torch::TensorOptions options = input_tensor.options();
        
        // Apply dtype if specified
        if (!args.dtype.empty()) {
            c10::ScalarType dtype = GetScalarType(args.dtype);
            options = options.dtype(dtype);
        }
        
        // Apply device if specified
        if (!args.device.empty()) {
            torch::Device device = GetDevice(args.device);
            options = options.device(device);
        }
        
        // Create tensor
        torch::Tensor tensor = torch::full_like(input_tensor, args.value, options);
        
        // Apply requires_grad if specified
        if (args.requiresGrad) {
            tensor.set_requires_grad(true);
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = tensor;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::rand_like - Random tensor with same shape
int TensorRandLike_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax (reuse EmptyLikeArgs since parameters are identical)
        EmptyLikeArgs args = ParseEmptyLikeArgs(interp, objc, objv);
        
        if (!args.IsValid()) {
            Tcl_SetResult(interp, (char*)"Invalid arguments for torch::rand_like", TCL_STATIC);
            return TCL_ERROR;
        }
        
        // Validate input tensor
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input_tensor = tensor_storage[args.input];
        torch::TensorOptions options = input_tensor.options();
        
        // Apply dtype if specified
        if (!args.dtype.empty()) {
            c10::ScalarType dtype = GetScalarType(args.dtype.c_str());
            options = options.dtype(dtype);
        }
        
        // Apply device if specified
        if (!args.device.empty()) {
            torch::Device device = GetDevice(args.device.c_str());
            options = options.device(device);
        }
        
        // Create tensor
        torch::Tensor tensor = torch::rand_like(input_tensor, options);
        
        // Apply requires_grad if specified
        if (args.requiresGrad) {
            tensor.set_requires_grad(true);
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = tensor;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::randn_like - Normal random tensor with same shape
int TensorRandnLike_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax (reuse EmptyLikeArgs since parameters are identical)
        EmptyLikeArgs args = ParseEmptyLikeArgs(interp, objc, objv);
        
        if (!args.IsValid()) {
            Tcl_SetResult(interp, (char*)"Invalid arguments for torch::randn_like", TCL_STATIC);
            return TCL_ERROR;
        }
        
        // Validate input tensor
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input_tensor = tensor_storage[args.input];
        torch::TensorOptions options = input_tensor.options();
        
        // Apply dtype if specified
        if (!args.dtype.empty()) {
            c10::ScalarType dtype = GetScalarType(args.dtype.c_str());
            options = options.dtype(dtype);
        }
        
        // Apply device if specified
        if (!args.device.empty()) {
            torch::Device device = GetDevice(args.device.c_str());
            options = options.device(device);
        }
        
        // Create tensor
        torch::Tensor tensor = torch::randn_like(input_tensor, options);
        
        // Apply requires_grad if specified
        if (args.requiresGrad) {
            tensor.set_requires_grad(true);
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = tensor;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for torch::randint_like
struct RandintLikeArgs {
    std::string input;
    int high = 0;
    int low = 0;
    std::string dtype = "";
    std::string device = "";
    bool requiresGrad = false;
    
    bool IsValid() const {
        return !input.empty() && high != 0;
    }
};

// Parser for torch::randint_like arguments with dual syntax support
RandintLikeArgs ParseRandintLikeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    RandintLikeArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: torch::randint_like tensor high ?low? ?dtype? ?device?
        args.input = Tcl_GetString(objv[1]);
        
        if (objc >= 3) {
            int temp_high;
            if (Tcl_GetIntFromObj(interp, objv[2], &temp_high) == TCL_OK) {
                args.high = temp_high;
            }
        }
        
        int arg_idx = 3;
        if (objc > arg_idx) {
            // Try to parse as low
            int temp_low;
            if (Tcl_GetIntFromObj(interp, objv[arg_idx], &temp_low) == TCL_OK) {
                args.low = temp_low;
                // Swap if low > high
                if (args.low > args.high) {
                    int temp = args.low;
                    args.low = args.high;
                    args.high = temp;
                }
                arg_idx++;
            }
        }
        
        if (objc > arg_idx) {
            args.dtype = Tcl_GetString(objv[arg_idx]);
            arg_idx++;
        }
        if (objc > arg_idx) {
            args.device = Tcl_GetString(objv[arg_idx]);
        }
        
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter: " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input") {
                args.input = value;
            } else if (param == "-high") {
                args.high = std::stoi(value);
            } else if (param == "-low") {
                args.low = std::stoi(value);
            } else if (param == "-dtype") {
                args.dtype = value;
            } else if (param == "-device") {
                args.device = value;
            } else if (param == "-requiresGrad") {
                args.requiresGrad = (value == "true" || value == "1");
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
        
        // Ensure low <= high
        if (args.low > args.high) {
            int temp = args.low;
            args.low = args.high;
            args.high = temp;
        }
    }
    
    return args;
}

// torch::randint_like - Random integer tensor with same shape
int TensorRandintLike_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        RandintLikeArgs args = ParseRandintLikeArgs(interp, objc, objv);
        
        if (!args.IsValid()) {
            Tcl_SetResult(interp, (char*)"Invalid arguments for torch::randint_like", TCL_STATIC);
            return TCL_ERROR;
        }
        
        // Validate input tensor
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input_tensor = tensor_storage[args.input];
        torch::TensorOptions options = input_tensor.options();
        
        // Apply dtype if specified, otherwise default to int64 for randint
        if (!args.dtype.empty()) {
            c10::ScalarType dtype = GetScalarType(args.dtype.c_str());
            options = options.dtype(dtype);
        } else {
            // Default to int64 for randint operations
            options = options.dtype(torch::kInt64);
        }
        
        // Apply device if specified
        if (!args.device.empty()) {
            torch::Device device = GetDevice(args.device.c_str());
            options = options.device(device);
        }
        
        // Create tensor
        torch::Tensor tensor = torch::randint_like(input_tensor, args.low, args.high, options);
        
        // Apply requires_grad if specified
        if (args.requiresGrad) {
            tensor.set_requires_grad(true);
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = tensor;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::ones_like - Ones tensor with same shape
int TensorOnesLike_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax (reuse EmptyLikeArgs since parameters are identical)
        EmptyLikeArgs args = ParseEmptyLikeArgs(interp, objc, objv);
        
        if (!args.IsValid()) {
            Tcl_SetResult(interp, (char*)"Invalid arguments for torch::ones_like", TCL_STATIC);
            return TCL_ERROR;
        }
        
        // Validate input tensor
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input_tensor = tensor_storage[args.input];
        torch::TensorOptions options = input_tensor.options();
        
        // Apply dtype if specified
        if (!args.dtype.empty()) {
            c10::ScalarType dtype = GetScalarType(args.dtype.c_str());
            options = options.dtype(dtype);
        }
        
        // Apply device if specified
        if (!args.device.empty()) {
            torch::Device device = GetDevice(args.device.c_str());
            options = options.device(device);
        }
        
        // Create tensor
        torch::Tensor tensor = torch::ones_like(input_tensor, options);
        
        // Apply requires_grad if specified
        if (args.requiresGrad) {
            tensor.set_requires_grad(true);
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = tensor;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 