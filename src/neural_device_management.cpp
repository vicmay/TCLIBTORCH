#include "libtorchtcl.h"

// Helper function to move a module to a specific device
static bool MoveModuleToDevice(std::shared_ptr<torch::nn::Module> module, const torch::Device& device) {
    try {
        module->to(device);
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

// Helper function to get the device of a module's first parameter
static torch::Device GetModuleDevice(std::shared_ptr<torch::nn::Module> module) {
    for (const auto& param : module->parameters()) {
        return param.device();
    }
    // If no parameters, return CPU as default
    return torch::kCPU;
}

// Parameter structure for layer_cpu command
struct LayerCpuArgs {
    std::string layer;
    
    bool IsValid() const {
        return !layer.empty();
    }
};

// Parse dual syntax for layer_cpu
LayerCpuArgs ParseLayerCpuArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LayerCpuArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): layer
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::layer_cpu layer");
        }
        
        args.layer = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-layer" || param == "-input") {
                args.layer = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: layer");
    }
    
    return args;
}

// Parameter structure for layer_cuda command
struct LayerCudaArgs {
    std::string layer;
    
    bool IsValid() const {
        return !layer.empty();
    }
};

// Parse dual syntax for layer_cuda
LayerCudaArgs ParseLayerCudaArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LayerCudaArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): layer
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::layer_cuda layer");
        }
        
        args.layer = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-layer" || param == "-input") {
                args.layer = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: layer");
    }
    
    return args;
}

// Parameter structure for layer_device command
struct LayerDeviceArgs {
    std::string layer;
    
    bool IsValid() const {
        return !layer.empty();
    }
};

// Parse dual syntax for layer_device
LayerDeviceArgs ParseLayerDeviceArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LayerDeviceArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): layer
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::layer_device layer");
        }
        
        args.layer = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-layer" || param == "-input") {
                args.layer = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: layer");
    }
    
    return args;
}

// Parameter structure for layer_to command
struct LayerToArgs {
    std::string layer;
    std::string device;
    
    bool IsValid() const {
        return !layer.empty() && !device.empty();
    }
};

// Parse dual syntax for layer_to command
LayerToArgs ParseLayerToArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LayerToArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::layer_to layer device");
        }
        args.layer = Tcl_GetString(objv[1]);
        args.device = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-layer") {
                args.layer = value;
            } else if (param == "-device") {
                args.device = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -layer and -device");
    }
    
    return args;
}

// Parameter structure for parameters_to command
struct ParametersToArgs {
    std::string parameters;  // parameter list (list of tensor names)
    std::string device = "cpu";  // target device
    
    bool IsValid() const {
        return !parameters.empty() && (device == "cpu" || device == "cuda");
    }
};

// Parse dual syntax for parameters_to
ParametersToArgs ParseParametersToArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ParametersToArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): parameters ?device?
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("Usage: torch::parameters_to parameters ?device?");
        }
        
        args.parameters = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            args.device = Tcl_GetString(objv[2]);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-parameters" || param == "-params") {
                args.parameters = Tcl_GetString(objv[i + 1]);
            } else if (param == "-device") {
                args.device = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid (parameters required, device must be 'cpu' or 'cuda')");
    }
    
    return args;
}

// torch::layer_to(layer, device) - Move layer to specific device
// New syntax: torch::layerTo -layer layername -device device
int LayerTo_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        LayerToArgs args = ParseLayerToArgs(interp, objc, objv);
        std::string layer_name = args.layer;
        std::string device_str = args.device;
        
        // Check if layer exists
        if (module_storage.find(layer_name) == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid layer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get the device
        torch::Device device = GetDevice(device_str.c_str());
        
        // Move the module to the device
        auto& module = module_storage[layer_name];
        if (!MoveModuleToDevice(module, device)) {
            Tcl_SetResult(interp, const_cast<char*>("Failed to move layer to device"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Return the layer name (for chaining operations)
        Tcl_SetResult(interp, const_cast<char*>(layer_name.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::layer_device(layer) - Get current device of layer with dual syntax support
int LayerDevice_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        LayerDeviceArgs args = ParseLayerDeviceArgs(interp, objc, objv);
        
        // Check if layer exists
        if (module_storage.find(args.layer) == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid layer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get the device of the module
        auto& module = module_storage[args.layer];
        torch::Device device = GetModuleDevice(module);
        
        // Convert device to string
        std::string device_str;
        if (device.is_cuda()) {
            device_str = "cuda:" + std::to_string(device.index());
        } else {
            device_str = "cpu";
        }
        
        Tcl_SetResult(interp, const_cast<char*>(device_str.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::layer_cuda(layer) - Move layer to CUDA with dual syntax support
int LayerCuda_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        LayerCudaArgs args = ParseLayerCudaArgs(interp, objc, objv);
        
        // Check if layer exists
        if (module_storage.find(args.layer) == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid layer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Check if CUDA is available
        if (!torch::cuda::is_available()) {
            Tcl_SetResult(interp, const_cast<char*>("CUDA is not available"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Move the module to CUDA
        auto& module = module_storage[args.layer];
        if (!MoveModuleToDevice(module, torch::kCUDA)) {
            Tcl_SetResult(interp, const_cast<char*>("Failed to move layer to CUDA"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Return the layer name (for chaining operations)
        Tcl_SetResult(interp, const_cast<char*>(args.layer.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::layer_cpu(layer) - Move layer to CPU with dual syntax support
int LayerCpu_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        LayerCpuArgs args = ParseLayerCpuArgs(interp, objc, objv);
        
        // Check if layer exists
        if (module_storage.find(args.layer) == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid layer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Move the module to CPU
        auto& module = module_storage[args.layer];
        if (!MoveModuleToDevice(module, torch::kCPU)) {
            Tcl_SetResult(interp, const_cast<char*>("Failed to move layer to CPU"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Return the layer name (for chaining operations)
        Tcl_SetResult(interp, const_cast<char*>(args.layer.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 

// torch::parameters_to - Move parameters to device with dual syntax support
int ParametersTo_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        ParametersToArgs args = ParseParametersToArgs(interp, objc, objv);
        
        // Parse parameters list
        Tcl_Obj* param_list_obj = Tcl_NewStringObj(args.parameters.c_str(), -1);
        int param_count;
        Tcl_ListObjLength(interp, param_list_obj, &param_count);
        
        // Get target device
        torch::Device device(args.device);
        
        // Move each parameter to the target device
        for (int i = 0; i < param_count; i++) {
            Tcl_Obj* param_obj;
            Tcl_ListObjIndex(interp, param_list_obj, i, &param_obj);
            std::string param_name = Tcl_GetString(param_obj);
            
            if (tensor_storage.find(param_name) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>(("Invalid parameter tensor: " + param_name).c_str()), TCL_VOLATILE);
                return TCL_ERROR;
            }
            
            // Move tensor to device
            tensor_storage[param_name] = tensor_storage[param_name].to(device);
        }
        
        Tcl_SetResult(interp, const_cast<char*>("ok"), TCL_STATIC);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in parameters_to: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 