#include "libtorchtcl.h"

// Parameter structure for layer_parameters command
struct LayerParametersArgs {
    std::string layer;
    
    bool IsValid() const {
        return !layer.empty();
    }
};

// Parse dual syntax for layer_parameters command
LayerParametersArgs ParseLayerParametersArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LayerParametersArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::layer_parameters layer");
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
            
            if (param == "-layer") {
                args.layer = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: -layer");
    }
    
    return args;
}

// torch::layer_parameters(layer) - Get list of trainable parameters
// New syntax: torch::layerParameters -layer layername
int LayerParameters_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        LayerParametersArgs args = ParseLayerParametersArgs(interp, objc, objv);
        std::string layer_name = args.layer;
        
        // Check if layer exists
        if (module_storage.find(layer_name) == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid layer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& module = module_storage[layer_name];
        
        // Get parameters and store them as tensors
        std::vector<std::string> param_names;
        for (const auto& param : module->parameters()) {
            std::string param_handle = GetNextHandle("tensor");
            tensor_storage[param_handle] = param;
            param_names.push_back(param_handle);
        }
        
        // Create TCL list of parameter handles
        Tcl_Obj* param_list = Tcl_NewListObj(0, NULL);
        for (const auto& name : param_names) {
            Tcl_ListObjAppendElement(interp, param_list, Tcl_NewStringObj(name.c_str(), -1));
        }
        
        Tcl_SetObjResult(interp, param_list);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for model_train command
struct ModelTrainArgs {
    std::string model;
    
    bool IsValid() const {
        return !model.empty();
    }
};

// Parse dual syntax for model_train
ModelTrainArgs ParseModelTrainArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ModelTrainArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::model_train model");
        }
        
        args.model = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-model") {
                args.model = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Model name is required");
    }
    
    return args;
}

int ModelTrain_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        ModelTrainArgs args = ParseModelTrainArgs(interp, objc, objv);
        
        // Check if model exists
        if (module_storage.find(args.model) == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid model name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& module = module_storage[args.model];
        
        // Set to training mode
        module->train();
        
        // Return the model name (for chaining operations)
        Tcl_SetResult(interp, const_cast<char*>(args.model.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for model_eval command
struct ModelEvalArgs {
    std::string model;
    
    bool IsValid() const {
        return !model.empty();
    }
};

// Parse dual syntax for model_eval
ModelEvalArgs ParseModelEvalArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ModelEvalArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::model_eval model");
        }
        
        args.model = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-model") {
                args.model = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Model name is required");
    }
    
    return args;
}

// torch::model_eval(model) - Set model to evaluation mode
int ModelEval_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        ModelEvalArgs args = ParseModelEvalArgs(interp, objc, objv);
        
        // Check if model exists
        if (module_storage.find(args.model) == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid model name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& module = module_storage[args.model];
        
        // Set to evaluation mode
        module->eval();
        
        // Return the model name (for chaining operations)
        Tcl_SetResult(interp, const_cast<char*>(args.model.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 