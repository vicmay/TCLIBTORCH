#include "libtorchtcl.h"

// Parameter structure for torch::optimizer_adam
struct OptimizerAdamArgs {
    std::string parameters;  // parameter list (list of tensor names)
    double lr = 0.001;       // learning rate
    double beta1 = 0.9;      // first moment decay rate
    double beta2 = 0.999;    // second moment decay rate
    double weightDecay = 0.0; // weight decay
    
    bool IsValid() const {
        return !parameters.empty() && lr > 0.0 && beta1 >= 0.0 && beta1 < 1.0 && 
               beta2 >= 0.0 && beta2 < 1.0 && weightDecay >= 0.0;
    }
};

// Parse dual syntax for torch::optimizer_adam
OptimizerAdamArgs ParseOptimizerAdamArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    OptimizerAdamArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): parameters lr ?beta1? ?beta2? ?weight_decay?
        if (objc < 3 || objc > 6) {
            throw std::runtime_error("Usage: torch::optimizer_adam parameter_list learning_rate ?beta1? ?beta2? ?weight_decay?");
        }
        
        args.parameters = Tcl_GetString(objv[1]);
        
        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.lr) != TCL_OK) {
            throw std::runtime_error("Required parameters missing");
        }
        
        if (objc > 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.beta1) != TCL_OK) {
                throw std::runtime_error("Required parameters missing");
            }
        }
        
        if (objc > 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &args.beta2) != TCL_OK) {
                throw std::runtime_error("Required parameters missing");
            }
        }
        
        if (objc > 5) {
            if (Tcl_GetDoubleFromObj(interp, objv[5], &args.weightDecay) != TCL_OK) {
                throw std::runtime_error("Required parameters missing");
            }
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
            } else if (param == "-lr" || param == "-learningRate") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.lr) != TCL_OK) {
                    throw std::runtime_error("Required parameters missing");
                }
            } else if (param == "-beta1") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.beta1) != TCL_OK) {
                    throw std::runtime_error("Required parameters missing");
                }
            } else if (param == "-beta2") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.beta2) != TCL_OK) {
                    throw std::runtime_error("Required parameters missing");
                }
            } else if (param == "-weightDecay" || param == "-weight_decay") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.weightDecay) != TCL_OK) {
                    throw std::runtime_error("Required parameters missing");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing");
    }
    
    return args;
}

// Parameter structure for torch::optimizer_sgd
struct OptimizerSGDArgs {
    std::string parameters;   // parameter list (list of tensor names)
    double lr = 0.01;         // learning rate (default: 0.01)
    double momentum = 0.0;    // momentum factor (default: 0.0)
    double dampening = 0.0;   // dampening for momentum (default: 0.0)
    double weightDecay = 0.0; // weight decay (L2 penalty) (default: 0.0)
    bool nesterov = false;    // enables Nesterov momentum (default: false)
    
    bool IsValid() const {
        return !parameters.empty() && lr > 0.0 && momentum >= 0.0 && 
               dampening >= 0.0 && weightDecay >= 0.0 &&
               (!nesterov || (momentum > 0.0 && dampening == 0.0)); // Nesterov requires momentum > 0 and dampening == 0
    }
};

// Parse dual syntax for torch::optimizer_sgd
OptimizerSGDArgs ParseOptimizerSGDArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    OptimizerSGDArgs args;
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        args.parameters = Tcl_GetString(objv[1]);
        
        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.lr) != TCL_OK) {
            throw std::runtime_error("Invalid learning rate");
        }
        
        if (objc > 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.momentum) != TCL_OK) {
                throw std::runtime_error("Invalid momentum value");
            }
        }
        
        if (objc > 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &args.dampening) != TCL_OK) {
                throw std::runtime_error("Invalid dampening value");
            }
        }
        
        if (objc > 5) {
            if (Tcl_GetDoubleFromObj(interp, objv[5], &args.weightDecay) != TCL_OK) {
                throw std::runtime_error("Invalid weight_decay value");
            }
        }
        
        if (objc > 6) {
            int nesterov_val;
            if (Tcl_GetBooleanFromObj(interp, objv[6], &nesterov_val) != TCL_OK) {
                throw std::runtime_error("Invalid nesterov value (must be boolean)");
            }
            args.nesterov = nesterov_val;
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
            } else if (param == "-lr" || param == "-learningRate") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.lr) != TCL_OK) {
                    throw std::runtime_error("Invalid learning rate");
                }
            } else if (param == "-momentum") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.momentum) != TCL_OK) {
                    throw std::runtime_error("Invalid momentum value");
                }
            } else if (param == "-dampening") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.dampening) != TCL_OK) {
                    throw std::runtime_error("Invalid dampening value");
                }
            } else if (param == "-weightDecay" || param == "-weight_decay") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.weightDecay) != TCL_OK) {
                    throw std::runtime_error("Invalid weight_decay value");
                }
            } else if (param == "-nesterov") {
                int nesterov_val;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &nesterov_val) != TCL_OK) {
                    throw std::runtime_error("Invalid nesterov value (must be boolean)");
                }
                args.nesterov = nesterov_val;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid (parameters and positive learning rate required, momentum/dampening/weight_decay must be non-negative, Nesterov requires momentum > 0 and dampening == 0)");
    }
    
    return args;
}

int OptimizerSGD_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        OptimizerSGDArgs args = ParseOptimizerSGDArgs(interp, objc, objv);
        
        // Parse parameter list
        Tcl_Obj* param_list_obj = Tcl_NewStringObj(args.parameters.c_str(), -1);
        int param_count;
        Tcl_ListObjLength(interp, param_list_obj, &param_count);
        
        std::vector<torch::Tensor> parameters;
        for (int i = 0; i < param_count; i++) {
            Tcl_Obj* param_obj;
            Tcl_ListObjIndex(interp, param_list_obj, i, &param_obj);
            std::string param_name = Tcl_GetString(param_obj);
            
            if (tensor_storage.find(param_name) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>(("Invalid parameter tensor: " + param_name).c_str()), TCL_VOLATILE);
                return TCL_ERROR;
            }
            parameters.push_back(tensor_storage[param_name]);
        }
        
        // Create SGD optimizer with all parameters
        auto optimizer = std::make_shared<torch::optim::SGD>(
            parameters, 
            torch::optim::SGDOptions(args.lr)
                .momentum(args.momentum)
                .dampening(args.dampening)
                .weight_decay(args.weightDecay)
                .nesterov(args.nesterov)
        );
        
        std::string handle = GetNextHandle("optimizer");
        optimizer_storage[handle] = optimizer;
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::optimizer_adam - Adam optimizer with dual syntax support
int OptimizerAdam_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        OptimizerAdamArgs args = ParseOptimizerAdamArgs(interp, objc, objv);
        
        // Parse parameter list
        Tcl_Obj* param_list_obj = Tcl_NewStringObj(args.parameters.c_str(), -1);
        int param_count;
        Tcl_ListObjLength(interp, param_list_obj, &param_count);
        
        std::vector<torch::Tensor> parameters;
        for (int i = 0; i < param_count; i++) {
            Tcl_Obj* param_obj;
            Tcl_ListObjIndex(interp, param_list_obj, i, &param_obj);
            std::string param_name = Tcl_GetString(param_obj);
            
            if (tensor_storage.find(param_name) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>(("Invalid parameter tensor: " + param_name).c_str()), TCL_VOLATILE);
                return TCL_ERROR;
            }
            parameters.push_back(tensor_storage[param_name]);
        }
        
        // Create Adam optimizer
        auto optimizer = std::make_shared<torch::optim::Adam>(
            parameters, 
            torch::optim::AdamOptions(args.lr)
                .betas(std::make_tuple(args.beta1, args.beta2))
                .weight_decay(args.weightDecay)
        );
        
        std::string handle = GetNextHandle("optimizer");
        optimizer_storage[handle] = optimizer;
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for torch::optimizer_step
struct OptimizerStepArgs {
    std::string optimizer;  // optimizer handle (required)
    
    bool IsValid() const {
        return !optimizer.empty();
    }
};

// Parse dual syntax for torch::optimizer_step
OptimizerStepArgs ParseOptimizerStepArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    OptimizerStepArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): optimizer
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::optimizer_step optimizer");
        }
        
        args.optimizer = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-optimizer" || param == "-opt") {
                args.optimizer = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing (optimizer handle required)");
    }
    
    return args;
}

// Parameter structure for torch::optimizer_zero_grad
struct OptimizerZeroGradArgs {
    std::string optimizer;  // optimizer handle (required)
    bool setToNone = true;  // whether to set gradients to None instead of zero (default: true)
    
    bool IsValid() const {
        return !optimizer.empty();
    }
};

// Parse dual syntax for torch::optimizer_zero_grad
OptimizerZeroGradArgs ParseOptimizerZeroGradArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    OptimizerZeroGradArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): optimizer ?set_to_none?
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("Usage: torch::optimizer_zero_grad optimizer ?set_to_none?");
        }
        
        args.optimizer = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            int setToNoneVal;
            if (Tcl_GetBooleanFromObj(interp, objv[2], &setToNoneVal) != TCL_OK) {
                throw std::runtime_error("Invalid set_to_none value (must be boolean)");
            }
            args.setToNone = setToNoneVal;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-optimizer" || param == "-opt") {
                args.optimizer = Tcl_GetString(objv[i + 1]);
            } else if (param == "-setToNone" || param == "-set_to_none") {
                int setToNoneVal;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &setToNoneVal) != TCL_OK) {
                    throw std::runtime_error("Invalid set_to_none value (must be boolean)");
                }
                args.setToNone = setToNoneVal;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing (optimizer handle required)");
    }
    
    return args;
}

// torch::optimizer_step - Step optimizer with dual syntax support
int OptimizerStep_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        OptimizerStepArgs args = ParseOptimizerStepArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer handle"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& optimizer = optimizer_storage[args.optimizer];
        optimizer->step();
        
        Tcl_SetResult(interp, const_cast<char*>("OK"), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::optimizer_zero_grad - Zero gradients with dual syntax support
int OptimizerZeroGrad_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        OptimizerZeroGradArgs args = ParseOptimizerZeroGradArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer handle"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& optimizer = optimizer_storage[args.optimizer];
        optimizer->zero_grad(args.setToNone);
        
        Tcl_SetResult(interp, const_cast<char*>("OK"), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 