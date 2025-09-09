#include "libtorchtcl.h"

// Parameter structure for torch::optimizer_adamw
struct OptimizerAdamWArgs {
    std::string parameters;  // parameter list (list of tensor names)
    double lr = 0.001;       // learning rate
    double beta1 = 0.9;      // first moment decay rate
    double beta2 = 0.999;    // second moment decay rate
    double eps = 1e-8;       // epsilon for numerical stability
    double weightDecay = 0.01; // weight decay (AdamW specific default)
    bool amsgrad = false;  // whether to use AMSGrad variant
    
    bool IsValid() const {
        return !parameters.empty() && lr > 0.0 && beta1 >= 0.0 && beta1 < 1.0 && 
               beta2 >= 0.0 && beta2 < 1.0 && eps > 0.0 && weightDecay >= 0.0;
    }
};

// Parse dual syntax for torch::optimizer_adamw
OptimizerAdamWArgs ParseOptimizerAdamWArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    OptimizerAdamWArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): parameters lr ?weight_decay?
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: torch::optimizer_adamw parameter_list learning_rate ?weight_decay?");
        }
        
        args.parameters = Tcl_GetString(objv[1]);
        
        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.lr) != TCL_OK) {
            throw std::runtime_error("Invalid learning rate");
        }
        
        if (objc > 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.weightDecay) != TCL_OK) {
                throw std::runtime_error("Invalid weight_decay value");
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
                    throw std::runtime_error("Invalid learning rate");
                }
            } else if (param == "-beta1") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.beta1) != TCL_OK) {
                    throw std::runtime_error("Invalid beta1 value");
                }
            } else if (param == "-beta2") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.beta2) != TCL_OK) {
                    throw std::runtime_error("Invalid beta2 value");
                }
            } else if (param == "-eps" || param == "-epsilon") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.eps) != TCL_OK) {
                    throw std::runtime_error("Invalid eps value");
                }
            } else if (param == "-weightDecay" || param == "-weight_decay") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.weightDecay) != TCL_OK) {
                    throw std::runtime_error("Invalid weight_decay value");
                }
            } else if (param == "-amsgrad") {
                int amsgrad_val;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &amsgrad_val) != TCL_OK) {
                    throw std::runtime_error("Invalid amsgrad value (must be boolean)");
                }
                args.amsgrad = amsgrad_val;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid (parameters and positive learning rate required, beta values must be in [0,1), eps and weight_decay must be non-negative)");
    }
    
    return args;
}

// torch::optimizer_adamw - AdamW optimizer with dual syntax support
int OptimizerAdamW_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        OptimizerAdamWArgs args = ParseOptimizerAdamWArgs(interp, objc, objv);
        
        // Parse parameters list
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
        
        // Create AdamW optimizer
        auto optimizer = std::make_shared<torch::optim::AdamW>(
            parameters, 
            torch::optim::AdamWOptions(args.lr)
                .betas(std::make_tuple(args.beta1, args.beta2))
                .eps(args.eps)
                .weight_decay(args.weightDecay)
                .amsgrad(args.amsgrad)
        );
        
        // Store optimizer
        std::string handle = GetNextHandle("optimizer");
        optimizer_storage[handle] = optimizer;
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for torch::optimizer_rmsprop
struct OptimizerRMSpropArgs {
    std::string parameters; // parameter list (tensor handles)
    double lr = 0.01;       // learning rate
    double alpha = 0.99;    // smoothing constant
    double eps = 1e-8;      // epsilon for numerical stability

    bool IsValid() const {
        return !parameters.empty() && lr > 0.0 && alpha > 0.0 && eps > 0.0;
    }
};

// Dual-syntax parser for torch::optimizer_rmsprop
static OptimizerRMSpropArgs ParseOptimizerRMSpropArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    OptimizerRMSpropArgs args;

    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Legacy positional syntax: parameters lr ?alpha? ?eps?
        if (objc < 3 || objc > 5) {
            throw std::runtime_error("Usage: torch::optimizer_rmsprop parameter_list learning_rate ?alpha? ?eps?");
        }

        args.parameters = Tcl_GetString(objv[1]);

        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.lr) != TCL_OK) {
            throw std::runtime_error("Invalid learning rate");
        }

        if (objc >= 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.alpha) != TCL_OK) {
                throw std::runtime_error("Invalid alpha value");
            }
        }

        if (objc == 5) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &args.eps) != TCL_OK) {
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

            if (param == "-parameters" || param == "-params") {
                args.parameters = Tcl_GetString(objv[i + 1]);
            } else if (param == "-lr" || param == "-learningRate") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.lr) != TCL_OK) {
                    throw std::runtime_error("Invalid learning rate");
                }
            } else if (param == "-alpha") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.alpha) != TCL_OK) {
                    throw std::runtime_error("Invalid alpha value");
                }
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
        throw std::runtime_error("Required parameters missing or invalid (parameters and positive learning rate required)");
    }

    return args;
}

// torch::rmsprop(params, lr, alpha?, eps?) - RMSprop optimizer
int OptimizerRMSprop_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData;

    try {
        OptimizerRMSpropArgs args = ParseOptimizerRMSpropArgs(interp, objc, objv);

        Tcl_Obj* param_list_obj = Tcl_NewStringObj(args.parameters.c_str(), -1);
        int param_count;
        Tcl_ListObjLength(interp, param_list_obj, &param_count);

        std::vector<torch::Tensor> parameters;
        for (int i = 0; i < param_count; ++i) {
            Tcl_Obj* param_obj;
            Tcl_ListObjIndex(interp, param_list_obj, i, &param_obj);
            std::string name = Tcl_GetString(param_obj);
            if (tensor_storage.find(name) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>(("Invalid parameter tensor: " + name).c_str()), TCL_VOLATILE);
                return TCL_ERROR;
            }
            parameters.push_back(tensor_storage[name]);
        }

        auto optimizer = std::make_shared<torch::optim::RMSprop>(
            parameters,
            torch::optim::RMSpropOptions(args.lr).alpha(args.alpha).eps(args.eps)
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

// Insert new argument structure and parser ABOVE the OptimizerMomentumSGD_Cmd implementation
struct OptimizerMomentumSGDArgs {
    std::string parameters;   // parameter list (list of tensor names)
    double lr = 0.01;         // learning rate
    double momentum = 0.9;    // momentum factor
    double weightDecay = 0.0; // weight decay factor

    bool IsValid() const {
        return !parameters.empty() && lr > 0.0 && momentum >= 0.0 && weightDecay >= 0.0;
    }
};

// Dual-syntax parser for torch::optimizer_momentum_sgd
static OptimizerMomentumSGDArgs ParseOptimizerMomentumSGDArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    OptimizerMomentumSGDArgs args;

    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Legacy positional syntax: parameters lr momentum ?weight_decay?
        if (objc < 4 || objc > 5) {
            throw std::runtime_error("Usage: torch::optimizer_momentum_sgd parameter_list learning_rate momentum ?weight_decay?");
        }

        args.parameters = Tcl_GetString(objv[1]);

        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.lr) != TCL_OK) {
            throw std::runtime_error("Invalid learning rate");
        }

        if (Tcl_GetDoubleFromObj(interp, objv[3], &args.momentum) != TCL_OK) {
            throw std::runtime_error("Invalid momentum value");
        }

        if (objc == 5) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &args.weightDecay) != TCL_OK) {
                throw std::runtime_error("Invalid weight_decay value");
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
                    throw std::runtime_error("Invalid learning rate");
                }
            } else if (param == "-momentum") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.momentum) != TCL_OK) {
                    throw std::runtime_error("Invalid momentum value");
                }
            } else if (param == "-weightDecay" || param == "-weight_decay") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.weightDecay) != TCL_OK) {
                    throw std::runtime_error("Invalid weight_decay value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid (parameters, positive learning rate, non-negative momentum and weight_decay required)");
    }

    return args;
}

// torch::momentum_sgd(params, lr, momentum, weight_decay?) - SGD with momentum
int OptimizerMomentumSGD_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        // Parse arguments using dual-syntax parser
        OptimizerMomentumSGDArgs args = ParseOptimizerMomentumSGDArgs(interp, objc, objv);

        // Convert parameter list string into vector<torch::Tensor>
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

        // Create SGD optimizer with momentum
        auto optimizer = std::make_shared<torch::optim::SGD>(
            parameters,
            torch::optim::SGDOptions(args.lr).momentum(args.momentum).weight_decay(args.weightDecay)
        );

        // Store optimizer handle
        std::string handle = GetNextHandle("optimizer");
        optimizer_storage[handle] = optimizer;

        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for torch::optimizer_adagrad
struct OptimizerAdagradArgs {
    std::string parameters;  // parameter list (list of tensor names)
    double lr = 0.01;        // learning rate
    double eps = 1e-10;      // epsilon for numerical stability
    
    bool IsValid() const {
        return !parameters.empty() && lr > 0.0 && eps > 0.0;
    }
};

// Parse dual syntax for torch::optimizer_adagrad
OptimizerAdagradArgs ParseOptimizerAdagradArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    OptimizerAdagradArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): parameters lr ?eps?
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: torch::optimizer_adagrad parameter_list learning_rate ?eps?");
        }
        
        args.parameters = Tcl_GetString(objv[1]);
        
        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.lr) != TCL_OK) {
            throw std::runtime_error("Invalid learning rate");
        }
        
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
            
            if (param == "-parameters" || param == "-params") {
                args.parameters = Tcl_GetString(objv[i + 1]);
            } else if (param == "-lr" || param == "-learningRate") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.lr) != TCL_OK) {
                    throw std::runtime_error("Invalid learning rate");
                }
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
        throw std::runtime_error("Required parameters missing or invalid (parameters and positive learning rate required)");
    }
    
    return args;
}

// torch::optimizer_adagrad - Adagrad optimizer with dual syntax support
int OptimizerAdagrad_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        OptimizerAdagradArgs args = ParseOptimizerAdagradArgs(interp, objc, objv);
        
        // Parse parameters list
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
        
        // Create Adagrad optimizer
        auto optimizer = std::make_shared<torch::optim::Adagrad>(
            parameters, 
            torch::optim::AdagradOptions(args.lr).eps(args.eps)
        );
        
        // Store optimizer
        std::string handle = GetNextHandle("optimizer");
        optimizer_storage[handle] = optimizer;
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 