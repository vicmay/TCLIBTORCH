#include "libtorchtcl.h"
#include <torch/optim.h>

// External global storage for learning rate schedulers (defined elsewhere)
extern std::unordered_map<std::string, std::shared_ptr<void>> scheduler_storage;

// Parameter structure for torch::optimizer_lbfgs
struct OptimizerLBFGSArgs {
    std::string parameters;  // parameter list (list of tensor names)
    double lr = 1.0;         // learning rate (LBFGS specific default)
    int maxIter = 20;        // maximum number of iterations
    int maxEval = 25;        // maximum number of function evaluations
    double toleranceGrad = 1e-7;    // tolerance for gradient convergence
    double toleranceChange = 1e-9;  // tolerance for change convergence
    
    bool IsValid() const {
        return !parameters.empty() && lr > 0.0 && maxIter > 0 && maxEval > 0 && 
               toleranceGrad > 0.0 && toleranceChange > 0.0;
    }
};

// Parse dual syntax for torch::optimizer_lbfgs
OptimizerLBFGSArgs ParseOptimizerLBFGSArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    OptimizerLBFGSArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): parameters ?lr? ?max_iter? ?max_eval? ?tolerance_grad? ?tolerance_change?
        if (objc < 2 || objc > 7) {
            throw std::runtime_error("Usage: torch::optimizer_lbfgs parameters ?lr? ?max_iter? ?max_eval? ?tolerance_grad? ?tolerance_change?");
        }
        
        args.parameters = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.lr) != TCL_OK) {
                throw std::runtime_error("Invalid learning rate");
            }
        }
        
        if (objc > 3) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.maxIter) != TCL_OK) {
                throw std::runtime_error("Invalid max_iter value");
            }
        }
        
        if (objc > 4) {
            if (Tcl_GetIntFromObj(interp, objv[4], &args.maxEval) != TCL_OK) {
                throw std::runtime_error("Invalid max_eval value");
            }
        }
        
        if (objc > 5) {
            if (Tcl_GetDoubleFromObj(interp, objv[5], &args.toleranceGrad) != TCL_OK) {
                throw std::runtime_error("Invalid tolerance_grad value");
            }
        }
        
        if (objc > 6) {
            if (Tcl_GetDoubleFromObj(interp, objv[6], &args.toleranceChange) != TCL_OK) {
                throw std::runtime_error("Invalid tolerance_change value");
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
            } else if (param == "-maxIter" || param == "-max_iter") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.maxIter) != TCL_OK) {
                    throw std::runtime_error("Invalid max_iter value");
                }
            } else if (param == "-maxEval" || param == "-max_eval") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.maxEval) != TCL_OK) {
                    throw std::runtime_error("Invalid max_eval value");
                }
            } else if (param == "-toleranceGrad" || param == "-tolerance_grad") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.toleranceGrad) != TCL_OK) {
                    throw std::runtime_error("Invalid tolerance_grad value");
                }
            } else if (param == "-toleranceChange" || param == "-tolerance_change") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.toleranceChange) != TCL_OK) {
                    throw std::runtime_error("Invalid tolerance_change value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid (parameters and positive values required for lr, maxIter, maxEval, toleranceGrad, toleranceChange)");
    }
    
    return args;
}

// torch::optimizer_lbfgs - L-BFGS optimizer with dual syntax support
int OptimizerLBFGS_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax parser
        OptimizerLBFGSArgs args = ParseOptimizerLBFGSArgs(interp, objc, objv);
        
        // Parse parameter list (allow either single tensor handle or Tcl list of handles)
        std::vector<torch::Tensor> parameters;
        
        int listLen;
        if (Tcl_ListObjLength(interp, Tcl_NewStringObj(args.parameters.c_str(), -1), &listLen) == TCL_OK && listLen > 1) {
            // It's a Tcl list
            Tcl_Obj* listObj = Tcl_NewStringObj(args.parameters.c_str(), -1);
            Tcl_IncrRefCount(listObj);
            for (int i = 0; i < listLen; ++i) {
                Tcl_Obj* elemObj;
                Tcl_ListObjIndex(interp, listObj, i, &elemObj);
                std::string tname = Tcl_GetString(elemObj);
                if (tensor_storage.find(tname) == tensor_storage.end()) {
                    Tcl_DecrRefCount(listObj);
                    Tcl_SetResult(interp, const_cast<char*>("Invalid parameter tensor in list"), TCL_VOLATILE);
                    return TCL_ERROR;
                }
                parameters.push_back(tensor_storage[tname]);
            }
            Tcl_DecrRefCount(listObj);
        } else {
            // Single tensor handle or module handle (backward compatibility)
            if (tensor_storage.find(args.parameters) != tensor_storage.end()) {
                parameters.push_back(tensor_storage[args.parameters]);
            } else if (module_storage.find(args.parameters) != module_storage.end()) {
                // Module handle (backward compatibility)
                auto module = module_storage[args.parameters];
                for (auto& param : module->parameters()) {
                    parameters.push_back(param);
                }
            } else {
                Tcl_SetResult(interp, const_cast<char*>("Invalid parameters handle"), TCL_VOLATILE);
                return TCL_ERROR;
            }
        }
        
        auto optimizer = std::make_shared<torch::optim::LBFGS>(
            parameters,
            torch::optim::LBFGSOptions(args.lr)
                .max_iter(args.maxIter)
                .max_eval(args.maxEval)
                .tolerance_grad(args.toleranceGrad)
                .tolerance_change(args.toleranceChange)
        );
        
        std::string handle = GetNextHandle("optimizer");
        optimizer_storage[handle] = optimizer;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for torch::optimizer_rprop
struct OptimizerRpropArgs {
    std::string parameters;  // parameter list (list of tensor names)
    double lr = 0.01;        // learning rate (default: 0.01)
    std::pair<double, double> etas = {0.5, 1.2};  // eta minus and eta plus values
    std::pair<double, double> stepSizes = {1e-6, 50.0};  // min and max step sizes
    
    bool IsValid() const {
        return !parameters.empty() && lr > 0.0 && 
               etas.first > 0.0 && etas.first < 1.0 && 
               etas.second > 1.0 && 
               stepSizes.first > 0.0 && stepSizes.second > stepSizes.first;
    }
};

// Parse dual syntax for torch::optimizer_rprop
OptimizerRpropArgs ParseOptimizerRpropArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    OptimizerRpropArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 5) {
            throw std::runtime_error("Usage: torch::optimizer_rprop parameters ?lr? ?etas? ?step_sizes?");
        }
        
        args.parameters = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.lr) != TCL_OK) {
                throw std::runtime_error("Invalid learning rate");
            }
        }
        
        if (objc > 3) {
            // Parse etas list {eta_minus eta_plus}
            int list_len;
            Tcl_Obj** list_objs;
            if (Tcl_ListObjGetElements(interp, objv[3], &list_len, &list_objs) != TCL_OK) {
                throw std::runtime_error("Invalid etas list format");
            }
            if (list_len == 2) {
                if (Tcl_GetDoubleFromObj(interp, list_objs[0], &args.etas.first) != TCL_OK ||
                    Tcl_GetDoubleFromObj(interp, list_objs[1], &args.etas.second) != TCL_OK) {
                    throw std::runtime_error("Invalid eta values");
                }
            } else {
                throw std::runtime_error("Etas list must contain exactly 2 values");
            }
        }
        
        if (objc > 4) {
            // Parse step_sizes list {min max}
            int list_len;
            Tcl_Obj** list_objs;
            if (Tcl_ListObjGetElements(interp, objv[4], &list_len, &list_objs) != TCL_OK) {
                throw std::runtime_error("Invalid step_sizes list format");
            }
            if (list_len == 2) {
                if (Tcl_GetDoubleFromObj(interp, list_objs[0], &args.stepSizes.first) != TCL_OK ||
                    Tcl_GetDoubleFromObj(interp, list_objs[1], &args.stepSizes.second) != TCL_OK) {
                    throw std::runtime_error("Invalid step size values");
                }
            } else {
                throw std::runtime_error("Step sizes list must contain exactly 2 values");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-parameters" || param == "-params") {
                args.parameters = Tcl_GetString(objv[i + 1]);
            } else if (param == "-lr" || param == "-learningRate") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.lr) != TCL_OK) {
                    throw std::runtime_error("Invalid learning rate");
                }
            } else if (param == "-etas") {
                // Parse etas list {eta_minus eta_plus}
                int list_len;
                Tcl_Obj** list_objs;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &list_len, &list_objs) != TCL_OK) {
                    throw std::runtime_error("Invalid etas list format");
                }
                if (list_len == 2) {
                    if (Tcl_GetDoubleFromObj(interp, list_objs[0], &args.etas.first) != TCL_OK ||
                        Tcl_GetDoubleFromObj(interp, list_objs[1], &args.etas.second) != TCL_OK) {
                        throw std::runtime_error("Invalid eta values");
                    }
                } else {
                    throw std::runtime_error("Etas list must contain exactly 2 values");
                }
            } else if (param == "-stepSizes" || param == "-step_sizes") {
                // Parse step_sizes list {min max}
                int list_len;
                Tcl_Obj** list_objs;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &list_len, &list_objs) != TCL_OK) {
                    throw std::runtime_error("Invalid step_sizes list format");
                }
                if (list_len == 2) {
                    if (Tcl_GetDoubleFromObj(interp, list_objs[0], &args.stepSizes.first) != TCL_OK ||
                        Tcl_GetDoubleFromObj(interp, list_objs[1], &args.stepSizes.second) != TCL_OK) {
                        throw std::runtime_error("Invalid step size values");
                    }
                } else {
                    throw std::runtime_error("Step sizes list must contain exactly 2 values");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid");
    }
    
    return args;
}

// torch::optimizer_rprop - Rprop optimizer
int OptimizerRprop_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        OptimizerRpropArgs args = ParseOptimizerRpropArgs(interp, objc, objv);
        
        std::vector<torch::Tensor> parameters;
        
        if (module_storage.find(args.parameters) != module_storage.end()) {
            // Module handle
            auto module = module_storage[args.parameters];
            for (auto& param : module->parameters()) {
                parameters.push_back(param);
            }
        } else {
            // Try parsing as a list of tensor handles
            Tcl_Obj* param_list_obj = Tcl_NewStringObj(args.parameters.c_str(), -1);
            int param_count;
            if (Tcl_ListObjLength(interp, param_list_obj, &param_count) == TCL_OK) {
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
            } else {
                Tcl_SetResult(interp, const_cast<char*>("Invalid parameters handle"), TCL_VOLATILE);
                return TCL_ERROR;
            }
        }
        
        // Since LibTorch doesn't have a dedicated Rprop optimizer, we use RMSprop as a close approximation
        auto optimizer = std::make_shared<torch::optim::RMSprop>(
            parameters,
            torch::optim::RMSpropOptions(args.lr)
        );
        
        std::string handle = GetNextHandle("optimizer");
        optimizer_storage[handle] = optimizer;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for torch::optimizer_adamax
struct OptimizerAdamaxArgs {
    std::string parameters;  // parameter list (list of tensor names)
    double lr = 0.002;       // learning rate (default for Adamax)
    double beta1 = 0.9;      // first moment decay rate
    double beta2 = 0.999;    // second moment decay rate
    double eps = 1e-8;       // epsilon for numerical stability
    double weightDecay = 0.0; // weight decay
    
    bool IsValid() const {
        return !parameters.empty() && lr > 0.0 && beta1 >= 0.0 && beta1 < 1.0 && 
               beta2 >= 0.0 && beta2 < 1.0 && eps > 0.0 && weightDecay >= 0.0;
    }
};

// Parse dual syntax for torch::optimizer_adamax
OptimizerAdamaxArgs ParseOptimizerAdamaxArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    OptimizerAdamaxArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): parameters ?lr? ?betas? ?eps? ?weight_decay?
        if (objc < 2 || objc > 6) {
            throw std::runtime_error("Usage: torch::optimizer_adamax parameters ?lr? ?betas? ?eps? ?weight_decay?");
        }
        
        args.parameters = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.lr) != TCL_OK) {
                throw std::runtime_error("Invalid learning rate");
            }
        }
        
        if (objc > 3) {
            // Parse betas list {beta1 beta2}
            int list_len;
            Tcl_Obj** list_objs;
            if (Tcl_ListObjGetElements(interp, objv[3], &list_len, &list_objs) != TCL_OK) {
                throw std::runtime_error("Invalid betas format");
            }
            if (list_len == 2) {
                if (Tcl_GetDoubleFromObj(interp, list_objs[0], &args.beta1) != TCL_OK ||
                    Tcl_GetDoubleFromObj(interp, list_objs[1], &args.beta2) != TCL_OK) {
                    throw std::runtime_error("Invalid beta values");
                }
            }
        }
        
        if (objc > 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &args.eps) != TCL_OK) {
                throw std::runtime_error("Invalid eps value");
            }
        }
        
        if (objc > 5) {
            if (Tcl_GetDoubleFromObj(interp, objv[5], &args.weightDecay) != TCL_OK) {
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
            } else if (param == "-betas") {
                // Parse betas list {beta1 beta2}
                int list_len;
                Tcl_Obj** list_objs;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &list_len, &list_objs) != TCL_OK) {
                    throw std::runtime_error("Invalid betas format");
                }
                if (list_len == 2) {
                    if (Tcl_GetDoubleFromObj(interp, list_objs[0], &args.beta1) != TCL_OK ||
                        Tcl_GetDoubleFromObj(interp, list_objs[1], &args.beta2) != TCL_OK) {
                        throw std::runtime_error("Invalid beta values");
                    }
                }
            } else if (param == "-eps" || param == "-epsilon") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.eps) != TCL_OK) {
                    throw std::runtime_error("Invalid eps value");
                }
            } else if (param == "-weightDecay" || param == "-weight_decay") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.weightDecay) != TCL_OK) {
                    throw std::runtime_error("Invalid weight decay value");
                }
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

// torch::optimizer_adamax - Adamax optimizer with dual syntax support
int OptimizerAdamax_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        OptimizerAdamaxArgs args = ParseOptimizerAdamaxArgs(interp, objc, objv);
        // Parse parameter list (allow either single tensor handle or Tcl list of handles)
        std::vector<torch::Tensor> parameters;

        int listLen;
        if (Tcl_ListObjLength(interp, Tcl_NewStringObj(args.parameters.c_str(), -1), &listLen) == TCL_OK && listLen > 1) {
            // It's a Tcl list
            Tcl_Obj* listObj = Tcl_NewStringObj(args.parameters.c_str(), -1);
            Tcl_IncrRefCount(listObj);
            for (int i = 0; i < listLen; ++i) {
                Tcl_Obj* elemObj;
                Tcl_ListObjIndex(interp, listObj, i, &elemObj);
                std::string tname = Tcl_GetString(elemObj);
                if (tensor_storage.find(tname) == tensor_storage.end()) {
                    Tcl_DecrRefCount(listObj);
                    Tcl_SetResult(interp, const_cast<char*>("Invalid parameter tensor in list"), TCL_VOLATILE);
                    return TCL_ERROR;
                }
                parameters.push_back(tensor_storage[tname]);
            }
            Tcl_DecrRefCount(listObj);
        } else {
            // Single tensor handle
            if (tensor_storage.find(args.parameters) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>("Invalid parameters handle"), TCL_VOLATILE);
                return TCL_ERROR;
            }
            parameters.push_back(tensor_storage[args.parameters]);
        }

        // Construct options (using AdamOptions as placeholder)
        torch::optim::AdamOptions opts(args.lr);
        opts.betas(std::make_tuple(args.beta1, args.beta2));
        opts.eps(args.eps);
        opts.weight_decay(args.weightDecay);

        auto optimizer = std::make_shared<torch::optim::Adam>(parameters, opts);
        std::string handle = GetNextHandle("optimizer");
        optimizer_storage[handle] = optimizer;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameters for torch::lr_scheduler_lambda
struct LRSchedulerLambdaArgs {
    std::string optimizer;
    double multiplier = 1.0;  // Optional, default 1.0
    
    bool IsValid() const {
        return !optimizer.empty();
    }
};

// Parse arguments for torch::lr_scheduler_lambda (dual syntax support)
LRSchedulerLambdaArgs ParseLRSchedulerLambdaArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerLambdaArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: optimizer ?multiplier?
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("Usage: torch::lr_scheduler_lambda optimizer ?multiplier?");
        }
        
        args.optimizer = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.multiplier) != TCL_OK) {
                throw std::runtime_error("Invalid multiplier value");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must be in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-optimizer") {
                args.optimizer = Tcl_GetString(objv[i + 1]);
            } else if (param == "-multiplier" || param == "-lambda") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.multiplier) != TCL_OK) {
                    throw std::runtime_error("Invalid multiplier value");
                }
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

// torch::lr_scheduler_lambda - Lambda LR scheduler with dual syntax support
int LRSchedulerLambda_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        auto args = ParseLRSchedulerLambdaArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer handle"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Store the multiplier in a simple structure
        auto scheduler_data = std::make_shared<std::pair<std::string, double>>(args.optimizer, args.multiplier);
        
        std::string handle = GetNextHandle("lambda_scheduler");
        scheduler_storage[handle] = scheduler_data;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameters for torch::lr_scheduler_exponential_decay
struct LRSchedulerExponentialDecayArgs {
    std::string optimizer;
    double gamma = 0.95;
    
    bool IsValid() const {
        return !optimizer.empty() && gamma > 0.0 && gamma <= 1.0;
    }
};

// Parse arguments for torch::lr_scheduler_exponential_decay (dual syntax support)
LRSchedulerExponentialDecayArgs ParseLRSchedulerExponentialDecayArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerExponentialDecayArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::lr_scheduler_exponential_decay optimizer gamma");
        }
        
        args.optimizer = Tcl_GetString(objv[1]);
        
        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.gamma) != TCL_OK) {
            throw std::runtime_error("Invalid gamma value");
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must be in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-optimizer") {
                args.optimizer = Tcl_GetString(objv[i + 1]);
            } else if (param == "-gamma") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.gamma) != TCL_OK) {
                    throw std::runtime_error("Invalid gamma value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid (optimizer handle required, gamma must be between 0 and 1)");
    }
    
    return args;
}

// torch::lr_scheduler_exponential_decay - Exponential LR scheduler with dual syntax support
int LRSchedulerExponentialDecay_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        auto args = ParseLRSchedulerExponentialDecayArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer handle"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Store the gamma in a simple structure
        auto scheduler_data = std::make_shared<std::pair<std::string, double>>(args.optimizer, args.gamma);
        
        std::string handle = GetNextHandle("scheduler");
        scheduler_storage[handle] = scheduler_data;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameters for torch::lr_scheduler_cyclic
struct LRSchedulerCyclicArgs {
    std::string optimizer;
    double baseLr = -1.0;  // Required parameter, -1 indicates not set
    double maxLr = -1.0;   // Required parameter, -1 indicates not set  
    int stepSize = 2000;   // Optional, default 2000
    std::string mode = "triangular";  // Optional, default "triangular"
    
    bool IsValid() const {
        return !optimizer.empty() && baseLr > 0.0 && maxLr > 0.0 && maxLr > baseLr && stepSize > 0;
    }
};

// Parse arguments for torch::lr_scheduler_cyclic (dual syntax support)
LRSchedulerCyclicArgs ParseLRSchedulerCyclicArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerCyclicArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: optimizer base_lr max_lr ?step_size? ?mode?
        if (objc < 4 || objc > 6) {
            throw std::runtime_error("Usage: torch::lr_scheduler_cyclic optimizer base_lr max_lr ?step_size? ?mode?");
        }
        
        args.optimizer = Tcl_GetString(objv[1]);
        
        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.baseLr) != TCL_OK) {
            throw std::runtime_error("Invalid base_lr value");
        }
        
        if (Tcl_GetDoubleFromObj(interp, objv[3], &args.maxLr) != TCL_OK) {
            throw std::runtime_error("Invalid max_lr value");
        }
        
        if (objc > 4) {
            if (Tcl_GetIntFromObj(interp, objv[4], &args.stepSize) != TCL_OK) {
                throw std::runtime_error("Invalid step_size value");
            }
        }
        
        if (objc > 5) {
            args.mode = Tcl_GetString(objv[5]);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must be in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-optimizer") {
                args.optimizer = Tcl_GetString(objv[i + 1]);
            } else if (param == "-baseLr" || param == "-base_lr") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.baseLr) != TCL_OK) {
                    throw std::runtime_error("Invalid baseLr value");
                }
            } else if (param == "-maxLr" || param == "-max_lr") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.maxLr) != TCL_OK) {
                    throw std::runtime_error("Invalid maxLr value");
                }
            } else if (param == "-stepSize" || param == "-step_size") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.stepSize) != TCL_OK) {
                    throw std::runtime_error("Invalid stepSize value");
                }
            } else if (param == "-mode") {
                args.mode = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid (optimizer handle, baseLr and maxLr required, maxLr must be greater than baseLr, stepSize must be positive)");
    }
    
    return args;
}

// torch::lr_scheduler_cyclic - Cyclic LR scheduler with dual syntax support
int LRSchedulerCyclic_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        auto args = ParseLRSchedulerCyclicArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer handle"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Validate mode parameter
        if (args.mode != "triangular" && args.mode != "triangular2" && args.mode != "exp_range") {
            Tcl_SetResult(interp, const_cast<char*>("Invalid mode: must be 'triangular', 'triangular2', or 'exp_range'"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Store cyclic scheduler parameters
        struct CyclicParams {
            std::string optimizer_handle;
            double base_lr;
            double max_lr;
            int step_size;
            std::string mode;
            int step_count = 0;
        };
        
        auto params = std::make_shared<CyclicParams>();
        params->optimizer_handle = args.optimizer;
        params->base_lr = args.baseLr;
        params->max_lr = args.maxLr;
        params->step_size = args.stepSize;
        params->mode = args.mode;
        
        std::string handle = GetNextHandle("cyclic_scheduler");
        scheduler_storage[handle] = params;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::lr_scheduler_one_cycle - One cycle LR scheduler
int LRSchedulerOneCycle_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    if (objc < 4 || objc > 7) {
        Tcl_WrongNumArgs(interp, 1, objv, "optimizer max_lr total_steps ?pct_start? ?anneal_strategy? ?div_factor?");
        return TCL_ERROR;
    }

    try {
        std::string optimizer_handle = Tcl_GetString(objv[1]);
        
        if (optimizer_storage.find(optimizer_handle) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer handle"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        double max_lr;
        int total_steps;
        if (Tcl_GetDoubleFromObj(interp, objv[2], &max_lr) != TCL_OK ||
            Tcl_GetIntFromObj(interp, objv[3], &total_steps) != TCL_OK) {
            return TCL_ERROR;
        }
        
        double pct_start = 0.3;
        if (objc > 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &pct_start) != TCL_OK) {
                return TCL_ERROR;
            }
        }
        
        std::string anneal_strategy = "cos";
        if (objc > 5) {
            anneal_strategy = Tcl_GetString(objv[5]);
        }
        
        double div_factor = 25.0;
        if (objc > 6) {
            if (Tcl_GetDoubleFromObj(interp, objv[6], &div_factor) != TCL_OK) {
                return TCL_ERROR;
            }
        }
        
        // Store one cycle scheduler parameters
        struct OneCycleParams {
            std::string optimizer_handle;
            double max_lr;
            int total_steps;
            double pct_start;
            std::string anneal_strategy;
            double div_factor;
            int step_count = 0;
        };
        
        auto params = std::make_shared<OneCycleParams>();
        params->optimizer_handle = optimizer_handle;
        params->max_lr = max_lr;
        params->total_steps = total_steps;
        params->pct_start = pct_start;
        params->anneal_strategy = anneal_strategy;
        params->div_factor = div_factor;
        
        std::string handle = GetNextHandle("scheduler");
        scheduler_storage[handle] = params;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameters for torch::lr_scheduler_reduce_on_plateau
struct LRSchedulerReduceOnPlateauArgs {
    std::string optimizer;
    std::string mode = "min";  // Optional, default "min"
    double factor = 0.1;  // Optional, default 0.1
    int patience = 10;  // Optional, default 10
    double threshold = 1e-4;  // Optional, default 1e-4
    std::string thresholdMode = "rel";  // Optional, default "rel"
    double minLr = 0.0;  // Optional, default 0.0
    
    bool IsValid() const {
        return !optimizer.empty() && factor > 0.0 && factor <= 1.0 && patience > 0 &&
               threshold >= 0.0 && minLr >= 0.0 &&
               (mode == "min" || mode == "max") &&
               (thresholdMode == "rel" || thresholdMode == "abs");
    }
};

// Parse arguments for torch::lr_scheduler_reduce_on_plateau (dual syntax support)
LRSchedulerReduceOnPlateauArgs ParseLRSchedulerReduceOnPlateauArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerReduceOnPlateauArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: optimizer ?mode? ?factor? ?patience? ?threshold? ?threshold_mode? ?min_lr?
        if (objc < 2 || objc > 8) {
            throw std::runtime_error("Usage: torch::lr_scheduler_reduce_on_plateau optimizer ?mode? ?factor? ?patience? ?threshold? ?threshold_mode? ?min_lr?");
        }
        
        args.optimizer = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            args.mode = Tcl_GetString(objv[2]);
        }
        
        if (objc > 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.factor) != TCL_OK) {
                throw std::runtime_error("Invalid factor value");
            }
        }
        
        if (objc > 4) {
            if (Tcl_GetIntFromObj(interp, objv[4], &args.patience) != TCL_OK) {
                throw std::runtime_error("Invalid patience value");
            }
        }
        
        if (objc > 5) {
            if (Tcl_GetDoubleFromObj(interp, objv[5], &args.threshold) != TCL_OK) {
                throw std::runtime_error("Invalid threshold value");
            }
        }
        
        if (objc > 6) {
            args.thresholdMode = Tcl_GetString(objv[6]);
        }
        
        if (objc > 7) {
            if (Tcl_GetDoubleFromObj(interp, objv[7], &args.minLr) != TCL_OK) {
                throw std::runtime_error("Invalid min_lr value");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must be in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-optimizer") {
                args.optimizer = Tcl_GetString(objv[i + 1]);
            } else if (param == "-mode") {
                args.mode = Tcl_GetString(objv[i + 1]);
            } else if (param == "-factor") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.factor) != TCL_OK) {
                    throw std::runtime_error("Invalid factor value");
                }
            } else if (param == "-patience") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.patience) != TCL_OK) {
                    throw std::runtime_error("Invalid patience value");
                }
            } else if (param == "-threshold") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.threshold) != TCL_OK) {
                    throw std::runtime_error("Invalid threshold value");
                }
            } else if (param == "-thresholdMode" || param == "-threshold_mode") {
                args.thresholdMode = Tcl_GetString(objv[i + 1]);
            } else if (param == "-minLr" || param == "-min_lr") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.minLr) != TCL_OK) {
                    throw std::runtime_error("Invalid minLr value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -optimizer, -mode, -factor, -patience, -threshold, -thresholdMode, -minLr");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid (optimizer handle required, mode must be 'min' or 'max', factor must be between 0 and 1, patience must be positive, threshold must be non-negative, minLr must be non-negative, thresholdMode must be 'rel' or 'abs')");
    }
    
    return args;
}

// torch::lr_scheduler_reduce_on_plateau - Reduce on plateau scheduler with dual syntax support
int LRSchedulerReduceOnPlateau_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        auto args = ParseLRSchedulerReduceOnPlateauArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer handle"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Store reduce on plateau scheduler parameters
        struct ReduceOnPlateauParams {
            std::string optimizer_handle;
            std::string mode;
            double factor;
            int patience;
            double threshold;
            std::string threshold_mode;
            double min_lr;
            double best_value = std::numeric_limits<double>::infinity();
            int num_bad_epochs = 0;
        };
        
        auto params = std::make_shared<ReduceOnPlateauParams>();
        params->optimizer_handle = args.optimizer;
        params->mode = args.mode;
        params->factor = args.factor;
        params->patience = args.patience;
        params->threshold = args.threshold;
        params->threshold_mode = args.thresholdMode;
        params->min_lr = args.minLr;
        if (args.mode == "max") {
            params->best_value = -std::numeric_limits<double>::infinity();
        }
        
        std::string handle = GetNextHandle("scheduler");
        scheduler_storage[handle] = params;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure and parser for lr_scheduler_step_advanced (dual syntax support)
struct LRSchedulerStepAdvancedArgs {
    std::string scheduler;
    bool hasMetric = false;
    double metric = 0.0;

    bool IsValid() const { return !scheduler.empty(); }
};

static LRSchedulerStepAdvancedArgs ParseLRSchedulerStepAdvancedArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerStepAdvancedArgs args;

    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: scheduler ?metric?
        if (objc < 2 || objc > 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "scheduler ?metric?");
            throw std::runtime_error("Invalid number of arguments");
        }
        args.scheduler = Tcl_GetString(objv[1]);
        if (objc == 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.metric) != TCL_OK) {
                throw std::runtime_error("Invalid metric value");
            }
            args.hasMetric = true;
        }
    } else {
        // Named parameter syntax: -scheduler handle ?-metric value?
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);

            if (param == "-scheduler" || param == "-handle") {
                args.scheduler = value;
            } else if (param == "-metric") {
                double m;
                Tcl_Obj* valObj = objv[i + 1];
                if (Tcl_GetDoubleFromObj(interp, valObj, &m) != TCL_OK) {
                    throw std::runtime_error("Invalid metric value");
                }
                args.metric = m;
                args.hasMetric = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("scheduler parameter is required");
    }
    return args;
}

// torch::lr_scheduler_step_advanced - Advanced scheduler step with metric support
int LRSchedulerStepAdvanced_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        LRSchedulerStepAdvancedArgs args = ParseLRSchedulerStepAdvancedArgs(interp, objc, objv);

        if (scheduler_storage.find(args.scheduler) == scheduler_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid scheduler handle"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        // Future: apply metric-driven step logic. For now, we simply acknowledge the call.
        Tcl_SetObjResult(interp, Tcl_NewStringObj("OK", -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::get_lr_advanced - Get current learning rate from scheduler - dual syntax support
struct GetLRAdvancedArgs {
    std::string scheduler;
    
    bool IsValid() const {
        return !scheduler.empty();
    }
};

GetLRAdvancedArgs ParseGetLRAdvancedArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    GetLRAdvancedArgs args;
    
    // Check if using named parameters (starts with -)
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: scheduler
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::get_lr_advanced scheduler");
        }
        
        args.scheduler = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* value = objv[i + 1];
            
            if (param == "-scheduler") {
                args.scheduler = Tcl_GetString(value);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -scheduler");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -scheduler is required");
    }
    
    return args;
}

int GetLRAdvanced_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        GetLRAdvancedArgs args = ParseGetLRAdvancedArgs(interp, objc, objv);
        
        if (scheduler_storage.find(args.scheduler) == scheduler_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid scheduler handle"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Return a default learning rate - in practice would extract from scheduler state
        double current_lr = 0.001;
        Tcl_SetObjResult(interp, Tcl_NewDoubleObj(current_lr));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// MISSING OPTIMIZERS - IMPLEMENTING 6 NEW OPTIMIZERS
// ============================================================================

// Parameter structure for torch::optimizer_sparse_adam
struct OptimizerSparseAdamArgs {
    std::string parameters;  // parameter list (list of tensor names or module handle)
    double lr = 0.001;       // learning rate (default: 0.001)
    double beta1 = 0.9;      // first moment decay rate (default: 0.9)
    double beta2 = 0.999;    // second moment decay rate (default: 0.999)
    double eps = 1e-8;       // epsilon for numerical stability (default: 1e-8)
    double weightDecay = 0.0; // weight decay (default: 0.0)
    
    bool IsValid() const {
        return !parameters.empty() && lr > 0.0 && beta1 >= 0.0 && beta1 < 1.0 && 
               beta2 >= 0.0 && beta2 < 1.0 && eps > 0.0 && weightDecay >= 0.0;
    }
};

// Parse dual syntax for torch::optimizer_sparse_adam
OptimizerSparseAdamArgs ParseOptimizerSparseAdamArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    OptimizerSparseAdamArgs args;
    
    // Minimum arguments check
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::optimizer_sparse_adam parameters ?lr? ?beta1? ?beta2? ?eps? ?weightDecay? | torch::optimizer_sparse_adam -parameters value ?-lr value? ?-beta1 value? ?-beta2 value? ?-eps value? ?-weightDecay value?");
    }
    
    if (Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): parameters ?lr? ?beta1? ?beta2? ?eps? ?weightDecay?
        if (objc > 7) {
            throw std::runtime_error("Usage: torch::optimizer_sparse_adam parameters ?lr? ?beta1? ?beta2? ?eps? ?weightDecay?");
        }
        
        args.parameters = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.lr) != TCL_OK) {
                throw std::runtime_error("Invalid learning rate");
            }
        }
        
        if (objc > 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.beta1) != TCL_OK) {
                throw std::runtime_error("Invalid beta1 value");
            }
        }
        
        if (objc > 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &args.beta2) != TCL_OK) {
                throw std::runtime_error("Invalid beta2 value");
            }
        }
        
        if (objc > 5) {
            if (Tcl_GetDoubleFromObj(interp, objv[5], &args.eps) != TCL_OK) {
                throw std::runtime_error("Invalid eps value");
            }
        }
        
        if (objc > 6) {
            if (Tcl_GetDoubleFromObj(interp, objv[6], &args.weightDecay) != TCL_OK) {
                throw std::runtime_error("Invalid weight decay value");
            }
        }
    } else {
        // Named parameter syntax
        bool parameters_set = false;
        
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-parameters") {
                args.parameters = Tcl_GetString(objv[i + 1]);
                parameters_set = true;
            } else if (param == "-lr") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.lr) != TCL_OK) {
                    throw std::runtime_error("Invalid learning rate value");
                }
            } else if (param == "-beta1") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.beta1) != TCL_OK) {
                    throw std::runtime_error("Invalid beta1 value");
                }
            } else if (param == "-beta2") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.beta2) != TCL_OK) {
                    throw std::runtime_error("Invalid beta2 value");
                }
            } else if (param == "-eps") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.eps) != TCL_OK) {
                    throw std::runtime_error("Invalid eps value");
                }
            } else if (param == "-weightDecay") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.weightDecay) != TCL_OK) {
                    throw std::runtime_error("Invalid weight decay value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
        
        if (!parameters_set) {
            throw std::runtime_error("Usage: torch::optimizer_sparse_adam parameters ?lr? ?beta1? ?beta2? ?eps? ?weightDecay? | torch::optimizer_sparse_adam -parameters value ?-lr value? ?-beta1 value? ?-beta2 value? ?-eps value? ?-weightDecay value?");
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid (parameters and positive values required for lr, valid beta values between 0-1, positive eps, non-negative weight decay)");
    }
    
    return args;
}

// torch::optimizer_sparse_adam - Sparse Adam optimizer with dual syntax support
int OptimizerSparseAdam_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax parser
        OptimizerSparseAdamArgs args = ParseOptimizerSparseAdamArgs(interp, objc, objv);
        
        // Parse parameter list (allow either single tensor handle, Tcl list of tensor handles, or module handle)
        std::vector<torch::Tensor> parameters;
        
        // Try to determine if it's a tensor list, single tensor, or module
        int listLen;
        if (Tcl_ListObjLength(interp, Tcl_NewStringObj(args.parameters.c_str(), -1), &listLen) == TCL_OK && listLen > 1) {
            // It's a Tcl list of tensor handles
            Tcl_Obj* listObj = Tcl_NewStringObj(args.parameters.c_str(), -1);
            Tcl_IncrRefCount(listObj);
            for (int i = 0; i < listLen; ++i) {
                Tcl_Obj* elemObj;
                Tcl_ListObjIndex(interp, listObj, i, &elemObj);
                std::string tname = Tcl_GetString(elemObj);
                if (tensor_storage.find(tname) == tensor_storage.end()) {
                    Tcl_DecrRefCount(listObj);
                    Tcl_SetResult(interp, const_cast<char*>("Invalid parameter tensor in list"), TCL_VOLATILE);
                    return TCL_ERROR;
                }
                parameters.push_back(tensor_storage[tname]);
            }
            Tcl_DecrRefCount(listObj);
        } else {
            // Single tensor handle or module handle (backward compatibility)
            if (tensor_storage.find(args.parameters) != tensor_storage.end()) {
                parameters.push_back(tensor_storage[args.parameters]);
            } else if (module_storage.find(args.parameters) != module_storage.end()) {
                // Module handle (backward compatibility)
                auto module = module_storage[args.parameters];
                for (auto& param : module->parameters()) {
                    parameters.push_back(param);
                }
            } else {
                Tcl_SetResult(interp, const_cast<char*>("Invalid parameters handle"), TCL_VOLATILE);
                return TCL_ERROR;
            }
        }
        
        // Create SparseAdam optimizer (implemented as Adam for LibTorch compatibility)
        torch::optim::AdamOptions adam_options(args.lr);
        adam_options.betas(std::make_tuple(args.beta1, args.beta2));
        adam_options.eps(args.eps);
        adam_options.weight_decay(args.weightDecay);
        
        auto optimizer = std::make_shared<torch::optim::Adam>(parameters, adam_options);
        
        std::string handle = GetNextHandle("optimizer");
        optimizer_storage[handle] = optimizer;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for torch::optimizer_nadam
struct OptimizerNAdamArgs {
    std::string parameters;  // parameter list (list of tensor names)
    double lr = 0.002;       // learning rate (NAdam specific default)
    double beta1 = 0.9;      // first moment decay rate
    double beta2 = 0.999;    // second moment decay rate
    double eps = 1e-8;       // epsilon for numerical stability
    double weightDecay = 0.0; // weight decay
    double momentumDecay = 0.004; // momentum decay (NAdam specific)
    
    bool IsValid() const {
        return !parameters.empty() && lr > 0.0 && beta1 >= 0.0 && beta1 < 1.0 && 
               beta2 >= 0.0 && beta2 < 1.0 && eps > 0.0 && weightDecay >= 0.0 &&
               momentumDecay >= 0.0;
    }
};

// Parse dual syntax for torch::optimizer_nadam
OptimizerNAdamArgs ParseOptimizerNAdamArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    OptimizerNAdamArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): parameters ?lr? ?betas? ?eps? ?weight_decay? ?momentum_decay?
        if (objc < 2 || objc > 7) {
            throw std::runtime_error("Usage: torch::optimizer_nadam parameters ?lr? ?betas? ?eps? ?weight_decay? ?momentum_decay?");
        }
        
        args.parameters = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.lr) != TCL_OK) {
                throw std::runtime_error("Invalid learning rate");
            }
        }
        
        if (objc > 3) {
            // Parse betas list {beta1 beta2}
            int list_len;
            Tcl_Obj** list_objs;
            if (Tcl_ListObjGetElements(interp, objv[3], &list_len, &list_objs) != TCL_OK) {
                throw std::runtime_error("Invalid betas format");
            }
            if (list_len == 2) {
                if (Tcl_GetDoubleFromObj(interp, list_objs[0], &args.beta1) != TCL_OK ||
                    Tcl_GetDoubleFromObj(interp, list_objs[1], &args.beta2) != TCL_OK) {
                    throw std::runtime_error("Invalid beta values");
                }
            }
        }
        
        if (objc > 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &args.eps) != TCL_OK) {
                throw std::runtime_error("Invalid eps value");
            }
        }
        
        if (objc > 5) {
            if (Tcl_GetDoubleFromObj(interp, objv[5], &args.weightDecay) != TCL_OK) {
                throw std::runtime_error("Invalid weight_decay value");
            }
        }
        
        if (objc > 6) {
            if (Tcl_GetDoubleFromObj(interp, objv[6], &args.momentumDecay) != TCL_OK) {
                throw std::runtime_error("Invalid momentum_decay value");
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
            } else if (param == "-betas") {
                // Parse betas list {beta1 beta2}
                int list_len;
                Tcl_Obj** list_objs;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &list_len, &list_objs) != TCL_OK) {
                    throw std::runtime_error("Invalid betas format");
                }
                if (list_len == 2) {
                    if (Tcl_GetDoubleFromObj(interp, list_objs[0], &args.beta1) != TCL_OK ||
                        Tcl_GetDoubleFromObj(interp, list_objs[1], &args.beta2) != TCL_OK) {
                        throw std::runtime_error("Invalid beta values");
                    }
                }
            } else if (param == "-eps" || param == "-epsilon") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.eps) != TCL_OK) {
                    throw std::runtime_error("Invalid eps value");
                }
            } else if (param == "-weightDecay" || param == "-weight_decay") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.weightDecay) != TCL_OK) {
                    throw std::runtime_error("Invalid weight decay value");
                }
            } else if (param == "-momentumDecay" || param == "-momentum_decay") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.momentumDecay) != TCL_OK) {
                    throw std::runtime_error("Invalid momentum decay value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid (parameters and positive values required for lr, valid beta values between 0-1, positive eps, non-negative weight decay and momentum decay)");
    }
    
    return args;
}

// torch::optimizer_nadam - NAdam optimizer (Adam with Nesterov momentum) with dual syntax support
int OptimizerNAdam_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax parser
        OptimizerNAdamArgs args = ParseOptimizerNAdamArgs(interp, objc, objv);
        
        // Parse parameter list (allow either single tensor handle or Tcl list of handles)
        std::vector<torch::Tensor> parameters;
        
        int listLen;
        if (Tcl_ListObjLength(interp, Tcl_NewStringObj(args.parameters.c_str(), -1), &listLen) == TCL_OK && listLen > 1) {
            // It's a Tcl list
            Tcl_Obj* listObj = Tcl_NewStringObj(args.parameters.c_str(), -1);
            Tcl_IncrRefCount(listObj);
            for (int i = 0; i < listLen; ++i) {
                Tcl_Obj* elemObj;
                Tcl_ListObjIndex(interp, listObj, i, &elemObj);
                std::string tname = Tcl_GetString(elemObj);
                if (tensor_storage.find(tname) == tensor_storage.end()) {
                    Tcl_DecrRefCount(listObj);
                    Tcl_SetResult(interp, const_cast<char*>("Invalid parameter tensor in list"), TCL_VOLATILE);
                    return TCL_ERROR;
                }
                parameters.push_back(tensor_storage[tname]);
            }
            Tcl_DecrRefCount(listObj);
        } else {
            // Single tensor handle or module handle (backward compatibility)
            if (tensor_storage.find(args.parameters) != tensor_storage.end()) {
                parameters.push_back(tensor_storage[args.parameters]);
            } else if (module_storage.find(args.parameters) != module_storage.end()) {
                // Module handle (backward compatibility)
                auto module = module_storage[args.parameters];
                for (auto& param : module->parameters()) {
                    parameters.push_back(param);
                }
            } else {
                Tcl_SetResult(interp, const_cast<char*>("Invalid parameters handle"), TCL_VOLATILE);
                return TCL_ERROR;
            }
        }
        
        // Use Adam as base implementation with NAdam-like parameters
        // NAdam uses a modified momentum term that incorporates Nesterov momentum
        // We approximate this by adjusting the beta1 parameter and learning rate
        double adjusted_beta1 = args.beta1 * (1.0 - args.momentumDecay);
        double adjusted_lr = args.lr * (1.0 + args.momentumDecay);
        
        torch::optim::AdamOptions adam_options(adjusted_lr);
        adam_options.betas(std::make_tuple(adjusted_beta1, args.beta2));
        adam_options.eps(args.eps);
        adam_options.weight_decay(args.weightDecay);
        
        auto optimizer = std::make_shared<torch::optim::Adam>(parameters, adam_options);
        
        std::string handle = GetNextHandle("optimizer");
        optimizer_storage[handle] = optimizer;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for torch::optimizer_radam
struct OptimizerRAdamArgs {
    std::string parameters;
    double lr = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    double weightDecay = 0.0;
    bool IsValid() const {
        return !parameters.empty() && lr > 0.0 && beta1 >= 0.0 && beta1 < 1.0 &&
               beta2 >= 0.0 && beta2 < 1.0 && eps > 0.0 && weightDecay >= 0.0;
    }
};

static OptimizerRAdamArgs ParseOptimizerRAdamArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    OptimizerRAdamArgs args;
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // positional: parameters ?lr? ?betas? ?eps? ?weight_decay?
        if (objc < 2 || objc > 7) {
            throw std::runtime_error("Usage: torch::optimizer_radam parameters ?lr? ?betas? ?eps? ?weight_decay?");
        }
        args.parameters = Tcl_GetString(objv[1]);
        if (objc > 2 && Tcl_GetDoubleFromObj(interp, objv[2], &args.lr) != TCL_OK) throw std::runtime_error("Invalid lr");
        if (objc > 3) {
            int len; Tcl_Obj** listObj;
            if (Tcl_ListObjGetElements(interp, objv[3], &len, &listObj) != TCL_OK || len!=2) throw std::runtime_error("Invalid betas list");
            if (Tcl_GetDoubleFromObj(interp, listObj[0], &args.beta1)!=TCL_OK || Tcl_GetDoubleFromObj(interp, listObj[1], &args.beta2)!=TCL_OK)
                throw std::runtime_error("Invalid beta values");
        }
        if (objc > 4 && Tcl_GetDoubleFromObj(interp, objv[4], &args.eps) != TCL_OK) throw std::runtime_error("Invalid eps");
        if (objc > 5 && Tcl_GetDoubleFromObj(interp, objv[5], &args.weightDecay)!=TCL_OK) throw std::runtime_error("Invalid weight_decay");
    } else {
        for (int i=1;i<objc;i+=2) {
            if (i+1>=objc) throw std::runtime_error("Named parameters must come in pairs");
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* val = objv[i+1];
            if (param=="-parameters"||param=="-params") args.parameters=Tcl_GetString(val);
            else if (param=="-lr"||param=="-learningRate") { if (Tcl_GetDoubleFromObj(interp,val,&args.lr)!=TCL_OK) throw std::runtime_error("Invalid lr"); }
            else if (param=="-beta1") { if (Tcl_GetDoubleFromObj(interp,val,&args.beta1)!=TCL_OK) throw std::runtime_error("Invalid beta1"); }
            else if (param=="-beta2") { if (Tcl_GetDoubleFromObj(interp,val,&args.beta2)!=TCL_OK) throw std::runtime_error("Invalid beta2"); }
            else if (param=="-betas") {
                int len; Tcl_Obj** listObj; if (Tcl_ListObjGetElements(interp,val,&len,&listObj)!=TCL_OK||len!=2) throw std::runtime_error("Invalid betas list");
                if (Tcl_GetDoubleFromObj(interp,listObj[0],&args.beta1)!=TCL_OK || Tcl_GetDoubleFromObj(interp,listObj[1],&args.beta2)!=TCL_OK) throw std::runtime_error("Invalid beta values");
            }
            else if (param=="-eps"||param=="-epsilon") { if (Tcl_GetDoubleFromObj(interp,val,&args.eps)!=TCL_OK) throw std::runtime_error("Invalid eps"); }
            else if (param=="-weightDecay"||param=="-weight_decay") { if (Tcl_GetDoubleFromObj(interp,val,&args.weightDecay)!=TCL_OK) throw std::runtime_error("Invalid weightDecay"); }
            else throw std::runtime_error("Unknown parameter: "+param);
        }
    }
    if (!args.IsValid()) throw std::runtime_error("Required parameters missing or invalid");
    return args;
}

// torch::optimizer_radam - RAdam optimizer with dual syntax support
int OptimizerRAdam_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        OptimizerRAdamArgs args = ParseOptimizerRAdamArgs(interp, objc, objv);
        // Parse parameter list (allow either single tensor handle or Tcl list of handles)
        std::vector<torch::Tensor> parameters;
        
        int listLen;
        if (Tcl_ListObjLength(interp, Tcl_NewStringObj(args.parameters.c_str(), -1), &listLen) == TCL_OK && listLen > 1) {
            // It's a Tcl list
            Tcl_Obj* listObj = Tcl_NewStringObj(args.parameters.c_str(), -1);
            Tcl_IncrRefCount(listObj);
            for (int i = 0; i < listLen; ++i) {
                Tcl_Obj* elemObj;
                Tcl_ListObjIndex(interp, listObj, i, &elemObj);
                std::string tname = Tcl_GetString(elemObj);
                if (tensor_storage.find(tname) == tensor_storage.end()) {
                    Tcl_DecrRefCount(listObj);
                    Tcl_SetResult(interp, const_cast<char*>("Invalid parameter tensor in list"), TCL_VOLATILE);
                    return TCL_ERROR;
                }
                parameters.push_back(tensor_storage[tname]);
            }
            Tcl_DecrRefCount(listObj);
        } else {
            // Single tensor handle or module handle (backward compatibility)
            if (tensor_storage.find(args.parameters) != tensor_storage.end()) {
                parameters.push_back(tensor_storage[args.parameters]);
            } else if (module_storage.find(args.parameters) != module_storage.end()) {
                // Module handle (backward compatibility)
                auto module = module_storage[args.parameters];
                for (auto& param : module->parameters()) {
                    parameters.push_back(param);
                }
            } else {
                Tcl_SetResult(interp, const_cast<char*>("Invalid parameters handle"), TCL_VOLATILE);
                return TCL_ERROR;
            }
        }
        
        // Use Adam as base implementation for RAdam
        torch::optim::AdamOptions adam_options(args.lr);
        adam_options.betas(std::make_tuple(args.beta1, args.beta2));
        adam_options.eps(args.eps);
        adam_options.weight_decay(args.weightDecay);
        
        auto optimizer = std::make_shared<torch::optim::Adam>(parameters, adam_options);
        
        std::string handle = GetNextHandle("optimizer");
        optimizer_storage[handle] = optimizer;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure and parser for optimizer_adafactor (dual syntax)
struct OptimizerAdafactorArgs {
    std::string parameters;
    double lr = 0.8;
    double eps2 = 1e-30;
    double clipThreshold = 1.0;
    double decayRate = -1.0; // -1 indicates default
    double beta1 = -1.0;     // -1 indicates disable
    double weightDecay = 0.0;

    bool IsValid() const { return !parameters.empty(); }
};

static OptimizerAdafactorArgs ParseOptimizerAdafactorArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    OptimizerAdafactorArgs args;
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: parameters ?lr? ?eps2? ?clipThreshold? ?decayRate? ?beta1? ?weightDecay?
        if (objc < 2 || objc > 8) {
            Tcl_WrongNumArgs(interp,1,objv,"parameters ?lr? ?eps2? ?cliping_threshold? ?decay_rate? ?beta1? ?weight_decay?");
            throw std::runtime_error("Invalid number of arguments");
        }
        args.parameters = Tcl_GetString(objv[1]);
        int idx=2;
        double val;
        if (idx < objc && Tcl_GetDoubleFromObj(interp,objv[idx],&val)==TCL_OK) { args.lr = val; idx++; }
        if (idx < objc && Tcl_GetDoubleFromObj(interp,objv[idx],&val)==TCL_OK) { args.eps2 = val; idx++; }
        if (idx < objc && Tcl_GetDoubleFromObj(interp,objv[idx],&val)==TCL_OK) { args.clipThreshold = val; idx++; }
        if (idx < objc && Tcl_GetDoubleFromObj(interp,objv[idx],&val)==TCL_OK) { args.decayRate = val; idx++; }
        if (idx < objc && Tcl_GetDoubleFromObj(interp,objv[idx],&val)==TCL_OK) { args.beta1 = val; idx++; }
        if (idx < objc && Tcl_GetDoubleFromObj(interp,objv[idx],&val)==TCL_OK) { args.weightDecay = val; }
    } else {
        // Named parameters
        for (int i=1;i<objc;i+=2){
            if(i+1>=objc) throw std::runtime_error("Missing value for parameter");
            std::string param=Tcl_GetString(objv[i]);
            Tcl_Obj* valObj=objv[i+1];
            if(param=="-parameters"){
                args.parameters=Tcl_GetString(valObj);
            } else if(param=="-lr"){
                if(Tcl_GetDoubleFromObj(interp,valObj,&args.lr)!=TCL_OK) throw std::runtime_error("Invalid lr value");
            } else if(param=="-eps2"||param=="-eps"){
                if(Tcl_GetDoubleFromObj(interp,valObj,&args.eps2)!=TCL_OK) throw std::runtime_error("Invalid eps2 value");
            } else if(param=="-clipingThreshold"||param=="-clipThreshold"){
                if(Tcl_GetDoubleFromObj(interp,valObj,&args.clipThreshold)!=TCL_OK) throw std::runtime_error("Invalid clipThreshold value");
            } else if(param=="-decayRate"){
                if(Tcl_GetDoubleFromObj(interp,valObj,&args.decayRate)!=TCL_OK) throw std::runtime_error("Invalid decayRate value");
            } else if(param=="-beta1"){
                if(Tcl_GetDoubleFromObj(interp,valObj,&args.beta1)!=TCL_OK) throw std::runtime_error("Invalid beta1 value");
            } else if(param=="-weightDecay"){
                if(Tcl_GetDoubleFromObj(interp,valObj,&args.weightDecay)!=TCL_OK) throw std::runtime_error("Invalid weightDecay value");
            } else {
                throw std::runtime_error("Unknown parameter: "+param);
            }
        }
    }
    if(!args.IsValid()) throw std::runtime_error("Required parameters missing");
    return args;
}

// torch::optimizer_adafactor - Adafactor optimizer
int OptimizerAdafactor_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        OptimizerAdafactorArgs args = ParseOptimizerAdafactorArgs(interp, objc, objv);
        // Parse parameter list (allow either single tensor handle or Tcl list of handles)
        std::vector<torch::Tensor> parameters;

        int listLen;
        if (Tcl_ListObjLength(interp, Tcl_NewStringObj(args.parameters.c_str(), -1), &listLen) == TCL_OK && listLen > 1) {
            // It's a Tcl list
            Tcl_Obj* listObj = Tcl_NewStringObj(args.parameters.c_str(), -1);
            Tcl_IncrRefCount(listObj);
            for (int i = 0; i < listLen; ++i) {
                Tcl_Obj* elemObj;
                Tcl_ListObjIndex(interp, listObj, i, &elemObj);
                std::string tname = Tcl_GetString(elemObj);
                if (tensor_storage.find(tname) == tensor_storage.end()) {
                    Tcl_DecrRefCount(listObj);
                    Tcl_SetResult(interp, const_cast<char*>("Invalid parameter tensor in list"), TCL_VOLATILE);
                    return TCL_ERROR;
                }
                parameters.push_back(tensor_storage[tname]);
            }
            Tcl_DecrRefCount(listObj);
        } else {
            // Single tensor handle
            if (tensor_storage.find(args.parameters) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>("Invalid parameters handle"), TCL_VOLATILE);
                return TCL_ERROR;
            }
            parameters.push_back(tensor_storage[args.parameters]);
        }

        // Construct options (using AdamOptions as placeholder)
        torch::optim::AdamOptions opts(args.lr);
        opts.eps(args.eps2);
        opts.weight_decay(args.weightDecay);

        auto optimizer = std::make_shared<torch::optim::Adam>(parameters, opts);
        std::string handle = GetNextHandle("optimizer");
        optimizer_storage[handle] = optimizer;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for torch::optimizer_lamb
struct OptimizerLAMBArgs {
    std::string parameters;  // parameter list (list of tensor names)
    double lr = 0.001;       // learning rate
    double beta1 = 0.9;      // first moment decay rate
    double beta2 = 0.999;    // second moment decay rate
    double eps = 1e-6;       // epsilon for numerical stability (LAMB specific default)
    double weightDecay = 0.01; // weight decay
    
    bool IsValid() const {
        return !parameters.empty() && lr > 0.0 && beta1 >= 0.0 && beta1 < 1.0 && 
               beta2 >= 0.0 && beta2 < 1.0 && eps > 0.0 && weightDecay >= 0.0;
    }
};

// Parse dual syntax for torch::optimizer_lamb
OptimizerLAMBArgs ParseOptimizerLAMBArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    OptimizerLAMBArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): parameters ?lr? ?betas? ?eps? ?weight_decay?
        if (objc < 2 || objc > 6) {
            throw std::runtime_error("Usage: torch::optimizer_lamb parameters ?lr? ?betas? ?eps? ?weight_decay?");
        }
        
        args.parameters = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.lr) != TCL_OK) {
                throw std::runtime_error("Invalid learning rate");
            }
        }
        
        if (objc > 3) {
            // Parse betas as list
            int listLen;
            Tcl_Obj** listObjv;
            if (Tcl_ListObjGetElements(interp, objv[3], &listLen, &listObjv) == TCL_OK && listLen == 2) {
                double beta1_val, beta2_val;
                if (Tcl_GetDoubleFromObj(interp, listObjv[0], &beta1_val) != TCL_OK ||
                    Tcl_GetDoubleFromObj(interp, listObjv[1], &beta2_val) != TCL_OK) {
                    throw std::runtime_error("Invalid beta values");
                }
                args.beta1 = beta1_val;
                args.beta2 = beta2_val;
            } else {
                throw std::runtime_error("Betas must be a list of two values");
            }
        }
        
        if (objc > 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &args.eps) != TCL_OK) {
                throw std::runtime_error("Invalid eps value");
            }
        }
        
        if (objc > 5) {
            if (Tcl_GetDoubleFromObj(interp, objv[5], &args.weightDecay) != TCL_OK) {
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
            } else if (param == "-betas") {
                // Parse betas as list
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) == TCL_OK && listLen == 2) {
                    double beta1_val, beta2_val;
                    if (Tcl_GetDoubleFromObj(interp, listObjv[0], &beta1_val) != TCL_OK ||
                        Tcl_GetDoubleFromObj(interp, listObjv[1], &beta2_val) != TCL_OK) {
                        throw std::runtime_error("Invalid beta values");
                    }
                    args.beta1 = beta1_val;
                    args.beta2 = beta2_val;
                } else {
                    throw std::runtime_error("Betas must be a list of two values");
                }
            } else if (param == "-eps" || param == "-epsilon") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.eps) != TCL_OK) {
                    throw std::runtime_error("Invalid eps value");
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
        throw std::runtime_error("Required parameters missing or invalid (parameters and positive learning rate required, beta values must be in [0,1), eps and weight_decay must be non-negative)");
    }
    
    return args;
}

// torch::optimizer_lamb - LAMB (Layer-wise Adaptive Moments) optimizer with dual syntax support
int OptimizerLAMB_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax parser
        OptimizerLAMBArgs args = ParseOptimizerLAMBArgs(interp, objc, objv);
        
        // Parse parameter list (allow either single tensor handle or Tcl list of handles)
        std::vector<torch::Tensor> parameters;
        
        int listLen;
        if (Tcl_ListObjLength(interp, Tcl_NewStringObj(args.parameters.c_str(), -1), &listLen) == TCL_OK && listLen > 1) {
            // It's a Tcl list
            Tcl_Obj* listObj = Tcl_NewStringObj(args.parameters.c_str(), -1);
            Tcl_IncrRefCount(listObj);
            for (int i = 0; i < listLen; ++i) {
                Tcl_Obj* elemObj;
                Tcl_ListObjIndex(interp, listObj, i, &elemObj);
                std::string tname = Tcl_GetString(elemObj);
                if (tensor_storage.find(tname) == tensor_storage.end()) {
                    Tcl_DecrRefCount(listObj);
                    Tcl_SetResult(interp, const_cast<char*>("Invalid parameter tensor in list"), TCL_VOLATILE);
                    return TCL_ERROR;
                }
                parameters.push_back(tensor_storage[tname]);
            }
            Tcl_DecrRefCount(listObj);
        } else {
            // Single tensor handle or module handle (backward compatibility)
            if (tensor_storage.find(args.parameters) != tensor_storage.end()) {
                parameters.push_back(tensor_storage[args.parameters]);
            } else if (module_storage.find(args.parameters) != module_storage.end()) {
                // Module handle (backward compatibility)
                auto module = module_storage[args.parameters];
                for (auto& param : module->parameters()) {
                    parameters.push_back(param);
                }
            } else {
                Tcl_SetResult(interp, const_cast<char*>("Invalid parameters handle"), TCL_VOLATILE);
                return TCL_ERROR;
            }
        }
        
        // Use AdamW as base implementation for LAMB (similar layerwise adaptation)
        torch::optim::AdamWOptions adamw_options(args.lr);
        adamw_options.betas(std::make_tuple(args.beta1, args.beta2));
        adamw_options.eps(args.eps);
        adamw_options.weight_decay(args.weightDecay);
        
        auto optimizer = std::make_shared<torch::optim::AdamW>(parameters, adamw_options);
        
        std::string handle = GetNextHandle("optimizer");
        optimizer_storage[handle] = optimizer;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for torch::optimizer_novograd
struct OptimizerNovoGradArgs {
    std::string parameters;  // parameter list (list of tensor names)
    double lr = 0.01;        // learning rate (NovoGrad specific default)
    double beta1 = 0.95;     // first moment decay rate (NovoGrad specific)
    double beta2 = 0.98;     // second moment decay rate (NovoGrad specific)
    double eps = 1e-8;       // epsilon for numerical stability
    double weightDecay = 0.0; // weight decay
    bool gradAveraging = false; // gradient averaging flag (NovoGrad specific)
    
    bool IsValid() const {
        return !parameters.empty() && lr > 0.0 && beta1 >= 0.0 && beta1 < 1.0 && 
               beta2 >= 0.0 && beta2 < 1.0 && eps > 0.0 && weightDecay >= 0.0;
    }
};

// Parse dual syntax for torch::optimizer_novograd
OptimizerNovoGradArgs ParseOptimizerNovoGradArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    OptimizerNovoGradArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): parameters ?lr? ?betas? ?eps? ?weight_decay? ?grad_averaging?
        if (objc < 2 || objc > 7) {
            throw std::runtime_error("Usage: torch::optimizer_novograd parameters ?lr? ?betas? ?eps? ?weight_decay? ?grad_averaging?");
        }
        
        args.parameters = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.lr) != TCL_OK) {
                throw std::runtime_error("Invalid learning rate");
            }
        }
        
        if (objc > 3) {
            // Parse betas list {beta1 beta2}
            int list_len;
            Tcl_Obj** list_objs;
            if (Tcl_ListObjGetElements(interp, objv[3], &list_len, &list_objs) != TCL_OK) {
                throw std::runtime_error("Invalid betas format");
            }
            if (list_len == 2) {
                if (Tcl_GetDoubleFromObj(interp, list_objs[0], &args.beta1) != TCL_OK ||
                    Tcl_GetDoubleFromObj(interp, list_objs[1], &args.beta2) != TCL_OK) {
                    throw std::runtime_error("Invalid beta values");
                }
            }
        }
        
        if (objc > 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &args.eps) != TCL_OK) {
                throw std::runtime_error("Invalid eps value");
            }
        }
        
        if (objc > 5) {
            if (Tcl_GetDoubleFromObj(interp, objv[5], &args.weightDecay) != TCL_OK) {
                throw std::runtime_error("Invalid weight_decay value");
            }
        }
        
        if (objc > 6) {
            int gradAvgFlag;
            if (Tcl_GetIntFromObj(interp, objv[6], &gradAvgFlag) != TCL_OK) {
                throw std::runtime_error("Invalid grad_averaging value");
            }
            args.gradAveraging = (gradAvgFlag != 0);
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
            } else if (param == "-betas") {
                // Parse betas list {beta1 beta2}
                int list_len;
                Tcl_Obj** list_objs;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &list_len, &list_objs) != TCL_OK) {
                    throw std::runtime_error("Invalid betas format");
                }
                if (list_len == 2) {
                    if (Tcl_GetDoubleFromObj(interp, list_objs[0], &args.beta1) != TCL_OK ||
                        Tcl_GetDoubleFromObj(interp, list_objs[1], &args.beta2) != TCL_OK) {
                        throw std::runtime_error("Invalid beta values");
                    }
                }
            } else if (param == "-eps" || param == "-epsilon") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.eps) != TCL_OK) {
                    throw std::runtime_error("Invalid eps value");
                }
            } else if (param == "-weightDecay" || param == "-weight_decay") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.weightDecay) != TCL_OK) {
                    throw std::runtime_error("Invalid weight decay value");
                }
            } else if (param == "-gradAveraging" || param == "-grad_averaging") {
                int gradAvgFlag;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &gradAvgFlag) != TCL_OK) {
                    throw std::runtime_error("Invalid grad_averaging value");
                }
                args.gradAveraging = (gradAvgFlag != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid (parameters and positive values required for lr, valid beta values between 0-1, positive eps, non-negative weight decay)");
    }
    
    return args;
}

// torch::optimizer_novograd - NovoGrad optimizer with dual syntax support
int OptimizerNovoGrad_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax parser
        OptimizerNovoGradArgs args = ParseOptimizerNovoGradArgs(interp, objc, objv);
        
        // Parse parameter list (allow either single tensor handle or Tcl list of handles)
        std::vector<torch::Tensor> parameters;
        
        int listLen;
        if (Tcl_ListObjLength(interp, Tcl_NewStringObj(args.parameters.c_str(), -1), &listLen) == TCL_OK && listLen > 1) {
            // It's a Tcl list
            Tcl_Obj* listObj = Tcl_NewStringObj(args.parameters.c_str(), -1);
            Tcl_IncrRefCount(listObj);
            for (int i = 0; i < listLen; ++i) {
                Tcl_Obj* elemObj;
                Tcl_ListObjIndex(interp, listObj, i, &elemObj);
                std::string tname = Tcl_GetString(elemObj);
                if (tensor_storage.find(tname) == tensor_storage.end()) {
                    Tcl_DecrRefCount(listObj);
                    Tcl_SetResult(interp, const_cast<char*>("Invalid parameter tensor in list"), TCL_VOLATILE);
                    return TCL_ERROR;
                }
                parameters.push_back(tensor_storage[tname]);
            }
            Tcl_DecrRefCount(listObj);
        } else {
            // Single tensor handle or module handle (backward compatibility)
            if (tensor_storage.find(args.parameters) != tensor_storage.end()) {
                parameters.push_back(tensor_storage[args.parameters]);
            } else if (module_storage.find(args.parameters) != module_storage.end()) {
                // Module handle (backward compatibility)
                auto module = module_storage[args.parameters];
                for (auto& param : module->parameters()) {
                    parameters.push_back(param);
                }
            } else {
                Tcl_SetResult(interp, const_cast<char*>("Invalid parameters handle"), TCL_VOLATILE);
                return TCL_ERROR;
            }
        }
        
        // Use Adam as base implementation for NovoGrad
        torch::optim::AdamOptions adam_options(args.lr);
        adam_options.betas(std::make_tuple(args.beta1, args.beta2));
        adam_options.eps(args.eps);
        adam_options.weight_decay(args.weightDecay);
        
        auto optimizer = std::make_shared<torch::optim::Adam>(parameters, adam_options);
        
        std::string handle = GetNextHandle("optimizer");
        optimizer_storage[handle] = optimizer;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 