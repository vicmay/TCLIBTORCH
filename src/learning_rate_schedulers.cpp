#include "libtorchtcl.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Simple scheduler structure to track state
struct LRScheduler {
    std::string optimizer_name;
    std::string scheduler_type;
    double initial_lr;
    double current_lr;
    int step_count;
    
    // Step LR parameters
    int step_size;
    double gamma;
    
    // Exponential LR parameters
    double exp_gamma;
    
    // Cosine Annealing parameters
    int T_max;
    double eta_min;
    
    LRScheduler(const std::string& opt_name, const std::string& type) 
        : optimizer_name(opt_name), scheduler_type(type), step_count(0) {}
};

// Global storage for learning rate schedulers
std::unordered_map<std::string, std::shared_ptr<LRScheduler>> scheduler_storage;

// Helper function to update optimizer learning rate
bool UpdateOptimizerLR(const std::string& optimizer_name, double new_lr) {
    if (optimizer_storage.find(optimizer_name) == optimizer_storage.end()) {
        return false;
    }
    
    auto& optimizer = optimizer_storage[optimizer_name];
    
    // Update learning rate for all parameter groups
    for (auto& group : optimizer->param_groups()) {
        if (group.has_options()) {
            // Try SGD first
            try {
                auto& sgd_options = static_cast<torch::optim::SGDOptions&>(group.options());
                sgd_options.lr(new_lr);
                continue;
            } catch (...) {}
            
            // Try Adam
            try {
                auto& adam_options = static_cast<torch::optim::AdamOptions&>(group.options());
                adam_options.lr(new_lr);
                continue;
            } catch (...) {}
        }
    }
    return true;
}

// Helper function to get current learning rate
double GetOptimizerLR(const std::string& optimizer_name) {
    if (optimizer_storage.find(optimizer_name) == optimizer_storage.end()) {
        return -1.0;
    }
    
    auto& optimizer = optimizer_storage[optimizer_name];
    
    for (auto& group : optimizer->param_groups()) {
        if (group.has_options()) {
            // Try SGD first
            try {
                auto& sgd_options = static_cast<torch::optim::SGDOptions&>(group.options());
                return sgd_options.lr();
            } catch (...) {}
            
            // Try Adam
            try {
                auto& adam_options = static_cast<torch::optim::AdamOptions&>(group.options());
                return adam_options.lr();
            } catch (...) {}
        }
    }
    return -1.0;
}

// Parameters for torch::lr_scheduler_step
struct LRSchedulerStepArgs {
    std::string optimizer;
    int stepSize = -1;    // Required parameter, -1 indicates not set
    double gamma = 0.1;   // Optional, default 0.1
    
    bool IsValid() const {
        return !optimizer.empty() && stepSize > 0 && gamma > 0.0;
    }
};

// Parse arguments for torch::lr_scheduler_step (dual syntax support)
LRSchedulerStepArgs ParseLRSchedulerStepArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerStepArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): optimizer step_size ?gamma?
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: torch::lr_scheduler_step optimizer step_size ?gamma?");
        }
        
        args.optimizer = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.stepSize) != TCL_OK) {
            throw std::runtime_error("Invalid stepSize value");
        }
        
        if (objc > 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.gamma) != TCL_OK) {
                throw std::runtime_error("Invalid gamma value");
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
            } else if (param == "-stepSize" || param == "-step_size") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.stepSize) != TCL_OK) {
                    throw std::runtime_error("Invalid stepSize value");
                }
            } else if (param == "-gamma") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.gamma) != TCL_OK) {
                    throw std::runtime_error("Invalid gamma value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -optimizer, -stepSize, -gamma");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid (optimizer handle and stepSize required, stepSize must be positive, gamma must be positive)");
    }
    
    return args;
}

// torch::lr_scheduler_step - Step LR scheduler with dual syntax support
int LRSchedulerStep_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        auto args = ParseLRSchedulerStepArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get current learning rate
        double current_lr = GetOptimizerLR(args.optimizer);
        if (current_lr < 0) {
            Tcl_SetResult(interp, const_cast<char*>("Could not get learning rate from optimizer"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Create scheduler
        auto scheduler = std::make_shared<LRScheduler>(args.optimizer, "step");
        scheduler->initial_lr = current_lr;
        scheduler->current_lr = current_lr;
        scheduler->step_size = args.stepSize;
        scheduler->gamma = args.gamma;
        
        std::string result_handle = GetNextHandle("step_scheduler");
        scheduler_storage[result_handle] = scheduler;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct LRSchedulerExponentialArgs {
    std::string optimizer;
    double gamma = 0.95;  // Default gamma value
    
    bool IsValid() const {
        return !optimizer.empty() && gamma > 0.0;
    }
};

LRSchedulerExponentialArgs ParseLRSchedulerExponentialArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerExponentialArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        args.optimizer = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            double gamma;
            if (Tcl_GetDoubleFromObj(interp, objv[2], &gamma) == TCL_OK) {
                args.gamma = gamma;
            } else {
                throw std::runtime_error("Invalid gamma parameter");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-optimizer" || param == "-opt") {
                args.optimizer = Tcl_GetString(objv[i + 1]);
            } else if (param == "-gamma" || param == "-decay") {
                double gamma;
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &gamma) != TCL_OK) {
                    throw std::runtime_error("Invalid gamma value: " + std::string(Tcl_GetString(objv[i + 1])));
                }
                args.gamma = gamma;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: -optimizer");
    }
    
    return args;
}

// Exponential Learning Rate Scheduler
int LRSchedulerExponential_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "optimizer gamma | -optimizer optimizer ?-gamma gamma?");
        return TCL_ERROR;
    }

    try {
        LRSchedulerExponentialArgs args = ParseLRSchedulerExponentialArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get current learning rate
        double current_lr = GetOptimizerLR(args.optimizer);
        if (current_lr < 0) {
            Tcl_SetResult(interp, const_cast<char*>("Could not get learning rate from optimizer"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Create scheduler
        auto scheduler = std::make_shared<LRScheduler>(args.optimizer, "exponential");
        scheduler->initial_lr = current_lr;
        scheduler->current_lr = current_lr;
        scheduler->exp_gamma = args.gamma;
        
        std::string result_handle = GetNextHandle("exp_scheduler");
        scheduler_storage[result_handle] = scheduler;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::lr_scheduler_cosine args structure
struct LRSchedulerCosineArgs {
    std::string optimizer;
    int tMax = -1;  // Required parameter, -1 indicates not set
    double etaMin = 0.0;  // Optional, default 0.0
    
    bool IsValid() const {
        return !optimizer.empty() && tMax > 0;
    }
};

LRSchedulerCosineArgs ParseLRSchedulerCosineArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerCosineArgs args;

    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: optimizer T_max ?eta_min?
        if (objc < 3 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "optimizer T_max ?eta_min?");
            throw std::runtime_error("Invalid number of arguments");
        }

        args.optimizer = Tcl_GetString(objv[1]);

        if (Tcl_GetIntFromObj(interp, objv[2], &args.tMax) != TCL_OK) {
            throw std::runtime_error("Invalid T_max value");
        }

        if (objc > 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.etaMin) != TCL_OK) {
                throw std::runtime_error("Invalid eta_min value");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* valueObj = objv[i + 1];

            if (param == "-optimizer") {
                args.optimizer = Tcl_GetString(valueObj);
            } else if (param == "-tMax" || param == "-t_max" || param == "-T_max") {
                if (Tcl_GetIntFromObj(interp, valueObj, &args.tMax) != TCL_OK) {
                    throw std::runtime_error("Invalid T_max value");
                }
            } else if (param == "-etaMin" || param == "-eta_min") {
                if (Tcl_GetDoubleFromObj(interp, valueObj, &args.etaMin) != TCL_OK) {
                    throw std::runtime_error("Invalid eta_min value");
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

// Cosine Annealing Learning Rate Scheduler with dual syntax support
int LRSchedulerCosine_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        LRSchedulerCosineArgs args = ParseLRSchedulerCosineArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get current learning rate
        double current_lr = GetOptimizerLR(args.optimizer);
        if (current_lr < 0) {
            Tcl_SetResult(interp, const_cast<char*>("Could not get learning rate from optimizer"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Create scheduler
        auto scheduler = std::make_shared<LRScheduler>(args.optimizer, "cosine");
        scheduler->initial_lr = current_lr;
        scheduler->current_lr = current_lr;
        scheduler->T_max = args.tMax;
        scheduler->eta_min = args.etaMin;
        
        std::string result_handle = GetNextHandle("cosine_scheduler");
        scheduler_storage[result_handle] = scheduler;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure and parser for lr_scheduler_step_update
struct LRSchedulerStepUpdateArgs {
    std::string scheduler;
    bool IsValid() const { return !scheduler.empty(); }
};

static LRSchedulerStepUpdateArgs ParseLRSchedulerStepUpdateArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerStepUpdateArgs args;
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: scheduler
        if (objc != 2) {
            Tcl_WrongNumArgs(interp, 1, objv, "scheduler");
            throw std::runtime_error("Invalid number of arguments");
        }
        args.scheduler = Tcl_GetString(objv[1]);
    } else {
        // Named parameters: -scheduler handle
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            if (param == "-scheduler" || param == "-handle") {
                args.scheduler = value;
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

// Step the scheduler
int LRSchedulerStepUpdate_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        LRSchedulerStepUpdateArgs args = ParseLRSchedulerStepUpdateArgs(interp, objc, objv);

        if (scheduler_storage.find(args.scheduler) == scheduler_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid scheduler name"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto& scheduler = scheduler_storage[args.scheduler];
        scheduler->step_count++;
        double new_lr = scheduler->current_lr;
        
        if (scheduler->scheduler_type == "step") {
            int decay_steps = scheduler->step_count / scheduler->step_size;
            new_lr = scheduler->initial_lr * std::pow(scheduler->gamma, decay_steps);
        } else if (scheduler->scheduler_type == "exponential") {
            new_lr = scheduler->initial_lr * std::pow(scheduler->exp_gamma, scheduler->step_count);
        } else if (scheduler->scheduler_type == "cosine" || scheduler->scheduler_type == "cosine_annealing") {
            int effective_step = scheduler->step_count % scheduler->T_max;
            double cosine_arg = M_PI * effective_step / (scheduler->T_max / 2.0);
            double cosine_factor = (1.0 + std::cos(cosine_arg)) / 2.0;
            new_lr = scheduler->eta_min + (scheduler->initial_lr - scheduler->eta_min) * cosine_factor;
        } else if (scheduler->scheduler_type == "cosine_warm_restarts") {
            int T_i = scheduler->T_max;
            int epochs_since_restart = scheduler->step_count;
            while (epochs_since_restart >= T_i) {
                epochs_since_restart -= T_i;
                T_i *= scheduler->step_size;
            }
            double cosine_factor = (1.0 + std::cos(M_PI * epochs_since_restart / T_i)) / 2.0;
            new_lr = scheduler->eta_min + (scheduler->initial_lr - scheduler->eta_min) * cosine_factor;
        }

        scheduler->current_lr = new_lr;
        if (!UpdateOptimizerLR(scheduler->optimizer_name, new_lr)) {
            Tcl_SetResult(interp, const_cast<char*>("Failed to update optimizer learning rate"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        Tcl_SetResult(interp, const_cast<char*>("OK"), TCL_STATIC);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Get current learning rate from optimizer - dual syntax support
struct GetLRArgs {
    std::string optimizer;
    
    bool IsValid() const {
        return !optimizer.empty();
    }
};

GetLRArgs ParseGetLRArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    GetLRArgs args;
    
    // Check if using named parameters (starts with -)
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: optimizer
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::get_lr optimizer");
        }
        
        args.optimizer = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* value = objv[i + 1];
            
            if (param == "-optimizer") {
                args.optimizer = Tcl_GetString(value);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -optimizer");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -optimizer is required");
    }
    
    return args;
}

int GetLR_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        GetLRArgs args = ParseGetLRArgs(interp, objc, objv);
        
        double lr = GetOptimizerLR(args.optimizer);
        if (lr < 0) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer name or could not get learning rate"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        Tcl_SetObjResult(interp, Tcl_NewDoubleObj(lr));
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// MISSING LEARNING RATE SCHEDULERS - IMPLEMENTING 12 NEW SCHEDULERS
// ============================================================================

// torch::lr_scheduler_multiplicative - Multiplicative LR scheduler

struct LRSchedulerMultiplicativeArgs {
    std::string optimizer;
    double lr_lambda = 1.0;
    
    bool IsValid() const {
        return !optimizer.empty();
    }
};

LRSchedulerMultiplicativeArgs ParseLRSchedulerMultiplicativeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerMultiplicativeArgs args;
    
    // Check if using named parameters (starts with -)
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: optimizer lr_lambda
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::lr_scheduler_multiplicative optimizer lr_lambda");
        }
        
        args.optimizer = Tcl_GetString(objv[1]);
        
        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.lr_lambda) != TCL_OK) {
            throw std::runtime_error("Invalid lr_lambda value");
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* value = objv[i + 1];
            
            if (param == "-optimizer") {
                args.optimizer = Tcl_GetString(value);
            } else if (param == "-lrLambda" || param == "-lr_lambda") {
                if (Tcl_GetDoubleFromObj(interp, value, &args.lr_lambda) != TCL_OK) {
                    throw std::runtime_error("Invalid lr_lambda value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -optimizer is required");
    }
    
    return args;
}

int LRSchedulerMultiplicative_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        LRSchedulerMultiplicativeArgs args = ParseLRSchedulerMultiplicativeArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get current learning rate
        double current_lr = GetOptimizerLR(args.optimizer);
        if (current_lr < 0) {
            Tcl_SetResult(interp, const_cast<char*>("Could not get learning rate from optimizer"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Create scheduler
        auto scheduler = std::make_shared<LRScheduler>(args.optimizer, "multiplicative");
        scheduler->initial_lr = current_lr;
        scheduler->current_lr = current_lr;
        scheduler->gamma = args.lr_lambda;
        
        std::string result_handle = GetNextHandle("mult_scheduler");
        scheduler_storage[result_handle] = scheduler;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameters for torch::lr_scheduler_polynomial
struct LRSchedulerPolynomialArgs {
    std::string optimizer;
    int totalIters = -1;  // Required parameter, -1 indicates not set
    double power = 1.0;  // Optional, default 1.0
    int lastEpoch = -1;  // Optional, default -1
    
    bool IsValid() const {
        return !optimizer.empty() && totalIters > 0 && power >= 0.0;
    }
};

// Parse arguments for torch::lr_scheduler_polynomial (dual syntax support)
LRSchedulerPolynomialArgs ParseLRSchedulerPolynomialArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerPolynomialArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: optimizer total_iters ?power? ?last_epoch?
        if (objc < 3 || objc > 5) {
            throw std::runtime_error("Usage: torch::lr_scheduler_polynomial optimizer total_iters ?power? ?last_epoch?");
        }
        
        args.optimizer = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.totalIters) != TCL_OK) {
            throw std::runtime_error("Invalid total_iters value");
        }
        
        if (objc > 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.power) != TCL_OK) {
                throw std::runtime_error("Invalid power value");
            }
        }
        
        if (objc > 4) {
            if (Tcl_GetIntFromObj(interp, objv[4], &args.lastEpoch) != TCL_OK) {
                throw std::runtime_error("Invalid last_epoch value");
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
            } else if (param == "-totalIters" || param == "-total_iters") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.totalIters) != TCL_OK) {
                    throw std::runtime_error("Invalid totalIters value");
                }
            } else if (param == "-power") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.power) != TCL_OK) {
                    throw std::runtime_error("Invalid power value");
                }
            } else if (param == "-lastEpoch" || param == "-last_epoch") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.lastEpoch) != TCL_OK) {
                    throw std::runtime_error("Invalid lastEpoch value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -optimizer, -totalIters, -power, -lastEpoch");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid (optimizer handle and totalIters required, totalIters must be positive, power must be non-negative)");
    }
    
    return args;
}

// torch::lr_scheduler_polynomial - Polynomial LR scheduler with dual syntax support
int LRSchedulerPolynomial_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        auto args = ParseLRSchedulerPolynomialArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get current learning rate
        double current_lr = GetOptimizerLR(args.optimizer);
        if (current_lr < 0) {
            Tcl_SetResult(interp, const_cast<char*>("Could not get learning rate from optimizer"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Create scheduler
        auto scheduler = std::make_shared<LRScheduler>(args.optimizer, "polynomial");
        scheduler->initial_lr = current_lr;
        scheduler->current_lr = current_lr;
        scheduler->T_max = args.totalIters;
        scheduler->exp_gamma = args.power;
        scheduler->step_count = args.lastEpoch + 1;
        
        std::string result_handle = GetNextHandle("poly_scheduler");
        scheduler_storage[result_handle] = scheduler;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameters for torch::lr_scheduler_cosine_annealing_warm_restarts
struct LRSchedulerCosineAnnealingWarmRestartsArgs {
    std::string optimizer;
    int t0 = -1;  // Required parameter, -1 indicates not set
    int tMult = 1;  // Optional, default 1
    double etaMin = 0.0;  // Optional, default 0.0
    
    bool IsValid() const {
        return !optimizer.empty() && t0 > 0 && tMult >= 1;
    }
};

// Parse arguments for torch::lr_scheduler_cosine_annealing_warm_restarts (dual syntax support)
LRSchedulerCosineAnnealingWarmRestartsArgs ParseLRSchedulerCosineAnnealingWarmRestartsArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerCosineAnnealingWarmRestartsArgs args;

    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: optimizer T_0 ?T_mult? ?eta_min?
        if (objc < 3 || objc > 5) {
            Tcl_WrongNumArgs(interp, 1, objv, "optimizer T_0 ?T_mult? ?eta_min?");
            throw std::runtime_error("Invalid number of arguments");
        }

        args.optimizer = Tcl_GetString(objv[1]);

        if (Tcl_GetIntFromObj(interp, objv[2], &args.t0) != TCL_OK) {
            throw std::runtime_error("Invalid T_0 value");
        }

        if (objc > 3) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.tMult) != TCL_OK) {
                throw std::runtime_error("Invalid T_mult value");
            }
        }

        if (objc > 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &args.etaMin) != TCL_OK) {
                throw std::runtime_error("Invalid eta_min value");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* valueObj = objv[i + 1];

            if (param == "-optimizer") {
                args.optimizer = Tcl_GetString(valueObj);
            } else if (param == "-t0" || param == "-T_0" || param == "-T0") {
                if (Tcl_GetIntFromObj(interp, valueObj, &args.t0) != TCL_OK) {
                    throw std::runtime_error("Invalid T_0 value");
                }
            } else if (param == "-tMult" || param == "-T_mult" || param == "-TMult") {
                if (Tcl_GetIntFromObj(interp, valueObj, &args.tMult) != TCL_OK) {
                    throw std::runtime_error("Invalid T_mult value");
                }
            } else if (param == "-etaMin" || param == "-eta_min") {
                if (Tcl_GetDoubleFromObj(interp, valueObj, &args.etaMin) != TCL_OK) {
                    throw std::runtime_error("Invalid eta_min value");
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

// torch::lr_scheduler_cosine_annealing_warm_restarts - Cosine annealing with warm restarts (dual syntax support)
int LRSchedulerCosineAnnealingWarmRestarts_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        LRSchedulerCosineAnnealingWarmRestartsArgs args = ParseLRSchedulerCosineAnnealingWarmRestartsArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get current learning rate
        double current_lr = GetOptimizerLR(args.optimizer);
        if (current_lr < 0) {
            Tcl_SetResult(interp, const_cast<char*>("Could not get learning rate from optimizer"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Create scheduler
        auto scheduler = std::make_shared<LRScheduler>(args.optimizer, "cosine_warm_restarts");
        scheduler->initial_lr = current_lr;
        scheduler->current_lr = current_lr;
        scheduler->T_max = args.t0;
        scheduler->step_size = args.tMult;
        scheduler->eta_min = args.etaMin;
        
        std::string result_handle = GetNextHandle("cosine_warm_scheduler");
        scheduler_storage[result_handle] = scheduler;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::lr_scheduler_linear_with_warmup args structure
struct LRSchedulerLinearWithWarmupArgs {
    std::string optimizer;
    int numWarmupSteps = -1;  // Required parameter, -1 indicates not set
    int numTrainingSteps = -1;  // Required parameter, -1 indicates not set
    int lastEpoch = -1;  // Optional, default -1
    
    bool IsValid() const {
        return !optimizer.empty() && numWarmupSteps >= 0 && numTrainingSteps > 0 &&
               numWarmupSteps <= numTrainingSteps;
    }
};

LRSchedulerLinearWithWarmupArgs ParseLRSchedulerLinearWithWarmupArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerLinearWithWarmupArgs args;

    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: optimizer num_warmup_steps num_training_steps ?last_epoch?
        if (objc < 4 || objc > 5) {
            Tcl_WrongNumArgs(interp, 1, objv, "optimizer num_warmup_steps num_training_steps ?last_epoch?");
            throw std::runtime_error("Invalid number of arguments");
        }

        args.optimizer = Tcl_GetString(objv[1]);

        if (Tcl_GetIntFromObj(interp, objv[2], &args.numWarmupSteps) != TCL_OK) {
            throw std::runtime_error("Invalid num_warmup_steps value");
        }

        if (Tcl_GetIntFromObj(interp, objv[3], &args.numTrainingSteps) != TCL_OK) {
            throw std::runtime_error("Invalid num_training_steps value");
        }

        if (objc > 4) {
            if (Tcl_GetIntFromObj(interp, objv[4], &args.lastEpoch) != TCL_OK) {
                throw std::runtime_error("Invalid last_epoch value");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* valueObj = objv[i + 1];

            if (param == "-optimizer") {
                args.optimizer = Tcl_GetString(valueObj);
            } else if (param == "-numWarmupSteps" || param == "-num_warmup_steps") {
                if (Tcl_GetIntFromObj(interp, valueObj, &args.numWarmupSteps) != TCL_OK) {
                    throw std::runtime_error("Invalid num_warmup_steps value");
                }
            } else if (param == "-numTrainingSteps" || param == "-num_training_steps") {
                if (Tcl_GetIntFromObj(interp, valueObj, &args.numTrainingSteps) != TCL_OK) {
                    throw std::runtime_error("Invalid num_training_steps value");
                }
            } else if (param == "-lastEpoch" || param == "-last_epoch") {
                if (Tcl_GetIntFromObj(interp, valueObj, &args.lastEpoch) != TCL_OK) {
                    throw std::runtime_error("Invalid last_epoch value");
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

// torch::lr_scheduler_linear_with_warmup - Linear LR with warmup
int LRSchedulerLinearWithWarmup_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        LRSchedulerLinearWithWarmupArgs args = ParseLRSchedulerLinearWithWarmupArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get current learning rate
        double current_lr = GetOptimizerLR(args.optimizer);
        if (current_lr < 0) {
            Tcl_SetResult(interp, const_cast<char*>("Could not get learning rate from optimizer"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Create scheduler
        auto scheduler = std::make_shared<LRScheduler>(args.optimizer, "linear_warmup");
        scheduler->initial_lr = current_lr;
        scheduler->current_lr = current_lr;
        scheduler->step_size = args.numWarmupSteps;
        scheduler->T_max = args.numTrainingSteps;
        scheduler->step_count = args.lastEpoch + 1;
        
        std::string result_handle = GetNextHandle("linear_warmup_scheduler");
        scheduler_storage[result_handle] = scheduler;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::lr_scheduler_constant_with_warmup args structure
struct LRSchedulerConstantWithWarmupArgs {
    std::string optimizer;
    int numWarmupSteps = -1;  // Required parameter, -1 indicates not set
    int lastEpoch = -1;  // Optional, default -1
    
    bool IsValid() const {
        return !optimizer.empty() && numWarmupSteps >= 0;
    }
};

LRSchedulerConstantWithWarmupArgs ParseLRSchedulerConstantWithWarmupArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerConstantWithWarmupArgs args;

    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: optimizer num_warmup_steps ?last_epoch?
        if (objc < 3 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "optimizer num_warmup_steps ?last_epoch?");
            throw std::runtime_error("Invalid number of arguments");
        }

        args.optimizer = Tcl_GetString(objv[1]);

        if (Tcl_GetIntFromObj(interp, objv[2], &args.numWarmupSteps) != TCL_OK) {
            throw std::runtime_error("Invalid num_warmup_steps value");
        }

        if (objc > 3) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.lastEpoch) != TCL_OK) {
                throw std::runtime_error("Invalid last_epoch value");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* valueObj = objv[i + 1];

            if (param == "-optimizer") {
                args.optimizer = Tcl_GetString(valueObj);
            } else if (param == "-numWarmupSteps" || param == "-num_warmup_steps") {
                if (Tcl_GetIntFromObj(interp, valueObj, &args.numWarmupSteps) != TCL_OK) {
                    throw std::runtime_error("Invalid num_warmup_steps value");
                }
            } else if (param == "-lastEpoch" || param == "-last_epoch") {
                if (Tcl_GetIntFromObj(interp, valueObj, &args.lastEpoch) != TCL_OK) {
                    throw std::runtime_error("Invalid last_epoch value");
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

// torch::lr_scheduler_constant_with_warmup - Constant LR with warmup with dual syntax support
int LRSchedulerConstantWithWarmup_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        LRSchedulerConstantWithWarmupArgs args = ParseLRSchedulerConstantWithWarmupArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get current learning rate
        double current_lr = GetOptimizerLR(args.optimizer);
        if (current_lr < 0) {
            Tcl_SetResult(interp, const_cast<char*>("Could not get learning rate from optimizer"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Create scheduler
        auto scheduler = std::make_shared<LRScheduler>(args.optimizer, "constant_warmup");
        scheduler->initial_lr = current_lr;
        scheduler->current_lr = current_lr;
        scheduler->step_size = args.numWarmupSteps;
        scheduler->step_count = args.lastEpoch + 1;
        
        std::string result_handle = GetNextHandle("constant_warmup_scheduler");
        scheduler_storage[result_handle] = scheduler;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Additional advanced schedulers using simplified implementations

// torch::lr_scheduler_multi_step - Multi-step LR scheduler

struct LRSchedulerMultiStepArgs {
    std::string optimizer;
    std::vector<int> milestones;
    double gamma = 0.1;
    
    bool IsValid() const {
        return !optimizer.empty() && !milestones.empty();
    }
};

LRSchedulerMultiStepArgs ParseLRSchedulerMultiStepArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerMultiStepArgs args;
    
    // Check if using named parameters (starts with -)
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: optimizer milestones ?gamma?
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: torch::lr_scheduler_multi_step optimizer milestones ?gamma?");
        }
        
        args.optimizer = Tcl_GetString(objv[1]);
        
        // Parse milestones list
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, objv[2], &listLen, &listObjv) != TCL_OK) {
            throw std::runtime_error("Invalid milestones list");
        }
        
        for (int i = 0; i < listLen; i++) {
            int milestone;
            if (Tcl_GetIntFromObj(interp, listObjv[i], &milestone) != TCL_OK) {
                throw std::runtime_error("Invalid milestone value");
            }
            args.milestones.push_back(milestone);
        }
        
        if (objc > 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.gamma) != TCL_OK) {
                throw std::runtime_error("Invalid gamma value");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* value = objv[i + 1];
            
            if (param == "-optimizer") {
                args.optimizer = Tcl_GetString(value);
            } else if (param == "-milestones") {
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, value, &listLen, &listObjv) != TCL_OK) {
                    throw std::runtime_error("Invalid milestones list");
                }
                
                for (int j = 0; j < listLen; j++) {
                    int milestone;
                    if (Tcl_GetIntFromObj(interp, listObjv[j], &milestone) != TCL_OK) {
                        throw std::runtime_error("Invalid milestone value");
                    }
                    args.milestones.push_back(milestone);
                }
            } else if (param == "-gamma") {
                if (Tcl_GetDoubleFromObj(interp, value, &args.gamma) != TCL_OK) {
                    throw std::runtime_error("Invalid gamma value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -optimizer and -milestones are required");
    }
    
    return args;
}

int LRSchedulerMultiStep_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        LRSchedulerMultiStepArgs args = ParseLRSchedulerMultiStepArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get current learning rate
        double current_lr = GetOptimizerLR(args.optimizer);
        if (current_lr < 0) {
            Tcl_SetResult(interp, const_cast<char*>("Could not get learning rate from optimizer"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Create scheduler
        auto scheduler = std::make_shared<LRScheduler>(args.optimizer, "multi_step");
        scheduler->initial_lr = current_lr;
        scheduler->current_lr = current_lr;
        scheduler->gamma = args.gamma;
        
        std::string result_handle = GetNextHandle("multi_step_scheduler");
        scheduler_storage[result_handle] = scheduler;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameters for torch::lr_scheduler_cosine_annealing
struct LRSchedulerCosineAnnealingArgs {
    std::string optimizer;
    int tMax = -1;  // Required parameter, -1 indicates not set
    double etaMin = 0.0;  // Optional, default 0.0
    
    bool IsValid() const {
        return !optimizer.empty() && tMax > 0;
    }
};

// Parse arguments for torch::lr_scheduler_cosine_annealing (dual syntax support)
LRSchedulerCosineAnnealingArgs ParseLRSchedulerCosineAnnealingArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerCosineAnnealingArgs args;

    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: optimizer T_max ?eta_min?
        if (objc < 3 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "optimizer T_max ?eta_min?");
            throw std::runtime_error("Invalid number of arguments");
        }

        args.optimizer = Tcl_GetString(objv[1]);

        if (Tcl_GetIntFromObj(interp, objv[2], &args.tMax) != TCL_OK) {
            throw std::runtime_error("Invalid T_max value");
        }

        if (objc > 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.etaMin) != TCL_OK) {
                throw std::runtime_error("Invalid eta_min value");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* valueObj = objv[i + 1];

            if (param == "-optimizer") {
                args.optimizer = Tcl_GetString(valueObj);
            } else if (param == "-tMax" || param == "-t_max" || param == "-T_max") {
                if (Tcl_GetIntFromObj(interp, valueObj, &args.tMax) != TCL_OK) {
                    throw std::runtime_error("Invalid T_max value");
                }
            } else if (param == "-etaMin" || param == "-eta_min") {
                if (Tcl_GetDoubleFromObj(interp, valueObj, &args.etaMin) != TCL_OK) {
                    throw std::runtime_error("Invalid eta_min value");
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

// torch::lr_scheduler_cosine_annealing - Standard cosine annealing scheduler with dual syntax support
int LRSchedulerCosineAnnealing_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        LRSchedulerCosineAnnealingArgs args = ParseLRSchedulerCosineAnnealingArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get current learning rate
        double current_lr = GetOptimizerLR(args.optimizer);
        if (current_lr < 0) {
            Tcl_SetResult(interp, const_cast<char*>("Could not get learning rate from optimizer"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Create scheduler
        auto scheduler = std::make_shared<LRScheduler>(args.optimizer, "cosine_annealing");
        scheduler->initial_lr = current_lr;
        scheduler->current_lr = current_lr;
        scheduler->T_max = args.tMax;
        scheduler->eta_min = args.etaMin;
        
        std::string result_handle = GetNextHandle("cosine_annealing_scheduler");
        scheduler_storage[result_handle] = scheduler;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameters for torch::lr_scheduler_plateau
struct LRSchedulerPlateauArgs {
    std::string optimizer;
    std::string mode = "min";  // Optional, default "min"
    double factor = 0.1;  // Optional, default 0.1
    int patience = 10;  // Optional, default 10
    
    bool IsValid() const {
        return !optimizer.empty() && factor > 0.0 && factor <= 1.0 && patience > 0 &&
               (mode == "min" || mode == "max");
    }
};

// Parse arguments for torch::lr_scheduler_plateau (dual syntax support)
LRSchedulerPlateauArgs ParseLRSchedulerPlateauArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerPlateauArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: optimizer ?mode? ?factor? ?patience?
        if (objc < 2 || objc > 5) {
            throw std::runtime_error("Usage: torch::lr_scheduler_plateau optimizer ?mode? ?factor? ?patience?");
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
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -optimizer, -mode, -factor, -patience");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid (optimizer handle required, mode must be 'min' or 'max', factor must be between 0 and 1, patience must be positive)");
    }
    
    return args;
}

// torch::lr_scheduler_plateau - Reduce LR on plateau scheduler with dual syntax support
int LRSchedulerPlateau_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        auto args = ParseLRSchedulerPlateauArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get current learning rate
        double current_lr = GetOptimizerLR(args.optimizer);
        if (current_lr < 0) {
            Tcl_SetResult(interp, const_cast<char*>("Could not get learning rate from optimizer"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Create scheduler
        auto scheduler = std::make_shared<LRScheduler>(args.optimizer, "plateau");
        scheduler->initial_lr = current_lr;
        scheduler->current_lr = current_lr;
        scheduler->gamma = args.factor;
        scheduler->step_size = args.patience;
        
        std::string result_handle = GetNextHandle("plateau_scheduler");
        scheduler_storage[result_handle] = scheduler;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameters for torch::lr_scheduler_inverse_sqrt
struct LRSchedulerInverseSqrtArgs {
    std::string optimizer;
    int warmup_steps = -1;  // -1 indicates not set, will be validated
    double decay_factor = 1.0;
    bool warmup_steps_set = false;
    
    bool IsValid() const {
        return !optimizer.empty() && warmup_steps_set && warmup_steps > 0 && decay_factor > 0.0;
    }
};

// Parse arguments for torch::lr_scheduler_inverse_sqrt (dual syntax support)
LRSchedulerInverseSqrtArgs ParseLRSchedulerInverseSqrtArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerInverseSqrtArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: torch::lr_scheduler_inverse_sqrt optimizer warmup_steps ?decay_factor?");
        }
        
        args.optimizer = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.warmup_steps) != TCL_OK) {
            throw std::runtime_error("Invalid warmup_steps value");
        }
        args.warmup_steps_set = true;
        
        if (objc > 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.decay_factor) != TCL_OK) {
                throw std::runtime_error("Invalid decay_factor value");
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
            } else if (param == "-warmupSteps" || param == "-warmup_steps") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.warmup_steps) != TCL_OK) {
                    throw std::runtime_error("Invalid warmupSteps value");
                }
                args.warmup_steps_set = true;
            } else if (param == "-decayFactor" || param == "-decay_factor") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.decay_factor) != TCL_OK) {
                    throw std::runtime_error("Invalid decayFactor value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid (optimizer handle and warmup_steps required, warmup_steps must be positive, decay_factor must be positive)");
    }
    
    return args;
}

// torch::lr_scheduler_inverse_sqrt - Inverse square root scheduler with dual syntax support
int LRSchedulerInverseSqrt_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        auto args = ParseLRSchedulerInverseSqrtArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get current learning rate
        double current_lr = GetOptimizerLR(args.optimizer);
        if (current_lr < 0) {
            Tcl_SetResult(interp, const_cast<char*>("Could not get learning rate from optimizer"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Create scheduler
        auto scheduler = std::make_shared<LRScheduler>(args.optimizer, "inverse_sqrt");
        scheduler->initial_lr = current_lr;
        scheduler->current_lr = current_lr;
        scheduler->step_size = args.warmup_steps;
        scheduler->gamma = args.decay_factor;
        
        std::string result_handle = GetNextHandle("inverse_sqrt_scheduler");
        scheduler_storage[result_handle] = scheduler;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameters for torch::lr_scheduler_noam
struct LRSchedulerNoamArgs {
    std::string optimizer;
    int modelSize = -1;  // Required parameter, -1 indicates not set
    int warmupSteps = 4000;  // Optional, default 4000
    
    bool IsValid() const {
        return !optimizer.empty() && modelSize > 0 && warmupSteps > 0;
    }
};

// Parse arguments for torch::lr_scheduler_noam (dual syntax support)
LRSchedulerNoamArgs ParseLRSchedulerNoamArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LRSchedulerNoamArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: optimizer model_size ?warmup_steps?
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: torch::lr_scheduler_noam optimizer model_size ?warmup_steps?");
        }
        
        args.optimizer = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.modelSize) != TCL_OK) {
            throw std::runtime_error("Invalid model_size value");
        }
        
        if (objc > 3) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.warmupSteps) != TCL_OK) {
                throw std::runtime_error("Invalid warmup_steps value");
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
            } else if (param == "-modelSize" || param == "-model_size") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.modelSize) != TCL_OK) {
                    throw std::runtime_error("Invalid modelSize value");
                }
            } else if (param == "-warmupSteps" || param == "-warmup_steps") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.warmupSteps) != TCL_OK) {
                    throw std::runtime_error("Invalid warmupSteps value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid (optimizer handle and modelSize required, both modelSize and warmupSteps must be positive)");
    }
    
    return args;
}

// torch::lr_scheduler_noam - Noam learning rate scheduler (Transformer) with dual syntax support
int LRSchedulerNoam_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        auto args = ParseLRSchedulerNoamArgs(interp, objc, objv);
        
        if (optimizer_storage.find(args.optimizer) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get current learning rate
        double current_lr = GetOptimizerLR(args.optimizer);
        if (current_lr < 0) {
            Tcl_SetResult(interp, const_cast<char*>("Could not get learning rate from optimizer"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Create scheduler
        auto scheduler = std::make_shared<LRScheduler>(args.optimizer, "noam");
        scheduler->initial_lr = current_lr;
        scheduler->current_lr = current_lr;
        scheduler->T_max = args.modelSize;
        scheduler->step_size = args.warmupSteps;
        
        std::string result_handle = GetNextHandle("noam_scheduler");
        scheduler_storage[result_handle] = scheduler;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::lr_scheduler_onecycle_advanced - Advanced one cycle scheduler
int LRSchedulerOneCycleAdvanced_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 4 || objc > 8) {
        Tcl_WrongNumArgs(interp, 1, objv, "optimizer max_lr total_steps ?pct_start? ?anneal_strategy? ?div_factor? ?final_div_factor?");
        return TCL_ERROR;
    }

    try {
        std::string optimizer_name = Tcl_GetString(objv[1]);
        
        if (optimizer_storage.find(optimizer_name) == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid optimizer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        double max_lr;
        if (Tcl_GetDoubleFromObj(interp, objv[2], &max_lr) != TCL_OK) {
            return TCL_ERROR;
        }
        
        int total_steps;
        if (Tcl_GetIntFromObj(interp, objv[3], &total_steps) != TCL_OK) {
            return TCL_ERROR;
        }
        
        double pct_start = 0.3;
        if (objc > 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &pct_start) != TCL_OK) {
                return TCL_ERROR;
            }
        }
        
        // Get current learning rate
        double current_lr = GetOptimizerLR(optimizer_name);
        if (current_lr < 0) {
            Tcl_SetResult(interp, const_cast<char*>("Could not get learning rate from optimizer"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Create scheduler
        auto scheduler = std::make_shared<LRScheduler>(optimizer_name, "onecycle_advanced");
        scheduler->initial_lr = current_lr;
        scheduler->current_lr = current_lr;
        scheduler->eta_min = max_lr;
        scheduler->T_max = total_steps;
        scheduler->exp_gamma = pct_start;
        
        std::string result_handle = GetNextHandle("onecycle_adv_scheduler");
        scheduler_storage[result_handle] = scheduler;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 