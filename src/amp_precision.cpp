#include "libtorchtcl.h"
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/autocast_mode.h>

// Forward declarations of global variables
extern std::unordered_map<std::string, torch::Tensor> tensor_storage;
extern std::unordered_map<std::string, std::shared_ptr<torch::nn::Module>> module_storage;
extern std::unordered_map<std::string, std::shared_ptr<torch::optim::Optimizer>> optimizer_storage;


// LibTorch-native gradient scaler implementation
struct NativeGradScaler {
    torch::Tensor scale;
    torch::Tensor growth_tracker;
    torch::Tensor found_inf;
    double growth_factor;
    double backoff_factor;
    int64_t growth_interval;
    
    NativeGradScaler(double init_scale = 65536.0, double growth = 2.0, double backoff = 0.5, int64_t interval = 2000)
        : growth_factor(growth), backoff_factor(backoff), growth_interval(interval) {
        scale = torch::tensor(init_scale, torch::kFloat32);
        growth_tracker = torch::tensor(0, torch::kInt32);
        found_inf = torch::tensor(0.0, torch::kFloat32);
    }
    
    torch::Tensor scale_tensor(const torch::Tensor& tensor) {
        return tensor * scale;
    }
    
    void step_optimizer(torch::optim::Optimizer& optimizer) {
        // Check for infinite gradients
        found_inf.zero_();
        std::vector<torch::Tensor> grads;
        for (auto& group : optimizer.param_groups()) {
            for (auto& param : group.params()) {
                if (param.grad().defined()) {
                    grads.push_back(param.grad());
                }
            }
        }
        
        if (!grads.empty()) {
            // Use LibTorch's native AMP unscaling
            torch::Tensor inv_scale = 1.0 / scale;
            at::_amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale);
        }
        
        // Only step if no infinities found
        if (found_inf.item<float>() == 0.0f) {
            optimizer.step();
        }
    }
    
    void update() {
        // Use LibTorch's native AMP scale update
        auto [new_scale, new_growth_tracker] = at::_amp_update_scale(
            scale, growth_tracker, found_inf, growth_factor, backoff_factor, growth_interval);
        scale = new_scale;
        growth_tracker = new_growth_tracker;
    }
    
    double get_scale() const { 
        return scale.item<double>(); 
    }
};

// Storage for gradient scalers
std::unordered_map<std::string, NativeGradScaler> g_grad_scalers;
int g_scaler_counter = 0;

// Note: Autocast state is now managed by LibTorch's native autocast system

extern "C" {

// ============================================================================
// Autocast Functions
// ============================================================================

// Parameter structure for autocast_enable command
struct AutocastEnableArgs {
    std::string device_type = "cuda";
    std::string dtype = "float16";
    
    bool IsValid() const {
        return (device_type == "cuda" || device_type == "cpu") &&
               (dtype == "float16" || dtype == "bfloat16" || dtype == "float32");
    }
    
    c10::ScalarType GetScalarType() const {
        if (dtype == "float16") return torch::kFloat16;
        if (dtype == "bfloat16") return torch::kBFloat16;
        if (dtype == "float32") return torch::kFloat32;
        return torch::kFloat16; // default
    }
};

// Parse dual syntax for autocast_enable
AutocastEnableArgs ParseAutocastEnableArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    AutocastEnableArgs args;
    
    if (objc == 1) {
        // No parameters - use defaults
        return args;
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc > 3) {
            throw std::runtime_error("Usage: torch::autocast_enable [device_type] [dtype]");
        }
        if (objc >= 2) {
            args.device_type = Tcl_GetString(objv[1]);
        }
        if (objc >= 3) {
            args.dtype = Tcl_GetString(objv[2]);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-device_type" || param == "-device") {
                args.device_type = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dtype" || param == "-data_type") {
                args.dtype = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -device_type, -device, -dtype, -data_type");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Invalid parameters. Device type: cuda or cpu. Dtype: float16, bfloat16, or float32");
    }
    
    return args;
}

int Torch_AutocastEnable_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        // Parse arguments using dual syntax
        AutocastEnableArgs args = ParseAutocastEnableArgs(interp, objc, objv);
        
        c10::ScalarType dtype = args.GetScalarType();

        if (args.device_type == "cuda") {
            at::autocast::set_autocast_enabled(at::kCUDA, true);
            at::autocast::set_autocast_dtype(at::kCUDA, dtype);
        } else if (args.device_type == "cpu") {
            at::autocast::set_autocast_enabled(at::kCPU, true);
            at::autocast::set_autocast_dtype(at::kCPU, dtype);
        } else {
            Tcl_SetResult(interp, const_cast<char*>("Invalid device type. Use cuda or cpu"), TCL_STATIC);
            return TCL_ERROR;
        }

        Tcl_SetResult(interp, const_cast<char*>("autocast enabled"), TCL_STATIC);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// Parameter structure for autocast_disable command
struct AutocastDisableArgs {
    std::string device_type = "cuda";
    
    bool IsValid() const {
        return device_type == "cuda" || device_type == "cpu";
    }
};

// Parse dual syntax for autocast_disable
AutocastDisableArgs ParseAutocastDisableArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    AutocastDisableArgs args;
    
    if (objc == 1) {
        // No parameters - use defaults
        return args;
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc > 2) {
            throw std::runtime_error("Usage: torch::autocast_disable [device_type]");
        }
        args.device_type = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-device_type" || param == "-device") {
                args.device_type = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -device_type, -device");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Invalid device type. Use cuda or cpu");
    }
    
    return args;
}

// Parameter structure for autocast_is_enabled command
struct AutocastIsEnabledArgs {
    std::string device_type = "cuda";
    
    bool IsValid() const {
        return device_type == "cuda" || device_type == "cpu";
    }
};

// Parse dual syntax for autocast_is_enabled
AutocastIsEnabledArgs ParseAutocastIsEnabledArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    AutocastIsEnabledArgs args;
    
    if (objc == 1) {
        // No parameters - use defaults
        return args;
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc > 2) {
            throw std::runtime_error("Usage: torch::autocast_is_enabled [device_type]");
        }
        args.device_type = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-device_type" || param == "-device") {
                args.device_type = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -device_type, -device");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Invalid device type. Use cuda or cpu");
    }
    
    return args;
}

int Torch_AutocastDisable_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax
        AutocastDisableArgs args = ParseAutocastDisableArgs(interp, objc, objv);

        if (args.device_type == "cuda") {
            at::autocast::set_autocast_enabled(at::kCUDA, false);
        } else if (args.device_type == "cpu") {
            at::autocast::set_autocast_enabled(at::kCPU, false);
        } else {
            Tcl_SetResult(interp, const_cast<char*>("Invalid device type. Use cuda or cpu"), TCL_STATIC);
            return TCL_ERROR;
        }

        Tcl_SetResult(interp, const_cast<char*>("autocast disabled"), TCL_STATIC);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

int Torch_AutocastIsEnabled_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        // Parse arguments using dual syntax
        AutocastIsEnabledArgs args = ParseAutocastIsEnabledArgs(interp, objc, objv);

        bool is_enabled = false;
        if (args.device_type == "cuda") {
            is_enabled = at::autocast::is_autocast_enabled(at::kCUDA);
        } else if (args.device_type == "cpu") {
            is_enabled = at::autocast::is_autocast_enabled(at::kCPU);
        } else {
            Tcl_SetResult(interp, const_cast<char*>("Invalid device type. Use cuda or cpu"), TCL_STATIC);
            return TCL_ERROR;
        }

        Tcl_SetObjResult(interp, Tcl_NewBooleanObj(is_enabled ? 1 : 0));
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// Parameter structure for autocast_set_dtype command
struct AutocastSetDtypeArgs {
    std::string dtype = "";  // Required parameter
    std::string device_type = "cuda";
    
    bool IsValid() const {
        return !dtype.empty() && 
               (device_type == "cuda" || device_type == "cpu") &&
               (dtype == "float16" || dtype == "bfloat16" || dtype == "float32");
    }
    
    c10::ScalarType GetScalarType() const {
        if (dtype == "float16") return torch::kFloat16;
        if (dtype == "bfloat16") return torch::kBFloat16;
        if (dtype == "float32") return torch::kFloat32;
        return torch::kFloat16; // default
    }
};

// Parse dual syntax for autocast_set_dtype
AutocastSetDtypeArgs ParseAutocastSetDtypeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    AutocastSetDtypeArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Missing required dtype parameter");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc > 3) {
            throw std::runtime_error("Usage: torch::autocast_set_dtype dtype [device_type]");
        }
        args.dtype = Tcl_GetString(objv[1]);
        if (objc >= 3) {
            args.device_type = Tcl_GetString(objv[2]);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-dtype" || param == "-data_type") {
                args.dtype = Tcl_GetString(objv[i + 1]);
            } else if (param == "-device_type" || param == "-device") {
                args.device_type = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -dtype, -data_type, -device_type, -device");
            }
        }
    }
    
    if (!args.IsValid()) {
        if (args.dtype.empty()) {
            throw std::runtime_error("Missing required dtype parameter. Valid dtypes: float16, bfloat16, float32");
        } else {
            throw std::runtime_error("Invalid parameters. Device type: cuda or cpu. Dtype: float16, bfloat16, or float32");
        }
    }
    
    return args;
}

int Torch_AutocastSetDtype_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        // Parse arguments using dual syntax
        AutocastSetDtypeArgs args = ParseAutocastSetDtypeArgs(interp, objc, objv);
        
        c10::ScalarType dtype = args.GetScalarType();

        if (args.device_type == "cuda") {
            at::autocast::set_autocast_dtype(at::kCUDA, dtype);
        } else if (args.device_type == "cpu") {
            at::autocast::set_autocast_dtype(at::kCPU, dtype);
        } else {
            Tcl_SetResult(interp, const_cast<char*>("Invalid device type. Use cuda or cpu"), TCL_STATIC);
            return TCL_ERROR;
        }

        Tcl_SetResult(interp, const_cast<char*>("autocast dtype set"), TCL_STATIC);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// Gradient Scaler Functions
// ============================================================================

// Dual Syntax Parser for torch::grad_scaler_new
struct GradScalerNewArgs {
    double init_scale = 65536.0;
    double growth_factor = 2.0;
    double backoff_factor = 0.5;
    int growth_interval = 2000;
    
    bool IsValid() const {
        return init_scale > 0.0 && growth_factor > 0.0 && backoff_factor > 0.0 && growth_interval > 0;
    }
};

GradScalerNewArgs ParseGradScalerNewArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    GradScalerNewArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): ?init_scale? ?growth_factor? ?backoff_factor? ?growth_interval?
        if (objc > 5) {
            throw std::runtime_error("Usage: torch::grad_scaler_new ?init_scale? ?growth_factor? ?backoff_factor? ?growth_interval?");
        }
        
        if (objc >= 2) {
            if (Tcl_GetDoubleFromObj(interp, objv[1], &args.init_scale) != TCL_OK) {
                throw std::runtime_error("Invalid init_scale parameter");
            }
        }
        if (objc >= 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.growth_factor) != TCL_OK) {
                throw std::runtime_error("Invalid growth_factor parameter");
            }
        }
        if (objc >= 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.backoff_factor) != TCL_OK) {
                throw std::runtime_error("Invalid backoff_factor parameter");
            }
        }
        if (objc >= 5) {
            if (Tcl_GetIntFromObj(interp, objv[4], &args.growth_interval) != TCL_OK) {
                throw std::runtime_error("Invalid growth_interval parameter");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-initScale" || param == "-init_scale") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.init_scale) != TCL_OK) {
                    throw std::runtime_error("Invalid init_scale parameter");
                }
            } else if (param == "-growthFactor" || param == "-growth_factor") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.growth_factor) != TCL_OK) {
                    throw std::runtime_error("Invalid growth_factor parameter");
                }
            } else if (param == "-backoffFactor" || param == "-backoff_factor") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.backoff_factor) != TCL_OK) {
                    throw std::runtime_error("Invalid backoff_factor parameter");
                }
            } else if (param == "-growthInterval" || param == "-growth_interval") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.growth_interval) != TCL_OK) {
                    throw std::runtime_error("Invalid growth_interval parameter");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Invalid parameters: all values must be positive");
    }
    
    return args;
}

// torch::grad_scaler_new - Create gradient scaler with dual syntax support
int Torch_GradScalerNew_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        // Parse arguments using dual syntax parser
        GradScalerNewArgs args = ParseGradScalerNewArgs(interp, objc, objv);

        NativeGradScaler scaler(args.init_scale, args.growth_factor, args.backoff_factor, args.growth_interval);
        
        std::string scaler_name = "scaler" + std::to_string(g_scaler_counter++);
        g_grad_scalers[scaler_name] = std::move(scaler);

        Tcl_SetResult(interp, const_cast<char*>(scaler_name.c_str()), TCL_VOLATILE);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// Dual Syntax Parser for torch::grad_scaler_scale
struct GradScalerScaleArgs {
    std::string scaler;  // gradient scaler handle
    std::string tensor;  // tensor handle
    
    bool IsValid() const {
        return !scaler.empty() && !tensor.empty();
    }
};

GradScalerScaleArgs ParseGradScalerScaleArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    GradScalerScaleArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): scaler tensor
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::grad_scaler_scale scaler tensor");
        }
        
        args.scaler = Tcl_GetString(objv[1]);
        args.tensor = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-scaler" || param == "-gradScaler") {
                args.scaler = Tcl_GetString(objv[i + 1]);
            } else if (param == "-tensor" || param == "-input") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: scaler and tensor handles required");
    }
    
    return args;
}

// torch::grad_scaler_scale - Scale tensor with gradient scaler with dual syntax support
int Torch_GradScalerScale_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        // Parse arguments using dual syntax parser
        GradScalerScaleArgs args = ParseGradScalerScaleArgs(interp, objc, objv);

        auto scaler_it = g_grad_scalers.find(args.scaler);
        if (scaler_it == g_grad_scalers.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Gradient scaler not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        auto tensor_it = tensor_storage.find(args.tensor);
        if (tensor_it == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Tensor not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        auto scaled_tensor = scaler_it->second.scale_tensor(tensor_it->second);
        
        std::string result_name = GetNextHandle("tensor");
        tensor_storage[result_name] = scaled_tensor;

        Tcl_SetResult(interp, const_cast<char*>(result_name.c_str()), TCL_VOLATILE);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// Dual Syntax Parser for torch::grad_scaler_step
struct GradScalerStepArgs {
    std::string scaler;     // gradient scaler handle
    std::string optimizer;  // optimizer handle
    
    bool IsValid() const {
        return !scaler.empty() && !optimizer.empty();
    }
};

GradScalerStepArgs ParseGradScalerStepArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    GradScalerStepArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): scaler optimizer
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::grad_scaler_step scaler optimizer");
        }
        
        args.scaler = Tcl_GetString(objv[1]);
        args.optimizer = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-scaler" || param == "-gradScaler") {
                args.scaler = Tcl_GetString(objv[i + 1]);
            } else if (param == "-optimizer" || param == "-optim") {
                args.optimizer = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: scaler and optimizer handles required");
    }
    
    return args;
}

// torch::grad_scaler_step - Step optimizer with gradient scaler with dual syntax support
int Torch_GradScalerStep_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        // Parse arguments using dual syntax parser
        GradScalerStepArgs args = ParseGradScalerStepArgs(interp, objc, objv);

        auto scaler_it = g_grad_scalers.find(args.scaler);
        if (scaler_it == g_grad_scalers.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Gradient scaler not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        auto optimizer_it = optimizer_storage.find(args.optimizer);
        if (optimizer_it == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Optimizer not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        scaler_it->second.step_optimizer(*optimizer_it->second);

        Tcl_SetResult(interp, const_cast<char*>("scaler step completed"), TCL_STATIC);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// Dual Syntax Parser for torch::grad_scaler_update
struct GradScalerUpdateArgs {
    std::string scaler;  // gradient scaler handle
    
    bool IsValid() const {
        return !scaler.empty();
    }
};

GradScalerUpdateArgs ParseGradScalerUpdateArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    GradScalerUpdateArgs args;
    
    if (objc == 1) {
        // No arguments provided
        throw std::runtime_error("Usage: torch::grad_scaler_update scaler");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): scaler
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::grad_scaler_update scaler");
        }
        
        args.scaler = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-scaler" || param == "-gradScaler") {
                args.scaler = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: scaler handle required");
    }
    
    return args;
}

// torch::grad_scaler_update - Update gradient scaler with dual syntax support
int Torch_GradScalerUpdate_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        // Parse arguments using dual syntax parser
        GradScalerUpdateArgs args = ParseGradScalerUpdateArgs(interp, objc, objv);

        auto scaler_it = g_grad_scalers.find(args.scaler);
        if (scaler_it == g_grad_scalers.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Gradient scaler not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        scaler_it->second.update();

        Tcl_SetResult(interp, const_cast<char*>("scaler updated"), TCL_STATIC);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for torch::grad_scaler_get_scale
// ============================================================================
struct GradScalerGetScaleArgs {
    std::string scaler;  // gradient scaler handle
    
    bool IsValid() const {
        return !scaler.empty();
    }
};

GradScalerGetScaleArgs ParseGradScalerGetScaleArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    GradScalerGetScaleArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): scaler
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::grad_scaler_get_scale scaler");
        }
        
        args.scaler = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-scaler" || param == "-gradscaler") {
                args.scaler = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: scaler handle required");
    }
    
    return args;
}

// torch::grad_scaler_get_scale - Get scale value from gradient scaler with dual syntax support
int Torch_GradScalerGetScale_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        // Parse arguments using dual syntax parser
        GradScalerGetScaleArgs args = ParseGradScalerGetScaleArgs(interp, objc, objv);

        // Validate scaler handle exists
        auto scaler_it = g_grad_scalers.find(args.scaler);
        if (scaler_it == g_grad_scalers.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Gradient scaler not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        double scale = scaler_it->second.get_scale();
        Tcl_SetObjResult(interp, Tcl_NewDoubleObj(scale));
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// Additional Mixed Precision Tensor Operations
// ============================================================================

// Parameter structure for tensor_masked_fill command
struct TensorMaskedFillArgs {
    std::string tensor;
    std::string mask;
    double value;
    
    bool IsValid() const {
        return !tensor.empty() && !mask.empty();
    }
};

// Parse dual syntax for tensor_masked_fill
TensorMaskedFillArgs ParseTensorMaskedFillArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorMaskedFillArgs args;
    
    if (objc < 4) {
        throw std::runtime_error("Usage: torch::tensor_masked_fill tensor mask value | torch::tensor_masked_fill -tensor tensor -mask mask -value value");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 4) {
            throw std::runtime_error("Usage: torch::tensor_masked_fill tensor mask value");
        }
        args.tensor = Tcl_GetString(objv[1]);
        args.mask = Tcl_GetString(objv[2]);
        
        if (Tcl_GetDoubleFromObj(interp, objv[3], &args.value) != TCL_OK) {
            throw std::runtime_error("Invalid value parameter");
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-tensor") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else if (param == "-mask") {
                args.mask = Tcl_GetString(objv[i + 1]);
            } else if (param == "-value") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.value) != TCL_OK) {
                    throw std::runtime_error("Invalid value parameter");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -tensor, -mask, -value");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: tensor and mask required");
    }
    
    return args;
}

int Torch_TensorMaskedFill_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        // Parse arguments using dual syntax parser
        TensorMaskedFillArgs args = ParseTensorMaskedFillArgs(interp, objc, objv);

        auto tensor_it = tensor_storage.find(args.tensor);
        auto mask_it = tensor_storage.find(args.mask);
        
        if (tensor_it == tensor_storage.end() || mask_it == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Tensor not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        auto result = tensor_it->second.masked_fill(mask_it->second, args.value);

        std::string result_name = GetNextHandle("tensor");
        tensor_storage[result_name] = result;

        Tcl_SetResult(interp, const_cast<char*>(result_name.c_str()), TCL_VOLATILE);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_clamp command
struct TensorClampArgs {
    std::string tensor;
    std::optional<double> min_val;
    std::optional<double> max_val;
    
    bool IsValid() const {
        return !tensor.empty();
    }
};

// Parse dual syntax for tensor_clamp
TensorClampArgs ParseTensorClampArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorClampArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::tensor_clamp tensor ?min? ?max? | torch::tensor_clamp -tensor tensor ?-min value? ?-max value?");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc > 4) {
            throw std::runtime_error("Usage: torch::tensor_clamp tensor ?min? ?max?");
        }
        args.tensor = Tcl_GetString(objv[1]);
        
        if (objc >= 3) {
            double min_val;
            if (Tcl_GetDoubleFromObj(interp, objv[2], &min_val) != TCL_OK) {
                throw std::runtime_error("Invalid min value");
            }
            args.min_val = min_val;
        }
        
        if (objc >= 4) {
            double max_val;
            if (Tcl_GetDoubleFromObj(interp, objv[3], &max_val) != TCL_OK) {
                throw std::runtime_error("Invalid max value");
            }
            args.max_val = max_val;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-tensor") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else if (param == "-min") {
                double min_val;
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &min_val) != TCL_OK) {
                    throw std::runtime_error("Invalid min value");
                }
                args.min_val = min_val;
            } else if (param == "-max") {
                double max_val;
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &max_val) != TCL_OK) {
                    throw std::runtime_error("Invalid max value");
                }
                args.max_val = max_val;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -tensor, -min, -max");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: tensor required");
    }
    
    return args;
}

int Torch_TensorClamp_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        TensorClampArgs args = ParseTensorClampArgs(interp, objc, objv);
        
        auto tensor_it = tensor_storage.find(args.tensor);
        if (tensor_it == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Tensor not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        torch::Tensor result;
        
        if (!args.min_val.has_value() && !args.max_val.has_value()) {
            // No clamping bounds specified
            result = tensor_it->second.clone();
        } else if (args.min_val.has_value() && !args.max_val.has_value()) {
            // Only min specified
            result = torch::clamp_min(tensor_it->second, args.min_val.value());
        } else if (!args.min_val.has_value() && args.max_val.has_value()) {
            // Only max specified
            result = torch::clamp_max(tensor_it->second, args.max_val.value());
        } else {
            // Both min and max specified
            result = torch::clamp(tensor_it->second, args.min_val.value(), args.max_val.value());
        }

        std::string result_name = GetNextHandle("tensor");
        tensor_storage[result_name] = result;

        Tcl_SetResult(interp, const_cast<char*>(result_name.c_str()), TCL_VOLATILE);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

} // extern "C" 