#include "libtorchtcl.h"
#include <map>
#include <string>

// Parameter structure for KLDivLoss
struct KLDivLossArgs {
    std::string input;
    std::string target;
    std::string reduction = "mean";  // Default reduction
    bool logTarget = false;          // Default log_target flag
    
    bool IsValid() const {
        return !input.empty() && !target.empty();
    }
};

// Dual syntax parser for KLDivLoss
KLDivLossArgs ParseKLDivLossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    KLDivLossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 5) {
            throw std::runtime_error("Usage: torch::kl_div_loss input target ?reduction? ?log_target?");
        }
        args.input = Tcl_GetString(objv[1]);
        args.target = Tcl_GetString(objv[2]);
        if (objc > 3) {
            args.reduction = Tcl_GetString(objv[3]);
        }
        if (objc > 4) {
            int logTargetInt;
            if (Tcl_GetIntFromObj(interp, objv[4], &logTargetInt) != TCL_OK) {
                throw std::runtime_error("log_target must be 0 or 1");
            }
            args.logTarget = (logTargetInt != 0);
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
            } else if (param == "-target") {
                args.target = Tcl_GetString(objv[i + 1]);
            } else if (param == "-reduction") {
                args.reduction = Tcl_GetString(objv[i + 1]);
            } else if (param == "-logTarget") {
                int logTargetInt;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &logTargetInt) != TCL_OK) {
                    throw std::runtime_error("logTarget must be 0 or 1");
                }
                args.logTarget = (logTargetInt != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input, target");
    }
    
    return args;
}

// Parameter structure for multilabel_margin_loss
struct MultilabelMarginLossArgs {
    std::string input;
    std::string target;
    std::string reduction = "mean";  // Default reduction (mean, sum, none)
    
    bool IsValid() const {
        return !input.empty() && !target.empty();
    }
};

// Dual syntax parser for multilabel_margin_loss
MultilabelMarginLossArgs ParseMultilabelMarginLossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MultilabelMarginLossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: torch::multilabel_margin_loss input target ?reduction?");
        }
        args.input = Tcl_GetString(objv[1]);
        args.target = Tcl_GetString(objv[2]);
        if (objc > 3) {
            // Convert old integer format to string format
            int red_val;
            if (Tcl_GetIntFromObj(interp, objv[3], &red_val) == TCL_OK) {
                if (red_val == 0) args.reduction = "none";
                else if (red_val == 1) args.reduction = "mean";
                else args.reduction = "sum";
            } else {
                args.reduction = Tcl_GetString(objv[3]);
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
            } else if (param == "-target") {
                args.target = Tcl_GetString(objv[i + 1]);
            } else if (param == "-reduction") {
                args.reduction = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input, target");
    }
    
    return args;
}

// Parameter structure for multilabel_soft_margin_loss
struct MultilabelSoftMarginLossArgs {
    std::string input;
    std::string target;
    std::string reduction = "mean";  // Default reduction (mean, sum, none)
    
    bool IsValid() const {
        return !input.empty() && !target.empty();
    }
};

// Dual syntax parser for multilabel_soft_margin_loss
MultilabelSoftMarginLossArgs ParseMultilabelSoftMarginLossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MultilabelSoftMarginLossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: torch::multilabel_soft_margin_loss input target ?reduction?");
        }
        args.input = Tcl_GetString(objv[1]);
        args.target = Tcl_GetString(objv[2]);
        if (objc > 3) {
            // Convert old integer format to string format
            int red_val;
            if (Tcl_GetIntFromObj(interp, objv[3], &red_val) == TCL_OK) {
                if (red_val == 0) args.reduction = "none";
                else if (red_val == 1) args.reduction = "mean";
                else args.reduction = "sum";
            } else {
                args.reduction = Tcl_GetString(objv[3]);
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
            } else if (param == "-target") {
                args.target = Tcl_GetString(objv[i + 1]);
            } else if (param == "-reduction") {
                args.reduction = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input, target");
    }
    
    return args;
}

// Parameter structure for cosine_embedding_loss command
struct CosineEmbeddingLossArgs {
    std::string input1;
    std::string input2;
    std::string target;
    double margin = 0.0;
    std::string reduction = "mean";
    
    bool IsValid() const {
        return !input1.empty() && !input2.empty() && !target.empty();
    }
};

// Parse dual syntax for cosine_embedding_loss
CosineEmbeddingLossArgs ParseCosineEmbeddingLossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    CosineEmbeddingLossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc >= 2) args.input1 = Tcl_GetString(objv[1]);
        if (objc >= 3) args.input2 = Tcl_GetString(objv[2]);
        if (objc >= 4) args.target = Tcl_GetString(objv[3]);
        if (objc >= 5) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &args.margin) != TCL_OK) {
                throw std::runtime_error("Invalid margin value");
            }
        }
        if (objc >= 6) {
            int red_val;
            if (Tcl_GetIntFromObj(interp, objv[5], &red_val) != TCL_OK) {
                throw std::runtime_error("Invalid reduction value");
            }
            if (red_val == 0) args.reduction = "none";
            else if (red_val == 1) args.reduction = "mean";
            else if (red_val == 2) args.reduction = "sum";
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must have values");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input1") {
                args.input1 = value;
            } else if (param == "-input2") {
                args.input2 = value;
            } else if (param == "-target") {
                args.target = value;
            } else if (param == "-margin") {
                args.margin = std::stod(value);
            } else if (param == "-reduction") {
                args.reduction = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters -input1, -input2, and -target must be provided");
    }
    
    return args;
}

// Parameter structure for dice_loss command
struct DiceLossArgs {
    std::string input;
    std::string target;
    double smooth = 1.0;
    std::string reduction = "mean";
    
    bool IsValid() const {
        return !input.empty() && !target.empty();
    }
};

// Parse dual syntax for dice_loss
DiceLossArgs ParseDiceLossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DiceLossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc >= 2) args.input = Tcl_GetString(objv[1]);
        if (objc >= 3) args.target = Tcl_GetString(objv[2]);
        if (objc >= 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.smooth) != TCL_OK) {
                throw std::runtime_error("Invalid smooth parameter");
            }
        }
        if (objc >= 5) {
            int red_val;
            if (Tcl_GetIntFromObj(interp, objv[4], &red_val) != TCL_OK) {
                throw std::runtime_error("Invalid reduction parameter");
            }
            if (red_val == 0) {
                args.reduction = "none";
            } else if (red_val == 1) {
                args.reduction = "mean";
            } else {
                args.reduction = "sum";
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must have values");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input") {
                args.input = value;
            } else if (param == "-target") {
                args.target = value;
            } else if (param == "-smooth") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.smooth) != TCL_OK) {
                    throw std::runtime_error("Invalid smooth parameter value");
                }
            } else if (param == "-reduction") {
                args.reduction = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters -input and -target must be provided");
    }
    
    return args;
}

// Parameter structure for focal_loss command
struct FocalLossArgs {
    std::string input;
    std::string target;
    double alpha = 1.0;
    double gamma = 2.0;
    std::string reduction = "mean";
    
    bool IsValid() const {
        return !input.empty() && !target.empty();
    }
};

// Parse dual syntax for focal_loss
FocalLossArgs ParseFocalLossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    FocalLossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc >= 2) args.input = Tcl_GetString(objv[1]);
        if (objc >= 3) args.target = Tcl_GetString(objv[2]);
        if (objc >= 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.alpha) != TCL_OK) {
                throw std::runtime_error("Invalid alpha parameter");
            }
        }
        if (objc >= 5) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &args.gamma) != TCL_OK) {
                throw std::runtime_error("Invalid gamma parameter");
            }
        }
        if (objc >= 6) {
            int red_val;
            if (Tcl_GetIntFromObj(interp, objv[5], &red_val) != TCL_OK) {
                throw std::runtime_error("Invalid reduction parameter");
            }
            if (red_val == 0) {
                args.reduction = "none";
            } else if (red_val == 1) {
                args.reduction = "mean";
            } else {
                args.reduction = "sum";
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must have values");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input") {
                args.input = value;
            } else if (param == "-target") {
                args.target = value;
            } else if (param == "-alpha") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.alpha) != TCL_OK) {
                    throw std::runtime_error("Invalid alpha parameter value");
                }
            } else if (param == "-gamma") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.gamma) != TCL_OK) {
                    throw std::runtime_error("Invalid gamma parameter value");
                }
            } else if (param == "-reduction") {
                args.reduction = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters -input and -target must be provided");
    }
    
    return args;
}

// Parameter structure for gaussian_nll_loss command
struct GaussianNLLLossArgs {
    std::string input;
    std::string target;
    std::string var;
    bool full = false;
    double eps = 1e-6;
    std::string reduction = "mean";
    
    bool IsValid() const {
        return !input.empty() && !target.empty() && !var.empty();
    }
};

// Parse dual syntax for gaussian_nll_loss
GaussianNLLLossArgs ParseGaussianNLLLossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    GaussianNLLLossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc >= 2) args.input = Tcl_GetString(objv[1]);
        if (objc >= 3) args.target = Tcl_GetString(objv[2]);
        if (objc >= 4) args.var = Tcl_GetString(objv[3]);
        if (objc >= 5) {
            int full_val;
            if (Tcl_GetIntFromObj(interp, objv[4], &full_val) != TCL_OK) {
                throw std::runtime_error("Invalid full parameter");
            }
            args.full = (full_val != 0);
        }
        if (objc >= 6) {
            if (Tcl_GetDoubleFromObj(interp, objv[5], &args.eps) != TCL_OK) {
                throw std::runtime_error("Invalid eps parameter");
            }
        }
        if (objc >= 7) {
            int red_val;
            if (Tcl_GetIntFromObj(interp, objv[6], &red_val) != TCL_OK) {
                throw std::runtime_error("Invalid reduction parameter");
            }
            if (red_val == 0) {
                args.reduction = "none";
            } else if (red_val == 1) {
                args.reduction = "mean";
            } else {
                args.reduction = "sum";
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must have values");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input") {
                args.input = value;
            } else if (param == "-target") {
                args.target = value;
            } else if (param == "-var") {
                args.var = value;
            } else if (param == "-full") {
                int full_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &full_val) != TCL_OK) {
                    throw std::runtime_error("Invalid full parameter value");
                }
                args.full = (full_val != 0);
            } else if (param == "-eps") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.eps) != TCL_OK) {
                    throw std::runtime_error("Invalid eps parameter value");
                }
            } else if (param == "-reduction") {
                args.reduction = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters -input, -target, and -var must be provided");
    }
    
    return args;
}

// Parameter structure for l1_loss command
struct L1LossArgs {
    std::string input;
    std::string target;
    std::string reduction = "mean";
    
    bool IsValid() const {
        return !input.empty() && !target.empty();
    }
};

// Parse dual syntax for l1_loss
L1LossArgs ParseL1LossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    L1LossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc >= 2) args.input = Tcl_GetString(objv[1]);
        if (objc >= 3) args.target = Tcl_GetString(objv[2]);
        if (objc >= 4) {
            // Convert integer reduction to string for consistency
            int red_val;
            if (Tcl_GetIntFromObj(interp, objv[3], &red_val) == TCL_OK) {
                if (red_val == 0) args.reduction = "none";
                else if (red_val == 1) args.reduction = "mean";
                else if (red_val == 2) args.reduction = "sum";
            } else {
                // Try as string
                args.reduction = Tcl_GetString(objv[3]);
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must have values");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input") {
                args.input = value;
            } else if (param == "-target") {
                args.target = value;
            } else if (param == "-reduction") {
                args.reduction = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters -input and -target must be provided");
    }
    
    return args;
}

// torch::l1_loss - L1/Mean Absolute Error loss
int TensorL1Loss_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Check minimum argument count
    if (objc < 3) {
        Tcl_WrongNumArgs(interp, 1, objv, "input target ?reduction? | -input tensor -target tensor ?-reduction string?");
        return TCL_ERROR;
    }

    try {
        L1LossArgs args = ParseL1LossArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.target) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid target tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto& target = tensor_storage[args.target];
        
        // Convert reduction string to integer for PyTorch API
        int64_t reduction = 1; // mean (default)
        if (args.reduction == "none") reduction = 0;
        else if (args.reduction == "mean") reduction = 1;
        else if (args.reduction == "sum") reduction = 2;
        
        torch::Tensor result = torch::l1_loss(input, target, reduction);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for smooth_l1_loss command
struct SmoothL1LossArgs {
    std::string input;
    std::string target;
    std::string reduction = "mean";
    double beta = 1.0;
    
    bool IsValid() const {
        return !input.empty() && !target.empty();
    }
};

// Parse dual syntax for smooth_l1_loss
SmoothL1LossArgs ParseSmoothL1LossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SmoothL1LossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc >= 2) args.input = Tcl_GetString(objv[1]);
        if (objc >= 3) args.target = Tcl_GetString(objv[2]);
        if (objc >= 4) {
            int red_val;
            if (Tcl_GetIntFromObj(interp, objv[3], &red_val) != TCL_OK) {
                throw std::runtime_error("Invalid reduction parameter");
            }
            if (red_val == 0) {
                args.reduction = "none";
            } else if (red_val == 1) {
                args.reduction = "mean";
            } else {
                args.reduction = "sum";
            }
        }
        if (objc >= 5) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &args.beta) != TCL_OK) {
                throw std::runtime_error("Invalid beta parameter");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must have values");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input") {
                args.input = value;
            } else if (param == "-target") {
                args.target = value;
            } else if (param == "-reduction") {
                args.reduction = value;
            } else if (param == "-beta") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.beta) != TCL_OK) {
                    throw std::runtime_error("Invalid beta parameter value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters -input and -target must be provided");
    }
    
    return args;
}

// torch::smooth_l1_loss - Smooth L1 loss (Huber loss with delta=1)
int TensorSmoothL1Loss_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Check minimum argument count
    if (objc < 3) {
        Tcl_WrongNumArgs(interp, 1, objv, "input target ?reduction? ?beta? | -input tensor -target tensor ?-reduction string? ?-beta double?");
        return TCL_ERROR;
    }

    try {
        SmoothL1LossArgs args = ParseSmoothL1LossArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.target) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid target tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto& target = tensor_storage[args.target];
        
        // Convert reduction string to integer for PyTorch API
        int64_t reduction = 1; // mean (default)
        if (args.reduction == "none") reduction = 0;
        else if (args.reduction == "mean") reduction = 1;
        else if (args.reduction == "sum") reduction = 2;
        
        torch::Tensor result = torch::smooth_l1_loss(input, target, reduction, args.beta);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for huber_loss command
struct HuberLossArgs {
    std::string input;
    std::string target;
    std::string reduction = "mean";
    double delta = 1.0;
    
    bool IsValid() const {
        return !input.empty() && !target.empty();
    }
};

// Parse dual syntax for huber_loss
HuberLossArgs ParseHuberLossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    HuberLossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc >= 2) args.input = Tcl_GetString(objv[1]);
        if (objc >= 3) args.target = Tcl_GetString(objv[2]);
        if (objc >= 4) {
            // Handle both string and integer reduction for backward compatibility
            const char* reduction_str = Tcl_GetString(objv[3]);
            if (strcmp(reduction_str, "none") == 0 || strcmp(reduction_str, "0") == 0) {
                args.reduction = "none";
            } else if (strcmp(reduction_str, "mean") == 0 || strcmp(reduction_str, "1") == 0) {
                args.reduction = "mean";
            } else if (strcmp(reduction_str, "sum") == 0 || strcmp(reduction_str, "2") == 0) {
                args.reduction = "sum";
            } else {
                args.reduction = reduction_str;
            }
        }
        if (objc >= 5) {
            double delta_val;
            if (Tcl_GetDoubleFromObj(interp, objv[4], &delta_val) == TCL_OK) {
                args.delta = delta_val;
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must have values");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input") {
                args.input = value;
            } else if (param == "-target") {
                args.target = value;
            } else if (param == "-reduction") {
                args.reduction = value;
            } else if (param == "-delta") {
                args.delta = std::stod(value);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters -input and -target must be provided");
    }
    
    return args;
}

// torch::huber_loss - Huber loss
int TensorHuberLoss_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Check minimum argument count
    if (objc < 3) {
        Tcl_WrongNumArgs(interp, 1, objv, "input target ?reduction? ?delta? | -input tensor -target tensor ?-reduction string? ?-delta double?");
        return TCL_ERROR;
    }

    try {
        HuberLossArgs args = ParseHuberLossArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.target) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid target tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto& target = tensor_storage[args.target];
        
        // Convert string reduction to integer for PyTorch
        int64_t reduction = 1; // mean (default)
        if (args.reduction == "none") {
            reduction = 0;
        } else if (args.reduction == "mean") {
            reduction = 1;
        } else if (args.reduction == "sum") {
            reduction = 2;
        }
        
        torch::Tensor result = torch::huber_loss(input, target, reduction, args.delta);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::kl_div_loss - KL Divergence loss with dual syntax support
int TensorKLDivLoss_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        KLDivLossArgs args = ParseKLDivLossArgs(interp, objc, objv);
        
        if (!args.IsValid()) {
            Tcl_SetResult(interp, (char*)"Invalid arguments for torch::kl_div_loss", TCL_STATIC);
            return TCL_ERROR;
        }
        
        // Validate tensor existence
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.target) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid target tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto& target = tensor_storage[args.target];
        
        // Convert reduction string to integer
        int64_t reduction = 1; // mean (default)
        if (args.reduction == "none") reduction = 0;
        else if (args.reduction == "mean") reduction = 1;
        else if (args.reduction == "sum") reduction = 2;
        
        // Compute KL divergence loss
        torch::Tensor result = torch::kl_div(input, target, reduction, args.logTarget);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::cosine_embedding_loss - Cosine embedding loss
int TensorCosineEmbeddingLoss_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Check minimum argument count
    if (objc < 4) {
        Tcl_WrongNumArgs(interp, 1, objv, "input1 input2 target ?margin? ?reduction? | -input1 tensor -input2 tensor -target tensor ?-margin double? ?-reduction string?");
        return TCL_ERROR;
    }

    try {
        CosineEmbeddingLossArgs args = ParseCosineEmbeddingLossArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input1 tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.input2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input2 tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.target) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid target tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input1 = tensor_storage[args.input1];
        auto& input2 = tensor_storage[args.input2];
        auto& target = tensor_storage[args.target];
        
        // Convert reduction string to integer
        int64_t reduction = 1; // mean (default)
        if (args.reduction == "none") reduction = 0;
        else if (args.reduction == "mean") reduction = 1;
        else if (args.reduction == "sum") reduction = 2;
        
        torch::Tensor result = torch::cosine_embedding_loss(input1, input2, target, args.margin, reduction);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for margin_ranking_loss command
struct MarginRankingLossArgs {
    std::string input1;
    std::string input2;
    std::string target;
    double margin = 0.0;
    std::string reduction = "mean";
    
    bool IsValid() const {
        return !input1.empty() && !input2.empty() && !target.empty();
    }
};

// Parse dual syntax for margin_ranking_loss
MarginRankingLossArgs ParseMarginRankingLossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MarginRankingLossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc >= 2) args.input1 = Tcl_GetString(objv[1]);
        if (objc >= 3) args.input2 = Tcl_GetString(objv[2]);
        if (objc >= 4) args.target = Tcl_GetString(objv[3]);
        if (objc >= 5) {
            double margin_val;
            if (Tcl_GetDoubleFromObj(interp, objv[4], &margin_val) == TCL_OK) {
                args.margin = margin_val;
            }
        }
        if (objc >= 6) {
            // Handle both string and integer reduction for backward compatibility
            const char* reduction_str = Tcl_GetString(objv[5]);
            if (strcmp(reduction_str, "none") == 0 || strcmp(reduction_str, "0") == 0) {
                args.reduction = "none";
            } else if (strcmp(reduction_str, "mean") == 0 || strcmp(reduction_str, "1") == 0) {
                args.reduction = "mean";
            } else if (strcmp(reduction_str, "sum") == 0 || strcmp(reduction_str, "2") == 0) {
                args.reduction = "sum";
            } else {
                args.reduction = reduction_str;
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must have values");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input1") {
                args.input1 = value;
            } else if (param == "-input2") {
                args.input2 = value;
            } else if (param == "-target") {
                args.target = value;
            } else if (param == "-margin") {
                args.margin = std::stod(value);
            } else if (param == "-reduction") {
                args.reduction = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters -input1, -input2, and -target must be provided");
    }
    
    return args;
}

// torch::margin_ranking_loss - Margin ranking loss
int TensorMarginRankingLoss_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Check minimum argument count
    if (objc < 4) {
        Tcl_WrongNumArgs(interp, 1, objv, "input1 input2 target ?margin? ?reduction? | -input1 tensor -input2 tensor -target tensor ?-margin double? ?-reduction string?");
        return TCL_ERROR;
    }

    try {
        MarginRankingLossArgs args = ParseMarginRankingLossArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input1 tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.input2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input2 tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.target) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid target tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input1 = tensor_storage[args.input1];
        auto& input2 = tensor_storage[args.input2];
        auto& target = tensor_storage[args.target];
        
        // Convert string reduction to integer for PyTorch
        int64_t reduction = 1; // mean (default)
        if (args.reduction == "none") {
            reduction = 0;
        } else if (args.reduction == "mean") {
            reduction = 1;
        } else if (args.reduction == "sum") {
            reduction = 2;
        }
        
        torch::Tensor result = torch::margin_ranking_loss(input1, input2, target, args.margin, reduction);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for triplet_margin_loss command
struct TripletMarginLossArgs {
    std::string anchor;
    std::string positive;
    std::string negative;
    double margin = 1.0;
    double p = 2.0;
    std::string reduction = "mean";
    
    bool IsValid() const {
        return !anchor.empty() && !positive.empty() && !negative.empty();
    }
};

// Parse dual syntax for triplet_margin_loss
TripletMarginLossArgs ParseTripletMarginLossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TripletMarginLossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc >= 2) args.anchor = Tcl_GetString(objv[1]);
        if (objc >= 3) args.positive = Tcl_GetString(objv[2]);
        if (objc >= 4) args.negative = Tcl_GetString(objv[3]);
        if (objc >= 5) {
            double margin_val;
            if (Tcl_GetDoubleFromObj(interp, objv[4], &margin_val) == TCL_OK) {
                args.margin = margin_val;
            }
        }
        if (objc >= 6) {
            double p_val;
            if (Tcl_GetDoubleFromObj(interp, objv[5], &p_val) == TCL_OK) {
                args.p = p_val;
            }
        }
        if (objc >= 7) {
            int red_val;
            if (Tcl_GetIntFromObj(interp, objv[6], &red_val) == TCL_OK) {
                if (red_val == 0) {
                    args.reduction = "none";
                } else if (red_val == 1) {
                    args.reduction = "mean";
                } else {
                    args.reduction = "sum";
                }
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must have values");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-anchor") {
                args.anchor = value;
            } else if (param == "-positive") {
                args.positive = value;
            } else if (param == "-negative") {
                args.negative = value;
            } else if (param == "-margin") {
                args.margin = std::stod(value);
            } else if (param == "-p") {
                args.p = std::stod(value);
            } else if (param == "-reduction") {
                args.reduction = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters -anchor, -positive, and -negative must be provided");
    }
    
    return args;
}

// torch::triplet_margin_loss - Triplet margin loss
int TensorTripletMarginLoss_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Check minimum argument count
    if (objc < 4) {
        Tcl_WrongNumArgs(interp, 1, objv, "anchor positive negative ?margin? ?p? ?reduction? | -anchor tensor -positive tensor -negative tensor ?-margin double? ?-p double? ?-reduction string?");
        return TCL_ERROR;
    }

    try {
        TripletMarginLossArgs args = ParseTripletMarginLossArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.anchor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid anchor tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.positive) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid positive tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.negative) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid negative tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& anchor = tensor_storage[args.anchor];
        auto& positive = tensor_storage[args.positive];
        auto& negative = tensor_storage[args.negative];
        
        // Convert string reduction to integer for PyTorch
        int64_t reduction = 1; // mean (default)
        if (args.reduction == "none") {
            reduction = 0;
        } else if (args.reduction == "mean") {
            reduction = 1;
        } else if (args.reduction == "sum") {
            reduction = 2;
        }
        
        torch::Tensor result = torch::triplet_margin_loss(anchor, positive, negative, args.margin, args.p, 1e-6, false, reduction);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for hinge_embedding_loss command
struct HingeEmbeddingLossArgs {
    std::string input;
    std::string target;
    double margin = 1.0;
    std::string reduction = "mean";
    
    bool IsValid() const {
        return !input.empty() && !target.empty();
    }
};

// Parse dual syntax for hinge_embedding_loss
HingeEmbeddingLossArgs ParseHingeEmbeddingLossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    HingeEmbeddingLossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc >= 2) args.input = Tcl_GetString(objv[1]);
        if (objc >= 3) args.target = Tcl_GetString(objv[2]);
        if (objc >= 4) {
            double margin_val;
            if (Tcl_GetDoubleFromObj(interp, objv[3], &margin_val) == TCL_OK) {
                args.margin = margin_val;
            }
        }
        if (objc >= 5) {
            // Handle both string and integer reduction for backward compatibility
            const char* reduction_str = Tcl_GetString(objv[4]);
            if (strcmp(reduction_str, "none") == 0 || strcmp(reduction_str, "0") == 0) {
                args.reduction = "none";
            } else if (strcmp(reduction_str, "mean") == 0 || strcmp(reduction_str, "1") == 0) {
                args.reduction = "mean";
            } else if (strcmp(reduction_str, "sum") == 0 || strcmp(reduction_str, "2") == 0) {
                args.reduction = "sum";
            } else {
                args.reduction = reduction_str;
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must have values");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input") {
                args.input = value;
            } else if (param == "-target") {
                args.target = value;
            } else if (param == "-margin") {
                args.margin = std::stod(value);
            } else if (param == "-reduction") {
                args.reduction = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters -input and -target must be provided");
    }
    
    return args;
}

// torch::hinge_embedding_loss - Hinge embedding loss  
int TensorHingeEmbeddingLoss_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Check minimum argument count
    if (objc < 3) {
        Tcl_WrongNumArgs(interp, 1, objv, "input target ?margin? ?reduction? | -input tensor -target tensor ?-margin double? ?-reduction string?");
        return TCL_ERROR;
    }

    try {
        HingeEmbeddingLossArgs args = ParseHingeEmbeddingLossArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.target) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid target tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto& target = tensor_storage[args.target];
        
        // Convert string reduction to integer for PyTorch
        int64_t reduction = 1; // mean (default)
        if (args.reduction == "none") {
            reduction = 0;
        } else if (args.reduction == "mean") {
            reduction = 1;
        } else if (args.reduction == "sum") {
            reduction = 2;
        }
        
        torch::Tensor result = torch::hinge_embedding_loss(input, target, args.margin, reduction);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for poisson_nll_loss command
struct PoissonNLLLossArgs {
    std::string input;
    std::string target;
    bool logInput = true;
    bool full = false;
    std::string reduction = "mean";
    
    bool IsValid() const {
        return !input.empty() && !target.empty();
    }
};

// Parse dual syntax for poisson_nll_loss
PoissonNLLLossArgs ParsePoissonNLLLossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    PoissonNLLLossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc >= 2) args.input = Tcl_GetString(objv[1]);
        if (objc >= 3) args.target = Tcl_GetString(objv[2]);
        if (objc >= 4) {
            int log_val;
            if (Tcl_GetIntFromObj(interp, objv[3], &log_val) != TCL_OK) {
                throw std::runtime_error("Invalid log_input parameter");
            }
            args.logInput = (log_val != 0);
        }
        if (objc >= 5) {
            int full_val;
            if (Tcl_GetIntFromObj(interp, objv[4], &full_val) != TCL_OK) {
                throw std::runtime_error("Invalid full parameter");
            }
            args.full = (full_val != 0);
        }
        if (objc >= 6) {
            int red_val;
            if (Tcl_GetIntFromObj(interp, objv[5], &red_val) != TCL_OK) {
                throw std::runtime_error("Invalid reduction parameter");
            }
            if (red_val == 0) {
                args.reduction = "none";
            } else if (red_val == 1) {
                args.reduction = "mean";
            } else {
                args.reduction = "sum";
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must have values");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input") {
                args.input = value;
            } else if (param == "-target") {
                args.target = value;
            } else if (param == "-logInput") {
                int log_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &log_val) != TCL_OK) {
                    throw std::runtime_error("Invalid logInput parameter value");
                }
                args.logInput = (log_val != 0);
            } else if (param == "-full") {
                int full_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &full_val) != TCL_OK) {
                    throw std::runtime_error("Invalid full parameter value");
                }
                args.full = (full_val != 0);
            } else if (param == "-reduction") {
                args.reduction = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters -input and -target must be provided");
    }
    
    return args;
}

// torch::poisson_nll_loss - Poisson negative log likelihood loss
int TensorPoissonNLLLoss_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Check minimum argument count
    if (objc < 3) {
        Tcl_WrongNumArgs(interp, 1, objv, "input target ?log_input? ?full? ?reduction? | -input tensor -target tensor ?-logInput bool? ?-full bool? ?-reduction string?");
        return TCL_ERROR;
    }

    try {
        PoissonNLLLossArgs args = ParsePoissonNLLLossArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.target) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid target tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto& target = tensor_storage[args.target];
        
        // Convert reduction string to integer for PyTorch API
        int64_t reduction = 1; // mean (default)
        if (args.reduction == "none") reduction = 0;
        else if (args.reduction == "mean") reduction = 1;
        else if (args.reduction == "sum") reduction = 2;
        
        torch::Tensor result = torch::poisson_nll_loss(input, target, args.logInput, args.full, 1e-8, reduction);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::gaussian_nll_loss - Gaussian negative log likelihood loss
int TensorGaussianNLLLoss_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Check minimum argument count
    if (objc < 4) {
        Tcl_WrongNumArgs(interp, 1, objv, "input target var ?full? ?eps? ?reduction? | -input tensor -target tensor -var tensor ?-full bool? ?-eps double? ?-reduction string?");
        return TCL_ERROR;
    }

    try {
        GaussianNLLLossArgs args = ParseGaussianNLLLossArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.target) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid target tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.var) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid var tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto& target = tensor_storage[args.target];
        auto& var = tensor_storage[args.var];
        
        // Manual implementation of Gaussian NLL loss
        torch::Tensor diff = input - target;
        torch::Tensor var_clamped = torch::clamp(var, args.eps);
        
        // Base loss: 0.5 * ((input-target)^2/var + log(var))
        torch::Tensor loss = 0.5 * ((diff * diff) / var_clamped + torch::log(var_clamped));
        
        if (args.full) {
            // Add constant term: 0.5 * log(2*pi)
            loss = loss + 0.5 * torch::log(torch::tensor(2 * M_PI));
        }
        
        torch::Tensor result;
        if (args.reduction == "none") {
            result = loss;
        } else if (args.reduction == "mean") {
            result = torch::mean(loss);
        } else if (args.reduction == "sum") {
            result = torch::sum(loss);
        } else {
            throw std::runtime_error("Invalid reduction type: " + args.reduction);
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::focal_loss - Focal Loss for addressing class imbalance
int TensorFocalLoss_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Check minimum argument count
    if (objc < 3) {
        Tcl_WrongNumArgs(interp, 1, objv, "input target ?alpha? ?gamma? ?reduction? | -input tensor -target tensor ?-alpha double? ?-gamma double? ?-reduction string?");
        return TCL_ERROR;
    }

    try {
        FocalLossArgs args = ParseFocalLossArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.target) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid target tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto& target = tensor_storage[args.target];
        
        // Manual implementation of Focal Loss: -alpha * (1-p)^gamma * log(p)
        // First apply softmax to get probabilities
        torch::Tensor probs = torch::softmax(input, -1);
        
        // Get the probability of the correct class
        // Ensure target is int64 for gather operation
        torch::Tensor target_int64 = target.to(torch::kLong);
        torch::Tensor p_t = torch::gather(probs, -1, target_int64.unsqueeze(-1)).squeeze(-1);
        
        // Compute (1-p)^gamma
        torch::Tensor one_minus_p = 1.0 - p_t;
        torch::Tensor modulating_factor = torch::pow(one_minus_p, args.gamma);
        
        // Compute -log(p)
        torch::Tensor log_p = -torch::log(torch::clamp(p_t, 1e-8, 1.0));
        
        // Combine: focal_loss = alpha * (1-p)^gamma * (-log(p))
        torch::Tensor focal_loss = args.alpha * modulating_factor * log_p;
        
        torch::Tensor result;
        if (args.reduction == "none") {
            result = focal_loss;
        } else if (args.reduction == "mean") {
            result = torch::mean(focal_loss);
        } else if (args.reduction == "sum") {
            result = torch::sum(focal_loss);
        } else {
            throw std::runtime_error("Invalid reduction type: " + args.reduction);
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::dice_loss - Dice Loss for segmentation tasks
int TensorDiceLoss_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Check minimum argument count
    if (objc < 3) {
        Tcl_WrongNumArgs(interp, 1, objv, "input target ?smooth? ?reduction? | -input tensor -target tensor ?-smooth double? ?-reduction string?");
        return TCL_ERROR;
    }

    try {
        DiceLossArgs args = ParseDiceLossArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.target) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid target tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto& target = tensor_storage[args.target];
        
        // Apply sigmoid to input to get probabilities
        torch::Tensor probs = torch::sigmoid(input);
        
        // Flatten tensors for easier computation
        torch::Tensor probs_flat = probs.view({-1});
        torch::Tensor target_flat = target.view({-1}).to(torch::kFloat);
        
        // Compute Dice coefficient: (2 * intersection + smooth) / (sum + smooth)
        torch::Tensor intersection = torch::sum(probs_flat * target_flat);
        torch::Tensor dice_coeff = (2.0 * intersection + args.smooth) / 
                                  (torch::sum(probs_flat) + torch::sum(target_flat) + args.smooth);
        
        // Dice loss = 1 - dice_coefficient
        torch::Tensor dice_loss = 1.0 - dice_coeff;
        
        torch::Tensor result;
        if (args.reduction == "none") {
            result = dice_loss.unsqueeze(0);
        } else if (args.reduction == "mean") {
            result = dice_loss;
        } else if (args.reduction == "sum") {
            result = dice_loss;
        } else {
            throw std::runtime_error("Invalid reduction type: " + args.reduction);
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tversky_loss command
struct TverskyLossArgs {
    std::string input;
    std::string target;
    double alpha = 0.7;         // Weight for false positives
    double beta = 0.3;          // Weight for false negatives
    double smooth = 1.0;
    std::string reduction = "mean";
    
    bool IsValid() const {
        return !input.empty() && !target.empty();
    }
};

// Parse dual syntax for tversky_loss
TverskyLossArgs ParseTverskyLossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TverskyLossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc >= 2) args.input = Tcl_GetString(objv[1]);
        if (objc >= 3) args.target = Tcl_GetString(objv[2]);
        if (objc >= 4) {
            double alpha_val;
            if (Tcl_GetDoubleFromObj(interp, objv[3], &alpha_val) == TCL_OK) {
                args.alpha = alpha_val;
            }
        }
        if (objc >= 5) {
            double beta_val;
            if (Tcl_GetDoubleFromObj(interp, objv[4], &beta_val) == TCL_OK) {
                args.beta = beta_val;
            }
        }
        if (objc >= 6) {
            double smooth_val;
            if (Tcl_GetDoubleFromObj(interp, objv[5], &smooth_val) == TCL_OK) {
                args.smooth = smooth_val;
            }
        }
        if (objc >= 7) {
            int red_val;
            if (Tcl_GetIntFromObj(interp, objv[6], &red_val) == TCL_OK) {
                if (red_val == 0) {
                    args.reduction = "none";
                } else if (red_val == 1) {
                    args.reduction = "mean";
                } else {
                    args.reduction = "sum";
                }
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must have values");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input") {
                args.input = value;
            } else if (param == "-target") {
                args.target = value;
            } else if (param == "-alpha") {
                args.alpha = std::stod(value);
            } else if (param == "-beta") {
                args.beta = std::stod(value);
            } else if (param == "-smooth") {
                args.smooth = std::stod(value);
            } else if (param == "-reduction") {
                args.reduction = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters -input and -target must be provided");
    }
    
    return args;
}

// torch::tversky_loss - Tversky Loss (generalization of Dice loss)
int TensorTverskyLoss_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Check minimum argument count
    if (objc < 3) {
        Tcl_WrongNumArgs(interp, 1, objv, "input target ?alpha? ?beta? ?smooth? ?reduction? | -input tensor -target tensor ?-alpha double? ?-beta double? ?-smooth double? ?-reduction string?");
        return TCL_ERROR;
    }

    try {
        TverskyLossArgs args = ParseTverskyLossArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.target) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid target tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto& target = tensor_storage[args.target];
        
        // Apply sigmoid to input to get probabilities
        torch::Tensor probs = torch::sigmoid(input);
        
        // Flatten tensors for easier computation
        torch::Tensor probs_flat = probs.view({-1});
        torch::Tensor target_flat = target.view({-1}).to(torch::kFloat);
        
        // Compute True Positives, False Positives, False Negatives
        torch::Tensor tp = torch::sum(probs_flat * target_flat);
        torch::Tensor fp = torch::sum(probs_flat * (1.0 - target_flat));
        torch::Tensor fn = torch::sum((1.0 - probs_flat) * target_flat);
        
        // Compute Tversky index: TP / (TP + alpha*FP + beta*FN + smooth)
        torch::Tensor tversky_index = (tp + args.smooth) / (tp + args.alpha * fp + args.beta * fn + args.smooth);
        
        // Tversky loss = 1 - tversky_index
        torch::Tensor tversky_loss = 1.0 - tversky_index;
        
        torch::Tensor result;
        if (args.reduction == "none") { // none - return per-sample loss
            result = tversky_loss.unsqueeze(0);
        } else if (args.reduction == "mean") { // mean
            result = tversky_loss;
        } else { // sum
            result = tversky_loss;
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for triplet_margin_with_distance_loss command
struct TripletMarginWithDistanceLossArgs {
    std::string anchor;
    std::string positive;
    std::string negative;
    std::string distanceFunction = "euclidean";
    double margin = 1.0;
    std::string reduction = "mean";
    
    bool IsValid() const {
        return !anchor.empty() && !positive.empty() && !negative.empty();
    }
};

// Parse dual syntax for triplet_margin_with_distance_loss
TripletMarginWithDistanceLossArgs ParseTripletMarginWithDistanceLossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TripletMarginWithDistanceLossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc >= 2) args.anchor = Tcl_GetString(objv[1]);
        if (objc >= 3) args.positive = Tcl_GetString(objv[2]);
        if (objc >= 4) args.negative = Tcl_GetString(objv[3]);
        if (objc >= 5) {
            int distance_func;
            if (Tcl_GetIntFromObj(interp, objv[4], &distance_func) == TCL_OK) {
                if (distance_func == 0) {
                    args.distanceFunction = "cosine";
                } else if (distance_func == 1) {
                    args.distanceFunction = "pairwise";
                } else {
                    args.distanceFunction = "euclidean";
                }
            }
        }
        if (objc >= 6) {
            double margin_val;
            if (Tcl_GetDoubleFromObj(interp, objv[5], &margin_val) == TCL_OK) {
                args.margin = margin_val;
            }
        }
        if (objc >= 7) {
            int red_val;
            if (Tcl_GetIntFromObj(interp, objv[6], &red_val) == TCL_OK) {
                if (red_val == 0) {
                    args.reduction = "none";
                } else if (red_val == 1) {
                    args.reduction = "mean";
                } else {
                    args.reduction = "sum";
                }
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must have values");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-anchor") {
                args.anchor = value;
            } else if (param == "-positive") {
                args.positive = value;
            } else if (param == "-negative") {
                args.negative = value;
            } else if (param == "-distanceFunction") {
                args.distanceFunction = value;
            } else if (param == "-margin") {
                args.margin = std::stod(value);
            } else if (param == "-reduction") {
                args.reduction = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters -anchor, -positive, and -negative must be provided");
    }
    
    return args;
}

// torch::triplet_margin_with_distance_loss - Triplet margin loss with custom distance function
int TensorTripletMarginWithDistanceLoss_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Check minimum argument count
    if (objc < 4) {
        Tcl_WrongNumArgs(interp, 1, objv, "anchor positive negative ?distance_function? ?margin? ?reduction? | -anchor tensor -positive tensor -negative tensor ?-distanceFunction string? ?-margin double? ?-reduction string?");
        return TCL_ERROR;
    }

    try {
        TripletMarginWithDistanceLossArgs args = ParseTripletMarginWithDistanceLossArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.anchor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid anchor tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.positive) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid positive tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.negative) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid negative tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& anchor = tensor_storage[args.anchor];
        auto& positive = tensor_storage[args.positive];
        auto& negative = tensor_storage[args.negative];
        
        // Compute distances based on distance function
        torch::Tensor pos_dist, neg_dist;
        if (args.distanceFunction == "cosine") {
            pos_dist = 1.0 - torch::cosine_similarity(anchor, positive, -1);
            neg_dist = 1.0 - torch::cosine_similarity(anchor, negative, -1);
        } else if (args.distanceFunction == "pairwise") {
            pos_dist = torch::pairwise_distance(anchor, positive, 2.0);
            neg_dist = torch::pairwise_distance(anchor, negative, 2.0);
        } else { // euclidean (default)
            pos_dist = torch::norm(anchor - positive, 2, -1);
            neg_dist = torch::norm(anchor - negative, 2, -1);
        }
        
        // Compute triplet loss: max(0, pos_dist - neg_dist + margin)
        torch::Tensor loss = torch::relu(pos_dist - neg_dist + args.margin);
        
        torch::Tensor result;
        if (args.reduction == "none") {
            result = loss;
        } else if (args.reduction == "mean") {
            result = torch::mean(loss);
        } else { // sum
            result = torch::sum(loss);
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for multi_margin_loss command
struct MultiMarginLossArgs {
    std::string input;
    std::string target;
    int p = 1;                      // Norm degree (default 1)
    double margin = 1.0;            // Margin value (default 1.0)
    std::string reduction = "mean"; // Default reduction
    
    bool IsValid() const {
        return !input.empty() && !target.empty();
    }
};

// Parse dual syntax for multi_margin_loss
MultiMarginLossArgs ParseMultiMarginLossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MultiMarginLossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: input target ?p? ?margin? ?reduction?
        if (objc < 3 || objc > 6) {
            Tcl_WrongNumArgs(interp, 1, objv, "input target ?p? ?margin? ?reduction?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.target = Tcl_GetString(objv[2]);
        
        if (objc > 3) {
            int p_val;
            if (Tcl_GetIntFromObj(interp, objv[3], &p_val) != TCL_OK) {
                throw std::runtime_error("Invalid p parameter");
            }
            args.p = p_val;
        }
        
        if (objc > 4) {
            double margin_val;
            if (Tcl_GetDoubleFromObj(interp, objv[4], &margin_val) != TCL_OK) {
                throw std::runtime_error("Invalid margin parameter");
            }
            args.margin = margin_val;
        }
        
        if (objc > 5) {
            int red_val;
            if (Tcl_GetIntFromObj(interp, objv[5], &red_val) != TCL_OK) {
                throw std::runtime_error("Invalid reduction parameter");
            }
            if (red_val == 0) args.reduction = "none";
            else if (red_val == 1) args.reduction = "mean";
            else if (red_val == 2) args.reduction = "sum";
            else throw std::runtime_error("Invalid reduction value (0=none, 1=mean, 2=sum)");
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
            } else if (param == "-target") {
                args.target = value;
            } else if (param == "-p") {
                try {
                    args.p = std::stoi(value);
                } catch (...) {
                    throw std::runtime_error("Invalid p value. Must be an integer.");
                }
            } else if (param == "-margin") {
                try {
                    args.margin = std::stod(value);
                } catch (...) {
                    throw std::runtime_error("Invalid margin value. Must be a number.");
                }
            } else if (param == "-reduction") {
                if (value == "none" || value == "mean" || value == "sum") {
                    args.reduction = value;
                } else {
                    throw std::runtime_error("Invalid reduction value. Use: none, mean, sum");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and target tensors required");
    }
    
    return args;
}

// torch::multi_margin_loss - Multi-class margin loss with dual syntax support
int TensorMultiMarginLoss_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        MultiMarginLossArgs args = ParseMultiMarginLossArgs(interp, objc, objv);
        
        if (!args.IsValid()) {
            Tcl_SetResult(interp, (char*)"Invalid arguments for torch::multi_margin_loss", TCL_STATIC);
            return TCL_ERROR;
        }
        
        // Validate tensor existence
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.target) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid target tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto& target = tensor_storage[args.target];
        
        // Convert reduction string to PyTorch reduction enum
        torch::Tensor result;
        if (args.reduction == "none") {
            result = torch::nn::functional::multi_margin_loss(input, target, 
                torch::nn::functional::MultiMarginLossFuncOptions()
                    .p(args.p)
                    .margin(args.margin)
                    .reduction(torch::kNone));
        } else if (args.reduction == "mean") {
            result = torch::nn::functional::multi_margin_loss(input, target, 
                torch::nn::functional::MultiMarginLossFuncOptions()
                    .p(args.p)
                    .margin(args.margin)
                    .reduction(torch::kMean));
        } else { // sum
            result = torch::nn::functional::multi_margin_loss(input, target, 
                torch::nn::functional::MultiMarginLossFuncOptions()
                    .p(args.p)
                    .margin(args.margin)
                    .reduction(torch::kSum));
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::multilabel_margin_loss - Multi-label margin loss
int TensorMultilabelMarginLoss_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    if (objc < 3) {
        Tcl_SetResult(interp, (char*)"Usage: torch::multilabel_margin_loss input target ?reduction?\n"
                      "   or: torch::multilabel_margin_loss -input TENSOR -target TENSOR -reduction STRING", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        MultilabelMarginLossArgs args = ParseMultilabelMarginLossArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.target) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid target tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto& target = tensor_storage[args.target];
        
        torch::Tensor result;
        if (args.reduction == "none") {
            result = torch::nn::functional::multilabel_margin_loss(input, target, torch::nn::functional::MultilabelMarginLossFuncOptions().reduction(torch::kNone));
        } else if (args.reduction == "mean") {
            result = torch::nn::functional::multilabel_margin_loss(input, target, torch::nn::functional::MultilabelMarginLossFuncOptions().reduction(torch::kMean));
        } else { // sum
            result = torch::nn::functional::multilabel_margin_loss(input, target, torch::nn::functional::MultilabelMarginLossFuncOptions().reduction(torch::kSum));
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::multilabel_soft_margin_loss - Multi-label soft margin loss
int TensorMultilabelSoftMarginLoss_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    if (objc < 3) {
        Tcl_SetResult(interp, (char*)"Usage: torch::multilabel_soft_margin_loss input target ?reduction?\n"
                      "   or: torch::multilabel_soft_margin_loss -input TENSOR -target TENSOR -reduction STRING", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        MultilabelSoftMarginLossArgs args = ParseMultilabelSoftMarginLossArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.target) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid target tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto& target = tensor_storage[args.target];
        
        torch::Tensor result;
        if (args.reduction == "none") {
            result = torch::nn::functional::multilabel_soft_margin_loss(input, target, torch::nn::functional::MultilabelSoftMarginLossFuncOptions().reduction(torch::kNone));
        } else if (args.reduction == "mean") {
            result = torch::nn::functional::multilabel_soft_margin_loss(input, target, torch::nn::functional::MultilabelSoftMarginLossFuncOptions().reduction(torch::kMean));
        } else { // sum
            result = torch::nn::functional::multilabel_soft_margin_loss(input, target, torch::nn::functional::MultilabelSoftMarginLossFuncOptions().reduction(torch::kSum));
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for soft_margin_loss command
struct SoftMarginLossArgs {
    std::string input;
    std::string target;
    std::string reduction = "mean";
    
    bool IsValid() const {
        return !input.empty() && !target.empty();
    }
};

// Parse dual syntax for soft_margin_loss
SoftMarginLossArgs ParseSoftMarginLossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SoftMarginLossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc >= 2) args.input = Tcl_GetString(objv[1]);
        if (objc >= 3) args.target = Tcl_GetString(objv[2]);
        if (objc >= 4) {
            int red_val;
            if (Tcl_GetIntFromObj(interp, objv[3], &red_val) != TCL_OK) {
                throw std::runtime_error("Invalid reduction parameter");
            }
            if (red_val == 0) {
                args.reduction = "none";
            } else if (red_val == 1) {
                args.reduction = "mean";
            } else {
                args.reduction = "sum";
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must have values");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input") {
                args.input = value;
            } else if (param == "-target") {
                args.target = value;
            } else if (param == "-reduction") {
                args.reduction = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters -input and -target must be provided");
    }
    
    return args;
}

// torch::soft_margin_loss - Soft margin loss
int TensorSoftMarginLoss_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    // Check minimum argument count
    if (objc < 3) {
        Tcl_WrongNumArgs(interp, 1, objv, "input target ?reduction? | -input tensor -target tensor ?-reduction string?");
        return TCL_ERROR;
    }

    try {
        SoftMarginLossArgs args = ParseSoftMarginLossArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.target) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid target tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto& target = tensor_storage[args.target];
        
        torch::Tensor result;
        if (args.reduction == "none") {
            result = torch::nn::functional::soft_margin_loss(input, target, torch::nn::functional::SoftMarginLossFuncOptions().reduction(torch::kNone));
        } else if (args.reduction == "mean") {
            result = torch::nn::functional::soft_margin_loss(input, target, torch::nn::functional::SoftMarginLossFuncOptions().reduction(torch::kMean));
        } else { // sum
            result = torch::nn::functional::soft_margin_loss(input, target, torch::nn::functional::SoftMarginLossFuncOptions().reduction(torch::kSum));
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 