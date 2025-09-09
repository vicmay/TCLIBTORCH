#include "libtorchtcl.h"

// Parameter structure for mse_loss command
struct MSELossArgs {
    std::string input;
    std::string target;
    std::string reduction = "mean";
    
    bool IsValid() const {
        return !input.empty() && !target.empty();
    }
};

// Parse dual syntax for mse_loss
MSELossArgs ParseMSELossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MSELossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc >= 2) args.input = Tcl_GetString(objv[1]);
        if (objc >= 3) args.target = Tcl_GetString(objv[2]);
        if (objc >= 4) args.reduction = Tcl_GetString(objv[3]);
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

// Mean Squared Error Loss
int MSELoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    // Check minimum argument count
    if (objc < 3) {
        Tcl_WrongNumArgs(interp, 1, objv, "input target ?reduction? | -input tensor -target tensor ?-reduction string?");
        return TCL_ERROR;
    }

    try {
        MSELossArgs args = ParseMSELossArgs(interp, objc, objv);
        
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
        
        // Parse reduction parameter
        at::Reduction::Reduction reduction = at::Reduction::Mean;
        if (args.reduction == "none") {
            reduction = at::Reduction::None;
        } else if (args.reduction == "sum") {
            reduction = at::Reduction::Sum;
        } else if (args.reduction == "mean") {
            reduction = at::Reduction::Mean;
        }
        
        torch::Tensor loss = torch::mse_loss(input, target, reduction);
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = loss;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for cross_entropy_loss command
struct CrossEntropyLossArgs {
    std::string input;
    std::string target;
    std::string weight = "none";
    std::string reduction = "mean";
    
    bool IsValid() const {
        return !input.empty() && !target.empty();
    }
};

// Parse dual syntax for cross_entropy_loss
CrossEntropyLossArgs ParseCrossEntropyLossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    CrossEntropyLossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc >= 2) args.input = Tcl_GetString(objv[1]);
        if (objc >= 3) args.target = Tcl_GetString(objv[2]);
        if (objc >= 4) args.weight = Tcl_GetString(objv[3]);
        if (objc >= 5) args.reduction = Tcl_GetString(objv[4]);
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
            } else if (param == "-weight") {
                args.weight = value;
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

// Cross Entropy Loss
int CrossEntropyLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    // Check minimum argument count
    if (objc < 3) {
        Tcl_WrongNumArgs(interp, 1, objv, "input target ?weight? ?reduction? | -input tensor -target tensor ?-weight tensor? ?-reduction string?");
        return TCL_ERROR;
    }

    try {
        CrossEntropyLossArgs args = ParseCrossEntropyLossArgs(interp, objc, objv);
        
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
        
        torch::Tensor weight;
        at::Reduction::Reduction reduction = at::Reduction::Mean;
        
        // Parse optional weight parameter
        if (args.weight != "none" && tensor_storage.find(args.weight) != tensor_storage.end()) {
            weight = tensor_storage[args.weight];
        }
        
        // Parse reduction parameter
        if (args.reduction == "none") {
            reduction = at::Reduction::None;
        } else if (args.reduction == "sum") {
            reduction = at::Reduction::Sum;
        } else if (args.reduction == "mean") {
            reduction = at::Reduction::Mean;
        }
        
        torch::Tensor loss;
        if (weight.defined()) {
            loss = torch::cross_entropy_loss(input, target, weight, reduction);
        } else {
            loss = torch::cross_entropy_loss(input, target, {}, reduction);
        }
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = loss;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for nll_loss command
struct NLLLossArgs {
    std::string input;
    std::string target;
    std::string weight = "none";
    std::string reduction = "mean";
    
    bool IsValid() const {
        return !input.empty() && !target.empty();
    }
};

// Parse dual syntax for nll_loss
NLLLossArgs ParseNLLLossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    NLLLossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc >= 2) args.input = Tcl_GetString(objv[1]);
        if (objc >= 3) args.target = Tcl_GetString(objv[2]);
        if (objc >= 4) args.weight = Tcl_GetString(objv[3]);
        if (objc >= 5) args.reduction = Tcl_GetString(objv[4]);
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
            } else if (param == "-weight") {
                args.weight = value;
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

// Negative Log Likelihood Loss
int NLLLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    // Check minimum argument count
    if (objc < 3) {
        Tcl_WrongNumArgs(interp, 1, objv, "input target ?weight? ?reduction? | -input tensor -target tensor ?-weight tensor? ?-reduction string?");
        return TCL_ERROR;
    }

    try {
        NLLLossArgs args = ParseNLLLossArgs(interp, objc, objv);
        
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
        
        torch::Tensor weight;
        at::Reduction::Reduction reduction = at::Reduction::Mean;
        
        // Parse optional weight parameter
        if (args.weight != "none" && tensor_storage.find(args.weight) != tensor_storage.end()) {
            weight = tensor_storage[args.weight];
        }
        
        // Parse reduction parameter
        if (args.reduction == "none") {
            reduction = at::Reduction::None;
        } else if (args.reduction == "sum") {
            reduction = at::Reduction::Sum;
        } else if (args.reduction == "mean") {
            reduction = at::Reduction::Mean;
        }
        
        torch::Tensor loss;
        if (weight.defined()) {
            loss = torch::nll_loss(input, target, weight, reduction);
        } else {
            loss = torch::nll_loss(input, target, {}, reduction);
        }
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = loss;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for bce_loss command
struct BCELossArgs {
    std::string input;
    std::string target;
    std::string weight = "none";
    std::string reduction = "mean";
    
    bool IsValid() const {
        return !input.empty() && !target.empty();
    }
};

// Parse dual syntax for bce_loss
BCELossArgs ParseBCELossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    BCELossArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc >= 2) args.input = Tcl_GetString(objv[1]);
        if (objc >= 3) args.target = Tcl_GetString(objv[2]);
        if (objc >= 4) args.weight = Tcl_GetString(objv[3]);
        if (objc >= 5) args.reduction = Tcl_GetString(objv[4]);
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
            } else if (param == "-weight") {
                args.weight = value;
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

// Binary Cross Entropy Loss
int BCELoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    // Check minimum argument count
    if (objc < 3) {
        Tcl_WrongNumArgs(interp, 1, objv, "input target ?weight? ?reduction? | -input tensor -target tensor ?-weight tensor? ?-reduction string?");
        return TCL_ERROR;
    }

    try {
        BCELossArgs args = ParseBCELossArgs(interp, objc, objv);
        
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
        
        torch::Tensor weight;
        at::Reduction::Reduction reduction = at::Reduction::Mean;
        
        // Parse optional weight parameter
        if (args.weight != "none" && tensor_storage.find(args.weight) != tensor_storage.end()) {
            weight = tensor_storage[args.weight];
        }
        
        // Parse reduction parameter
        if (args.reduction == "none") {
            reduction = at::Reduction::None;
        } else if (args.reduction == "sum") {
            reduction = at::Reduction::Sum;
        } else if (args.reduction == "mean") {
            reduction = at::Reduction::Mean;
        }
        
        torch::Tensor loss;
        if (weight.defined()) {
            loss = torch::binary_cross_entropy(input, target, weight, reduction);
        } else {
            loss = torch::binary_cross_entropy(input, target, {}, reduction);
        }
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = loss;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 