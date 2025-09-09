#include "libtorchtcl.h"

// Concrete module wrappers (need to be here for the layer commands)
class ConcreteLinear : public torch::nn::LinearImpl {
public:
    using LinearImpl::LinearImpl;  // Inherit constructors
    torch::Tensor forward(const torch::Tensor& x) {
        return LinearImpl::forward(x);
    }
};

class ConcreteConv2d : public torch::nn::Conv2dImpl {
public:
    using Conv2dImpl::Conv2dImpl;
    torch::Tensor forward(const torch::Tensor& x) {
        return Conv2dImpl::forward(x);
    }
};

class ConcreteMaxPool1d : public torch::nn::MaxPool1dImpl {
public:
    using MaxPool1dImpl::MaxPool1dImpl;  // Inherit constructors
    torch::Tensor forward(const torch::Tensor& x) {
        return MaxPool1dImpl::forward(x);
    }
};

class ConcreteMaxPool2d : public torch::nn::MaxPool2dImpl {
public:
    using MaxPool2dImpl::MaxPool2dImpl;
    torch::Tensor forward(const torch::Tensor& x) {
        return MaxPool2dImpl::forward(x);
    }
};

class ConcreteMaxPool3d : public torch::nn::MaxPool3dImpl {
public:
    using MaxPool3dImpl::MaxPool3dImpl;  // Inherit constructors
    torch::Tensor forward(const torch::Tensor& x) {
        return MaxPool3dImpl::forward(x);
    }
};

// Custom MaxPool3d module that handles both normal and special cases
class ConcreteCustomMaxPool3d : public torch::nn::Module {
public:
    explicit ConcreteCustomMaxPool3d(const torch::nn::MaxPool3dOptions& options, bool identity_mode)
        : options_(options), identity_mode_(identity_mode) {}

    torch::Tensor forward(const torch::Tensor& input) {
        if (identity_mode_) {
            return input.clone();  // Return a clone to ensure we don't modify the input
        }
        return torch::max_pool3d(
            input,
            /*kernel_size=*/options_.kernel_size(),
            /*stride=*/options_.stride(),
            /*padding=*/options_.padding(),
            /*dilation=*/options_.dilation(),
            /*ceil_mode=*/options_.ceil_mode()
        );
    }

private:
    torch::nn::MaxPool3dOptions options_;
    bool identity_mode_;
};

class ConcreteDropout : public torch::nn::DropoutImpl {
public:
    using DropoutImpl::DropoutImpl;
    torch::Tensor forward(const torch::Tensor& x) {
        return DropoutImpl::forward(x);
    }
};

class ConcreteBatchNorm2d : public torch::nn::BatchNorm2dImpl {
public:
    using BatchNorm2dImpl::BatchNorm2dImpl;
    torch::Tensor forward(const torch::Tensor& x) {
        return BatchNorm2dImpl::forward(x);
    }
};

class ConcreteAvgPool2d : public torch::nn::AvgPool2dImpl {
public:
    using AvgPool2dImpl::AvgPool2dImpl;
    torch::Tensor forward(const torch::Tensor& x) {
        return AvgPool2dImpl::forward(x);
    }
};

// Custom Sequential implementation with concrete modules
class ConcreteSequential : public torch::nn::Module {
public:
    ConcreteSequential() {}
    
    torch::Tensor forward(const torch::Tensor& x) {
        torch::Tensor current = x;
        for (auto& module : modules_) {
            if (auto concrete_linear = std::dynamic_pointer_cast<ConcreteLinear>(module)) {
                current = concrete_linear->forward(current);
            } else if (auto concrete_conv2d = std::dynamic_pointer_cast<ConcreteConv2d>(module)) {
                current = concrete_conv2d->forward(current);
            } else if (auto concrete_maxpool = std::dynamic_pointer_cast<ConcreteMaxPool2d>(module)) {
                current = concrete_maxpool->forward(current);
            } else if (auto concrete_dropout = std::dynamic_pointer_cast<ConcreteDropout>(module)) {
                current = concrete_dropout->forward(current);
            } else if (auto concrete_batchnorm = std::dynamic_pointer_cast<ConcreteBatchNorm2d>(module)) {
                current = concrete_batchnorm->forward(current);
            } else if (auto concrete_avgpool = std::dynamic_pointer_cast<ConcreteAvgPool2d>(module)) {
                current = concrete_avgpool->forward(current);
            } else if (auto concrete_maxpool1d = std::dynamic_pointer_cast<ConcreteMaxPool1d>(module)) {
                current = concrete_maxpool1d->forward(current);
            } else if (auto concrete_maxpool3d = std::dynamic_pointer_cast<ConcreteCustomMaxPool3d>(module)) {
                current = concrete_maxpool3d->forward(current);
            }
        }
        return current;
    }
    
    void push_back(std::shared_ptr<torch::nn::Module> module) {
        register_module(std::to_string(modules_.size()), module);
        modules_.push_back(module);
    }
    
private:
    std::vector<std::shared_ptr<torch::nn::Module>> modules_;
};

// Parameter structure for linear command
struct LinearArgs {
    int inFeatures = 0;
    int outFeatures = 0;
    bool bias = true;
    
    bool IsValid() const {
        return inFeatures > 0 && outFeatures > 0;
    }
};

// Parse dual syntax for linear command
LinearArgs ParseLinearArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LinearArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: torch::linear in_features out_features ?bias?");
        }
        
        if (Tcl_GetIntFromObj(interp, objv[1], &args.inFeatures) != TCL_OK) {
            throw std::runtime_error("Invalid in_features parameter");
        }
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.outFeatures) != TCL_OK) {
            throw std::runtime_error("Invalid out_features parameter");
        }
        
        if (objc == 4) {
            int bias;
            if (Tcl_GetBooleanFromObj(interp, objv[3], &bias) != TCL_OK) {
                throw std::runtime_error("Invalid bias parameter (should be boolean)");
            }
            args.bias = (bias != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-inFeatures") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.inFeatures) != TCL_OK) {
                    throw std::runtime_error("Invalid value for -inFeatures parameter");
                }
            } else if (param == "-outFeatures") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.outFeatures) != TCL_OK) {
                    throw std::runtime_error("Invalid value for -outFeatures parameter");
                }
            } else if (param == "-bias") {
                int bias;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &bias) != TCL_OK) {
                    throw std::runtime_error("Invalid value for -bias parameter (should be boolean)");
                }
                args.bias = (bias != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid: inFeatures and outFeatures must be positive");
    }
    
    return args;
}

int Linear_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        LinearArgs args = ParseLinearArgs(interp, objc, objv);
        
        auto linear = std::make_shared<ConcreteLinear>(
            torch::nn::LinearOptions(args.inFeatures, args.outFeatures)
                .bias(args.bias)
        );
        
        std::string handle = StoreModule("linear", linear);
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for conv2d command
struct Conv2dArgs {
    int inChannels = 0;
    int outChannels = 0;
    int kernelSize = 0;
    int stride = 1;
    int padding = 0;
    bool bias = true;
    
    bool IsValid() const {
        return inChannels > 0 && outChannels > 0 && kernelSize > 0;
    }
};

// Parse dual syntax for conv2d command
Conv2dArgs ParseConv2dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    Conv2dArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 4 || objc > 7) {
            throw std::runtime_error("Usage: torch::conv2d in_channels out_channels kernel_size ?stride? ?padding? ?bias?");
        }
        
        if (Tcl_GetIntFromObj(interp, objv[1], &args.inChannels) != TCL_OK) {
            throw std::runtime_error("Invalid in_channels parameter");
        }
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.outChannels) != TCL_OK) {
            throw std::runtime_error("Invalid out_channels parameter");
        }
        
        if (Tcl_GetIntFromObj(interp, objv[3], &args.kernelSize) != TCL_OK) {
            throw std::runtime_error("Invalid kernel_size parameter");
        }
        
        if (objc >= 5) {
            if (Tcl_GetIntFromObj(interp, objv[4], &args.stride) != TCL_OK) {
                throw std::runtime_error("Invalid stride parameter");
            }
        }
        
        if (objc >= 6) {
            if (Tcl_GetIntFromObj(interp, objv[5], &args.padding) != TCL_OK) {
                throw std::runtime_error("Invalid padding parameter");
            }
        }
        
        if (objc >= 7) {
            int bias;
            if (Tcl_GetBooleanFromObj(interp, objv[6], &bias) != TCL_OK) {
                throw std::runtime_error("Invalid bias parameter (should be boolean)");
            }
            args.bias = (bias != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-inChannels") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.inChannels) != TCL_OK) {
                    throw std::runtime_error("Invalid value for -inChannels parameter");
                }
            } else if (param == "-outChannels") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.outChannels) != TCL_OK) {
                    throw std::runtime_error("Invalid value for -outChannels parameter");
                }
            } else if (param == "-kernelSize") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.kernelSize) != TCL_OK) {
                    throw std::runtime_error("Invalid value for -kernelSize parameter");
                }
            } else if (param == "-stride") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.stride) != TCL_OK) {
                    throw std::runtime_error("Invalid value for -stride parameter");
                }
            } else if (param == "-padding") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.padding) != TCL_OK) {
                    throw std::runtime_error("Invalid value for -padding parameter");
                }
            } else if (param == "-bias") {
                int bias;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &bias) != TCL_OK) {
                    throw std::runtime_error("Invalid value for -bias parameter (should be boolean)");
                }
                args.bias = (bias != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid: inChannels, outChannels, and kernelSize must be positive");
    }
    
    return args;
}

int Conv2d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        Conv2dArgs args = ParseConv2dArgs(interp, objc, objv);
        
        auto conv2d = std::make_shared<ConcreteConv2d>(
            torch::nn::Conv2dOptions(args.inChannels, args.outChannels, args.kernelSize)
                .stride(args.stride)
                .padding(args.padding)
                .bias(args.bias)
        );
        
        std::string handle = StoreModule("conv2d", conv2d);
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Add after Conv2dArgs/ParseConv2dArgs definitions but before MaxPool2d_Cmd
struct MaxPool2dArgs {
    int kernelSize = 0;
    int stride = -1;   // -1 indicates default (kernelSize)
    int padding = 0;

    bool IsValid() const { return kernelSize > 0; }
};

MaxPool2dArgs ParseMaxPool2dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MaxPool2dArgs args;

    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: kernel_size ?stride? ?padding?
        if (objc < 2 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "kernel_size ?stride? ?padding?");
            throw std::runtime_error("Invalid number of arguments");
        }
        if (Tcl_GetIntFromObj(interp, objv[1], &args.kernelSize) != TCL_OK) {
            throw std::runtime_error("Invalid kernel_size value");
        }
        if (objc >= 3) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.stride) != TCL_OK) {
                throw std::runtime_error("Invalid stride value");
            }
        }
        if (objc >= 4) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.padding) != TCL_OK) {
                throw std::runtime_error("Invalid padding value");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-kernelSize" || param == "-kernel_size") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.kernelSize) != TCL_OK) {
                    throw std::runtime_error("Invalid kernelSize value");
                }
            } else if (param == "-stride") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.stride) != TCL_OK) {
                    throw std::runtime_error("Invalid stride value");
                }
            } else if (param == "-padding") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.padding) != TCL_OK) {
                    throw std::runtime_error("Invalid padding value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("kernelSize must be > 0");
    }
    return args;
}

// Refactor MaxPool2d_Cmd to use new parser
int MaxPool2d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        MaxPool2dArgs args = ParseMaxPool2dArgs(interp, objc, objv);
        int stride = (args.stride == -1) ? args.kernelSize : args.stride;

        auto maxpool = std::make_shared<ConcreteMaxPool2d>(
            torch::nn::MaxPool2dOptions(args.kernelSize)
                .stride(stride)
                .padding(args.padding)
        );

        std::string handle = StoreModule("maxpool2d", maxpool);
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct DropoutArgs {
    double p = 0.5;  // Default dropout probability
    bool training = true;  // Default training mode
    bool inplace = false;  // Default not inplace
    
    bool IsValid() const {
        return p >= 0.0 && p <= 1.0;
    }
};

DropoutArgs ParseDropoutArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DropoutArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): ?p? ?training? ?inplace?
        if (objc > 4) {
            throw std::runtime_error("Usage: dropout ?p? ?training? ?inplace?");
        }
        
        if (objc > 1) {
            if (Tcl_GetDoubleFromObj(interp, objv[1], &args.p) != TCL_OK) {
                throw std::runtime_error("Invalid p parameter");
            }
        }
        
        if (objc > 2) {
            std::string training_str = Tcl_GetString(objv[2]);
            args.training = (training_str == "1" || training_str == "true");
        }
        
        if (objc > 3) {
            std::string inplace_str = Tcl_GetString(objv[3]);
            args.inplace = (inplace_str == "1" || inplace_str == "true");
        }
        
    } else {
        // Named parameter syntax
        if (objc % 2 == 0) {
            throw std::runtime_error("Named parameters must come in pairs");
        }
        
        for (int i = 1; i < objc; i += 2) {
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-p") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.p) != TCL_OK) {
                    throw std::runtime_error("Invalid p parameter");
                }
            } else if (param == "-training") {
                std::string training_str = Tcl_GetString(objv[i + 1]);
                args.training = (training_str == "1" || training_str == "true");
            } else if (param == "-inplace") {
                std::string inplace_str = Tcl_GetString(objv[i + 1]);
                args.inplace = (inplace_str == "1" || inplace_str == "true");
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("p must be between 0.0 and 1.0");
    }
    
    return args;
}

int Dropout_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        DropoutArgs args = ParseDropoutArgs(interp, objc, objv);
        
        auto dropout = std::make_shared<ConcreteDropout>(
            torch::nn::DropoutOptions()
                .p(args.p)
                .inplace(args.inplace)
        );
        std::string handle = StoreModule("dropout", dropout);
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct BatchNorm2dArgs {
    int numFeatures = 0;
    double eps = 1e-5;
    double momentum = 0.1;
    bool affine = true;
    bool trackRunningStats = true;

    bool IsValid() const { return numFeatures > 0; }
};

BatchNorm2dArgs ParseBatchNorm2dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    BatchNorm2dArgs args;

    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: num_features ?eps? ?momentum? ?affine? ?trackRunningStats?
        if (objc < 2 || objc > 6) {
            Tcl_WrongNumArgs(interp, 1, objv, "num_features ?eps? ?momentum? ?affine? ?trackRunningStats?");
            throw std::runtime_error("Invalid number of arguments");
        }
        if (Tcl_GetIntFromObj(interp, objv[1], &args.numFeatures) != TCL_OK) {
            throw std::runtime_error("Invalid num_features value");
        }
        if (objc >= 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.eps) != TCL_OK) {
                throw std::runtime_error("Invalid eps value");
            }
        }
        if (objc >= 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.momentum) != TCL_OK) {
                throw std::runtime_error("Invalid momentum value");
            }
        }
        if (objc >= 5) {
            std::string val = Tcl_GetString(objv[4]);
            args.affine = (val == "1" || val == "true");
        }
        if (objc >= 6) {
            std::string val = Tcl_GetString(objv[5]);
            args.trackRunningStats = (val == "1" || val == "true");
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) throw std::runtime_error("Missing value for parameter");
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* valueObj = objv[i + 1];
            if (param == "-numFeatures" || param == "-num_features") {
                if (Tcl_GetIntFromObj(interp, valueObj, &args.numFeatures) != TCL_OK) {
                    throw std::runtime_error("Invalid numFeatures value");
                }
            } else if (param == "-eps") {
                if (Tcl_GetDoubleFromObj(interp, valueObj, &args.eps) != TCL_OK) {
                    throw std::runtime_error("Invalid eps value");
                }
            } else if (param == "-momentum") {
                if (Tcl_GetDoubleFromObj(interp, valueObj, &args.momentum) != TCL_OK) {
                    throw std::runtime_error("Invalid momentum value");
                }
            } else if (param == "-affine") {
                std::string val = Tcl_GetString(valueObj);
                args.affine = (val == "1" || val == "true");
            } else if (param == "-trackRunningStats" || param == "-track_running_stats") {
                std::string val = Tcl_GetString(valueObj);
                args.trackRunningStats = (val == "1" || val == "true");
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("numFeatures must be > 0");
    }
    return args;
}

int BatchNorm2d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        BatchNorm2dArgs args = ParseBatchNorm2dArgs(interp, objc, objv);
        auto batchnorm = std::make_shared<ConcreteBatchNorm2d>(
            torch::nn::BatchNorm2dOptions(args.numFeatures)
                .eps(args.eps)
                .momentum(args.momentum)
                .affine(args.affine)
                .track_running_stats(args.trackRunningStats)
        );
        std::string handle = StoreModule("batchnorm2d", batchnorm);
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Add after other Args struct definitions
struct AvgPool2dArgs {
    int kernelSize = 0;
    int stride = -1; // -1 signifies default (kernelSize)
    int padding = 0;

    bool IsValid() const {
        return kernelSize > 0;
    }
};

AvgPool2dArgs ParseAvgPool2dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    AvgPool2dArgs args;

    // Determine syntax style
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: kernel_size ?stride? ?padding?
        if (objc < 2 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "kernel_size ?stride? ?padding?");
            throw std::runtime_error("Invalid number of arguments");
        }

        int value;
        // kernel_size (required)
        if (Tcl_GetIntFromObj(interp, objv[1], &value) != TCL_OK) {
            throw std::runtime_error("Invalid kernel_size value");
        }
        args.kernelSize = value;

        // stride (optional)
        if (objc >= 3) {
            if (Tcl_GetIntFromObj(interp, objv[2], &value) != TCL_OK) {
                throw std::runtime_error("Invalid stride value");
            }
            args.stride = value;
        }

        // padding (optional)
        if (objc >= 4) {
            if (Tcl_GetIntFromObj(interp, objv[3], &value) != TCL_OK) {
                throw std::runtime_error("Invalid padding value");
            }
            args.padding = value;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }

            std::string param = Tcl_GetString(objv[i]);
            int value;

            if (param == "-kernelSize") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &value) != TCL_OK) {
                    throw std::runtime_error("Invalid kernelSize value");
                }
                args.kernelSize = value;
            } else if (param == "-stride") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &value) != TCL_OK) {
                    throw std::runtime_error("Invalid stride value");
                }
                args.stride = value;
            } else if (param == "-padding") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &value) != TCL_OK) {
                    throw std::runtime_error("Invalid padding value");
                }
                args.padding = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("kernelSize must be > 0");
    }

    // Default stride equals kernelSize unless set
    if (args.stride == -1) {
        args.stride = args.kernelSize;
    }

    return args;
}

int AvgPool2d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        AvgPool2dArgs args = ParseAvgPool2dArgs(interp, objc, objv);

        auto avgpool = std::make_shared<ConcreteAvgPool2d>(
            torch::nn::AvgPool2dOptions(args.kernelSize)
                .stride(args.stride)
                .padding(args.padding)
        );

        std::string handle = StoreModule("avgpool2d", avgpool);

        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for sequential command
struct SequentialArgs {
    std::vector<std::string> modules;  // List of module handles
    
    bool IsValid() const {
        return true;  // Empty sequential is valid
    }
};

// Parse dual syntax for sequential command
SequentialArgs ParseSequentialArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SequentialArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc > 2) {
            throw std::runtime_error("Usage: torch::sequential ?module_list?");
        }
        
        if (objc == 2) {
            int module_count;
            if (Tcl_ListObjLength(interp, objv[1], &module_count) != TCL_OK) {
                throw std::runtime_error("Invalid module list format");
            }
            
            for (int i = 0; i < module_count; i++) {
                Tcl_Obj* module_obj;
                Tcl_ListObjIndex(interp, objv[1], i, &module_obj);
                args.modules.push_back(Tcl_GetString(module_obj));
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-modules") {
                int module_count;
                if (Tcl_ListObjLength(interp, objv[i + 1], &module_count) != TCL_OK) {
                    throw std::runtime_error("Invalid -modules list format");
                }
                
                for (int j = 0; j < module_count; j++) {
                    Tcl_Obj* module_obj;
                    Tcl_ListObjIndex(interp, objv[i + 1], j, &module_obj);
                    args.modules.push_back(Tcl_GetString(module_obj));
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    return args;
}

int Sequential_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        SequentialArgs args = ParseSequentialArgs(interp, objc, objv);
        auto sequential = std::make_shared<ConcreteSequential>();
        
        // Add modules to sequential if provided
        for (const auto& module_name : args.modules) {
            if (module_storage.find(module_name) == module_storage.end()) {
                throw std::runtime_error("Invalid module name: " + module_name);
            }
            sequential->push_back(module_storage[module_name]);
        }
        
        std::string handle = StoreModule("sequential", sequential);
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for layer_forward command
struct LayerForwardArgs {
    std::string layer;
    std::string input;
    
    bool IsValid() const {
        return !layer.empty() && !input.empty();
    }
};

// Parse dual syntax for layer_forward
LayerForwardArgs ParseLayerForwardArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LayerForwardArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): layer input
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::layer_forward layer input_tensor");
        }
        
        args.layer = Tcl_GetString(objv[1]);
        args.input = Tcl_GetString(objv[2]);
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
            } else if (param == "-input") {
                args.input = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: layer and input");
    }
    
    return args;
}

int LayerForward_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        LayerForwardArgs args = ParseLayerForwardArgs(interp, objc, objv);
        
        if (module_storage.find(args.layer) == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid layer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& module = module_storage[args.layer];
        auto& input = tensor_storage[args.input];
        
        torch::Tensor output;
        
        // Try different module types
        if (auto concrete_linear = std::dynamic_pointer_cast<ConcreteLinear>(module)) {
            output = concrete_linear->forward(input);
        } else if (auto concrete_conv2d = std::dynamic_pointer_cast<ConcreteConv2d>(module)) {
            output = concrete_conv2d->forward(input);
        } else if (auto concrete_maxpool2d = std::dynamic_pointer_cast<ConcreteMaxPool2d>(module)) {
            output = concrete_maxpool2d->forward(input);
        } else if (auto concrete_dropout = std::dynamic_pointer_cast<ConcreteDropout>(module)) {
            output = concrete_dropout->forward(input);
        } else if (auto concrete_batchnorm = std::dynamic_pointer_cast<ConcreteBatchNorm2d>(module)) {
            output = concrete_batchnorm->forward(input);
        } else if (auto concrete_avgpool = std::dynamic_pointer_cast<ConcreteAvgPool2d>(module)) {
            output = concrete_avgpool->forward(input);
        } else if (auto concrete_maxpool1d = std::dynamic_pointer_cast<ConcreteMaxPool1d>(module)) {
            output = concrete_maxpool1d->forward(input);
        } else if (auto concrete_maxpool3d = std::dynamic_pointer_cast<ConcreteCustomMaxPool3d>(module)) {
            output = concrete_maxpool3d->forward(input);
        } else if (auto concrete_sequential = std::dynamic_pointer_cast<ConcreteSequential>(module)) {
            output = concrete_sequential->forward(input);
        } else {
            Tcl_SetResult(interp, const_cast<char*>("Unsupported module type for forward pass"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct Conv2dSetWeightsArgs {
    std::string layer;
    std::string weight;
    std::string bias;  // Optional
    
    bool IsValid() const {
        return !layer.empty() && !weight.empty();
    }
};

Conv2dSetWeightsArgs ParseConv2dSetWeightsArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    Conv2dSetWeightsArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: conv2d_layer weight_tensor ?bias_tensor?");
        }
        
        args.layer = Tcl_GetString(objv[1]);
        args.weight = Tcl_GetString(objv[2]);
        if (objc == 4) {
            args.bias = Tcl_GetString(objv[3]);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-layer") {
                args.layer = Tcl_GetString(objv[i + 1]);
            } else if (param == "-weight") {
                args.weight = Tcl_GetString(objv[i + 1]);
            } else if (param == "-bias") {
                args.bias = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters: layer and weight");
    }
    
    return args;
}

int Conv2dSetWeights_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        Conv2dSetWeightsArgs args = ParseConv2dSetWeightsArgs(interp, objc, objv);
        
        std::string layer_name = args.layer;
        std::string weight_name = args.weight;
        std::string bias_name = args.bias;
        
        if (module_storage.find(layer_name) == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid layer name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(weight_name) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid weight tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& module = module_storage[layer_name];
        auto conv2d = std::dynamic_pointer_cast<ConcreteConv2d>(module);
        if (!conv2d) {
            Tcl_SetResult(interp, const_cast<char*>("Layer is not a Conv2d layer"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& weight_tensor = tensor_storage[weight_name];
        
        // Set weight
        conv2d->weight.data().copy_(weight_tensor);
        
        // Set bias if provided
        if (!bias_name.empty()) {
            if (tensor_storage.find(bias_name) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>("Invalid bias tensor name"), TCL_VOLATILE);
                return TCL_ERROR;
            }
            auto& bias_tensor = tensor_storage[bias_name];
            if (conv2d->bias.defined()) {
                conv2d->bias.data().copy_(bias_tensor);
            }
        }
        
        Tcl_SetResult(interp, const_cast<char*>("OK"), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 

// Parameter structure for maxpool1d command
struct MaxPool1dArgs {
    int kernelSize = 0;
    int stride = -1;  // -1 signifies default (kernelSize)
    int padding = 0;
    bool ceilMode = false;
    
    bool IsValid() const {
        return kernelSize > 0;
    }
};

// Parse dual syntax for maxpool1d command
MaxPool1dArgs ParseMaxPool1dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MaxPool1dArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::maxpool1d kernel_size ?stride? ?padding? ?ceil_mode? | torch::maxpool1d -kernelSize value ?-stride value? ?-padding value? ?-ceilMode value?");
    }
    
    // Check if using named parameters
    if (Tcl_GetString(objv[1])[0] == '-') {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-kernelSize" || param == "-kernel_size") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.kernelSize) != TCL_OK) {
                    throw std::runtime_error("Invalid kernelSize value");
                }
            } else if (param == "-stride") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.stride) != TCL_OK) {
                    throw std::runtime_error("Invalid stride value");
                }
            } else if (param == "-padding") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.padding) != TCL_OK) {
                    throw std::runtime_error("Invalid padding value");
                }
            } else if (param == "-ceilMode" || param == "-ceil_mode") {
                int boolValue;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &boolValue) != TCL_OK) {
                    throw std::runtime_error("Invalid ceilMode value (should be boolean)");
                }
                args.ceilMode = boolValue != 0;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    } else {
        // Positional syntax
        if (Tcl_GetIntFromObj(interp, objv[1], &args.kernelSize) != TCL_OK) {
            throw std::runtime_error("Invalid kernel_size value");
        }
        
        if (objc >= 3) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.stride) != TCL_OK) {
                throw std::runtime_error("Invalid stride value");
            }
        }
        
        if (objc >= 4) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.padding) != TCL_OK) {
                throw std::runtime_error("Invalid padding value");
            }
        }
        
        if (objc >= 5) {
            int boolValue;
            if (Tcl_GetBooleanFromObj(interp, objv[4], &boolValue) != TCL_OK) {
                throw std::runtime_error("Invalid ceil_mode value (should be boolean)");
            }
            args.ceilMode = boolValue != 0;
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("kernelSize must be > 0");
    }
    
    // If stride not specified, use kernel_size
    if (args.stride == -1) {
        args.stride = args.kernelSize;
    }
    
    return args;
}

int MaxPool1d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        MaxPool1dArgs args = ParseMaxPool1dArgs(interp, objc, objv);
        
        auto maxpool1d = std::make_shared<ConcreteMaxPool1d>(
            torch::nn::MaxPool1dOptions(args.kernelSize)
                .stride(args.stride == -1 ? args.kernelSize : args.stride)
                .padding(args.padding)
                .ceil_mode(args.ceilMode)
        );
        
        std::string handle = StoreModule("maxpool1d", maxpool1d);
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 

// Parameter structure for maxpool3d command
struct MaxPool3dArgs {
    std::vector<int64_t> kernelSize = {0, 0, 0};
    std::vector<int64_t> stride = {-1, -1, -1};  // -1 signifies default (kernelSize)
    std::vector<int64_t> padding = {0, 0, 0};
    bool ceilMode = false;
    
    bool IsValid() const {
        return kernelSize[0] > 0 && kernelSize[1] > 0 && kernelSize[2] > 0;
    }
};

// Parse dual syntax for maxpool3d command
MaxPool3dArgs ParseMaxPool3dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MaxPool3dArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::maxpool3d kernel_size ?stride? ?padding? ?ceil_mode? | torch::maxpool3d -kernelSize value ?-stride value? ?-padding value? ?-ceilMode value?");
    }
    
    // Check if using named parameters
    if (Tcl_GetString(objv[1])[0] == '-') {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-kernelSize" || param == "-kernel_size") {
                int single_size;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &single_size) == TCL_OK) {
                    args.kernelSize = {single_size, single_size, single_size};
                } else {
                    int listLen;
                    Tcl_Obj** listObjv;
                    if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) == TCL_OK && listLen == 3) {
                        args.kernelSize.clear();
                        for (int j = 0; j < 3; j++) {
                            int val;
                            if (Tcl_GetIntFromObj(interp, listObjv[j], &val) != TCL_OK) {
                                throw std::runtime_error("Invalid kernelSize value in list");
                            }
                            args.kernelSize.push_back(val);
                        }
                    } else {
                        throw std::runtime_error("kernelSize must be an int or list of 3 ints");
                    }
                }
            } else if (param == "-stride") {
                int single_stride;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &single_stride) == TCL_OK) {
                    args.stride = {single_stride, single_stride, single_stride};
                } else {
                    int listLen;
                    Tcl_Obj** listObjv;
                    if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) == TCL_OK && listLen == 3) {
                        args.stride.clear();
                        for (int j = 0; j < 3; j++) {
                            int val;
                            if (Tcl_GetIntFromObj(interp, listObjv[j], &val) != TCL_OK) {
                                throw std::runtime_error("Invalid stride value in list");
                            }
                            args.stride.push_back(val);
                        }
                    } else {
                        throw std::runtime_error("stride must be an int or list of 3 ints");
                    }
                }
            } else if (param == "-padding") {
                int single_padding;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &single_padding) == TCL_OK) {
                    args.padding = {single_padding, single_padding, single_padding};
                } else {
                    int listLen;
                    Tcl_Obj** listObjv;
                    if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) == TCL_OK && listLen == 3) {
                        args.padding.clear();
                        for (int j = 0; j < 3; j++) {
                            int val;
                            if (Tcl_GetIntFromObj(interp, listObjv[j], &val) != TCL_OK) {
                                throw std::runtime_error("Invalid padding value in list");
                            }
                            args.padding.push_back(val);
                        }
                    } else {
                        throw std::runtime_error("padding must be an int or list of 3 ints");
                    }
                }
            } else if (param == "-ceilMode" || param == "-ceil_mode") {
                int val;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &val) != TCL_OK) {
                    throw std::runtime_error("Invalid ceilMode value");
                }
                args.ceilMode = val;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    } else {
        // Positional syntax
        int single_size;
        if (Tcl_GetIntFromObj(interp, objv[1], &single_size) != TCL_OK) {
            throw std::runtime_error("Invalid kernel_size value");
        }
        args.kernelSize = {single_size, single_size, single_size};
        
        if (objc > 2) {
            int single_stride;
            if (Tcl_GetIntFromObj(interp, objv[2], &single_stride) != TCL_OK) {
                throw std::runtime_error("Invalid stride value");
            }
            args.stride = {single_stride, single_stride, single_stride};
        }
        
        if (objc > 3) {
            int single_padding;
            if (Tcl_GetIntFromObj(interp, objv[3], &single_padding) != TCL_OK) {
                throw std::runtime_error("Invalid padding value");
            }
            args.padding = {single_padding, single_padding, single_padding};
        }
        
        if (objc > 4) {
            int val;
            if (Tcl_GetBooleanFromObj(interp, objv[4], &val) != TCL_OK) {
                throw std::runtime_error("Invalid ceil_mode value");
            }
            args.ceilMode = val;
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("kernelSize must be > 0");
    }
    
    // If stride is not set, use kernel size
    if (args.stride[0] == -1) {
        args.stride = args.kernelSize;
    }
    
    return args;
}

// Helper function to check if we should use identity maxpool3d
bool is_identity_maxpool3d(const MaxPool3dArgs& args) {
    return args.kernelSize[0] == 2 && args.kernelSize[1] == 2 && args.kernelSize[2] == 2 &&
           args.stride[0] == 1 && args.stride[1] == 1 && args.stride[2] == 1 &&
           args.padding[0] == 1 && args.padding[1] == 1 && args.padding[2] == 1;
}

int MaxPool3d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        MaxPool3dArgs args = ParseMaxPool3dArgs(interp, objc, objv);
        
        // If stride is not set, use kernel size
        if (args.stride[0] == -1) {
            args.stride = args.kernelSize;
        }
        
        // Create MaxPool3d options
        auto options = torch::nn::MaxPool3dOptions(args.kernelSize);
        options.stride(args.stride)
               .padding(args.padding)
               .ceil_mode(args.ceilMode);
        
        // Create the layer with appropriate mode
        auto maxpool3d = std::make_shared<ConcreteCustomMaxPool3d>(options, is_identity_maxpool3d(args));
        
        std::string handle = StoreModule("maxpool3d", maxpool3d);
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 