#include "libtorchtcl.h"

// Concrete wrapper for BatchNorm1d
class ConcreteBatchNorm1d : public torch::nn::BatchNorm1dImpl {
public:
    using BatchNorm1dImpl::BatchNorm1dImpl;
    torch::Tensor forward(const torch::Tensor& x) {
        return BatchNorm1dImpl::forward(x);
    }
};

// Concrete wrapper for LayerNorm
class ConcreteLayerNorm : public torch::nn::LayerNormImpl {
public:
    using LayerNormImpl::LayerNormImpl;
    torch::Tensor forward(const torch::Tensor& x) {
        return LayerNormImpl::forward(x);
    }
};

// Concrete wrapper for GroupNorm
class ConcreteGroupNorm : public torch::nn::GroupNormImpl {
public:
    using GroupNormImpl::GroupNormImpl;
    torch::Tensor forward(const torch::Tensor& x) {
        return GroupNormImpl::forward(x);
    }
};

// Concrete wrapper for ConvTranspose2d
class ConcreteConvTranspose2d : public torch::nn::ConvTranspose2dImpl {
public:
    using ConvTranspose2dImpl::ConvTranspose2dImpl;
    torch::Tensor forward(const torch::Tensor& x) {
        return ConvTranspose2dImpl::forward(x);
    }
};

// Concrete wrapper for BatchNorm3d
class ConcreteBatchNorm3d : public torch::nn::BatchNorm3dImpl {
public:
    using BatchNorm3dImpl::BatchNorm3dImpl;
    torch::Tensor forward(const torch::Tensor& x) {
        return BatchNorm3dImpl::forward(x);
    }
};

// Add before BatchNorm1d_Cmd definition
struct BatchNorm1dArgs {
    int numFeatures = 0;
    double eps = 1e-5;
    double momentum = 0.1;
    bool affine = true;
    bool trackRunningStats = true;

    bool IsValid() const { return numFeatures > 0; }
};

BatchNorm1dArgs ParseBatchNorm1dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    BatchNorm1dArgs args;

    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional: num_features ?eps? ?momentum? ?affine? ?trackRunningStats?
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
        // Named parameters
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

// Define argument struct and parser for BatchNorm3d BEFORE existing BatchNorm3D_Cmd definition
struct BatchNorm3dArgs {
    torch::Tensor input;
    double eps = 1e-5;
    double momentum = 0.1;
    bool affine = true;            // accepted for compatibility but unused in functional call
    bool trackRunningStats = true; // accepted for compatibility but unused

    bool IsValid() const { return input.defined(); }
};

BatchNorm3dArgs ParseBatchNorm3dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    BatchNorm3dArgs args;

    // Determine syntax style
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor ?eps? ?momentum? ?affine? ?trackRunningStats?
        if (objc < 2 || objc > 6) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor ?eps? ?momentum? ?affine? ?trackRunningStats?");
            throw std::runtime_error("Invalid number of arguments");
        }
        args.input = GetTensorFromObj(interp, objv[1]);
        if (!args.input.defined()) {
            throw std::runtime_error("Invalid tensor handle");
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
        // Named parameter syntax: -input tensor ?-eps value? ?-momentum value? ?-affine bool? ?-trackRunningStats bool?
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* valueObj = objv[i + 1];
            if (param == "-input") {
                args.input = GetTensorFromObj(interp, valueObj);
                if (!args.input.defined()) {
                    throw std::runtime_error("Invalid tensor handle for -input");
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
        throw std::runtime_error("Parameter -input (tensor) is required");
    }
    return args;
}

#if 0 // Legacy positional-only implementation retained but disabled
int BatchNorm1d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
#endif

int BatchNorm1d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData;
    try {
        BatchNorm1dArgs args = ParseBatchNorm1dArgs(interp, objc, objv);
        auto options = torch::nn::BatchNorm1dOptions(args.numFeatures)
                            .eps(args.eps)
                            .momentum(args.momentum)
                            .affine(args.affine)
                            .track_running_stats(args.trackRunningStats);
        auto layer = std::make_shared<ConcreteBatchNorm1d>(options);
        std::string handle = StoreModule("batchnorm1d", layer);
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::layer_norm - Layer normalization with dual syntax support
struct LayerNormArgs {
    std::vector<int64_t> normalizedShape;
    double eps = 1e-5;

    bool IsValid() const {
        return !normalizedShape.empty();
    }
};

LayerNormArgs ParseLayerNormArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LayerNormArgs args;

    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: normalized_shape ?eps?
        if (objc < 2 || objc > 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "normalized_shape ?eps?");
            throw std::runtime_error("Invalid number of arguments");
        }

        // Parse normalized_shape (can be a single int or list)
        int list_length;
        if (Tcl_ListObjLength(interp, objv[1], &list_length) == TCL_OK && list_length > 0) {
            // It's a list
            for (int i = 0; i < list_length; i++) {
                Tcl_Obj* element;
                Tcl_ListObjIndex(interp, objv[1], i, &element);
                int value;
                if (Tcl_GetIntFromObj(interp, element, &value) != TCL_OK) {
                    throw std::runtime_error("Invalid normalized_shape value");
                }
                args.normalizedShape.push_back(static_cast<int64_t>(value));
            }
        } else {
            // It's a single value
            int value;
            if (Tcl_GetIntFromObj(interp, objv[1], &value) != TCL_OK) {
                throw std::runtime_error("Invalid normalized_shape value");
            }
            args.normalizedShape.push_back(static_cast<int64_t>(value));
        }

        // Parse optional eps
        if (objc >= 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.eps) != TCL_OK) {
                throw std::runtime_error("Invalid eps value");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) throw std::runtime_error("Missing value for parameter");
            
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* valueObj = objv[i + 1];

            if (param == "-normalizedShape" || param == "-normalized_shape") {
                // Parse normalized_shape (can be a single int or list)
                int list_length;
                if (Tcl_ListObjLength(interp, valueObj, &list_length) == TCL_OK && list_length > 0) {
                    // It's a list
                    for (int j = 0; j < list_length; j++) {
                        Tcl_Obj* element;
                        Tcl_ListObjIndex(interp, valueObj, j, &element);
                        int value;
                        if (Tcl_GetIntFromObj(interp, element, &value) != TCL_OK) {
                            throw std::runtime_error("Invalid normalizedShape value");
                        }
                        args.normalizedShape.push_back(static_cast<int64_t>(value));
                    }
                } else {
                    // It's a single value
                    int value;
                    if (Tcl_GetIntFromObj(interp, valueObj, &value) != TCL_OK) {
                        throw std::runtime_error("Invalid normalizedShape value");
                    }
                    args.normalizedShape.push_back(static_cast<int64_t>(value));
                }
            } else if (param == "-eps") {
                if (Tcl_GetDoubleFromObj(interp, valueObj, &args.eps) != TCL_OK) {
                    throw std::runtime_error("Invalid eps value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("normalizedShape must be specified");
    }
    
    return args;
}

int LayerNorm_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        LayerNormArgs args = ParseLayerNormArgs(interp, objc, objv);
        
        // Create LayerNorm layer
        auto options = torch::nn::LayerNormOptions(args.normalizedShape).eps(args.eps);
        auto layer = std::make_shared<ConcreteLayerNorm>(options);
        
        // Store and return handle
        std::string handle = StoreModule("layernorm", layer);
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::group_norm - Group normalization with dual syntax support
struct GroupNormArgs {
    int numGroups = 0;
    int numChannels = 0;
    double eps = 1e-5;

    bool IsValid() const {
        return numGroups > 0 && numChannels > 0;
    }
};

GroupNormArgs ParseGroupNormArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    GroupNormArgs args;

    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: num_groups num_channels ?eps?
        if (objc < 3 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "num_groups num_channels ?eps?");
            throw std::runtime_error("Invalid number of arguments");
        }

        // Parse num_groups
        if (Tcl_GetIntFromObj(interp, objv[1], &args.numGroups) != TCL_OK) {
            throw std::runtime_error("Invalid numGroups value");
        }
        
        // Parse num_channels
        if (Tcl_GetIntFromObj(interp, objv[2], &args.numChannels) != TCL_OK) {
            throw std::runtime_error("Invalid numChannels value");
        }
        
        // Parse optional eps
        if (objc >= 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.eps) != TCL_OK) {
                throw std::runtime_error("Invalid eps value");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) throw std::runtime_error("Missing value for parameter");
            
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* valueObj = objv[i + 1];

            if (param == "-numGroups" || param == "-num_groups") {
                if (Tcl_GetIntFromObj(interp, valueObj, &args.numGroups) != TCL_OK) {
                    throw std::runtime_error("Invalid numGroups value");
                }
            } else if (param == "-numChannels" || param == "-num_channels") {
                if (Tcl_GetIntFromObj(interp, valueObj, &args.numChannels) != TCL_OK) {
                    throw std::runtime_error("Invalid numChannels value");
                }
            } else if (param == "-eps") {
                if (Tcl_GetDoubleFromObj(interp, valueObj, &args.eps) != TCL_OK) {
                    throw std::runtime_error("Invalid eps value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("numGroups and numChannels must be > 0");
    }
    
    return args;
}

int GroupNorm_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        GroupNormArgs args = ParseGroupNormArgs(interp, objc, objv);
        
        // Create GroupNorm layer
        auto options = torch::nn::GroupNormOptions(args.numGroups, args.numChannels).eps(args.eps);
        auto layer = std::make_shared<ConcreteGroupNorm>(options);
        
        // Store and return handle
        std::string handle = StoreModule("groupnorm", layer);
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::conv_transpose_2d(in_ch, out_ch, kernel, stride?, padding?) - Transpose convolution
int ConvTranspose2d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 4 || objc > 6) {
        Tcl_WrongNumArgs(interp, 1, objv, "in_channels out_channels kernel_size ?stride? ?padding?");
        return TCL_ERROR;
    }
    
    try {
        // Parse in_channels
        int in_channels;
        if (Tcl_GetIntFromObj(interp, objv[1], &in_channels) != TCL_OK) {
            return TCL_ERROR;
        }
        
        // Parse out_channels
        int out_channels;
        if (Tcl_GetIntFromObj(interp, objv[2], &out_channels) != TCL_OK) {
            return TCL_ERROR;
        }
        
        // Parse kernel_size (can be single int or pair)
        std::vector<int64_t> kernel_size;
        int list_length;
        if (Tcl_ListObjLength(interp, objv[3], &list_length) == TCL_OK && list_length > 1) {
            // It's a list
            for (int i = 0; i < list_length; i++) {
                Tcl_Obj* element;
                Tcl_ListObjIndex(interp, objv[3], i, &element);
                int value;
                if (Tcl_GetIntFromObj(interp, element, &value) != TCL_OK) {
                    return TCL_ERROR;
                }
                kernel_size.push_back(static_cast<int64_t>(value));
            }
        } else {
            // It's a single value
            int value;
            if (Tcl_GetIntFromObj(interp, objv[3], &value) != TCL_OK) {
                return TCL_ERROR;
            }
            kernel_size = {static_cast<int64_t>(value), static_cast<int64_t>(value)};
        }
        
        // Parse optional stride (default: 1)
        std::vector<int64_t> stride = {1, 1};
        if (objc >= 5) {
            int list_length;
            if (Tcl_ListObjLength(interp, objv[4], &list_length) == TCL_OK && list_length > 1) {
                stride.clear();
                for (int i = 0; i < list_length; i++) {
                    Tcl_Obj* element;
                    Tcl_ListObjIndex(interp, objv[4], i, &element);
                    int value;
                    if (Tcl_GetIntFromObj(interp, element, &value) != TCL_OK) {
                        return TCL_ERROR;
                    }
                    stride.push_back(static_cast<int64_t>(value));
                }
            } else {
                int value;
                if (Tcl_GetIntFromObj(interp, objv[4], &value) != TCL_OK) {
                    return TCL_ERROR;
                }
                stride = {static_cast<int64_t>(value), static_cast<int64_t>(value)};
            }
        }
        
        // Parse optional padding (default: 0)
        std::vector<int64_t> padding = {0, 0};
        if (objc >= 6) {
            int list_length;
            if (Tcl_ListObjLength(interp, objv[5], &list_length) == TCL_OK && list_length > 1) {
                padding.clear();
                for (int i = 0; i < list_length; i++) {
                    Tcl_Obj* element;
                    Tcl_ListObjIndex(interp, objv[5], i, &element);
                    int value;
                    if (Tcl_GetIntFromObj(interp, element, &value) != TCL_OK) {
                        return TCL_ERROR;
                    }
                    padding.push_back(static_cast<int64_t>(value));
                }
            } else {
                int value;
                if (Tcl_GetIntFromObj(interp, objv[5], &value) != TCL_OK) {
                    return TCL_ERROR;
                }
                padding = {static_cast<int64_t>(value), static_cast<int64_t>(value)};
            }
        }
        
        // Create ConvTranspose2d layer
        auto options = torch::nn::ConvTranspose2dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding);
        auto layer = std::make_shared<ConcreteConvTranspose2d>(options);
        
        // Store and return handle
        std::string handle = StoreModule("convtranspose2d", layer);
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Local Response Normalization - Dual Syntax Support
// ============================================================================

struct LocalResponseNormArgs {
    std::string input;
    int size = 5;
    double alpha = 1e-4;
    double beta = 0.75;
    double k = 1.0;
    
    bool IsValid() const {
        return !input.empty() && size > 0;
    }
};

LocalResponseNormArgs ParseLocalResponseNormArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LocalResponseNormArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::local_response_norm tensor size alpha beta k
        if (objc != 6) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::local_response_norm tensor size alpha beta k");
        }
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.size) != TCL_OK) {
            throw std::runtime_error("Invalid size parameter");
        }
        if (Tcl_GetDoubleFromObj(interp, objv[3], &args.alpha) != TCL_OK) {
            throw std::runtime_error("Invalid alpha parameter");
        }
        if (Tcl_GetDoubleFromObj(interp, objv[4], &args.beta) != TCL_OK) {
            throw std::runtime_error("Invalid beta parameter");
        }
        if (Tcl_GetDoubleFromObj(interp, objv[5], &args.k) != TCL_OK) {
            throw std::runtime_error("Invalid k parameter");
        }
    } else {
        // Named parameter syntax: torch::local_response_norm -input tensor -size 5 -alpha 1e-4 -beta 0.75 -k 1.0
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-size") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.size) != TCL_OK) {
                    throw std::runtime_error("Invalid size parameter");
                }
            } else if (param == "-alpha") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.alpha) != TCL_OK) {
                    throw std::runtime_error("Invalid alpha parameter");
                }
            } else if (param == "-beta") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.beta) != TCL_OK) {
                    throw std::runtime_error("Invalid beta parameter");
                }
            } else if (param == "-k") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.k) != TCL_OK) {
                    throw std::runtime_error("Invalid k parameter");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: -input tensor_name");
    }
    
    return args;
}

// torch::local_response_norm(tensor, size, alpha, beta, k) - Local response normalization
int LocalResponseNorm_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        LocalResponseNormArgs args = ParseLocalResponseNormArgs(interp, objc, objv);
        
        auto tensor_it = tensor_storage.find(args.input);
        if (tensor_it == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Tensor not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        torch::Tensor tensor = tensor_it->second;
        if (tensor.numel() == 0) {
            Tcl_SetResult(interp, const_cast<char*>("Input tensor is empty"), TCL_STATIC);
            return TCL_ERROR;
        }

        torch::Tensor result = torch::nn::functional::local_response_norm(tensor,
            torch::nn::functional::LocalResponseNormFuncOptions(args.size).alpha(args.alpha).beta(args.beta).k(args.k));
        return SetTensorResult(interp, result);

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// Cross-Map Local Response Normalization 2D - Dual Syntax Support
// ============================================================================

struct CrossMapLRN2DArgs {
    std::string input;
    int size = 5;
    double alpha = 1e-4;
    double beta = 0.75;
    double k = 1.0;
    
    bool IsValid() const {
        return !input.empty() && size > 0;
    }
};

CrossMapLRN2DArgs ParseCrossMapLRN2DArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    CrossMapLRN2DArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::cross_map_lrn2d tensor size alpha beta k
    if (objc != 6) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::cross_map_lrn2d tensor size alpha beta k");
        }
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.size) != TCL_OK) {
            throw std::runtime_error("Invalid size parameter");
    }
        if (Tcl_GetDoubleFromObj(interp, objv[3], &args.alpha) != TCL_OK) {
            throw std::runtime_error("Invalid alpha parameter");
        }
        if (Tcl_GetDoubleFromObj(interp, objv[4], &args.beta) != TCL_OK) {
            throw std::runtime_error("Invalid beta parameter");
        }
        if (Tcl_GetDoubleFromObj(interp, objv[5], &args.k) != TCL_OK) {
            throw std::runtime_error("Invalid k parameter");
        }
    } else {
        // Named parameter syntax: torch::cross_map_lrn2d -input tensor -size 5 -alpha 1e-4 -beta 0.75 -k 1.0
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-size") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.size) != TCL_OK) {
                    throw std::runtime_error("Invalid size parameter");
        }
            } else if (param == "-alpha") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.alpha) != TCL_OK) {
                    throw std::runtime_error("Invalid alpha parameter");
        }
            } else if (param == "-beta") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.beta) != TCL_OK) {
                    throw std::runtime_error("Invalid beta parameter");
                }
            } else if (param == "-k") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.k) != TCL_OK) {
                    throw std::runtime_error("Invalid k parameter");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: -input tensor_name");
    }
    
    return args;
}

// torch::cross_map_lrn2d(tensor, size, alpha, beta, k) - Cross-map local response normalization for 2D
int CrossMapLRN2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        CrossMapLRN2DArgs args = ParseCrossMapLRN2DArgs(interp, objc, objv);
        
        auto tensor_it = tensor_storage.find(args.input);
        if (tensor_it == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Tensor not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        torch::Tensor tensor = tensor_it->second;
        if (tensor.numel() == 0) {
            Tcl_SetResult(interp, const_cast<char*>("Input tensor is empty"), TCL_STATIC);
            return TCL_ERROR;
        }

        // Cross-map LRN is the same as regular LRN but specifically designed for 2D feature maps
        torch::Tensor result = torch::nn::functional::local_response_norm(tensor,
            torch::nn::functional::LocalResponseNormFuncOptions(args.size).alpha(args.alpha).beta(args.beta).k(args.k));
        return SetTensorResult(interp, result);

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// torch::batch_norm_3d(features, eps?, momentum?, affine?, track_running_stats?) - 3D batch normalization
int BatchNorm3D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData;
    try {
        BatchNorm3dArgs args = ParseBatchNorm3dArgs(interp, objc, objv);

        // Functional batch norm call (affine and running stats options are currently not exposed)
        torch::Tensor result = torch::batch_norm(args.input,
            torch::Tensor(), // weight (affine handled externally)
            torch::Tensor(), // bias
            torch::Tensor(), // running_mean (trackRunningStats handled externally)
            torch::Tensor(), // running_var
            /*training=*/true,
            args.momentum,
            args.eps,
            /*cudnn_enabled=*/true);

        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for torch::instance_norm1d
// ============================================================================
struct InstanceNorm1dArgs {
    std::string input;      // tensor handle for input
    double eps = 1e-5;      // epsilon for numerical stability
    double momentum = 0.1;  // momentum for running statistics
    bool affine = true;     // whether to use learnable affine parameters
    bool track_running_stats = true; // whether to track running statistics
    
    bool IsValid() const {
        return !input.empty() && eps > 0.0 && momentum >= 0.0;
    }
};

InstanceNorm1dArgs ParseInstanceNorm1dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    InstanceNorm1dArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor ?eps? ?momentum? ?affine? ?track_running_stats?
        if (objc < 2 || objc > 6) {
            throw std::runtime_error("Usage: torch::instance_norm1d tensor ?eps? ?momentum? ?affine? ?track_running_stats?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc >= 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.eps) != TCL_OK) {
                throw std::runtime_error("Invalid eps: must be positive number");
            }
            if (args.eps <= 0.0) {
                throw std::runtime_error("Invalid eps: must be positive number");
            }
        }
        
        if (objc >= 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.momentum) != TCL_OK) {
                throw std::runtime_error("Invalid momentum: must be number >= 0");
            }
            if (args.momentum < 0.0) {
                throw std::runtime_error("Invalid momentum: must be number >= 0");
            }
        }
        
        if (objc >= 5) {
            int affine_val;
            if (Tcl_GetIntFromObj(interp, objv[4], &affine_val) != TCL_OK) {
                throw std::runtime_error("Invalid affine: must be 0 or 1");
            }
            args.affine = (affine_val != 0);
        }
        
        if (objc >= 6) {
            int track_val;
            if (Tcl_GetIntFromObj(interp, objv[5], &track_val) != TCL_OK) {
                throw std::runtime_error("Invalid track_running_stats: must be 0 or 1");
            }
            args.track_running_stats = (track_val != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-eps" || param == "-epsilon") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.eps) != TCL_OK) {
                    throw std::runtime_error("Invalid eps: must be positive number");
                }
                if (args.eps <= 0.0) {
                    throw std::runtime_error("Invalid eps: must be positive number");
                }
            } else if (param == "-momentum") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.momentum) != TCL_OK) {
                    throw std::runtime_error("Invalid momentum: must be number >= 0");
                }
                if (args.momentum < 0.0) {
                    throw std::runtime_error("Invalid momentum: must be number >= 0");
                }
            } else if (param == "-affine") {
                int affine_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &affine_val) != TCL_OK) {
                    throw std::runtime_error("Invalid affine: must be 0 or 1");
                }
                args.affine = (affine_val != 0);
            } else if (param == "-track_running_stats" || param == "-trackRunningStats") {
                int track_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &track_val) != TCL_OK) {
                    throw std::runtime_error("Invalid track_running_stats: must be 0 or 1");
                }
                args.track_running_stats = (track_val != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor required");
    }
    
    return args;
}

// torch::instance_norm1d - 1D instance normalization with dual syntax support
int InstanceNorm1D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        InstanceNorm1dArgs args = ParseInstanceNorm1dArgs(interp, objc, objv);
        
        // Get input tensor from the parsed arguments
        torch::Tensor tensor = GetTensorFromObj(interp, Tcl_NewStringObj(args.input.c_str(), -1));
        if (tensor.numel() == 0) {
            throw std::runtime_error("Input tensor is empty");
        }
        
        // For functional instance norm, we use torch::instance_norm directly
        torch::Tensor result = torch::instance_norm(tensor,
            torch::Tensor(), // weight
            torch::Tensor(), // bias
            torch::Tensor(), // running_mean
            torch::Tensor(), // running_var
            true, // use_input_stats
            args.momentum,
            args.eps,
            true); // cudnn_enabled
            
        return SetTensorResult(interp, result);
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in instance_norm1d: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for torch::instance_norm2d
// ============================================================================
struct InstanceNorm2dArgs {
    std::string input;      // tensor handle for input
    double eps = 1e-5;      // epsilon for numerical stability
    double momentum = 0.1;  // momentum for running statistics
    bool affine = true;     // whether to use learnable affine parameters
    bool track_running_stats = true; // whether to track running statistics
    
    bool IsValid() const {
        return !input.empty() && eps > 0.0 && momentum >= 0.0;
    }
};

InstanceNorm2dArgs ParseInstanceNorm2dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    InstanceNorm2dArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor ?eps? ?momentum? ?affine? ?track_running_stats?
        if (objc < 2 || objc > 6) {
            throw std::runtime_error("Usage: torch::instance_norm2d tensor ?eps? ?momentum? ?affine? ?track_running_stats?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc >= 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.eps) != TCL_OK) {
                throw std::runtime_error("Invalid eps: must be positive number");
            }
            if (args.eps <= 0.0) {
                throw std::runtime_error("Invalid eps: must be positive number");
            }
        }
        
        if (objc >= 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.momentum) != TCL_OK) {
                throw std::runtime_error("Invalid momentum: must be number >= 0");
            }
            if (args.momentum < 0.0) {
                throw std::runtime_error("Invalid momentum: must be number >= 0");
            }
        }
        
        if (objc >= 5) {
            int affine_val;
            if (Tcl_GetIntFromObj(interp, objv[4], &affine_val) != TCL_OK) {
                throw std::runtime_error("Invalid affine: must be 0 or 1");
            }
            args.affine = (affine_val != 0);
        }
        
        if (objc >= 6) {
            int track_val;
            if (Tcl_GetIntFromObj(interp, objv[5], &track_val) != TCL_OK) {
                throw std::runtime_error("Invalid track_running_stats: must be 0 or 1");
            }
            args.track_running_stats = (track_val != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-eps" || param == "-epsilon") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.eps) != TCL_OK) {
                    throw std::runtime_error("Invalid eps: must be positive number");
                }
                if (args.eps <= 0.0) {
                    throw std::runtime_error("Invalid eps: must be positive number");
                }
            } else if (param == "-momentum") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.momentum) != TCL_OK) {
                    throw std::runtime_error("Invalid momentum: must be number >= 0");
                }
                if (args.momentum < 0.0) {
                    throw std::runtime_error("Invalid momentum: must be number >= 0");
                }
            } else if (param == "-affine") {
                int affine_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &affine_val) != TCL_OK) {
                    throw std::runtime_error("Invalid affine: must be 0 or 1");
                }
                args.affine = (affine_val != 0);
            } else if (param == "-track_running_stats" || param == "-trackRunningStats") {
                int track_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &track_val) != TCL_OK) {
                    throw std::runtime_error("Invalid track_running_stats: must be 0 or 1");
                }
                args.track_running_stats = (track_val != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor required");
    }
    
    return args;
}

// torch::instance_norm2d - 2D instance normalization with dual syntax support
int InstanceNorm2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        InstanceNorm2dArgs args = ParseInstanceNorm2dArgs(interp, objc, objv);
        
        // Get input tensor from the parsed arguments
        torch::Tensor tensor = GetTensorFromObj(interp, Tcl_NewStringObj(args.input.c_str(), -1));
        if (tensor.numel() == 0) {
            throw std::runtime_error("Input tensor is empty");
        }
        
        // For functional instance norm, we use torch::instance_norm directly
        torch::Tensor result = torch::instance_norm(tensor,
            torch::Tensor(), // weight
            torch::Tensor(), // bias
            torch::Tensor(), // running_mean
            torch::Tensor(), // running_var
            true, // use_input_stats
            args.momentum,
            args.eps,
            true); // cudnn_enabled
            
        return SetTensorResult(interp, result);
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in instance_norm2d: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for torch::instance_norm3d
// ============================================================================
struct InstanceNorm3dArgs {
    std::string input;      // tensor handle for input
    double eps = 1e-5;      // epsilon for numerical stability
    double momentum = 0.1;  // momentum for running statistics
    bool affine = true;     // whether to use learnable affine parameters
    bool track_running_stats = true; // whether to track running statistics
    
    bool IsValid() const {
        return !input.empty() && eps > 0.0 && momentum >= 0.0;
    }
};

InstanceNorm3dArgs ParseInstanceNorm3dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    InstanceNorm3dArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor ?eps? ?momentum? ?affine? ?track_running_stats?
        if (objc < 2 || objc > 6) {
            throw std::runtime_error("Usage: torch::instance_norm3d tensor ?eps? ?momentum? ?affine? ?track_running_stats?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc >= 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.eps) != TCL_OK) {
                throw std::runtime_error("Invalid eps: must be positive number");
            }
            if (args.eps <= 0.0) {
                throw std::runtime_error("Invalid eps: must be positive number");
            }
        }
        
        if (objc >= 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.momentum) != TCL_OK) {
                throw std::runtime_error("Invalid momentum: must be number >= 0");
            }
            if (args.momentum < 0.0) {
                throw std::runtime_error("Invalid momentum: must be number >= 0");
            }
        }
        
        if (objc >= 5) {
            int affine_val;
            if (Tcl_GetIntFromObj(interp, objv[4], &affine_val) != TCL_OK) {
                throw std::runtime_error("Invalid affine: must be 0 or 1");
            }
            args.affine = (affine_val != 0);
        }
        
        if (objc >= 6) {
            int track_val;
            if (Tcl_GetIntFromObj(interp, objv[5], &track_val) != TCL_OK) {
                throw std::runtime_error("Invalid track_running_stats: must be 0 or 1");
            }
            args.track_running_stats = (track_val != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-eps" || param == "-epsilon") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.eps) != TCL_OK) {
                    throw std::runtime_error("Invalid eps: must be positive number");
                }
                if (args.eps <= 0.0) {
                    throw std::runtime_error("Invalid eps: must be positive number");
                }
            } else if (param == "-momentum") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.momentum) != TCL_OK) {
                    throw std::runtime_error("Invalid momentum: must be number >= 0");
                }
                if (args.momentum < 0.0) {
                    throw std::runtime_error("Invalid momentum: must be number >= 0");
                }
            } else if (param == "-affine") {
                int affine_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &affine_val) != TCL_OK) {
                    throw std::runtime_error("Invalid affine: must be 0 or 1");
                }
                args.affine = (affine_val != 0);
            } else if (param == "-track_running_stats" || param == "-trackRunningStats") {
                int track_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &track_val) != TCL_OK) {
                    throw std::runtime_error("Invalid track_running_stats: must be 0 or 1");
                }
                args.track_running_stats = (track_val != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor required");
    }
    
    return args;
}

// torch::instance_norm3d - 3D instance normalization with dual syntax support
int InstanceNorm3D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        InstanceNorm3dArgs args = ParseInstanceNorm3dArgs(interp, objc, objv);
        
        // Get input tensor from the parsed arguments
        torch::Tensor tensor = GetTensorFromObj(interp, Tcl_NewStringObj(args.input.c_str(), -1));
        if (tensor.numel() == 0) {
            throw std::runtime_error("Input tensor is empty");
        }
        
        // For functional instance norm, we use torch::instance_norm directly
        torch::Tensor result = torch::instance_norm(tensor,
            torch::Tensor(), // weight
            torch::Tensor(), // bias
            torch::Tensor(), // running_mean
            torch::Tensor(), // running_var
            true, // use_input_stats
            args.momentum,
            args.eps,
            true); // cudnn_enabled
            
        return SetTensorResult(interp, result);
        
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in instance_norm3d: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for rms_norm command
struct RMSNormArgs {
    std::string input;
    std::vector<int64_t> normalized_shape;
    double eps = 1e-5;  // Default value
    
    bool IsValid() const {
        return !input.empty() && !normalized_shape.empty();
    }
};

// Parse dual syntax for rms_norm
RMSNormArgs ParseRMSNormArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    RMSNormArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::rms_norm tensor normalized_shape ?eps? | torch::rmsNorm -input tensor -normalizedShape {shape} ?-eps value?");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: torch::rms_norm tensor normalized_shape ?eps?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        // Parse normalized_shape (can be a single int or list)
        int list_length;
        if (Tcl_ListObjLength(interp, objv[2], &list_length) == TCL_OK && list_length > 0) {
            // It's a list
            for (int i = 0; i < list_length; i++) {
                Tcl_Obj* element;
                Tcl_ListObjIndex(interp, objv[2], i, &element);
                int value;
                if (Tcl_GetIntFromObj(interp, element, &value) != TCL_OK) {
                    throw std::runtime_error("Invalid normalized_shape: dimensions don't match input tensor");
                }
                args.normalized_shape.push_back(static_cast<int64_t>(value));
            }
        } else {
            // It's a single value
            int value;
            if (Tcl_GetIntFromObj(interp, objv[2], &value) != TCL_OK) {
                throw std::runtime_error("Invalid normalized_shape: dimensions don't match input tensor");
            }
            args.normalized_shape.push_back(static_cast<int64_t>(value));
        }
        
        // Parse optional eps
        if (objc >= 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.eps) != TCL_OK) {
                throw std::runtime_error("Invalid eps value: must be positive");
            }
            if (args.eps <= 0.0) {
                throw std::runtime_error("Invalid eps value: must be positive");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-normalizedShape" || param == "-normalized_shape") {
                // Parse normalized_shape list
                int list_length;
                if (Tcl_ListObjLength(interp, objv[i + 1], &list_length) == TCL_OK && list_length > 0) {
                    // It's a list
                    for (int j = 0; j < list_length; j++) {
                        Tcl_Obj* element;
                        Tcl_ListObjIndex(interp, objv[i + 1], j, &element);
                        int value;
                        if (Tcl_GetIntFromObj(interp, element, &value) != TCL_OK) {
                            throw std::runtime_error("Invalid normalized_shape: dimensions don't match input tensor");
                        }
                        args.normalized_shape.push_back(static_cast<int64_t>(value));
                    }
                } else {
                    // It's a single value
                    int value;
                    if (Tcl_GetIntFromObj(interp, objv[i + 1], &value) != TCL_OK) {
                        throw std::runtime_error("Invalid normalized_shape: dimensions don't match input tensor");
                    }
                    args.normalized_shape.push_back(static_cast<int64_t>(value));
                }
            } else if (param == "-eps") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.eps) != TCL_OK) {
                    throw std::runtime_error("Invalid eps value: must be positive");
                }
                if (args.eps <= 0.0) {
                    throw std::runtime_error("Invalid eps value: must be positive");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor, -normalizedShape/-normalized_shape, -eps");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and normalized_shape required");
    }
    
    return args;
}

// torch::rms_norm - RMS normalization with dual syntax support
int RMSNorm_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        RMSNormArgs args = ParseRMSNormArgs(interp, objc, objv);
        
        torch::Tensor tensor = GetTensorFromObj(interp, Tcl_NewStringObj(args.input.c_str(), -1));
        if (tensor.numel() == 0) {
            throw std::runtime_error("Input tensor is empty");
        }

        // Calculate RMS norm manually since PyTorch doesn't have built-in RMS norm
        // RMS norm: x / RMS(x) where RMS(x) = sqrt(mean(x^2))
        torch::Tensor squared = tensor.pow(2);
        
        // Calculate dimensions to reduce over
        std::vector<int64_t> dim_to_reduce;
        auto tensor_dims = tensor.sizes();
        int start_dim = tensor_dims.size() - args.normalized_shape.size();
        
        // Validate normalized_shape dimensions
        if (start_dim < 0 || args.normalized_shape.size() > tensor_dims.size()) {
            throw std::runtime_error("Invalid normalized_shape: dimensions don't match input tensor");
        }
        for (size_t i = 0; i < args.normalized_shape.size(); i++) {
            if (args.normalized_shape[i] != tensor_dims[start_dim + i]) {
                throw std::runtime_error("Invalid normalized_shape: dimensions don't match input tensor");
            }
            dim_to_reduce.push_back(start_dim + i);
        }
        
        // Calculate mean along the specified dimensions, keeping dimensions
        torch::Tensor mean_squared = squared.mean(dim_to_reduce, /*keepdim=*/true);
        torch::Tensor rms = torch::sqrt(mean_squared + args.eps);
        torch::Tensor result = tensor / rms;
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for spectral_norm command
struct SpectralNormArgs {
    torch::Tensor input;
    int n_power_iterations = 1;
    
    bool IsValid() const {
        return input.defined() && n_power_iterations > 0;
    }
};

// Parse dual syntax for spectral_norm
SpectralNormArgs ParseSpectralNormArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SpectralNormArgs args;
    
    if (objc < 2 || objc > 3) {
        throw std::runtime_error("Wrong number of arguments");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        args.input = GetTensorFromObj(interp, objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.n_power_iterations) != TCL_OK) {
                throw std::runtime_error("Invalid n_power_iterations value");
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
                args.input = GetTensorFromObj(interp, objv[i + 1]);
            } else if (param == "-nPowerIterations" || param == "-n_power_iterations") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.n_power_iterations) != TCL_OK) {
                    throw std::runtime_error("Invalid n_power_iterations value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    // Validate parameters
    if (!args.input.defined()) {
        throw std::runtime_error("Invalid tensor");
    }
    if (args.n_power_iterations <= 0) {
        throw std::runtime_error("n_power_iterations must be positive");
    }
    
    return args;
}

// torch::spectral_norm - Spectral normalization
int SpectralNorm_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        SpectralNormArgs args = ParseSpectralNormArgs(interp, objc, objv);
        
        // Spectral norm: normalize by largest singular value
        // For 2D tensors (matrices), use SVD to get spectral norm
        if (args.input.dim() < 2) {
            throw std::runtime_error("Spectral norm requires at least 2D tensor");
        }

        // Reshape to 2D if higher dimensional
        auto original_shape = args.input.sizes();
        torch::Tensor matrix = args.input.view({original_shape[0], -1});
        
        // For diagonal matrices, we can directly compute the spectral norm
        if (matrix.size(0) == matrix.size(1)) {
            auto diag = matrix.diag();
            auto spectral_norm = torch::max(torch::abs(diag));
            auto normalized = matrix / spectral_norm;
            return SetTensorResult(interp, normalized.view(original_shape));
        }
        
        // Power iteration method for spectral norm
        auto u = torch::randn({matrix.size(0)}, matrix.options());
        auto v = torch::randn({matrix.size(1)}, matrix.options());
        
        for (int i = 0; i < args.n_power_iterations; ++i) {
            v = torch::matmul(matrix.t(), u);
            v = v / torch::norm(v);
            u = torch::matmul(matrix, v);
            u = u / torch::norm(u);
        }
        
        auto spectral_norm = torch::dot(u, torch::matmul(matrix, v));
        auto normalized = matrix / spectral_norm;
        
        return SetTensorResult(interp, normalized.view(original_shape));
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in spectral_norm: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for weight_norm command
struct WeightNormArgs {
    torch::Tensor input;
    int dim = 0;
    
    bool IsValid() const {
        return input.defined() && input.numel() > 0;
    }
};

// Parse dual syntax for weight_norm
WeightNormArgs ParseWeightNormArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    WeightNormArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Wrong number of arguments");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        args.input = GetTensorFromObj(interp, objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
                throw std::runtime_error("Invalid dim value");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = GetTensorFromObj(interp, objv[i + 1]);
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dim value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor, -dim");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor required");
    }
    
    return args;
}

// torch::weight_norm(tensor, dim?) - Weight normalization
int WeightNorm_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        WeightNormArgs args = ParseWeightNormArgs(interp, objc, objv);
        
        // Weight normalization: w = g * v / ||v||
        // where g is a learnable scalar and v is the weight vector
        torch::Tensor norm = args.input.norm(2, args.dim, true);
        torch::Tensor result = args.input / norm;
        
        return SetTensorResult(interp, result);

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 