#include "libtorchtcl.h"
#include <torch/torch.h>

// Forward declarations of global variables
extern std::unordered_map<std::string, torch::Tensor> tensor_storage;
extern std::unordered_map<std::string, std::shared_ptr<torch::nn::Module>> module_storage;
extern std::unordered_map<std::string, std::shared_ptr<torch::optim::Optimizer>> optimizer_storage;


extern "C" {

// ============================================================================
// Advanced Indexing and Slicing Operations
// ============================================================================

// Parameter structure for tensor_slice command
struct TensorSliceArgs {
    std::string tensor;
    int dim;
    int start;
    int end = -1;
    int step = 1;
    bool has_end = false;
    bool has_step = false;
    
    bool IsValid() const {
        return !tensor.empty();
    }
};

// Parse dual syntax for tensor_slice
TensorSliceArgs ParseTensorSliceArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorSliceArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor dim start ?end? ?step?
        if (objc < 4 || objc > 6) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor dim start ?end? ?step?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.tensor = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
            throw std::runtime_error("Invalid dimension value");
        }
        
        if (Tcl_GetIntFromObj(interp, objv[3], &args.start) != TCL_OK) {
            throw std::runtime_error("Invalid start value");
        }
        
        if (objc >= 5) {
            if (Tcl_GetIntFromObj(interp, objv[4], &args.end) != TCL_OK) {
                throw std::runtime_error("Invalid end value");
            }
            args.has_end = true;
        }
        
        if (objc >= 6) {
            if (Tcl_GetIntFromObj(interp, objv[5], &args.step) != TCL_OK) {
                throw std::runtime_error("Invalid step value");
            }
            args.has_step = true;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-tensor" || param == "-input") {
                args.tensor = value;
            } else if (param == "-dim" || param == "-dimension") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dimension value");
                }
            } else if (param == "-start") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.start) != TCL_OK) {
                    throw std::runtime_error("Invalid start value");
                }
            } else if (param == "-end") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.end) != TCL_OK) {
                    throw std::runtime_error("Invalid end value");
                }
                args.has_end = true;
            } else if (param == "-step") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.step) != TCL_OK) {
                    throw std::runtime_error("Invalid step value");
                }
                args.has_step = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required tensor parameter missing");
    }
    
    return args;
}

int Torch_TensorSlice_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        TensorSliceArgs args = ParseTensorSliceArgs(interp, objc, objv);
        
        auto tensor_it = tensor_storage.find(args.tensor);
        if (tensor_it == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Tensor not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        auto tensor = tensor_it->second;
        torch::Tensor result;
        
        if (args.has_end) {
            result = tensor.slice(args.dim, args.start, args.end, args.step);
        } else {
            result = tensor.slice(args.dim, args.start);
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

// Parameter structure for tensor_advanced_index command
struct TensorAdvancedIndexArgs {
    std::string tensor;
    std::vector<std::string> indices;
    
    bool IsValid() const {
        return !tensor.empty() && !indices.empty();
    }
};

// Parse dual syntax for tensor_advanced_index
TensorAdvancedIndexArgs ParseTensorAdvancedIndexArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorAdvancedIndexArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::tensor_advanced_index tensor indices_list | torch::tensor_advanced_index -tensor tensor -indices indices_list");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::tensor_advanced_index tensor indices_list");
        }
        args.tensor = Tcl_GetString(objv[1]);
        
        // Parse indices list
        int list_length;
        Tcl_Obj **list_items;
        if (Tcl_ListObjGetElements(interp, objv[2], &list_length, &list_items) != TCL_OK) {
            throw std::runtime_error("Invalid indices list format");
        }
        
        for (int i = 0; i < list_length; i++) {
            args.indices.push_back(Tcl_GetString(list_items[i]));
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
            } else if (param == "-indices") {
                // Parse indices list
                int list_length;
                Tcl_Obj **list_items;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &list_length, &list_items) != TCL_OK) {
                    throw std::runtime_error("Invalid indices list format");
                }
                
                for (int j = 0; j < list_length; j++) {
                    args.indices.push_back(Tcl_GetString(list_items[j]));
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -tensor, -indices");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: tensor and indices list required");
    }
    
    return args;
}

int Torch_TensorAdvancedIndex_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        TensorAdvancedIndexArgs args = ParseTensorAdvancedIndexArgs(interp, objc, objv);
        
        auto tensor_it = tensor_storage.find(args.tensor);
        if (tensor_it == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Tensor not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        std::vector<torch::Tensor> indices;
        for (const auto& index_name : args.indices) {
            auto index_it = tensor_storage.find(index_name);
            if (index_it == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>("Index tensor not found"), TCL_STATIC);
                return TCL_ERROR;
            }
            indices.push_back(index_it->second);
        }

        // Real advanced indexing implementation using LibTorch indexing
        if (indices.empty()) {
            Tcl_SetResult(interp, const_cast<char*>("No indices provided"), TCL_STATIC);
            return TCL_ERROR;
        }
        
        // Use proper LibTorch advanced indexing with TensorIndex
        std::vector<torch::indexing::TensorIndex> tensor_indices;
        for (const auto& idx : indices) {
            tensor_indices.push_back(torch::indexing::TensorIndex(idx));
        }
        auto result = tensor_it->second.index(tensor_indices);

        std::string result_name = GetNextHandle("tensor");
        tensor_storage[result_name] = result;

        Tcl_SetResult(interp, const_cast<char*>(result_name.c_str()), TCL_VOLATILE);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// Sparse Tensor Operations
// ============================================================================

// Parameter structure for sparse_tensor_create
struct SparseTensorCreateArgs {
    std::string indices;
    std::string values;
    std::vector<int64_t> size;
    
    bool IsValid() const {
        return !indices.empty() && !values.empty() && !size.empty();
    }
};

// Dual syntax parser for sparse_tensor_create
SparseTensorCreateArgs ParseSparseTensorCreateArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SparseTensorCreateArgs args;
    
    // Check for named parameter syntax
    bool use_named_params = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');
    
    if (!use_named_params) {
        // Positional syntax (backward compatibility)
        if (objc != 4) {
            throw std::runtime_error("Usage: torch::sparse_tensor_create indices values size");
        }
        args.indices = Tcl_GetString(objv[1]);
        args.values = Tcl_GetString(objv[2]);
        args.size = TclListToShape(interp, objv[3]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-indices") {
                args.indices = Tcl_GetString(objv[i + 1]);
            } else if (param == "-values") {
                args.values = Tcl_GetString(objv[i + 1]);
            } else if (param == "-size") {
                args.size = TclListToShape(interp, objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: indices, values, size");
    }
    
    return args;
}

int Torch_SparseTensorCreate_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 4) {
        Tcl_SetResult(interp, (char*)"Usage: torch::sparse_tensor_create indices values size\n"
                      "   or: torch::sparse_tensor_create -indices TENSOR -values TENSOR -size LIST", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        SparseTensorCreateArgs args = ParseSparseTensorCreateArgs(interp, objc, objv);
        
        auto indices_it = tensor_storage.find(args.indices);
        auto values_it = tensor_storage.find(args.values);
        
        if (indices_it == tensor_storage.end() || values_it == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor handle"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        torch::Tensor indices_tensor = indices_it->second;
        torch::Tensor values_tensor  = values_it->second;

        // If indices are given in (nnz, ndim) form, transpose to (ndim, nnz)
        if (indices_tensor.dim() == 2 &&
            indices_tensor.size(0) != static_cast<long>(args.size.size()) &&
            indices_tensor.size(1) == static_cast<long>(args.size.size())) {
            indices_tensor = indices_tensor.t().contiguous();
        }
        // Validate final layout
        if (indices_tensor.dim() != 2 || indices_tensor.size(0) != static_cast<long>(args.size.size())) {
            Tcl_SetResult(interp, const_cast<char*>("Indices tensor has incorrect shape"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto result = torch::sparse_coo_tensor(indices_tensor, values_tensor, args.size);

        std::string result_name = GetNextHandle("tensor");
        tensor_storage[result_name] = result;

        Tcl_SetResult(interp, const_cast<char*>(result_name.c_str()), TCL_VOLATILE);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

int Torch_SparseTensorDense_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc != 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "sparse_tensor");
        return TCL_ERROR;
    }

    try {
        std::string tensor_name = Tcl_GetString(objv[1]);
        
        auto tensor_it = tensor_storage.find(tensor_name);
        if (tensor_it == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Tensor not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        auto result = tensor_it->second.to_dense();

        std::string result_name = GetNextHandle("tensor");
        tensor_storage[result_name] = result;

        Tcl_SetResult(interp, const_cast<char*>(result_name.c_str()), TCL_VOLATILE);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// Advanced Model Management
// ============================================================================

// Parameter structure for model_summary command
struct ModelSummaryArgs {
    std::string model;
    
    bool IsValid() const {
        return !model.empty();
    }
};

// Parse dual syntax for model_summary
ModelSummaryArgs ParseModelSummaryArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ModelSummaryArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::model_summary model");
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

int Torch_ModelSummary_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        ModelSummaryArgs args = ParseModelSummaryArgs(interp, objc, objv);
        
        auto model_it = module_storage.find(args.model);
        if (model_it == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Model not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        // Count parameters
        int64_t total_params = 0;
        int64_t trainable_params = 0;
        
        for (const auto& param : model_it->second->parameters()) {
            total_params += param.numel();
            if (param.requires_grad()) {
                trainable_params += param.numel();
            }
        }

        std::string summary = "Model Summary:\n";
        summary += "Total parameters: " + std::to_string(total_params) + "\n";
        summary += "Trainable parameters: " + std::to_string(trainable_params) + "\n";
        summary += "Non-trainable parameters: " + std::to_string(total_params - trainable_params);

        Tcl_SetResult(interp, const_cast<char*>(summary.c_str()), TCL_VOLATILE);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// Count Parameters Command - Dual Syntax Support
// ============================================================================

struct CountParametersArgs {
    std::string model;
    
    bool IsValid() const {
        return !model.empty();
    }
};

CountParametersArgs ParseCountParametersArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    CountParametersArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::count_parameters model
    if (objc != 2) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::count_parameters model");
        }
        args.model = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax: torch::count_parameters -model model_name
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-model") {
                args.model = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
    }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: -model model_name");
    }
    
    return args;
}

int Torch_CountParameters_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        CountParametersArgs args = ParseCountParametersArgs(interp, objc, objv);
        
        auto model_it = module_storage.find(args.model);
        if (model_it == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Model not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        int64_t total_params = 0;
        for (const auto& param : model_it->second->parameters()) {
            total_params += param.numel();
        }

        Tcl_SetObjResult(interp, Tcl_NewWideIntObj(total_params));
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// Distributed Training Utilities
// ============================================================================

// Note: This function is deprecated - use Torch_RealAllReduce_Cmd instead
int Torch_AllReduce_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    // Redirect to the real implementation
    return Torch_RealAllReduce_Cmd(clientData, interp, objc, objv);
}

// Note: This function is deprecated - use Torch_RealBroadcast_Cmd instead
int Torch_Broadcast_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    // Redirect to the real implementation
    return Torch_RealBroadcast_Cmd(clientData, interp, objc, objv);
}

// ============================================================================
// Additional Advanced Tensor Operations
// ============================================================================

// Parameter structure for tensor_norm command
struct TensorNormArgs {
    std::string tensor;
    double p = 2.0;  // Default L2 norm
    c10::optional<int64_t> dim = c10::nullopt;
    
    bool IsValid() const {
        return !tensor.empty();
    }
};

// Parse dual syntax for tensor_norm
TensorNormArgs ParseTensorNormArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorNormArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::tensor_norm tensor ?p? ?dim? | torch::tensor_norm -tensor tensor ?-p value? ?-dim value?");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc > 4) {
            throw std::runtime_error("Usage: torch::tensor_norm tensor ?p? ?dim?");
        }
        args.tensor = Tcl_GetString(objv[1]);
        
        if (objc >= 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.p) != TCL_OK) {
                throw std::runtime_error("Invalid p value");
            }
        }
        
        if (objc >= 4) {
            int dim_val;
            if (Tcl_GetIntFromObj(interp, objv[3], &dim_val) != TCL_OK) {
                throw std::runtime_error("Invalid dim value");
            }
            args.dim = dim_val;
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
            } else if (param == "-p") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.p) != TCL_OK) {
                    throw std::runtime_error("Invalid p value");
                }
            } else if (param == "-dim") {
                int dim_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &dim_val) != TCL_OK) {
                    throw std::runtime_error("Invalid dim value");
                }
                args.dim = dim_val;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -tensor, -p, -dim");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: tensor required");
    }
    
    return args;
}

int Torch_TensorNorm_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        // Parse arguments using dual syntax parser
        TensorNormArgs args = ParseTensorNormArgs(interp, objc, objv);

        auto tensor_it = tensor_storage.find(args.tensor);
        if (tensor_it == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Tensor not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        torch::Tensor result;
        if (args.dim.has_value()) {
            result = torch::norm(tensor_it->second, args.p, {args.dim.value()});
        } else {
            result = torch::norm(tensor_it->second, args.p);
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

// Parameter structure for tensor_normalize command
struct TensorNormalizeArgs {
    std::string tensor;
    double p = 2.0;  // Default L2 norm
    c10::optional<int64_t> dim = c10::nullopt;  // Default to all dimensions
    
    bool IsValid() const {
        return !tensor.empty();
    }
};

// Parse dual syntax for tensor_normalize
TensorNormalizeArgs ParseTensorNormalizeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorNormalizeArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::tensor_normalize tensor ?p? ?dim? | torch::tensor_normalize -tensor tensor ?-p value? ?-dim value?");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc > 4) {
            throw std::runtime_error("Usage: torch::tensor_normalize tensor ?p? ?dim?");
        }
        args.tensor = Tcl_GetString(objv[1]);
        
        if (objc >= 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.p) != TCL_OK) {
                throw std::runtime_error("Invalid p value");
            }
        }
        
        if (objc >= 4) {
            int dim_val;
            if (Tcl_GetIntFromObj(interp, objv[3], &dim_val) != TCL_OK) {
                throw std::runtime_error("Invalid dim value");
            }
            args.dim = dim_val;
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
            } else if (param == "-p") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.p) != TCL_OK) {
                    throw std::runtime_error("Invalid p value");
                }
            } else if (param == "-dim") {
                int dim_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &dim_val) != TCL_OK) {
                    throw std::runtime_error("Invalid dim value");
                }
                args.dim = dim_val;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -tensor, -p, -dim");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: tensor required");
    }
    
    return args;
}

int Torch_TensorNormalize_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        // Parse arguments using dual syntax parser
        TensorNormalizeArgs args = ParseTensorNormalizeArgs(interp, objc, objv);

        auto tensor_it = tensor_storage.find(args.tensor);
        if (tensor_it == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Tensor not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        torch::Tensor result;
        
        if (args.dim.has_value()) {
            // Normalize along specific dimension
            auto norm_tensor = torch::norm(tensor_it->second, args.p, {args.dim.value()}, true);
            // Add small epsilon to avoid division by zero
            norm_tensor = norm_tensor + 1e-8;
            result = tensor_it->second / norm_tensor;
        } else {
            // Normalize the entire tensor (flatten first)
            auto flat_tensor = tensor_it->second.flatten();
            auto norm_val = torch::norm(flat_tensor, args.p);
            result = tensor_it->second / (norm_val + 1e-8);
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

// Parameter structure for tensor_unique
struct TensorUniqueArgs {
    std::string tensor;
    bool sorted = true;
    bool return_inverse = false;
    
    bool IsValid() const {
        return !tensor.empty();
    }
};

// Dual syntax parser for tensor_unique
TensorUniqueArgs ParseTensorUniqueArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorUniqueArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 4) {
            throw std::runtime_error("Usage: torch::tensor_unique tensor ?sorted? ?return_inverse?");
        }
        args.tensor = Tcl_GetString(objv[1]);
        
        if (objc >= 3) {
            int sorted_val;
            if (Tcl_GetIntFromObj(interp, objv[2], &sorted_val) != TCL_OK) {
                throw std::runtime_error("Invalid sorted parameter");
            }
            args.sorted = (sorted_val != 0);
        }
        
        if (objc >= 4) {
            int return_inverse_val;
            if (Tcl_GetIntFromObj(interp, objv[3], &return_inverse_val) != TCL_OK) {
                throw std::runtime_error("Invalid return_inverse parameter");
            }
            args.return_inverse = (return_inverse_val != 0);
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
            } else if (param == "-sorted") {
                int sorted_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &sorted_val) != TCL_OK) {
                    throw std::runtime_error("Invalid sorted parameter");
                }
                args.sorted = (sorted_val != 0);
            } else if (param == "-returnInverse") {
                int return_inverse_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &return_inverse_val) != TCL_OK) {
                    throw std::runtime_error("Invalid returnInverse parameter");
                }
                args.return_inverse = (return_inverse_val != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: tensor");
    }
    
    return args;
}

int Torch_TensorUnique_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "tensor ?sorted? ?return_inverse? OR -tensor tensor -sorted bool -returnInverse bool");
        return TCL_ERROR;
    }

    try {
        TensorUniqueArgs args = ParseTensorUniqueArgs(interp, objc, objv);

        auto tensor_it = tensor_storage.find(args.tensor);
        if (tensor_it == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Tensor not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        if (args.return_inverse) {
            // Use the actual _unique function with return_inverse
            auto [unique_result, inverse_result] = at::_unique(tensor_it->second, args.sorted, args.return_inverse);
            
            std::string unique_name = GetNextHandle("tensor");
            std::string inverse_name = GetNextHandle("tensor");
            tensor_storage[unique_name] = unique_result;
            tensor_storage[inverse_name] = inverse_result;

            std::string result = "{unique " + unique_name + " inverse " + inverse_name + "}";
            Tcl_SetResult(interp, const_cast<char*>(result.c_str()), TCL_VOLATILE);
        } else {
            // Use the actual _unique function without return_inverse
            auto [unique_result, _] = at::_unique(tensor_it->second, args.sorted, false);
            
            std::string result_name = GetNextHandle("tensor");
            tensor_storage[result_name] = unique_result;
            Tcl_SetResult(interp, const_cast<char*>(result_name.c_str()), TCL_VOLATILE);
        }

        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// torch::block_diag - Create block diagonal matrix
int TensorBlockDiag_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "tensor1 ?tensor2 ...?");
        return TCL_ERROR;
    }

    try {
        std::vector<torch::Tensor> tensors;
        
        for (int i = 1; i < objc; i++) {
            auto tensor = GetTensorFromObj(interp, objv[i]);
            tensors.push_back(tensor);
        }
        
        auto result = torch::block_diag(tensors);
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in block_diag: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::broadcast_shapes - Get broadcast shape
int TensorBroadcastShapes_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 3) {
        Tcl_WrongNumArgs(interp, 1, objv, "shape1 shape2 ?shape3 ...?");
        return TCL_ERROR;
    }

    try {
        std::vector<std::vector<int64_t>> shapes;
        
        for (int i = 1; i < objc; i++) {
            auto shape = GetIntVectorFromObj(interp, objv[i]);
            std::vector<int64_t> shape64;
            for (int dim : shape) {
                shape64.push_back(dim);
            }
            shapes.push_back(shape64);
        }
        
        // Compute broadcast shape manually (PyTorch C++ doesn't expose this directly)
        std::vector<int64_t> result_shape;
        if (!shapes.empty()) {
            size_t max_ndim = 0;
            for (const auto& shape : shapes) {
                max_ndim = std::max(max_ndim, shape.size());
            }
            
            result_shape.resize(max_ndim, 1);
            
            for (const auto& shape : shapes) {
                size_t offset = max_ndim - shape.size();
                for (size_t i = 0; i < shape.size(); i++) {
                    size_t result_idx = offset + i;
                    if (result_shape[result_idx] == 1) {
                        result_shape[result_idx] = shape[i];
                    } else if (shape[i] != 1 && shape[i] != result_shape[result_idx]) {
                        throw std::runtime_error("Shapes cannot be broadcast");
                    }
                }
            }
        }
        
        // Return shape as list
        Tcl_Obj* resultList = Tcl_NewListObj(0, nullptr);
        for (int64_t dim : result_shape) {
            Tcl_ListObjAppendElement(interp, resultList, Tcl_NewWideIntObj(dim));
        }
        Tcl_SetObjResult(interp, resultList);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in broadcast_shapes: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for squeeze_multiple
struct SqueezeMultipleArgs {
    std::string tensor;
    std::vector<long int> dims;  // Empty vector means squeeze all dimensions
    bool has_dims = false;  // Track if dims were provided
    
    bool IsValid() const {
        return !tensor.empty();
    }
};

// Dual syntax parser for squeeze_multiple
SqueezeMultipleArgs ParseSqueezeMultipleArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SqueezeMultipleArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("Usage: torch::squeeze_multiple tensor ?dims?");
        }
        args.tensor = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            args.dims = GetIntVectorFromObj(interp, objv[2]);
            args.has_dims = true;
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
            } else if (param == "-dims") {
                args.dims = GetIntVectorFromObj(interp, objv[i + 1]);
                args.has_dims = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: tensor");
    }
    
    return args;
}

// torch::squeeze_multiple - Squeeze multiple dimensions
int TensorSqueezeMultiple_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 2) {
        Tcl_SetResult(interp, (char*)"Usage: torch::squeeze_multiple tensor ?dims?\n"
                      "   or: torch::squeeze_multiple -tensor TENSOR [-dims DIMS]", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        SqueezeMultipleArgs args = ParseSqueezeMultipleArgs(interp, objc, objv);
        // Convert tensor name back to Tcl_Obj for GetTensorFromObj
        if (tensor_storage.find(args.tensor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        auto tensor = tensor_storage[args.tensor];
        
        torch::Tensor result;
        if (!args.has_dims) {
            // Squeeze all dimensions
            result = torch::squeeze(tensor);
        } else {
            // Squeeze specific dimensions
            result = tensor;
            for (long int dim : args.dims) {
                result = torch::squeeze(result, dim);
            }
        }
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in squeeze_multiple: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for unsqueeze_multiple
struct UnsqueezeMultipleArgs {
    std::string tensor;
    std::vector<long int> dims;  // Required list of dimensions to unsqueeze
    
    bool IsValid() const {
        return !tensor.empty() && !dims.empty();
    }
};

// Dual syntax parser for unsqueeze_multiple
UnsqueezeMultipleArgs ParseUnsqueezeMultipleArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    UnsqueezeMultipleArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::unsqueeze_multiple tensor dims");
        }
        args.tensor = Tcl_GetString(objv[1]);
        args.dims = GetIntVectorFromObj(interp, objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-tensor") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dims") {
                args.dims = GetIntVectorFromObj(interp, objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: tensor, dims");
    }
    
    return args;
}

// torch::unsqueeze_multiple - Unsqueeze multiple dimensions
int TensorUnsqueezeMultiple_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        UnsqueezeMultipleArgs args = ParseUnsqueezeMultipleArgs(interp, objc, objv);
        
        // Convert tensor name to tensor object
        if (tensor_storage.find(args.tensor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        auto tensor = tensor_storage[args.tensor];
        
        torch::Tensor result = tensor;
        // Sort dims in descending order to avoid shifting indices
        auto dims = args.dims;
        std::sort(dims.rbegin(), dims.rend());
        
        for (long int dim : dims) {
            result = torch::unsqueeze(result, dim);
        }
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_split command
struct TensorSplitArgs {
    std::string input;
    std::string sections_or_indices;
    int dim = 0;
    bool has_dim = false;
    
    bool IsValid() const {
        return !input.empty() && !sections_or_indices.empty();
    }
};

// Parse dual syntax for tensor_split
TensorSplitArgs ParseTensorSplitArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorSplitArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor sections_or_indices ?dim?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.sections_or_indices = Tcl_GetString(objv[2]);
        
        if (objc == 4) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.dim) != TCL_OK) {
                throw std::runtime_error("Invalid dimension value");
            }
            args.has_dim = true;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = value;
            } else if (param == "-sections" || param == "-indices") {
                args.sections_or_indices = value;
            } else if (param == "-dim" || param == "-dimension") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dimension value");
                }
                args.has_dim = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and sections/indices are required");
    }
    
    return args;
}

// torch::tensor_split - Split tensor into sections
int TensorTensorSplit_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        TensorSplitArgs args = ParseTensorSplitArgs(interp, objc, objv);
        
        // Get tensor from storage
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        auto& tensor = tensor_storage[args.input];
        
        // Create Tcl_Obj for the sections_or_indices argument
        Tcl_Obj* sectionsObj = Tcl_NewStringObj(args.sections_or_indices.c_str(), -1);
        
        // Check if second argument is a number or list
        int listLen;
        if (Tcl_ListObjLength(interp, sectionsObj, &listLen) == TCL_OK && listLen > 1) {
            // It's a list of indices
            auto indices = GetIntVectorFromObj(interp, sectionsObj);
            std::vector<int64_t> indices64;
            for (int idx : indices) {
                indices64.push_back(idx);
            }
            auto result = torch::tensor_split(tensor, indices64, args.dim);
            
            // Return list of tensors
            Tcl_Obj* resultList = Tcl_NewListObj(0, nullptr);
            for (const auto& t : result) {
                std::string handle = GetNextHandle("tensor");
                tensor_storage[handle] = t;
                Tcl_ListObjAppendElement(interp, resultList, Tcl_NewStringObj(handle.c_str(), -1));
            }
            Tcl_SetObjResult(interp, resultList);
            return TCL_OK;
        } else {
            // It's a number of sections
            int sections = GetIntFromObj(interp, sectionsObj);
            auto result = torch::tensor_split(tensor, sections, args.dim);
            
            // Return list of tensors
            Tcl_Obj* resultList = Tcl_NewListObj(0, nullptr);
            for (const auto& t : result) {
                std::string handle = GetNextHandle("tensor");
                tensor_storage[handle] = t;
                Tcl_ListObjAppendElement(interp, resultList, Tcl_NewStringObj(handle.c_str(), -1));
            }
            Tcl_SetObjResult(interp, resultList);
            return TCL_OK;
        }
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in tensor_split: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for hsplit command
struct HSplitArgs {
    std::string tensor;
    std::string sections_or_indices;
    
    bool IsValid() const {
        return !tensor.empty() && !sections_or_indices.empty();
    }
};

// Parse dual syntax for hsplit
HSplitArgs ParseHSplitArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    HSplitArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::hsplit tensor sections_or_indices | torch::hsplit -tensor tensor -sections sections_or_indices");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::hsplit tensor sections_or_indices");
        }
        
        args.tensor = Tcl_GetString(objv[1]);
        args.sections_or_indices = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-tensor" || param == "-input") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else if (param == "-sections" || param == "-indices") {
                args.sections_or_indices = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters: -tensor/-input, -sections/-indices");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing. Both -tensor and -sections are required");
    }
    
    return args;
}

// torch::hsplit - Horizontal split
int TensorHSplit_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        HSplitArgs args = ParseHSplitArgs(interp, objc, objv);
        Tcl_Obj* tensorObj = Tcl_NewStringObj(args.tensor.c_str(), -1);
        auto tensor = GetTensorFromObj(interp, tensorObj);
        
        // Create Tcl_Obj for the sections_or_indices argument
        Tcl_Obj* sectionsObj = Tcl_NewStringObj(args.sections_or_indices.c_str(), -1);
        
        // Check if second argument is a number or list
        int listLen;
        if (Tcl_ListObjLength(interp, sectionsObj, &listLen) == TCL_OK && listLen > 1) {
            // It's a list of indices
            auto indices = GetIntVectorFromObj(interp, sectionsObj);
            std::vector<int64_t> indices64;
            for (int idx : indices) {
                indices64.push_back(idx);
            }
            auto result = torch::hsplit(tensor, indices64);
            
            // Return list of tensors
            Tcl_Obj* resultList = Tcl_NewListObj(0, nullptr);
            for (const auto& t : result) {
                std::string handle = GetNextHandle("tensor");
                tensor_storage[handle] = t;
                Tcl_ListObjAppendElement(interp, resultList, Tcl_NewStringObj(handle.c_str(), -1));
            }
            Tcl_SetObjResult(interp, resultList);
            return TCL_OK;
        } else {
            // It's a number of sections
            int sections = GetIntFromObj(interp, sectionsObj);
            auto result = torch::hsplit(tensor, sections);
            
            // Return list of tensors
            Tcl_Obj* resultList = Tcl_NewListObj(0, nullptr);
            for (const auto& t : result) {
                std::string handle = GetNextHandle("tensor");
                tensor_storage[handle] = t;
                Tcl_ListObjAppendElement(interp, resultList, Tcl_NewStringObj(handle.c_str(), -1));
            }
            Tcl_SetObjResult(interp, resultList);
            return TCL_OK;
        }
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in hsplit: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for vsplit command
struct VSplitArgs {
    std::string tensor;
    std::string sections_or_indices;
    
    bool IsValid() const {
        return !tensor.empty() && !sections_or_indices.empty();
    }
};

// Parse dual syntax for vsplit
VSplitArgs ParseVSplitArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    VSplitArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::vsplit tensor sections_or_indices | torch::vsplit -tensor tensor -sections sections_or_indices");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::vsplit tensor sections_or_indices");
        }
        
        args.tensor = Tcl_GetString(objv[1]);
        args.sections_or_indices = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-tensor" || param == "-input") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else if (param == "-sections" || param == "-indices") {
                args.sections_or_indices = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters: -tensor/-input, -sections/-indices");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing. Both -tensor and -sections are required");
    }
    
    return args;
}

// torch::vsplit - Vertical split
int TensorVSplit_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        VSplitArgs args = ParseVSplitArgs(interp, objc, objv);
        Tcl_Obj* tensorObj = Tcl_NewStringObj(args.tensor.c_str(), -1);
        auto tensor = GetTensorFromObj(interp, tensorObj);
        
        // Create Tcl_Obj for the sections_or_indices argument
        Tcl_Obj* sectionsObj = Tcl_NewStringObj(args.sections_or_indices.c_str(), -1);
        
        // Check if second argument is a number or list
        int listLen;
        if (Tcl_ListObjLength(interp, sectionsObj, &listLen) == TCL_OK && listLen > 1) {
            // It's a list of indices
            auto indices = GetIntVectorFromObj(interp, sectionsObj);
            std::vector<int64_t> indices64;
            for (int idx : indices) {
                indices64.push_back(idx);
            }
            auto result = torch::vsplit(tensor, indices64);
            
            // Return list of tensors
            Tcl_Obj* resultList = Tcl_NewListObj(0, nullptr);
            for (const auto& t : result) {
                std::string handle = GetNextHandle("tensor");
                tensor_storage[handle] = t;
                Tcl_ListObjAppendElement(interp, resultList, Tcl_NewStringObj(handle.c_str(), -1));
            }
            Tcl_SetObjResult(interp, resultList);
            return TCL_OK;
        } else {
            // It's a number of sections
            int sections = GetIntFromObj(interp, sectionsObj);
            auto result = torch::vsplit(tensor, sections);
            
            // Return list of tensors
            Tcl_Obj* resultList = Tcl_NewListObj(0, nullptr);
            for (const auto& t : result) {
                std::string handle = GetNextHandle("tensor");
                tensor_storage[handle] = t;
                Tcl_ListObjAppendElement(interp, resultList, Tcl_NewStringObj(handle.c_str(), -1));
            }
            Tcl_SetObjResult(interp, resultList);
            return TCL_OK;
        }
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in vsplit: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for dsplit command
struct DSplitArgs {
    std::string tensor;
    std::string sections_or_indices;
    
    bool IsValid() const {
        return !tensor.empty() && !sections_or_indices.empty();
    }
};

// Parse dual syntax for dsplit
DSplitArgs ParseDSplitArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DSplitArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::dsplit tensor sections_or_indices | torch::dsplit -tensor tensor -sections sections_or_indices");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::dsplit tensor sections_or_indices");
        }
        
        args.tensor = Tcl_GetString(objv[1]);
        args.sections_or_indices = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-tensor" || param == "-input") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else if (param == "-sections" || param == "-indices") {
                args.sections_or_indices = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters: -tensor/-input, -sections/-indices");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing. Both -tensor and -sections are required");
    }
    
    return args;
}

// torch::dsplit - Depth split
int TensorDSplit_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        DSplitArgs args = ParseDSplitArgs(interp, objc, objv);
        Tcl_Obj* tensorObj = Tcl_NewStringObj(args.tensor.c_str(), -1);
        auto tensor = GetTensorFromObj(interp, tensorObj);
        
        // Create Tcl_Obj for the sections_or_indices argument
        Tcl_Obj* sectionsObj = Tcl_NewStringObj(args.sections_or_indices.c_str(), -1);
        
        // Check if second argument is a number or list
        int listLen;
        if (Tcl_ListObjLength(interp, sectionsObj, &listLen) == TCL_OK && listLen > 1) {
            // It's a list of indices
            auto indices = GetIntVectorFromObj(interp, sectionsObj);
            std::vector<int64_t> indices64;
            for (int idx : indices) {
                indices64.push_back(idx);
            }
            auto result = torch::dsplit(tensor, indices64);
            
            // Return list of tensors
            Tcl_Obj* resultList = Tcl_NewListObj(0, nullptr);
            for (const auto& t : result) {
                std::string handle = GetNextHandle("tensor");
                tensor_storage[handle] = t;
                Tcl_ListObjAppendElement(interp, resultList, Tcl_NewStringObj(handle.c_str(), -1));
            }
            Tcl_SetObjResult(interp, resultList);
            return TCL_OK;
        } else {
            // It's a number of sections
            int sections = GetIntFromObj(interp, sectionsObj);
            auto result = torch::dsplit(tensor, sections);
            
            // Return list of tensors
            Tcl_Obj* resultList = Tcl_NewListObj(0, nullptr);
            for (const auto& t : result) {
                std::string handle = GetNextHandle("tensor");
                tensor_storage[handle] = t;
                Tcl_ListObjAppendElement(interp, resultList, Tcl_NewStringObj(handle.c_str(), -1));
            }
            Tcl_SetObjResult(interp, resultList);
            return TCL_OK;
        }
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in dsplit: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::column_stack - Stack tensors column-wise
int TensorColumnStack_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "tensor_list");
        return TCL_ERROR;
    }

    try {
        std::vector<torch::Tensor> tensors;
        
        if (objc == 2) {
            // Single argument - should be a list of tensors
            int listLen;
            Tcl_Obj** listElements;
            if (Tcl_ListObjGetElements(interp, objv[1], &listLen, &listElements) != TCL_OK) {
                return TCL_ERROR;
            }
            
            for (int i = 0; i < listLen; i++) {
                auto tensor = GetTensorFromObj(interp, listElements[i]);
                tensors.push_back(tensor);
            }
        } else {
            // Multiple arguments
            for (int i = 1; i < objc; i++) {
                auto tensor = GetTensorFromObj(interp, objv[i]);
                tensors.push_back(tensor);
            }
        }
        
        auto result = torch::column_stack(tensors);
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in column_stack: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for row_stack command
struct RowStackArgs {
    std::vector<std::string> tensors;
    
    bool IsValid() const {
        return !tensors.empty();
    }
};

// Parse dual syntax for row_stack
RowStackArgs ParseRowStackArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    RowStackArgs args;
    
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "tensor_list");
        throw std::runtime_error("");  // Empty error message since we use Tcl_WrongNumArgs
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc == 2) {
            // Single argument - should be a list of tensors
            int listLen;
            Tcl_Obj** listElements;
            if (Tcl_ListObjGetElements(interp, objv[1], &listLen, &listElements) == TCL_OK) {
                for (int i = 0; i < listLen; i++) {
                    args.tensors.push_back(Tcl_GetString(listElements[i]));
                }
            }
        } else {
            // Multiple arguments
            for (int i = 1; i < objc; i++) {
                args.tensors.push_back(Tcl_GetString(objv[i]));
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter: " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string key = Tcl_GetString(objv[i]);
            
            if (key == "-tensors" || key == "-inputs") {
                // Single argument - should be a list of tensors
                int listLen;
                Tcl_Obj** listElements;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listElements) == TCL_OK) {
                    for (int j = 0; j < listLen; j++) {
                        args.tensors.push_back(Tcl_GetString(listElements[j]));
                    }
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + key);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Missing required parameter: tensors");
    }
    
    return args;
}

// torch::row_stack - Stack tensors row-wise (alias for vstack)
int TensorRowStack_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        RowStackArgs args = ParseRowStackArgs(interp, objc, objv);
        
        std::vector<torch::Tensor> tensors;
        for (const auto& tensor_str : args.tensors) {
            Tcl_Obj* tensorObj = Tcl_NewStringObj(tensor_str.c_str(), -1);
            try {
                auto tensor = GetTensorFromObj(interp, tensorObj);
                tensors.push_back(tensor);
            } catch (const std::exception&) {
                throw std::runtime_error("Invalid tensor name");
            }
        }
        
        auto result = torch::row_stack(tensors);
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        if (std::string(e.what()).empty()) {
            return TCL_ERROR;  // Error message already set by Tcl_WrongNumArgs
        }
        Tcl_SetResult(interp, const_cast<char*>(("Error in row_stack: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct DStackArgs {
    std::vector<std::string> tensors;
    
    bool IsValid() const {
        return !tensors.empty();
    }
};

DStackArgs ParseDStackArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DStackArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc == 2) {
            // Single argument - should be a list of tensors
            int listLen;
            Tcl_Obj** listElements;
            if (Tcl_ListObjGetElements(interp, objv[1], &listLen, &listElements) == TCL_OK) {
                for (int i = 0; i < listLen; i++) {
                    args.tensors.push_back(Tcl_GetString(listElements[i]));
                }
            }
        } else {
            // Multiple arguments
            for (int i = 1; i < objc; i++) {
                args.tensors.push_back(Tcl_GetString(objv[i]));
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter: " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string key = Tcl_GetString(objv[i]);
            
            if (key == "-tensors" || key == "-inputs") {
                // Single argument - should be a list of tensors
                int listLen;
                Tcl_Obj** listElements;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listElements) == TCL_OK) {
                    for (int j = 0; j < listLen; j++) {
                        args.tensors.push_back(Tcl_GetString(listElements[j]));
                    }
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + key);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Missing required parameter: tensors");
    }
    
    return args;
}

// torch::dstack - Stack tensors depth-wise
int TensorDStack_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "tensor_list or -tensors tensor_list");
        return TCL_ERROR;
    }

    try {
        DStackArgs args = ParseDStackArgs(interp, objc, objv);
        
        std::vector<torch::Tensor> tensors;
        for (const auto& tensor_str : args.tensors) {
            Tcl_Obj* tensorObj = Tcl_NewStringObj(tensor_str.c_str(), -1);
            auto tensor = GetTensorFromObj(interp, tensorObj);
            tensors.push_back(tensor);
        }
        
        auto result = torch::dstack(tensors);
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in dstack: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for hstack command
struct HStackArgs {
    std::vector<std::string> tensors;
    
    bool IsValid() const {
        return !tensors.empty();
    }
};

// Parse dual syntax for hstack
HStackArgs ParseHStackArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    HStackArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc == 2) {
            // Single argument - should be a list of tensors
            int listLen;
            Tcl_Obj** listElements;
            if (Tcl_ListObjGetElements(interp, objv[1], &listLen, &listElements) == TCL_OK) {
                for (int i = 0; i < listLen; i++) {
                    args.tensors.push_back(Tcl_GetString(listElements[i]));
                }
            }
        } else {
            // Multiple arguments
            for (int i = 1; i < objc; i++) {
                args.tensors.push_back(Tcl_GetString(objv[i]));
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter: " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string key = Tcl_GetString(objv[i]);
            
            if (key == "-tensors" || key == "-inputs") {
                // Single argument - should be a list of tensors
                int listLen;
                Tcl_Obj** listElements;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listElements) == TCL_OK) {
                    for (int j = 0; j < listLen; j++) {
                        args.tensors.push_back(Tcl_GetString(listElements[j]));
                    }
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + key + ". Valid parameters: -tensors/-inputs");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Missing required parameter: tensors");
    }
    
    return args;
}

// torch::hstack - Stack tensors horizontally
int TensorHStack_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "tensor_list or -tensors tensor_list");
        return TCL_ERROR;
    }

    try {
        HStackArgs args = ParseHStackArgs(interp, objc, objv);
        
        std::vector<torch::Tensor> tensors;
        for (const auto& tensor_str : args.tensors) {
            Tcl_Obj* tensorObj = Tcl_NewStringObj(tensor_str.c_str(), -1);
            auto tensor = GetTensorFromObj(interp, tensorObj);
            tensors.push_back(tensor);
        }
        
        auto result = torch::hstack(tensors);
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in hstack: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for vstack command
struct VStackArgs {
    std::vector<std::string> tensors;
    
    bool IsValid() const {
        return !tensors.empty();
    }
};

// Parse dual syntax for vstack
VStackArgs ParseVStackArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    VStackArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc == 2) {
            // Single argument - should be a list of tensors
            int listLen;
            Tcl_Obj** listElements;
            if (Tcl_ListObjGetElements(interp, objv[1], &listLen, &listElements) == TCL_OK) {
                for (int i = 0; i < listLen; i++) {
                    args.tensors.push_back(Tcl_GetString(listElements[i]));
                }
            }
        } else {
            // Multiple arguments
            for (int i = 1; i < objc; i++) {
                args.tensors.push_back(Tcl_GetString(objv[i]));
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter: " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string key = Tcl_GetString(objv[i]);
            
            if (key == "-tensors" || key == "-inputs") {
                // Single argument - should be a list of tensors
                int listLen;
                Tcl_Obj** listElements;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listElements) == TCL_OK) {
                    for (int j = 0; j < listLen; j++) {
                        args.tensors.push_back(Tcl_GetString(listElements[j]));
                    }
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + key + ". Valid parameters: -tensors/-inputs");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Missing required parameter: tensors");
    }
    
    return args;
}

// torch::vstack - Stack tensors vertically
int TensorVStack_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "tensor_list or -tensors tensor_list");
        return TCL_ERROR;
    }

    try {
        VStackArgs args = ParseVStackArgs(interp, objc, objv);
        
        std::vector<torch::Tensor> tensors;
        for (const auto& tensor_str : args.tensors) {
            Tcl_Obj* tensorObj = Tcl_NewStringObj(tensor_str.c_str(), -1);
            auto tensor = GetTensorFromObj(interp, tensorObj);
            tensors.push_back(tensor);
        }
        
        auto result = torch::vstack(tensors);
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in vstack: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

} // extern "C" 