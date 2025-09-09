#include "libtorchtcl.h"

// ============================================================================
// SPARSE TENSOR OPERATIONS - BATCH IMPLEMENTATION OF 13 OPERATIONS
// ============================================================================

// Parameter structure for sparse_coo_tensor
struct SparseCOOArgs {
    std::string indices;
    std::string values;
    std::vector<int64_t> size;
    std::string dtype;
    std::string device;
    bool requires_grad = false;
    
    bool IsValid() const {
        return !indices.empty() && !values.empty() && !size.empty();
    }
};

// Dual syntax parser for sparse_coo_tensor
SparseCOOArgs ParseSparseCOOArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SparseCOOArgs args;
    
    // Check for named parameter syntax
    bool use_named_params = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');
    
    if (!use_named_params) {
        // Positional syntax (backward compatibility)
        if (objc < 4 || objc > 7) {
            Tcl_WrongNumArgs(interp, 1, objv, "indices values size ?dtype? ?device? ?requires_grad?");
            throw std::runtime_error("");  // Empty message since Tcl_WrongNumArgs already set the error
        }
        args.indices = Tcl_GetString(objv[1]);
        args.values = Tcl_GetString(objv[2]);
        args.size = TclListToShape(interp, objv[3]);
        
        if (objc >= 5) {
            args.dtype = Tcl_GetString(objv[4]);
        }
        if (objc >= 6) {
            args.device = Tcl_GetString(objv[5]);
        }
        if (objc >= 7) {
            int req_grad;
            if (Tcl_GetIntFromObj(interp, objv[6], &req_grad) != TCL_OK) {
                throw std::runtime_error("Invalid requires_grad value");
            }
            args.requires_grad = (req_grad != 0);
        }
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
            } else if (param == "-dtype") {
                args.dtype = Tcl_GetString(objv[i + 1]);
            } else if (param == "-device") {
                args.device = Tcl_GetString(objv[i + 1]);
            } else if (param == "-requires_grad") {
                int req_grad;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &req_grad) != TCL_OK) {
                    throw std::runtime_error("Invalid requires_grad value");
                }
                args.requires_grad = (req_grad != 0);
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

// torch::sparse_coo_tensor - Create COO sparse tensor
int TensorSparseCOO_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        SparseCOOArgs args = ParseSparseCOOArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.indices) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid indices tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.values) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid values tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto indices = tensor_storage[args.indices];
        auto values = tensor_storage[args.values];
        
        c10::ScalarType dtype = torch::kFloat32;
        if (!args.dtype.empty()) {
            try {
                dtype = GetScalarType(args.dtype.c_str());
            } catch (const std::exception& e) {
                Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
                return TCL_ERROR;
            }
            // Convert values tensor to match target dtype if needed
            if (values.scalar_type() != dtype) {
                values = values.to(dtype);
            }
        }
        
        torch::Device device = torch::kCPU;
        if (!args.device.empty()) {
            device = GetDevice(args.device.c_str());
        }

        auto options = torch::TensorOptions().dtype(dtype).device(device).requires_grad(args.requires_grad);
        auto output = torch::sparse_coo_tensor(indices, values, args.size, options);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        if (strlen(e.what()) > 0) {
            Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        }
        return TCL_ERROR;
    }
}

// Parameter structure for sparse_csr_tensor
struct SparseCSRArgs {
    std::string crow_indices;
    std::string col_indices;
    std::string values;
    std::vector<int64_t> size;
    std::string dtype;
    std::string device;
    bool requires_grad = false;
    
    bool IsValid() const {
        return !crow_indices.empty() && !col_indices.empty() && !values.empty() && !size.empty();
    }
};

// Dual syntax parser for sparse_csr_tensor
SparseCSRArgs ParseSparseCSRArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SparseCSRArgs args;
    
    // Check for named parameter syntax
    bool use_named_params = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');
    
    if (!use_named_params) {
        // Positional syntax (backward compatibility)
        if (objc < 5 || objc > 8) {
            Tcl_WrongNumArgs(interp, 1, objv, "crow_indices col_indices values size ?dtype? ?device? ?requires_grad?");
            throw std::runtime_error("");  // Empty message since Tcl_WrongNumArgs already set the error
        }
        args.crow_indices = Tcl_GetString(objv[1]);
        args.col_indices = Tcl_GetString(objv[2]);
        args.values = Tcl_GetString(objv[3]);
        args.size = TclListToShape(interp, objv[4]);
        
        if (objc >= 6) {
            args.dtype = Tcl_GetString(objv[5]);
        }
        if (objc >= 7) {
            args.device = Tcl_GetString(objv[6]);
        }
        if (objc >= 8) {
            int req_grad;
            if (Tcl_GetIntFromObj(interp, objv[7], &req_grad) != TCL_OK) {
                throw std::runtime_error("Invalid requires_grad value");
            }
            args.requires_grad = (req_grad != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-crow_indices") {
                args.crow_indices = Tcl_GetString(objv[i + 1]);
            } else if (param == "-col_indices") {
                args.col_indices = Tcl_GetString(objv[i + 1]);
            } else if (param == "-values") {
                args.values = Tcl_GetString(objv[i + 1]);
            } else if (param == "-size") {
                args.size = TclListToShape(interp, objv[i + 1]);
            } else if (param == "-dtype") {
                args.dtype = Tcl_GetString(objv[i + 1]);
            } else if (param == "-device") {
                args.device = Tcl_GetString(objv[i + 1]);
            } else if (param == "-requires_grad") {
                int req_grad;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &req_grad) != TCL_OK) {
                    throw std::runtime_error("Invalid requires_grad value");
                }
                args.requires_grad = (req_grad != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: crow_indices, col_indices, values, size");
    }
    
    return args;
}

// torch::sparse_csr_tensor - Create CSR sparse tensor
int TensorSparseCSR_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        SparseCSRArgs args = ParseSparseCSRArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.crow_indices) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid crow_indices tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.col_indices) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid col_indices tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.values) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid values tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto crow_indices = tensor_storage[args.crow_indices];
        auto col_indices = tensor_storage[args.col_indices];
        auto values = tensor_storage[args.values];
        
        c10::ScalarType dtype = torch::kFloat32;
        if (!args.dtype.empty()) {
            try {
                dtype = GetScalarType(args.dtype.c_str());
            } catch (const std::exception& e) {
                Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
                return TCL_ERROR;
            }
            // Convert values tensor to match target dtype if needed
            if (values.scalar_type() != dtype) {
                values = values.to(dtype);
            }
        }
        
        torch::Device device = torch::kCPU;
        if (!args.device.empty()) {
            device = GetDevice(args.device.c_str());
        }

        auto options = torch::TensorOptions().dtype(dtype).device(device).requires_grad(args.requires_grad);
        auto output = torch::sparse_csr_tensor(crow_indices, col_indices, values, args.size, options);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for sparse_csc_tensor
struct SparseCSCArgs {
    std::string ccol_indices;
    std::string row_indices;
    std::string values;
    std::vector<int64_t> size;
    std::string dtype;
    std::string device;
    bool requires_grad = false;
    
    bool IsValid() const {
        return !ccol_indices.empty() && !row_indices.empty() && !values.empty() && !size.empty();
    }
};

// Dual syntax parser for sparse_csc_tensor
SparseCSCArgs ParseSparseCSCArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SparseCSCArgs args;
    
    // Check for named parameter syntax
    bool use_named_params = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');
    
    if (!use_named_params) {
        // Positional syntax (backward compatibility)
        if (objc < 5 || objc > 8) {
            Tcl_WrongNumArgs(interp, 1, objv, "ccol_indices row_indices values size ?dtype? ?device? ?requires_grad?");
            throw std::runtime_error("");  // Empty message since Tcl_WrongNumArgs already set the error
        }
        args.ccol_indices = Tcl_GetString(objv[1]);
        args.row_indices = Tcl_GetString(objv[2]);
        args.values = Tcl_GetString(objv[3]);
        args.size = TclListToShape(interp, objv[4]);
        
        if (objc >= 6) {
            args.dtype = Tcl_GetString(objv[5]);
        }
        if (objc >= 7) {
            args.device = Tcl_GetString(objv[6]);
        }
        if (objc >= 8) {
            int req_grad;
            if (Tcl_GetIntFromObj(interp, objv[7], &req_grad) != TCL_OK) {
                throw std::runtime_error("Invalid requires_grad value");
            }
            args.requires_grad = (req_grad != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-ccol_indices") {
                args.ccol_indices = Tcl_GetString(objv[i + 1]);
            } else if (param == "-row_indices") {
                args.row_indices = Tcl_GetString(objv[i + 1]);
            } else if (param == "-values") {
                args.values = Tcl_GetString(objv[i + 1]);
            } else if (param == "-size") {
                args.size = TclListToShape(interp, objv[i + 1]);
            } else if (param == "-dtype") {
                args.dtype = Tcl_GetString(objv[i + 1]);
            } else if (param == "-device") {
                args.device = Tcl_GetString(objv[i + 1]);
            } else if (param == "-requires_grad") {
                int req_grad;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &req_grad) != TCL_OK) {
                    throw std::runtime_error("Invalid requires_grad value");
                }
                args.requires_grad = (req_grad != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: ccol_indices, row_indices, values, size");
    }
    
    return args;
}

// torch::sparse_csc_tensor - Create CSC sparse tensor
int TensorSparseCSC_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        SparseCSCArgs args = ParseSparseCSCArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.ccol_indices) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid ccol_indices tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.row_indices) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid row_indices tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.values) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid values tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto ccol_indices = tensor_storage[args.ccol_indices];
        auto row_indices = tensor_storage[args.row_indices];
        auto values = tensor_storage[args.values];
        
        c10::ScalarType dtype = torch::kFloat32;
        if (!args.dtype.empty()) {
            dtype = GetScalarType(args.dtype.c_str());
            // Convert values tensor to match target dtype if needed
            if (values.scalar_type() != dtype) {
                values = values.to(dtype);
            }
        }
        
        torch::Device device = torch::kCPU;
        if (!args.device.empty()) {
            device = GetDevice(args.device.c_str());
        }

        auto options = torch::TensorOptions().dtype(dtype).device(device).requires_grad(args.requires_grad);
        auto output = torch::sparse_csc_tensor(ccol_indices, row_indices, values, args.size, options);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for sparse_tensor_dense
struct SparseTensorDenseArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Dual syntax parser for sparse_tensor_dense
SparseTensorDenseArgs ParseSparseTensorDenseArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SparseTensorDenseArgs args;
    
    // Check for named parameter syntax
    bool use_named_params = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');
    
    if (!use_named_params) {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::sparse_to_dense sparse_tensor");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input");
    }
    
    return args;
}

// torch::sparse_tensor_dense - Convert sparse tensor to dense
int TensorSparseToDense_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 2) {
        Tcl_SetResult(interp, (char*)"Usage: torch::sparse_to_dense sparse_tensor\n"
                      "   or: torch::sparse_to_dense -input TENSOR", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        SparseTensorDenseArgs args = ParseSparseTensorDenseArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid sparse tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto output = input.to_dense();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for sparse_add
struct SparseAddArgs {
    std::string tensor1;
    std::string tensor2;
    double alpha = 1.0;  // Default alpha value
    
    bool IsValid() const {
        return !tensor1.empty() && !tensor2.empty();
    }
};

// Dual syntax parser for sparse_add
SparseAddArgs ParseSparseAddArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SparseAddArgs args;
    
    // Check for named parameter syntax
    bool use_named_params = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');
    
    if (!use_named_params) {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: torch::sparse_add tensor1 tensor2 ?alpha?");
        }
        args.tensor1 = Tcl_GetString(objv[1]);
        args.tensor2 = Tcl_GetString(objv[2]);
        
        if (objc > 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[3], &args.alpha) != TCL_OK) {
                throw std::runtime_error("Invalid alpha value");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-tensor1") {
                args.tensor1 = Tcl_GetString(objv[i + 1]);
            } else if (param == "-tensor2") {
                args.tensor2 = Tcl_GetString(objv[i + 1]);
            } else if (param == "-alpha") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.alpha) != TCL_OK) {
                    throw std::runtime_error("Invalid alpha value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: tensor1, tensor2");
    }
    
    return args;
}

// torch::sparse_add - Sparse tensor addition
int TensorSparseAdd_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 3) {
        Tcl_SetResult(interp, (char*)"Usage: torch::sparse_add tensor1 tensor2 ?alpha?\n"
                      "   or: torch::sparse_add -tensor1 TENSOR -tensor2 TENSOR [-alpha DOUBLE]", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        SparseAddArgs args = ParseSparseAddArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.tensor1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor1"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.tensor2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor2"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto tensor1 = tensor_storage[args.tensor1];
        auto tensor2 = tensor_storage[args.tensor2];

        auto output = torch::add(tensor1, tensor2, args.alpha);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for sparse_mm
struct SparseMMArgs {
    std::string sparse_tensor;
    std::string dense_tensor;
    
    bool IsValid() const {
        return !sparse_tensor.empty() && !dense_tensor.empty();
    }
};

// Dual syntax parser for sparse_mm
SparseMMArgs ParseSparseMMArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SparseMMArgs args;
    
    // Check for named parameter syntax
    bool use_named_params = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');
    
    if (!use_named_params) {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "sparse_tensor dense_tensor");
            throw std::runtime_error("");  // Empty message since Tcl_WrongNumArgs already set the error
        }
        args.sparse_tensor = Tcl_GetString(objv[1]);
        args.dense_tensor = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-sparse_tensor") {
                args.sparse_tensor = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dense_tensor") {
                args.dense_tensor = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: sparse_tensor, dense_tensor");
    }
    
    return args;
}

// torch::sparse_mm - Sparse matrix multiplication
int TensorSparseMM_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 3) {
        Tcl_SetResult(interp, (char*)"Usage: torch::sparse_mm sparse_tensor dense_tensor\n"
                      "   or: torch::sparse_mm -sparse_tensor TENSOR -dense_tensor TENSOR", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        SparseMMArgs args = ParseSparseMMArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.sparse_tensor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid sparse tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.dense_tensor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid dense tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto sparse_tensor = tensor_storage[args.sparse_tensor];
        auto dense_tensor = tensor_storage[args.dense_tensor];
        
        auto output = torch::mm(sparse_tensor, dense_tensor);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct SparseSoftmaxArgs {
    std::string input;
    int dim = 0;
    bool dim_provided = false;
    
    bool IsValid() const {
        return !input.empty() && dim_provided;
    }
};

// Dual syntax parser for sparse_softmax
SparseSoftmaxArgs ParseSparseSoftmaxArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SparseSoftmaxArgs args;
    
    // Check for named parameter syntax
    bool use_named_params = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');
    
    if (!use_named_params) {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::sparse_softmax sparse_tensor dim");
        }
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
            throw std::runtime_error("Invalid dim value");
        }
        args.dim_provided = true;
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dim value");
                }
                args.dim_provided = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and dim");
    }
    
    return args;
}

// torch::sparse_softmax - Sparse softmax
int TensorSparseSoftmax_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 3) {
        Tcl_SetResult(interp, (char*)"Usage: torch::sparse_softmax sparse_tensor dim\n"
                      "   or: torch::sparse_softmax -input TENSOR -dim INT", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        SparseSoftmaxArgs args = ParseSparseSoftmaxArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid sparse tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto output = torch::softmax(input, args.dim);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct SparseLogSoftmaxArgs {
    std::string input;
    int dim = 0;
    bool dim_provided = false;
    
    bool IsValid() const {
        return !input.empty() && dim_provided;
    }
};

// Dual syntax parser for sparse_log_softmax
SparseLogSoftmaxArgs ParseSparseLogSoftmaxArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SparseLogSoftmaxArgs args;
    
    // Check for named parameter syntax
    bool use_named_params = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');
    
    if (!use_named_params) {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::sparse_log_softmax sparse_tensor dim");
        }
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
            throw std::runtime_error("Invalid dim value");
        }
        args.dim_provided = true;
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dim value");
                }
                args.dim_provided = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and dim");
    }
    
    return args;
}

// torch::sparse_log_softmax - Sparse log softmax
int TensorSparseLogSoftmax_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 3) {
        Tcl_SetResult(interp, (char*)"Usage: torch::sparse_log_softmax sparse_tensor dim\n"
                      "   or: torch::sparse_log_softmax -input TENSOR -dim INT", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        SparseLogSoftmaxArgs args = ParseSparseLogSoftmaxArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid sparse tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto output = torch::log_softmax(input, args.dim);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for sparse_mask
struct SparseMaskArgs {
    std::string tensor;
    std::string mask;
    
    bool IsValid() const {
        return !tensor.empty() && !mask.empty();
    }
};

// Dual syntax parser for sparse_mask
SparseMaskArgs ParseSparseMaskArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SparseMaskArgs args;
    
    // Check for named parameter syntax
    bool use_named_params = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');
    
    if (!use_named_params) {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor mask");
            throw std::runtime_error("");  // Empty message since Tcl_WrongNumArgs already set the error
        }
        args.tensor = Tcl_GetString(objv[1]);
        args.mask = Tcl_GetString(objv[2]);
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
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: tensor, mask");
    }
    
    return args;
}

// torch::sparse_mask - Apply mask to sparse tensor
int TensorSparseMask_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 3) {
        Tcl_SetResult(interp, (char*)"Usage: torch::sparse_mask tensor mask\n"
                      "   or: torch::sparse_mask -tensor TENSOR -mask TENSOR", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        SparseMaskArgs args = ParseSparseMaskArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.tensor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.mask) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid mask tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto tensor = tensor_storage[args.tensor];
        auto mask = tensor_storage[args.mask];
        
        auto output = tensor.sparse_mask(mask);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for sparse_transpose
struct SparseTransposeArgs {
    std::string tensor;
    int dim0;
    int dim1;
    bool dim0_provided = false;
    bool dim1_provided = false;
    
    bool IsValid() const {
        return !tensor.empty() && dim0_provided && dim1_provided;
    }
};

// Dual syntax parser for sparse_transpose
SparseTransposeArgs ParseSparseTransposeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SparseTransposeArgs args;
    
    if (objc == 4 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: sparse_tensor dim0 dim1
        args.tensor = Tcl_GetString(objv[1]);
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dim0) != TCL_OK) {
            throw std::runtime_error("Invalid dim0 value");
        }
        if (Tcl_GetIntFromObj(interp, objv[3], &args.dim1) != TCL_OK) {
            throw std::runtime_error("Invalid dim1 value");
        }
        args.dim0_provided = true;
        args.dim1_provided = true;
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            std::string arg = Tcl_GetString(objv[i]);
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter " + arg);
            }
            
            if (arg == "-tensor") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else if (arg == "-dim0") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim0) != TCL_OK) {
                    throw std::runtime_error("Invalid dim0 value");
                }
                args.dim0_provided = true;
            } else if (arg == "-dim1") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim1) != TCL_OK) {
                    throw std::runtime_error("Invalid dim1 value");
                }
                args.dim1_provided = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + arg);
            }
        }
    }
    
    if (!args.IsValid()) {
        if (!args.tensor.empty()) {
            if (!args.dim0_provided) {
                throw std::runtime_error("Missing required parameter: dim0");
            }
            if (!args.dim1_provided) {
                throw std::runtime_error("Missing required parameter: dim1");
            }
        }
        throw std::runtime_error("Missing required parameter: tensor");
    }
    
    return args;
}

// torch::sparse_transpose - Sparse tensor transpose
int TensorSparseTranspose_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 2) {
        Tcl_SetResult(interp, (char*)"Usage: torch::sparse_transpose sparse_tensor dim0 dim1\n"
                      "   or: torch::sparse_transpose -tensor TENSOR -dim0 INT -dim1 INT", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        SparseTransposeArgs args = ParseSparseTransposeArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.tensor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid sparse tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.tensor];
        
        try {
            auto output = input.transpose(args.dim0, args.dim1);
            std::string handle = GetNextHandle("tensor");
            tensor_storage[handle] = output;
            Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
            return TCL_OK;
        } catch (const c10::Error& e) {
            // Convert PyTorch error to our expected error message
            Tcl_SetResult(interp, const_cast<char*>("Invalid dimension"), TCL_VOLATILE);
            return TCL_ERROR;
        }
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct SparseCoalesceArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Dual syntax parser for sparse_coalesce
SparseCoalesceArgs ParseSparseCoalesceArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SparseCoalesceArgs args;
    
    // Check for named parameter syntax
    bool use_named_params = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');
    
    if (!use_named_params) {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::sparse_coalesce sparse_tensor");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input");
    }
    
    return args;
}

// torch::sparse_coalesce - Coalesce sparse tensor
int TensorSparseCoalesce_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 2) {
        Tcl_SetResult(interp, (char*)"Usage: torch::sparse_coalesce sparse_tensor\n"
                      "   or: torch::sparse_coalesce -input TENSOR", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        SparseCoalesceArgs args = ParseSparseCoalesceArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid sparse tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto output = input.coalesce();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for sparse_reshape
struct SparseReshapeArgs {
    std::string input;
    std::vector<int64_t> shape;
    
    bool IsValid() const {
        return !input.empty() && !shape.empty();
    }
};

// Dual syntax parser for sparse_reshape
SparseReshapeArgs ParseSparseReshapeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SparseReshapeArgs args;
    
    // Check for named parameter syntax
    bool use_named_params = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');
    
    if (!use_named_params) {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::sparse_reshape sparse_tensor shape");
        }
        args.input = Tcl_GetString(objv[1]);
        args.shape = TclListToShape(interp, objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-shape") {
                args.shape = TclListToShape(interp, objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and shape");
    }
    
    return args;
}

// torch::sparse_reshape - Reshape sparse tensor
int TensorSparseReshape_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 3) {
        Tcl_SetResult(interp, (char*)"Usage: torch::sparse_reshape sparse_tensor shape\n"
                      "   or: torch::sparse_reshape -input TENSOR -shape SHAPE", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        SparseReshapeArgs args = ParseSparseReshapeArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid sparse tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        
        // Calculate total size
        int64_t total_size = 1;
        for (const auto& dim : args.shape) {
            total_size *= dim;
        }
        
        // Check if total size matches
        int64_t current_size = 1;
        for (const auto& dim : input.sizes()) {
            current_size *= dim;
        }
        
        if (total_size != current_size) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid integer in shape list"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        try {
            // Create a new tensor with the reshaped values
            auto values = input._values();
            auto indices = input._indices();
            auto output = torch::sparse_coo_tensor(indices, values, args.shape);
            
            std::string handle = GetNextHandle("tensor");
            tensor_storage[handle] = output;
            Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
            return TCL_OK;
        } catch (const c10::Error& e) {
            // Convert PyTorch error to our expected error message
            Tcl_SetResult(interp, const_cast<char*>("Invalid integer in shape list"), TCL_VOLATILE);
            return TCL_ERROR;
        }
    } catch (const std::exception& e) {
        if (std::string(e.what()) == "Required parameters missing: input and shape") {
            Tcl_SetResult(interp, const_cast<char*>("Missing value for parameter"), TCL_VOLATILE);
        } else {
            Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        }
        return TCL_ERROR;
    }
}

struct SparseSumArgs {
    std::string input;
    int dim = -1;
    bool dim_provided = false;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Dual syntax parser for sparse_sum
SparseSumArgs ParseSparseSumArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SparseSumArgs args;
    
    // Check for named parameter syntax
    bool use_named_params = (objc >= 2 && Tcl_GetString(objv[1])[0] == '-');
    
    if (!use_named_params) {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("wrong # args: should be \"torch::sparse_sum sparse_tensor ?dim?\"");
        }
        args.input = Tcl_GetString(objv[1]);
        
        if (objc == 3) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
                throw std::runtime_error("expected integer but got \"" + std::string(Tcl_GetString(objv[2])) + "\"");
            }
            args.dim_provided = true;
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
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("expected integer but got \"" + std::string(Tcl_GetString(objv[i + 1])) + "\"");
                }
                args.dim_provided = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input");
    }
    
    return args;
}

// torch::sparse_sum - Sparse tensor sum
int TensorSparseSum_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "sparse_tensor ?dim?");
        return TCL_ERROR;
    }

    try {
        SparseSumArgs args = ParseSparseSumArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid sparse tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        torch::Tensor output;
        
        try {
            if (args.dim_provided) {
                // Convert sparse tensor to dense before summing along dimension
                auto dense_input = input.to_dense();
                output = torch::sum(dense_input, args.dim);
                // Convert back to sparse tensor
                output = output.to_sparse();
            } else {
                output = torch::sum(input);
            }
            
            std::string handle = GetNextHandle("tensor");
            tensor_storage[handle] = output;
            Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
            return TCL_OK;
        } catch (const c10::Error& e) {
            Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
            return TCL_ERROR;
        }
    } catch (const std::exception& e) {
        if (strlen(e.what()) > 0) {
            Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        }
        return TCL_ERROR;
    }
}

 