#include "libtorchtcl.h"

// Parameter structure for tensor_svd command
struct TensorSVDArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_svd
TensorSVDArgs ParseTensorSVDArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorSVDArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor
        if (objc != 2) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required input parameter missing");
    }
    
    return args;
}

int TensorSVD_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorSVDArgs args = ParseTensorSVDArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        auto svd_result = torch::svd(tensor);
        
        // Store U, S, V tensors
        std::string u_name = GetNextHandle("tensor");
        std::string s_name = GetNextHandle("tensor");
        std::string v_name = GetNextHandle("tensor");
        
        tensor_storage[u_name] = std::get<0>(svd_result);
        tensor_storage[s_name] = std::get<1>(svd_result);
        tensor_storage[v_name] = std::get<2>(svd_result);
        
        std::ostringstream result;
        result << "{U " << u_name << " S " << s_name << " V " << v_name << "}";
        Tcl_SetResult(interp, const_cast<char*>(result.str().c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_eigen command
struct TensorEigenArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_eigen
TensorEigenArgs ParseTensorEigenArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorEigenArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor
        if (objc != 2) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: -input");
    }
    
    return args;
}

int TensorEigen_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorEigenArgs args = ParseTensorEigenArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        auto eigen_result = torch::linalg::eigh(tensor, "L");
        
        // Store eigenvalues and eigenvectors
        std::string vals_name = GetNextHandle("tensor");
        std::string vecs_name = GetNextHandle("tensor");
        
        tensor_storage[vals_name] = std::get<0>(eigen_result);
        tensor_storage[vecs_name] = std::get<1>(eigen_result);
        
        // Create a proper Tcl list object
        Tcl_Obj* result_list = Tcl_NewListObj(0, nullptr);
        
        // Add "eigenvalues" and the tensor handle
        Tcl_ListObjAppendElement(interp, result_list, Tcl_NewStringObj("eigenvalues", -1));
        Tcl_ListObjAppendElement(interp, result_list, Tcl_NewStringObj(vals_name.c_str(), -1));
        
        // Add "eigenvectors" and the tensor handle
        Tcl_ListObjAppendElement(interp, result_list, Tcl_NewStringObj("eigenvectors", -1));
        Tcl_ListObjAppendElement(interp, result_list, Tcl_NewStringObj(vecs_name.c_str(), -1));
        
        Tcl_SetObjResult(interp, result_list);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_qr command
struct TensorQRArgs {
    std::string tensor;
    
    bool IsValid() const {
        return !tensor.empty();
    }
};

// Parse dual syntax for tensor_qr
TensorQRArgs ParseTensorQRArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorQRArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::tensor_qr tensor | torch::tensor_qr -tensor tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::tensor_qr tensor");
        }
        args.tensor = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-tensor") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -tensor");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: tensor required");
    }
    
    return args;
}

int TensorQR_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax parser
        TensorQRArgs args = ParseTensorQRArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.tensor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.tensor];
        auto qr_result = torch::qr(tensor);
        
        // Store Q and R tensors
        std::string q_name = GetNextHandle("tensor");
        std::string r_name = GetNextHandle("tensor");
        
        tensor_storage[q_name] = std::get<0>(qr_result);
        tensor_storage[r_name] = std::get<1>(qr_result);
        
        std::ostringstream result;
        result << "{Q " << q_name << " R " << r_name << "}";
        Tcl_SetResult(interp, const_cast<char*>(result.str().c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_cholesky command
struct TensorCholeskyArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_cholesky
TensorCholeskyArgs ParseTensorCholeskyArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorCholeskyArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::tensor_cholesky tensor | torch::tensor_cholesky -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::tensor_cholesky tensor");
        }
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor required");
    }
    
    return args;
}

// torch::tensor_cholesky(tensor) - Cholesky decomposition
int TensorCholesky_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorCholeskyArgs args = ParseTensorCholeskyArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        auto cholesky_result = torch::linalg::cholesky(tensor);
        
        // Store result tensor
        std::string result_name = GetNextHandle("tensor");
        tensor_storage[result_name] = cholesky_result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_name.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_matrix_exp command
struct TensorMatrixExpArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_matrix_exp
TensorMatrixExpArgs ParseTensorMatrixExpArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorMatrixExpArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor
        if (objc != 2) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: -input");
    }
    
    return args;
}

// torch::tensor_matrix_exp(tensor) - Matrix exponential
int TensorMatrixExp_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorMatrixExpArgs args = ParseTensorMatrixExpArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        auto matrix_exp_result = torch::linalg::matrix_exp(tensor);
        
        // Store result tensor
        std::string result_name = GetNextHandle("tensor");
        tensor_storage[result_name] = matrix_exp_result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_name.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_pinv command
struct TensorPinvArgs {
    std::string input;
    double rcond = 1e-15;  // Default rcond value
    bool has_rcond = false;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_pinv
TensorPinvArgs ParseTensorPinvArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorPinvArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor ?rcond?
        if (objc < 2 || objc > 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor ?rcond?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc == 3) {
            if (Tcl_GetDoubleFromObj(interp, objv[2], &args.rcond) != TCL_OK) {
                throw std::runtime_error("Invalid rcond value");
            }
            args.has_rcond = true;
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
            } else if (param == "-rcond") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.rcond) != TCL_OK) {
                    throw std::runtime_error("Invalid rcond value");
                }
                args.has_rcond = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required input parameter missing");
    }
    
    return args;
}

// torch::tensor_pinv(tensor, rcond?) - Pseudo-inverse (Moore-Penrose)
int TensorPinv_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorPinvArgs args = ParseTensorPinvArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        torch::Tensor pinv_result;
        if (args.has_rcond) {
            // With rcond parameter
            pinv_result = torch::linalg::pinv(tensor, args.rcond);
        } else {
            // Default rcond
            pinv_result = torch::linalg::pinv(tensor);
        }
        
        // Store result tensor
        std::string result_name = GetNextHandle("tensor");
        tensor_storage[result_name] = pinv_result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_name.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 