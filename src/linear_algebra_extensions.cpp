#include "libtorchtcl.h"

// Parameter structure for cross command
struct TensorCrossArgs {
    std::string input;
    std::string other;
    int dim = -1;  // Default value, -1 means use default
    bool has_dim = false;
    
    bool IsValid() const {
        return !input.empty() && !other.empty();
    }
};

// Parse dual syntax for cross
TensorCrossArgs ParseTensorCrossArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorCrossArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::cross input other ?dim? | torch::cross -input tensor -other tensor ?-dim int?");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: torch::cross input other ?dim?");
        }
        args.input = Tcl_GetString(objv[1]);
        args.other = Tcl_GetString(objv[2]);
        
        if (objc > 3) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.dim) != TCL_OK) {
                throw std::runtime_error("Invalid dim parameter");
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
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-other") {
                args.other = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dim parameter");
                }
                args.has_dim = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor, -other, -dim");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and other tensors required");
    }
    
    return args;
}

// torch::cross - Cross product
int TensorCross_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        TensorCrossArgs args = ParseTensorCrossArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.other) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid other tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto other = tensor_storage[args.other];

        torch::Tensor output;
        if (args.has_dim) {
            output = torch::cross(input, other, args.dim);
        } else {
            output = torch::cross(input, other);
        }

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for dot command
struct DotArgs {
    std::string input;
    std::string other;
    
    bool IsValid() const {
        return !input.empty() && !other.empty();
    }
};

// Parse dual syntax for dot
DotArgs ParseDotArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DotArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::dot input other | torch::dot -input input -other other");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::dot input other");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.other = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-other") {
                args.other = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters: -input, -other");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing. Both -input and -other are required");
    }
    
    return args;
}

// torch::dot - Dot product
int TensorDot_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        DotArgs args = ParseDotArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.other) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid other tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto other = tensor_storage[args.other];

        auto output = torch::dot(input, other);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for outer command
struct TensorOuterArgs {
    std::string input;
    std::string other;
    
    bool IsValid() const {
        return !input.empty() && !other.empty();
    }
};

// Parse dual syntax for outer
TensorOuterArgs ParseTensorOuterArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorOuterArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::outer input other | torch::outer -input tensor -other tensor");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::outer input other");
        }
        args.input = Tcl_GetString(objv[1]);
        args.other = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-other") {
                args.other = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and other tensors required");
    }
    
    return args;
}

// torch::outer - Outer product with dual syntax support
int TensorOuter_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        TensorOuterArgs args = ParseTensorOuterArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.other) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid other tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto other = tensor_storage[args.other];

        auto output = torch::outer(input, other);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for trace command
struct TraceArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for trace
static TraceArgs ParseTraceArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TraceArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: input
        if (objc != 2) {
            Tcl_WrongNumArgs(interp, 1, objv, "input");
            throw std::runtime_error("Usage: torch::trace input");
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
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor required");
    }
    
    return args;
}

// torch::trace - Matrix trace
int TensorTrace_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TraceArgs args = ParseTraceArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto output = torch::trace(input);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for diag command
struct DiagArgs {
    std::string input;
    int diagonal = 0;  // Default to main diagonal
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for diag
DiagArgs ParseDiagArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DiagArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::diag input ?diagonal? | torch::diag -input input ?-diagonal diagonal?");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("Usage: torch::diag input ?diagonal?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            int diagonal;
            if (Tcl_GetIntFromObj(interp, objv[2], &diagonal) != TCL_OK) {
                throw std::runtime_error("Invalid diagonal value");
            }
            args.diagonal = diagonal;
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
            } else if (param == "-diagonal") {
                int diagonal;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &diagonal) != TCL_OK) {
                    throw std::runtime_error("Invalid diagonal value");
                }
                args.diagonal = diagonal;
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

// torch::diag - Diagonal elements or diagonal matrix
int TensorDiag_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        DiagArgs args = ParseDiagArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];

        torch::Tensor output;
        if (args.diagonal != 0) {
            output = torch::diag(input, args.diagonal);
        } else {
            output = torch::diag(input);
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

// Parameter structure for diagflat command
struct DiagflatArgs {
    std::string input;
    int offset = 0;  // Default to main diagonal
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for diagflat
DiagflatArgs ParseDiagflatArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DiagflatArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::diagflat input ?offset? | torch::diagflat -input input ?-offset offset?");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("Usage: torch::diagflat input ?offset?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            int offset;
            if (Tcl_GetIntFromObj(interp, objv[2], &offset) != TCL_OK) {
                throw std::runtime_error("Invalid offset value");
            }
            args.offset = offset;
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
            } else if (param == "-offset") {
                int offset;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &offset) != TCL_OK) {
                    throw std::runtime_error("Invalid offset value");
                }
                args.offset = offset;
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

// torch::diagflat - Diagonal matrix from flattened tensor
int TensorDiagflat_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        DiagflatArgs args = ParseDiagflatArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];

        torch::Tensor output;
        if (args.offset != 0) {
            output = torch::diagflat(input, args.offset);
        } else {
            output = torch::diagflat(input);
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

// Parameter structure for tril
struct TrilArgs {
    std::string input;
    int diagonal = 0;
    bool has_diagonal = false;
    bool IsValid() const { return !input.empty(); }
};

TrilArgs ParseTrilArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TrilArgs args;
    // Check for named parameter syntax
    bool hasNamedParams = false;
    for (int i = 1; i < objc; i++) {
        std::string arg = Tcl_GetString(objv[i]);
        if (arg.length() > 1 && arg[0] == '-' && !isdigit(arg[1])) {
            hasNamedParams = true;
            break;
        }
    }
    if (!hasNamedParams) {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("Usage: torch::tril input ?diagonal?");
        }
        args.input = Tcl_GetString(objv[1]);
        if (objc == 3) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.diagonal) != TCL_OK) {
                throw std::runtime_error("Invalid diagonal parameter: must be an integer");
            }
            args.has_diagonal = true;
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
            } else if (param == "-diagonal") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.diagonal) != TCL_OK) {
                    throw std::runtime_error("Invalid diagonal parameter: must be an integer");
                }
                args.has_diagonal = true;
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

// torch::tril - Lower triangular matrix
int TensorTril_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TrilArgs args = ParseTrilArgs(interp, objc, objv);
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        auto input = tensor_storage[args.input];
        torch::Tensor output;
        if (args.has_diagonal) {
            output = torch::tril(input, args.diagonal);
        } else {
            output = torch::tril(input);
        }
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for triu
struct TriuArgs {
    std::string input;
    int diagonal = 0;
    bool has_diagonal = false;
    bool IsValid() const { return !input.empty(); }
};

TriuArgs ParseTriuArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TriuArgs args;
    // Check for named parameter syntax
    bool hasNamedParams = false;
    for (int i = 1; i < objc; i++) {
        std::string arg = Tcl_GetString(objv[i]);
        if (arg.length() > 1 && arg[0] == '-' && !isdigit(arg[1])) {
            hasNamedParams = true;
            break;
        }
    }
    if (!hasNamedParams) {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("Usage: torch::triu input ?diagonal?");
        }
        args.input = Tcl_GetString(objv[1]);
        if (objc == 3) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.diagonal) != TCL_OK) {
                throw std::runtime_error("Invalid diagonal parameter: must be an integer");
            }
            args.has_diagonal = true;
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
            } else if (param == "-diagonal") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.diagonal) != TCL_OK) {
                    throw std::runtime_error("Invalid diagonal parameter: must be an integer");
                }
                args.has_diagonal = true;
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

// torch::triu - Upper triangular matrix
int TensorTriu_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TriuArgs args = ParseTriuArgs(interp, objc, objv);
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        auto input = tensor_storage[args.input];
        torch::Tensor output;
        if (args.has_diagonal) {
            output = torch::triu(input, args.diagonal);
        } else {
            output = torch::triu(input);
        }
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for matrix_power command
struct MatrixPowerArgs {
    std::string input;
    int n = 2;  // Default power value
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for matrix_power
MatrixPowerArgs ParseMatrixPowerArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MatrixPowerArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::matrix_power input n | torch::matrix_power -input tensor -n integer");
    }
    
    // Check if we have named parameters (look for parameters starting with '-' that are not negative numbers)
    bool hasNamedParams = false;
    for (int i = 1; i < objc; i++) {
        std::string arg = Tcl_GetString(objv[i]);
        // Check if it starts with '-' and is not a negative number
        if (arg.length() > 1 && arg[0] == '-' && !isdigit(arg[1])) {
            hasNamedParams = true;
            break;
        }
    }
    
    if (!hasNamedParams) {
        // Pure positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::matrix_power input n");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        int n_val;
        if (Tcl_GetIntFromObj(interp, objv[2], &n_val) != TCL_OK) {
            throw std::runtime_error("Invalid n parameter: must be an integer");
        }
        args.n = n_val;
    } else {
        // We have named parameters - check for mixed or pure named syntax
        if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
            // Mixed syntax: positional input, then named parameters
            args.input = Tcl_GetString(objv[1]);
            
            // Parse remaining named parameters starting from index 2
            for (int i = 2; i < objc; i += 2) {
                if (i + 1 >= objc) {
                    throw std::runtime_error("Missing value for parameter");
                }
                
                std::string param = Tcl_GetString(objv[i]);
                
                if (param == "-n") {
                    int n_val;
                    if (Tcl_GetIntFromObj(interp, objv[i + 1], &n_val) != TCL_OK) {
                        throw std::runtime_error("Invalid n parameter: must be an integer");
                    }
                    args.n = n_val;
                } else {
                    throw std::runtime_error("Unknown parameter: " + param);
                }
            }
        } else {
            // Pure named parameter syntax
            for (int i = 1; i < objc; i += 2) {
                if (i + 1 >= objc) {
                    throw std::runtime_error("Missing value for parameter");
                }
                
                std::string param = Tcl_GetString(objv[i]);
                
                if (param == "-input") {
                    args.input = Tcl_GetString(objv[i + 1]);
                } else if (param == "-n") {
                    int n_val;
                    if (Tcl_GetIntFromObj(interp, objv[i + 1], &n_val) != TCL_OK) {
                        throw std::runtime_error("Invalid n parameter: must be an integer");
                    }
                    args.n = n_val;
                } else {
                    throw std::runtime_error("Unknown parameter: " + param);
                }
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input");
    }
    
    return args;
}

// torch::matrix_power - Matrix power
int TensorMatrixPower_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        MatrixPowerArgs args = ParseMatrixPowerArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];

        auto output = torch::matrix_power(input, args.n);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for matrix_rank command
struct MatrixRankArgs {
    std::string input;
    double tol = 1e-12;  // Default tolerance
    bool hermitian = false;  // Default hermitian flag
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for matrix_rank
MatrixRankArgs ParseMatrixRankArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MatrixRankArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::matrix_rank input ?tol? ?hermitian? | torch::matrix_rank -input tensor ?-tol double? ?-hermitian bool?");
    }
    
    // Check if we have named parameters (look for parameters starting with '-' that are not negative numbers)
    bool hasNamedParams = false;
    for (int i = 1; i < objc; i++) {
        std::string arg = Tcl_GetString(objv[i]);
        // Check if it starts with '-' and is not a negative number
        if (arg.length() > 1 && arg[0] == '-' && !isdigit(arg[1])) {
            hasNamedParams = true;
            break;
        }
    }
    
    if (!hasNamedParams) {
        // Pure positional syntax (backward compatibility)
        if (objc < 2 || objc > 4) {
            throw std::runtime_error("Usage: torch::matrix_rank input ?tol? ?hermitian?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            double tol_val;
            if (Tcl_GetDoubleFromObj(interp, objv[2], &tol_val) != TCL_OK) {
                throw std::runtime_error("Invalid tol parameter: must be a number");
            }
            args.tol = tol_val;
        }
        
        if (objc > 3) {
            int hermitian_val;
            if (Tcl_GetIntFromObj(interp, objv[3], &hermitian_val) != TCL_OK) {
                throw std::runtime_error("Invalid hermitian parameter: must be a boolean (0 or 1)");
            }
            args.hermitian = (hermitian_val != 0);
        }
    } else {
        // We have named parameters - check for mixed or pure named syntax
        if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
            // Mixed syntax: positional input, then named parameters
            args.input = Tcl_GetString(objv[1]);
            
            // Parse remaining named parameters starting from index 2
            for (int i = 2; i < objc; i += 2) {
                if (i + 1 >= objc) {
                    throw std::runtime_error("Missing value for parameter");
                }
                
                std::string param = Tcl_GetString(objv[i]);
                
                if (param == "-tol" || param == "-tolerance") {
                    double tol_val;
                    if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &tol_val) != TCL_OK) {
                        throw std::runtime_error("Invalid tol parameter: must be a number");
                    }
                    args.tol = tol_val;
                } else if (param == "-hermitian") {
                    int hermitian_val;
                    if (Tcl_GetIntFromObj(interp, objv[i + 1], &hermitian_val) != TCL_OK) {
                        throw std::runtime_error("Invalid hermitian parameter: must be a boolean (0 or 1)");
                    }
                    args.hermitian = (hermitian_val != 0);
                } else {
                    throw std::runtime_error("Unknown parameter: " + param);
                }
            }
        } else {
            // Pure named parameter syntax
            for (int i = 1; i < objc; i += 2) {
                if (i + 1 >= objc) {
                    throw std::runtime_error("Missing value for parameter");
                }
                
                std::string param = Tcl_GetString(objv[i]);
                
                if (param == "-input") {
                    args.input = Tcl_GetString(objv[i + 1]);
                } else if (param == "-tol" || param == "-tolerance") {
                    double tol_val;
                    if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &tol_val) != TCL_OK) {
                        throw std::runtime_error("Invalid tol parameter: must be a number");
                    }
                    args.tol = tol_val;
                } else if (param == "-hermitian") {
                    int hermitian_val;
                    if (Tcl_GetIntFromObj(interp, objv[i + 1], &hermitian_val) != TCL_OK) {
                        throw std::runtime_error("Invalid hermitian parameter: must be a boolean (0 or 1)");
                    }
                    args.hermitian = (hermitian_val != 0);
                } else {
                    throw std::runtime_error("Unknown parameter: " + param);
                }
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input");
    }
    
    return args;
}

// torch::matrix_rank - Matrix rank
int TensorMatrixRank_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        MatrixRankArgs args = ParseMatrixRankArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];

        auto output = torch::linalg::matrix_rank(input, args.tol, args.hermitian);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for cond command
struct CondArgs {
    std::string input;
    std::string p = "";  // Empty means use default (2-norm)
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for cond
CondArgs ParseCondArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    CondArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::cond input ?p? | torch::cond -input tensor -p value");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        args.input = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            args.p = Tcl_GetString(objv[2]);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-p" || param == "-norm") {
                args.p = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor, -p/-norm");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor required");
    }
    
    return args;
}

// torch::cond - Condition number
int TensorCond_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        CondArgs args = ParseCondArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];

        torch::Tensor output;
        if (!args.p.empty()) {
            if (args.p == "fro" || args.p == "nuc") {
                // Use Frobenius norm as approximation for matrix condition number
                output = torch::norm(input) / torch::norm(torch::pinverse(input));
            } else {
                // Try to parse as number
                char* endptr;
                double p_val = std::strtod(args.p.c_str(), &endptr);
                if (*endptr != '\0') {
                    Tcl_SetResult(interp, const_cast<char*>("Invalid p parameter: must be a number or 'fro' or 'nuc'"), TCL_VOLATILE);
                    return TCL_ERROR;
                }
                // Compute condition number using SVD
                auto svd_result = torch::svd(input);
                auto s = std::get<1>(svd_result);
                output = s[0] / s[-1];
            }
        } else {
            // Default: use 2-norm condition number
            auto svd_result = torch::svd(input);
            auto s = std::get<1>(svd_result);
            output = s[0] / s[-1];
        }

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for matrix_norm command
struct MatrixNormArgs {
    std::string input;
    std::string ord = "fro";  // Default value
    std::vector<int64_t> dim;
    bool keepdim = false;
    bool has_dim = false;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for matrix_norm
MatrixNormArgs ParseMatrixNormArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MatrixNormArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::matrix_norm input ?ord? ?dim? ?keepdim? | torch::matrix_norm -input tensor ?-ord string/double? ?-dim list? ?-keepdim bool?");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 5) {
            throw std::runtime_error("Usage: torch::matrix_norm input ?ord? ?dim? ?keepdim?");
        }
        args.input = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            args.ord = Tcl_GetString(objv[2]);
        }
        
        if (objc > 3) {
            // Parse dim list
            int listLen;
            Tcl_Obj** listObjv;
            if (Tcl_ListObjGetElements(interp, objv[3], &listLen, &listObjv) == TCL_OK) {
                args.dim.clear();
                for (int i = 0; i < listLen; i++) {
                    int dim_val;
                    if (Tcl_GetIntFromObj(interp, listObjv[i], &dim_val) == TCL_OK) {
                        args.dim.push_back(dim_val);
                    }
                }
                if (!args.dim.empty()) {
                    args.has_dim = true;
                }
            }
        }
        
        if (objc > 4) {
            int keepdim_int;
            if (Tcl_GetIntFromObj(interp, objv[4], &keepdim_int) == TCL_OK) {
                args.keepdim = (keepdim_int != 0);
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameter missing value");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-ord") {
                args.ord = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dim") {
                // Parse dim list
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) == TCL_OK) {
                    args.dim.clear();
                    for (int j = 0; j < listLen; j++) {
                        int dim_val;
                        if (Tcl_GetIntFromObj(interp, listObjv[j], &dim_val) == TCL_OK) {
                            args.dim.push_back(dim_val);
                        }
                    }
                    if (!args.dim.empty()) {
                        args.has_dim = true;
                    }
                }
            } else if (param == "-keepdim") {
                int keepdim_int;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &keepdim_int) == TCL_OK) {
                    args.keepdim = (keepdim_int != 0);
                }
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

// torch::matrix_norm - Matrix norm
int TensorMatrixNorm_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        MatrixNormArgs args = ParseMatrixNormArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];

        // Handle ord parameter (string or numeric)
        bool use_string_ord = true;
        std::string ord_str = args.ord;
        c10::Scalar scalar_ord = 2.0;

        if (ord_str != "fro" && ord_str != "nuc") {
            // Try to parse as number
            try {
                double ord_val = std::stod(ord_str);
                scalar_ord = c10::Scalar(ord_val);
                use_string_ord = false;
            } catch (...) {
                // Keep as string, will use default "fro"
                ord_str = "fro";
            }
        }

        // Handle dim parameter
        c10::optional<c10::IntArrayRef> dim = c10::nullopt;
        if (args.has_dim && !args.dim.empty()) {
            dim = c10::IntArrayRef(args.dim);
        }

        torch::Tensor output;
        if (use_string_ord) {
            if (dim.has_value()) {
                output = torch::linalg_matrix_norm(input, ord_str, dim.value(), args.keepdim);
            } else {
                output = torch::linalg_matrix_norm(input, ord_str, {-2, -1}, args.keepdim);
            }
        } else {
            if (dim.has_value()) {
                output = torch::linalg_matrix_norm(input, scalar_ord, dim.value(), args.keepdim);
            } else {
                output = torch::linalg_matrix_norm(input, scalar_ord, {-2, -1}, args.keepdim);
            }
        }

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// =====================
// Dual Syntax Support for torch::vector_norm / torch::vectorNorm  
// =====================
struct TensorVectorNormArgs {
    std::string input;
    double ord = 2.0;
    std::vector<int64_t> dim;
    bool has_dim = false;
    bool keepdim = false;
    
    bool IsValid() const {
        return !input.empty();
    }
};

TensorVectorNormArgs ParseTensorVectorNormArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorVectorNormArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::vector_norm input ?ord? ?dim? ?keepdim?
        if (objc < 2 || objc > 5) {
            Tcl_WrongNumArgs(interp, 1, objv, "input ?ord? ?dim? ?keepdim?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            double ord_val;
            if (Tcl_GetDoubleFromObj(interp, objv[2], &ord_val) != TCL_OK) {
                throw std::runtime_error("Invalid ord value");
            }
            args.ord = ord_val;
        }
        
        if (objc > 3) {
            // Parse dim list
            int listLen;
            Tcl_Obj** listObjv;
            if (Tcl_ListObjGetElements(interp, objv[3], &listLen, &listObjv) == TCL_OK) {
                std::vector<int64_t> dim_vec;
                for (int i = 0; i < listLen; i++) {
                    int dim_val;
                    if (Tcl_GetIntFromObj(interp, listObjv[i], &dim_val) == TCL_OK) {
                        dim_vec.push_back(dim_val);
                    }
                }
                if (!dim_vec.empty()) {
                    args.dim = dim_vec;
                    args.has_dim = true;
                }
            }
        }
        
        if (objc > 4) {
            int keepdim_int;
            if (Tcl_GetIntFromObj(interp, objv[4], &keepdim_int) != TCL_OK) {
                throw std::runtime_error("Invalid keepdim value");
            }
            args.keepdim = (keepdim_int != 0);
        }
    } else {
        // Named parameter syntax: torch::vector_norm -input tensor ?-ord double? ?-dim list? ?-keepdim bool?
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-ord") {
                double ord_val;
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &ord_val) != TCL_OK) {
                    throw std::runtime_error("Invalid ord value");
                }
                args.ord = ord_val;
            } else if (param == "-dim") {
                // Parse dim list
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) == TCL_OK) {
                    std::vector<int64_t> dim_vec;
                    for (int j = 0; j < listLen; j++) {
                        int dim_val;
                        if (Tcl_GetIntFromObj(interp, listObjv[j], &dim_val) == TCL_OK) {
                            dim_vec.push_back(dim_val);
                        }
                    }
                    if (!dim_vec.empty()) {
                        args.dim = dim_vec;
                        args.has_dim = true;
                    }
                }
            } else if (param == "-keepdim") {
                int keepdim_int;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &keepdim_int) != TCL_OK) {
                    throw std::runtime_error("Invalid keepdim value");
                }
                args.keepdim = (keepdim_int != 0);
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

// torch::vector_norm - Vector norm
int TensorVectorNorm_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorVectorNormArgs args = ParseTensorVectorNormArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];

        // Prepare parameters
        c10::Scalar ord = args.ord;
        c10::optional<c10::IntArrayRef> dim = c10::nullopt;
        if (args.has_dim) {
            dim = c10::IntArrayRef(args.dim);
        }

        auto output = torch::linalg_vector_norm(input, ord, dim, args.keepdim);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for lstsq command
struct LstsqArgs {
    std::string B;
    std::string A;
    c10::optional<double> rcond = c10::nullopt;
    
    bool IsValid() const {
        return !B.empty() && !A.empty();
    }
};

// Parse dual syntax for lstsq
LstsqArgs ParseLstsqArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LstsqArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::lstsq B A ?rcond? | torch::lstsq -b tensor -a tensor ?-rcond double?");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: torch::lstsq B A ?rcond?");
        }
        args.B = Tcl_GetString(objv[1]);
        args.A = Tcl_GetString(objv[2]);
        
        if (objc > 3) {
            double rcond_val;
            if (Tcl_GetDoubleFromObj(interp, objv[3], &rcond_val) != TCL_OK) {
                throw std::runtime_error("Invalid rcond parameter");
            }
            args.rcond = rcond_val;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-b" || param == "-B") {
                args.B = Tcl_GetString(objv[i + 1]);
            } else if (param == "-a" || param == "-A") {
                args.A = Tcl_GetString(objv[i + 1]);
            } else if (param == "-rcond") {
                double rcond_val;
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &rcond_val) != TCL_OK) {
                    throw std::runtime_error("Invalid rcond parameter value");
                }
                args.rcond = rcond_val;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -b, -B, -a, -A, -rcond");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: B and A tensors required");
    }
    
    return args;
}

// torch::lstsq - Least squares solution
int TensorLstsq_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        LstsqArgs args = ParseLstsqArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.B) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid B tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.A) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid A tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto B = tensor_storage[args.B];
        auto A = tensor_storage[args.A];

        auto result = torch::linalg_lstsq(B, A, args.rcond);
        auto solution = std::get<0>(result);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = solution;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for solve_triangular command
struct SolveTriangularArgs {
    std::string B;
    std::string A;
    bool upper = true;           // Default value
    bool left = true;            // Default value  
    bool unitriangular = false;  // Default value
    
    bool IsValid() const {
        return !B.empty() && !A.empty();
    }
};

// Parse dual syntax for solve_triangular
SolveTriangularArgs ParseSolveTriangularArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SolveTriangularArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::solve_triangular B A ?upper? ?left? ?unitriangular? | torch::solveTriangular -B tensor -A tensor ?-upper bool? ?-left bool? ?-unitriangular bool?");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 6) {
            throw std::runtime_error("Usage: torch::solve_triangular B A ?upper? ?left? ?unitriangular?");
        }
        args.B = Tcl_GetString(objv[1]);
        args.A = Tcl_GetString(objv[2]);
        
        if (objc > 3) {
            int upper_int;
            if (Tcl_GetIntFromObj(interp, objv[3], &upper_int) != TCL_OK) {
                throw std::runtime_error("Invalid upper parameter");
            }
            args.upper = (upper_int != 0);
        }
        
        if (objc > 4) {
            int left_int;
            if (Tcl_GetIntFromObj(interp, objv[4], &left_int) != TCL_OK) {
                throw std::runtime_error("Invalid left parameter");
            }
            args.left = (left_int != 0);
        }
        
        if (objc > 5) {
            int unit_int;
            if (Tcl_GetIntFromObj(interp, objv[5], &unit_int) != TCL_OK) {
                throw std::runtime_error("Invalid unitriangular parameter");
            }
            args.unitriangular = (unit_int != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-B" || param == "-b") {
                args.B = Tcl_GetString(objv[i + 1]);
            } else if (param == "-A" || param == "-a") {
                args.A = Tcl_GetString(objv[i + 1]);
            } else if (param == "-upper") {
                int upper_int;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &upper_int) != TCL_OK) {
                    throw std::runtime_error("Invalid upper parameter value");
                }
                args.upper = (upper_int != 0);
            } else if (param == "-left") {
                int left_int;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &left_int) != TCL_OK) {
                    throw std::runtime_error("Invalid left parameter value");
                }
                args.left = (left_int != 0);
            } else if (param == "-unitriangular") {
                int unit_int;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &unit_int) != TCL_OK) {
                    throw std::runtime_error("Invalid unitriangular parameter value");
                }
                args.unitriangular = (unit_int != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -B, -b, -A, -a, -upper, -left, -unitriangular");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: B and A tensors required");
    }
    
    return args;
}

// torch::solve_triangular - Solve triangular system
int TensorSolveTriangular_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        SolveTriangularArgs args = ParseSolveTriangularArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.B) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid B tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.A) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid A tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto B = tensor_storage[args.B];
        auto A = tensor_storage[args.A];

        auto output = torch::linalg_solve_triangular(A, B, args.upper, args.left, args.unitriangular);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for cholesky_solve command
struct CholeskySolveArgs {
    std::string B;
    std::string L;
    bool upper = false;  // Default value
    
    bool IsValid() const {
        return !B.empty() && !L.empty();
    }
};

// Parse dual syntax for cholesky_solve
CholeskySolveArgs ParseCholeskySolveArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    CholeskySolveArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::cholesky_solve B L ?upper? | torch::choleskySolve -b tensor -l tensor -upper bool");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: torch::cholesky_solve B L ?upper?");
        }
        args.B = Tcl_GetString(objv[1]);
        args.L = Tcl_GetString(objv[2]);
        
        if (objc > 3) {
            int upper_int;
            if (Tcl_GetIntFromObj(interp, objv[3], &upper_int) != TCL_OK) {
                throw std::runtime_error("Invalid upper parameter");
            }
            args.upper = (upper_int != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-b" || param == "-B") {
                args.B = Tcl_GetString(objv[i + 1]);
            } else if (param == "-l" || param == "-L") {
                args.L = Tcl_GetString(objv[i + 1]);
            } else if (param == "-upper") {
                int upper_int;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &upper_int) != TCL_OK) {
                    throw std::runtime_error("Invalid upper parameter value");
                }
                args.upper = (upper_int != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -b, -B, -l, -L, -upper");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: B and L tensors required");
    }
    
    return args;
}

// torch::cholesky_solve - Cholesky solve
int TensorCholeskySolve_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        CholeskySolveArgs args = ParseCholeskySolveArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.B) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid B tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.L) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid L tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto B = tensor_storage[args.B];
        auto L = tensor_storage[args.L];

        auto output = torch::cholesky_solve(B, L, args.upper);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for lu_solve command
struct LUSolveArgs {
    std::string B;
    std::string LU_data;
    std::string LU_pivots;
    
    bool IsValid() const {
        return !B.empty() && !LU_data.empty() && !LU_pivots.empty();
    }
};

// Parse dual syntax for lu_solve
LUSolveArgs ParseLUSolveArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LUSolveArgs args;
    
    if (objc < 4) {
        throw std::runtime_error("Usage: torch::lu_solve B LU_data LU_pivots | torch::luSolve -B tensor -LU_data tensor -LU_pivots tensor");
    }
    
    if (objc >= 4 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 4) {
            throw std::runtime_error("Usage: torch::lu_solve B LU_data LU_pivots");
        }
        args.B = Tcl_GetString(objv[1]);
        args.LU_data = Tcl_GetString(objv[2]);
        args.LU_pivots = Tcl_GetString(objv[3]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-B" || param == "-b") {
                args.B = Tcl_GetString(objv[i + 1]);
            } else if (param == "-LU_data" || param == "-luData") {
                args.LU_data = Tcl_GetString(objv[i + 1]);
            } else if (param == "-LU_pivots" || param == "-luPivots") {
                args.LU_pivots = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -B, -b, -LU_data, -luData, -LU_pivots, -luPivots");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: B, LU_data, and LU_pivots tensors required");
    }
    
    return args;
}

// torch::lu_solve - LU solve
int TensorLUSolve_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        LUSolveArgs args = ParseLUSolveArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.B) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid B tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.LU_data) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid LU_data tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.LU_pivots) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid LU_pivots tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto B = tensor_storage[args.B];
        auto LU_data = tensor_storage[args.LU_data];
        auto LU_pivots = tensor_storage[args.LU_pivots];

        auto output = torch::lu_solve(B, LU_data, LU_pivots);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 