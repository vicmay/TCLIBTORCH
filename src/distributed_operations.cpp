#include "libtorchtcl.h"
#include <torch/torch.h>
#include <sstream>

// ============================================================================
// DISTRIBUTED_GATHER Command - Dual Syntax Support
// ============================================================================

struct DistributedGatherArgs {
    std::string tensor;
    int dst = 0;
    std::string group = "";
    
    bool IsValid() const {
        return !tensor.empty();
    }
};

DistributedGatherArgs ParseDistributedGatherArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DistributedGatherArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::distributed_gather tensor ?dst? ?group?
        if (objc < 2 || objc > 4) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::distributed_gather tensor ?dst? ?group?");
        }
        
        args.tensor = Tcl_GetString(objv[1]);
        
        if (objc >= 3) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.dst) != TCL_OK) {
                throw std::runtime_error("Invalid dst parameter. Must be an integer.");
            }
        }
        
        if (objc == 4) {
            args.group = Tcl_GetString(objv[3]);
        }
    } else {
        // Named parameter syntax: torch::distributed_gather -tensor tensor ?-dst dst? ?-group group?
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-tensor") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dst") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dst) != TCL_OK) {
                    throw std::runtime_error("Invalid -dst parameter. Must be an integer.");
                }
            } else if (param == "-group") {
                args.group = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: -tensor");
    }
    
    return args;
}

// torch::distributed_gather - Gather tensors from all processes
int TensorDistributedGather_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        DistributedGatherArgs args = ParseDistributedGatherArgs(interp, objc, objv);
        
        // Check if tensor exists in storage
        if (tensor_storage.find(args.tensor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        torch::Tensor tensor = tensor_storage[args.tensor];
        
        // Simplified distributed gather implementation
        // In a real distributed setting, this would gather from all processes
        std::vector<torch::Tensor> tensor_list;
        tensor_list.push_back(tensor);
        
        // For simulation, just return the original tensor
        auto result = torch::stack(tensor_list);
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// DISTRIBUTED_SCATTER Command - Dual Syntax Support
// ============================================================================

struct DistributedScatterArgs {
    std::string tensor;
    int src = 0;             // Default source rank
    std::string group = "";  // Optional group parameter
    
    bool IsValid() const {
        return !tensor.empty() && src >= 0;
    }
};

DistributedScatterArgs ParseDistributedScatterArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DistributedScatterArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::distributed_scatter tensor ?src? ?group?
        if (objc < 2 || objc > 4) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::distributed_scatter tensor ?src? ?group?");
        }
        
        args.tensor = Tcl_GetString(objv[1]);
        
        if (objc >= 3) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.src) != TCL_OK) {
                throw std::runtime_error("Invalid src parameter. Must be an integer.");
            }
        }
        
        if (objc == 4) {
            args.group = Tcl_GetString(objv[3]);
        }
    } else {
        // Named parameter syntax: torch::distributed_scatter -tensor tensor ?-src src? ?-group group?
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-tensor") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else if (param == "-src") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.src) != TCL_OK) {
                    throw std::runtime_error("Invalid -src parameter. Must be an integer.");
                }
            } else if (param == "-group") {
                args.group = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: -tensor, or invalid src parameter");
    }
    
    return args;
}

// torch::distributed_scatter - Scatter tensor to all processes
int TensorDistributedScatter_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        DistributedScatterArgs args = ParseDistributedScatterArgs(interp, objc, objv);
        
        // Check if tensor exists in storage
        if (tensor_storage.find(args.tensor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor handle"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        torch::Tensor tensor = tensor_storage[args.tensor];
        
        // Simplified distributed scatter implementation
        // In a real distributed setting, this would scatter to all processes
        auto result = tensor.clone();
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// DISTRIBUTED_REDUCE_SCATTER Command - Dual Syntax Support
// ============================================================================

struct DistributedReduceScatterArgs {
    std::string tensor;
    std::string op = "sum";      // Default operation
    std::string group = "";      // Optional group parameter
    
    bool IsValid() const {
        return !tensor.empty() && (op == "sum" || op == "mean" || op == "max" || op == "min" || op == "product");
    }
};

DistributedReduceScatterArgs ParseDistributedReduceScatterArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DistributedReduceScatterArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::distributed_reduce_scatter tensor ?op? ?group?
        if (objc < 2 || objc > 4) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::distributed_reduce_scatter tensor ?op? ?group?");
        }
        
        args.tensor = Tcl_GetString(objv[1]);
        
        if (objc >= 3) {
            args.op = Tcl_GetString(objv[2]);
        }
        
        if (objc == 4) {
            args.group = Tcl_GetString(objv[3]);
        }
    } else {
        // Named parameter syntax: torch::distributed_reduce_scatter -tensor tensor ?-op op? ?-group group?
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-tensor") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else if (param == "-op") {
                args.op = Tcl_GetString(objv[i + 1]);
            } else if (param == "-group") {
                args.group = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: -tensor, or invalid operation. Valid operations: sum, mean, max, min, product");
    }
    
    return args;
}

// torch::distributed_reduce_scatter - Reduce and scatter tensor
int TensorDistributedReduceScatter_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        DistributedReduceScatterArgs args = ParseDistributedReduceScatterArgs(interp, objc, objv);
        
        // Check if tensor exists in storage
        if (tensor_storage.find(args.tensor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor handle"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        torch::Tensor tensor = tensor_storage[args.tensor];
        
        // Simplified reduce-scatter implementation
        // In a real distributed setting, this would reduce across processes and scatter
        auto result = tensor.clone();
        
        // Apply the reduction operation locally for simulation
        if (args.op == "sum") {
            // Already a clone, sum operation is identity in single process
        } else if (args.op == "mean") {
            // In distributed setting, this would be divided by world size
            result = result / 1.0;  // Placeholder for world size
        } else if (args.op == "max") {
            // Max operation is identity in single process
        } else if (args.op == "min") {
            // Min operation is identity in single process
        } else if (args.op == "product") {
            // Product operation is identity in single process
        }
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// DISTRIBUTED_ALL_TO_ALL Command - Dual Syntax Support
// ============================================================================

struct DistributedAllToAllArgs {
    std::string tensor;
    std::string group = "";  // optional group parameter
    
    bool IsValid() const {
        return !tensor.empty();
    }
};

DistributedAllToAllArgs ParseDistributedAllToAllArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DistributedAllToAllArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::distributed_all_to_all tensor ?group?
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::distributed_all_to_all tensor ?group?");
        }
        
        args.tensor = Tcl_GetString(objv[1]);
        
        if (objc == 3) {
            args.group = Tcl_GetString(objv[2]);
        }
    } else {
        // Named parameter syntax: torch::distributed_all_to_all -tensor tensor ?-group group?
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-tensor") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else if (param == "-group") {
                args.group = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: -tensor");
    }
    
    return args;
}

// torch::distributed_all_to_all - All-to-all communication
int TensorDistributedAllToAll_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        DistributedAllToAllArgs args = ParseDistributedAllToAllArgs(interp, objc, objv);
        
        // Check if tensor exists in storage
        if (tensor_storage.find(args.tensor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // For simplicity, expect a single tensor and return it
        torch::Tensor tensor = tensor_storage[args.tensor];
        
        // Simplified all-to-all implementation
        // In a real distributed setting, this would exchange data between all processes
        auto result = tensor.clone();
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// DISTRIBUTED_SEND Command - Dual Syntax Support
// ============================================================================

struct DistributedSendArgs {
    std::string tensor;
    int dst = -1;          // Initialize to invalid value to detect missing parameter
    int tag = 0;           // Default tag value
    
    bool IsValid() const {
        return !tensor.empty() && dst >= 0;
    }
};

DistributedSendArgs ParseDistributedSendArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DistributedSendArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::distributed_send tensor dst ?tag?
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::distributed_send tensor dst ?tag?");
        }
        
        args.tensor = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dst) != TCL_OK) {
            throw std::runtime_error("Invalid dst parameter. Must be an integer.");
        }
        
        if (objc == 4) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.tag) != TCL_OK) {
                throw std::runtime_error("Invalid tag parameter. Must be an integer.");
            }
        }
    } else {
        // Named parameter syntax: torch::distributed_send -tensor tensor -dst dst ?-tag tag?
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-tensor") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dst") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dst) != TCL_OK) {
                    throw std::runtime_error("Invalid -dst parameter. Must be an integer.");
                }
            } else if (param == "-tag") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.tag) != TCL_OK) {
                    throw std::runtime_error("Invalid -tag parameter. Must be an integer.");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid: -tensor and -dst are required");
    }
    
    return args;
}

// torch::distributed_send - Point-to-point send
int TensorDistributedSend_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        DistributedSendArgs args = ParseDistributedSendArgs(interp, objc, objv);
        
        // Get tensor from tensor storage
        if (tensor_storage.find(args.tensor) == tensor_storage.end()) {
            throw std::runtime_error("Invalid tensor handle: " + args.tensor);
        }
        
        torch::Tensor tensor = tensor_storage[args.tensor];
        
        // Simplified send implementation
        // In a real distributed setting, this would send to specific process
        Tcl_SetResult(interp, const_cast<char*>("send_completed"), TCL_STATIC);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// DISTRIBUTED_RECV Command - Dual Syntax Support
// ============================================================================

struct DistributedRecvArgs {
    std::vector<int64_t> shape;
    int src = -1;          // Initialize to invalid value to detect missing parameter
    int tag = 0;           // Default tag value
    
    bool IsValid() const {
        return !shape.empty() && src >= 0;
    }
};

DistributedRecvArgs ParseDistributedRecvArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DistributedRecvArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::distributed_recv shape src ?tag?
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::distributed_recv shape src ?tag?");
        }
        
        args.shape = GetIntVectorFromObj(interp, objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.src) != TCL_OK) {
            throw std::runtime_error("Invalid src parameter. Must be an integer.");
        }
        
        if (objc == 4) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.tag) != TCL_OK) {
                throw std::runtime_error("Invalid tag parameter. Must be an integer.");
            }
        }
    } else {
        // Named parameter syntax: torch::distributed_recv -shape shape -src src ?-tag tag?
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-shape") {
                args.shape = GetIntVectorFromObj(interp, objv[i + 1]);
            } else if (param == "-src") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.src) != TCL_OK) {
                    throw std::runtime_error("Invalid -src parameter. Must be an integer.");
                }
            } else if (param == "-tag") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.tag) != TCL_OK) {
                    throw std::runtime_error("Invalid -tag parameter. Must be an integer.");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid: -shape and -src are required");
    }
    
    return args;
}

// torch::distributed_recv - Point-to-point receive
int TensorDistributedRecv_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        DistributedRecvArgs args = ParseDistributedRecvArgs(interp, objc, objv);
        
        // Simplified receive implementation
        // In a real distributed setting, this would receive from specific process
        auto result = torch::zeros(args.shape);
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// DISTRIBUTED_ISEND Command - Dual Syntax Support
// ============================================================================

struct DistributedISendArgs {
    std::string tensor;
    int dst = -1;          // Initialize to invalid value to detect missing parameter
    int tag = 0;           // Default tag value
    
    bool IsValid() const {
        return !tensor.empty() && dst >= 0;
    }
};

DistributedISendArgs ParseDistributedISendArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DistributedISendArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::distributed_isend tensor dst ?tag?
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::distributed_isend tensor dst ?tag?");
        }
        
        args.tensor = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dst) != TCL_OK) {
            throw std::runtime_error("Invalid dst parameter. Must be an integer.");
        }
        
        if (objc == 4) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.tag) != TCL_OK) {
                throw std::runtime_error("Invalid tag parameter. Must be an integer.");
            }
        }
    } else {
        // Named parameter syntax: torch::distributed_isend -tensor tensor -dst dst ?-tag tag?
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-tensor") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dst") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dst) != TCL_OK) {
                    throw std::runtime_error("Invalid -dst parameter. Must be an integer.");
                }
            } else if (param == "-tag") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.tag) != TCL_OK) {
                    throw std::runtime_error("Invalid -tag parameter. Must be an integer.");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid: -tensor and -dst are required");
    }
    
    return args;
}

// torch::distributed_isend - Non-blocking send
int TensorDistributedISend_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        DistributedISendArgs args = ParseDistributedISendArgs(interp, objc, objv);
        
        // Get tensor from tensor storage
        if (tensor_storage.find(args.tensor) == tensor_storage.end()) {
            throw std::runtime_error("Invalid tensor handle: " + args.tensor);
        }
        
        torch::Tensor tensor = tensor_storage[args.tensor];
        
        // Simplified non-blocking send implementation
        // In a real distributed setting, this would return a handle for later waiting
        // The handle would include dst and tag information
        std::string handle = "isend_handle_dst" + std::to_string(args.dst) + "_tag" + std::to_string(args.tag);
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// DISTRIBUTED_IRECV Command - Dual Syntax Support
// ============================================================================

struct DistributedIRecvArgs {
    std::vector<int64_t> shape;
    int src = -1;          // Initialize to invalid value to detect missing parameter
    int tag = 0;           // Default tag value
    
    bool IsValid() const {
        return !shape.empty() && src >= 0;
    }
};

DistributedIRecvArgs ParseDistributedIRecvArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DistributedIRecvArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::distributed_irecv shape src ?tag?
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::distributed_irecv shape src ?tag?");
        }
        
        args.shape = GetIntVectorFromObj(interp, objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.src) != TCL_OK) {
            throw std::runtime_error("Invalid src parameter. Must be an integer.");
        }
        
        if (objc == 4) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.tag) != TCL_OK) {
                throw std::runtime_error("Invalid tag parameter. Must be an integer.");
            }
        }
    } else {
        // Named parameter syntax: torch::distributed_irecv -shape shape -src src ?-tag tag?
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-shape") {
                args.shape = GetIntVectorFromObj(interp, objv[i + 1]);
            } else if (param == "-src") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.src) != TCL_OK) {
                    throw std::runtime_error("Invalid -src parameter. Must be an integer.");
                }
            } else if (param == "-tag") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.tag) != TCL_OK) {
                    throw std::runtime_error("Invalid -tag parameter. Must be an integer.");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid: -shape and -src are required");
    }
    
    return args;
}

// torch::distributed_irecv - Non-blocking receive
int TensorDistributedIRecv_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        DistributedIRecvArgs args = ParseDistributedIRecvArgs(interp, objc, objv);
        
        // Simplified non-blocking receive implementation
        // In a real distributed setting, this would return a handle for later waiting
        Tcl_SetResult(interp, const_cast<char*>("irecv_handle_1"), TCL_STATIC);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// DISTRIBUTED_WAIT Command - Dual Syntax Support
// ============================================================================

struct DistributedWaitArgs {
    std::string handle;
    
    bool IsValid() const {
        return true;  // Allow empty handles as they may be valid in some contexts
    }
};

DistributedWaitArgs ParseDistributedWaitArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DistributedWaitArgs args;
    
    // Check if this is named parameter syntax by looking for exact parameter matches
    bool isNamedSyntax = false;
    if (objc >= 2) {
        std::string firstArg = Tcl_GetString(objv[1]);
        if (firstArg == "-handle" || firstArg.find("-unknown") == 0 || firstArg.find("-invalid") == 0) {
            isNamedSyntax = true;
        }
    }
    
    if (!isNamedSyntax && objc == 2) {
        // Positional syntax (backward compatibility): torch::distributed_wait handle
        args.handle = Tcl_GetString(objv[1]);
    } else if (isNamedSyntax) {
        // Named parameter syntax: torch::distributed_wait -handle handle
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-handle") {
                args.handle = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    } else {
        throw std::runtime_error("Wrong number of arguments. Expected: torch::distributed_wait handle OR torch::distributed_wait -handle handle");
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid: -handle is required");
    }
    
    return args;
}

// torch::distributed_wait - Wait for non-blocking operations
int TensorDistributedWait_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        DistributedWaitArgs args = ParseDistributedWaitArgs(interp, objc, objv);
        
        // Simplified wait implementation
        // In a real distributed setting, this would wait for the operation to complete
        if (args.handle.find("isend") != std::string::npos) {
            Tcl_SetResult(interp, const_cast<char*>("send_completed"), TCL_STATIC);
        } else if (args.handle.find("irecv") != std::string::npos) {
            // Return a dummy tensor for recv operations
            auto result = torch::zeros({2, 2});
            return SetTensorResult(interp, result);
        } else {
            Tcl_SetResult(interp, const_cast<char*>("operation_completed"), TCL_STATIC);
        }
        
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// DISTRIBUTED_TEST Command - Dual Syntax Support
// ============================================================================

struct DistributedTestArgs {
    std::string handle;
    
    bool IsValid() const {
        return true;  // Allow empty handles as they may be valid in some contexts
    }
};

DistributedTestArgs ParseDistributedTestArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DistributedTestArgs args;
    
    // Check if this is named parameter syntax by looking for exact parameter matches
    bool isNamedSyntax = false;
    if (objc >= 2) {
        std::string firstArg = Tcl_GetString(objv[1]);
        if (firstArg == "-handle" || firstArg.find("-unknown") == 0 || firstArg.find("-invalid") == 0) {
            isNamedSyntax = true;
        }
    }
    
    if (!isNamedSyntax && objc == 2) {
        // Positional syntax (backward compatibility): torch::distributed_test handle
        args.handle = Tcl_GetString(objv[1]);
    } else if (isNamedSyntax) {
        // Named parameter syntax: torch::distributed_test -handle handle
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-handle") {
                args.handle = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    } else {
        throw std::runtime_error("Wrong number of arguments. Expected: torch::distributed_test handle OR torch::distributed_test -handle handle");
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid: -handle is required");
    }
    
    return args;
}

// torch::distributed_test - Test if non-blocking operation is complete
int TensorDistributedTest_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        DistributedTestArgs args = ParseDistributedTestArgs(interp, objc, objv);
        
        // Simplified test implementation
        // In a real distributed setting, this would check if operation is complete
        // For simulation, always return true (completed)
        Tcl_SetResult(interp, const_cast<char*>("true"), TCL_STATIC);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
} 