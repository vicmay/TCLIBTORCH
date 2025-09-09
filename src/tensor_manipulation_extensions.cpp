#include "libtorchtcl.h"

// Parameter structure for flip command
struct TensorFlipArgs {
    std::string input;
    std::vector<int64_t> dims;
    
    bool IsValid() const {
        return !input.empty() && !dims.empty();
    }
};

// Parse dual syntax for flip
TensorFlipArgs ParseTensorFlipArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorFlipArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::flip input dims | torch::flip -input tensor -dims list");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::flip input dims");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        // Parse dimensions list
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, objv[2], &listLen, &listObjv) != TCL_OK) {
            throw std::runtime_error("Invalid dims parameter");
        }
        
        for (int i = 0; i < listLen; i++) {
            int dim_val;
            if (Tcl_GetIntFromObj(interp, listObjv[i], &dim_val) != TCL_OK) {
                throw std::runtime_error("Invalid dimension value");
            }
            args.dims.push_back(dim_val);
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
            } else if (param == "-dims" || param == "-dimensions") {
                // Parse dimensions list
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) != TCL_OK) {
                    throw std::runtime_error("Invalid dims parameter");
                }
                
                for (int j = 0; j < listLen; j++) {
                    int dim_val;
                    if (Tcl_GetIntFromObj(interp, listObjv[j], &dim_val) != TCL_OK) {
                        throw std::runtime_error("Invalid dimension value");
                    }
                    args.dims.push_back(dim_val);
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -input and -dims");
    }
    
    return args;
}

// torch::flip - Flip tensor along specified dimensions with dual syntax support
int TensorFlip_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorFlipArgs args = ParseTensorFlipArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto output = torch::flip(input, args.dims);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for roll command
struct RollArgs {
    std::string input;
    std::vector<int64_t> shifts;
    std::vector<int64_t> dims;
    
    bool IsValid() const {
        return !input.empty() && !shifts.empty() && (dims.empty() || dims.size() == shifts.size());
    }
};

// Parse dual syntax for roll
RollArgs ParseRollArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    RollArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::roll input shifts ?dims? | torch::roll -input tensor -shifts {shift1 ?shift2 ...?} ?-dims {dim1 ?dim2 ...?}?");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc > 4) {
            throw std::runtime_error("Usage: torch::roll input shifts ?dims?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        // Parse shifts
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, objv[2], &listLen, &listObjv) != TCL_OK) {
            throw std::runtime_error("Invalid shifts list");
        }
        
        for (int i = 0; i < listLen; i++) {
            int shift_val;
            if (Tcl_GetIntFromObj(interp, listObjv[i], &shift_val) != TCL_OK) {
                throw std::runtime_error("Invalid shift value");
            }
            args.shifts.push_back(shift_val);
        }
        
        if (objc > 3) {
            // Parse dims
            int dims_len;
            Tcl_Obj** dims_objv;
            if (Tcl_ListObjGetElements(interp, objv[3], &dims_len, &dims_objv) != TCL_OK) {
                throw std::runtime_error("Invalid dims list");
            }
            
            for (int i = 0; i < dims_len; i++) {
                int dim_val;
                if (Tcl_GetIntFromObj(interp, dims_objv[i], &dim_val) != TCL_OK) {
                    throw std::runtime_error("Invalid dimension value");
                }
                args.dims.push_back(dim_val);
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
            } else if (param == "-shifts") {
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) != TCL_OK) {
                    throw std::runtime_error("Invalid shifts list");
                }
                
                for (int j = 0; j < listLen; j++) {
                    int shift_val;
                    if (Tcl_GetIntFromObj(interp, listObjv[j], &shift_val) != TCL_OK) {
                        throw std::runtime_error("Invalid shift value");
                    }
                    args.shifts.push_back(shift_val);
                }
            } else if (param == "-dims") {
                int dims_len;
                Tcl_Obj** dims_objv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &dims_len, &dims_objv) != TCL_OK) {
                    throw std::runtime_error("Invalid dims list");
                }
                
                for (int j = 0; j < dims_len; j++) {
                    int dim_val;
                    if (Tcl_GetIntFromObj(interp, dims_objv[j], &dim_val) != TCL_OK) {
                        throw std::runtime_error("Invalid dimension value");
                    }
                    args.dims.push_back(dim_val);
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid");
    }
    
    return args;
}

// torch::roll - Roll tensor elements along specified dimension
int TensorRoll_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        RollArgs args = ParseRollArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        torch::Tensor output;

        if (args.dims.empty()) {
            output = torch::roll(input, args.shifts);
        } else {
            output = torch::roll(input, args.shifts, args.dims);
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

// Parameter structure for rot90 command
struct Rot90Args {
    std::string input;
    int k = 1;
    std::vector<int64_t> dims = {0, 1};
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for rot90
Rot90Args ParseRot90Args(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    Rot90Args args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::rot90 input ?k? ?dims? OR torch::rot90 -input tensor ?-k number? ?-dims list?");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        args.input = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.k) != TCL_OK) {
                throw std::runtime_error("Invalid k value");
            }
        }
        
        if (objc > 3) {
            int listLen;
            Tcl_Obj** listObjv;
            if (Tcl_ListObjGetElements(interp, objv[3], &listLen, &listObjv) != TCL_OK) {
                throw std::runtime_error("Invalid dims list");
            }
            
            args.dims.clear();
            for (int i = 0; i < listLen; i++) {
                int dim_val;
                if (Tcl_GetIntFromObj(interp, listObjv[i], &dim_val) != TCL_OK) {
                    throw std::runtime_error("Invalid dims list");
                }
                args.dims.push_back(dim_val);
            }
        }
    } else {
        // Named parameter syntax
        bool has_input = false;
        
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
                has_input = true;
            } else if (param == "-k") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.k) != TCL_OK) {
                    throw std::runtime_error("Invalid k value");
                }
            } else if (param == "-dims") {
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) != TCL_OK) {
                    throw std::runtime_error("Invalid dims list");
                }
                
                args.dims.clear();
                for (int j = 0; j < listLen; j++) {
                    int dim_val;
                    if (Tcl_GetIntFromObj(interp, listObjv[j], &dim_val) != TCL_OK) {
                        throw std::runtime_error("Invalid dims list");
                    }
                    args.dims.push_back(dim_val);
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
        
        if (!has_input) {
            throw std::runtime_error("Input tensor is required");
        }
    }
    
    return args;
}

// torch::rot90 - Rotate tensor by 90 degrees
int TensorRot90_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        Rot90Args args = ParseRot90Args(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto output = torch::rot90(input, args.k, args.dims);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::narrow_copy - Narrow copy of tensor
// Parameter structure for narrow_copy command
struct NarrowCopyArgs {
    std::string input;
    int dim = 0;
    int start = 0;
    int length = 0;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for narrow_copy
NarrowCopyArgs ParseNarrowCopyArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    NarrowCopyArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 5) {
            throw std::runtime_error("Usage: torch::narrow_copy input dim start length");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
            throw std::runtime_error("Invalid dimension value");
        }
        
        if (Tcl_GetIntFromObj(interp, objv[3], &args.start) != TCL_OK) {
            throw std::runtime_error("Invalid start value");
        }
        
        if (Tcl_GetIntFromObj(interp, objv[4], &args.length) != TCL_OK) {
            throw std::runtime_error("Invalid length value");
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
                    throw std::runtime_error("Invalid dimension value");
                }
            } else if (param == "-start") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.start) != TCL_OK) {
                    throw std::runtime_error("Invalid start value");
                }
            } else if (param == "-length") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.length) != TCL_OK) {
                    throw std::runtime_error("Invalid length value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Input tensor is required");
    }
    
    return args;
}

int TensorNarrowCopy_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        NarrowCopyArgs args = ParseNarrowCopyArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto output = input.narrow_copy(args.dim, args.start, args.length);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for take_along_dim command
struct TensorTakeAlongDimArgs {
    std::string input;
    std::string indices;
    int dim = -1;  // -1 means no dimension specified
    
    bool IsValid() const {
        return !input.empty() && !indices.empty();
    }
};

// Parse dual syntax for take_along_dim
TensorTakeAlongDimArgs ParseTensorTakeAlongDimArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorTakeAlongDimArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::take_along_dim input indices ?dim? | torch::take_along_dim -input input -indices indices ?-dim dim?");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc > 4) {
            throw std::runtime_error("Usage: torch::take_along_dim input indices ?dim?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.indices = Tcl_GetString(objv[2]);
        
        if (objc > 3) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.dim) != TCL_OK) {
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
            
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-indices") {
                args.indices = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dim value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and indices");
    }
    
    return args;
}

// torch::take_along_dim - Take values along a dimension using indices
int TensorTakeAlongDim_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorTakeAlongDimArgs args = ParseTensorTakeAlongDimArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.indices) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid indices tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto indices = tensor_storage[args.indices];

        torch::Tensor output;
        if (args.dim >= 0) {
            output = torch::take_along_dim(input, indices, args.dim);
        } else {
            output = torch::take_along_dim(input, indices);
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

// ========== Dual syntax support for torch::gather_nd =========================
struct TensorGatherNdArgs {
    std::string input;
    std::string indices;

    bool IsValid() const {
        return !input.empty() && !indices.empty();
    }
};

TensorGatherNdArgs ParseTensorGatherNdArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorGatherNdArgs args;

    if (objc < 3) {
        throw std::runtime_error("Usage: torch::gather_nd input indices OR torch::gather_nd -input handle -indices handle");
    }

    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax
        args.input   = Tcl_GetString(objv[1]);
        args.indices = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameter requires a value");
            }
            std::string param = Tcl_GetString(objv[i]);

            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-indices") {
                args.indices = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and indices must be provided");
    }

    return args;
}

// torch::gather_nd - N-dimensional gather
int TensorGatherNd_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorGatherNdArgs args = ParseTensorGatherNdArgs(interp, objc, objv);

        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        if (tensor_storage.find(args.indices) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid indices tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input   = tensor_storage[args.input];
        auto indices = tensor_storage[args.indices];

        // Simplified gather_nd implementation using index_select
        auto output = input.index_select(0, indices.flatten()).view_as(indices);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for scatter_nd command
struct ScatterNdArgs {
    std::string input;
    std::string indices;
    std::string updates;
    
    bool IsValid() const {
        return !input.empty() && !indices.empty() && !updates.empty();
    }
};

// Parse dual syntax for scatter_nd
ScatterNdArgs ParseScatterNdArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ScatterNdArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 4) {
            throw std::runtime_error("Usage: torch::scatter_nd input indices updates");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.indices = Tcl_GetString(objv[2]);
        args.updates = Tcl_GetString(objv[3]);
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
            } else if (param == "-indices") {
                args.indices = value;
            } else if (param == "-updates") {
                args.updates = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing");
    }
    
    return args;
}

// torch::scatter_nd - N-dimensional scatter
int TensorScatterNd_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        ScatterNdArgs args = ParseScatterNdArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.indices) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid indices tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.updates) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid updates tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto indices = tensor_storage[args.indices];
        auto updates = tensor_storage[args.updates];

        // Simplified scatter_nd implementation
        auto output = input.clone();
        output.scatter_(0, indices, updates);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for meshgrid command
struct MeshgridArgs {
    std::vector<std::string> tensors;
    
    bool IsValid() const {
        return !tensors.empty();
    }
};

// Parse dual syntax for meshgrid
MeshgridArgs ParseMeshgridArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MeshgridArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2) {
            throw std::runtime_error("Usage: torch::meshgrid tensor1 tensor2 ...");
        }
        
        for (int i = 1; i < objc; i++) {
            args.tensors.push_back(Tcl_GetString(objv[i]));
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-tensors") {
                // Parse list of tensors
                int list_len;
                Tcl_Obj** tensor_list;
                
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &list_len, &tensor_list) != TCL_OK) {
                    throw std::runtime_error("Invalid tensor list");
                }
                
                for (int j = 0; j < list_len; j++) {
                    args.tensors.push_back(Tcl_GetString(tensor_list[j]));
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("At least one tensor is required");
    }
    
    return args;
}

// torch::meshgrid - Create coordinate grids
int TensorMeshgrid_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        MeshgridArgs args = ParseMeshgridArgs(interp, objc, objv);
        std::vector<torch::Tensor> tensors;
        
        for (const auto& tensor_name : args.tensors) {
            if (tensor_storage.find(tensor_name) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>("Invalid tensor"), TCL_VOLATILE);
                return TCL_ERROR;
            }
            
            tensors.push_back(tensor_storage[tensor_name]);
        }

        auto grids = torch::meshgrid(tensors);

        // Return list of tensor handles
        Tcl_Obj* result_list = Tcl_NewListObj(0, nullptr);
        
        for (const auto& grid : grids) {
            std::string handle = GetNextHandle("tensor");
            tensor_storage[handle] = grid;
            
            Tcl_Obj* handle_obj = Tcl_NewStringObj(handle.c_str(), -1);
            Tcl_ListObjAppendElement(interp, result_list, handle_obj);
        }

        Tcl_SetObjResult(interp, result_list);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for combinations command
struct CombinationsArgs {
    std::string input;
    int r = 2;
    bool with_replacement = false;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for combinations
CombinationsArgs ParseCombinationsArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    CombinationsArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 4) {
            throw std::runtime_error("wrong # args: should be \"torch::combinations input ?r? ?with_replacement?\"");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.r) != TCL_OK) {
                throw std::runtime_error("Invalid r parameter");
            }
        }
        
        if (objc > 3) {
            int replacement_flag;
            if (Tcl_GetIntFromObj(interp, objv[3], &replacement_flag) != TCL_OK) {
                throw std::runtime_error("Invalid with_replacement parameter");
            }
            args.with_replacement = (replacement_flag != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = value;
            } else if (param == "-r") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.r) != TCL_OK) {
                    throw std::runtime_error("Invalid -r parameter");
                }
            } else if (param == "-with_replacement" || param == "-replacement") {
                int replacement_flag;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &replacement_flag) != TCL_OK) {
                    throw std::runtime_error("Invalid -with_replacement parameter");
                }
                args.with_replacement = (replacement_flag != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter -input missing");
    }
    
    return args;
}

// torch::combinations - Generate combinations of elements
int TensorCombinations_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        auto args = ParseCombinationsArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto output = torch::combinations(input, args.r, args.with_replacement);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for cartesian_prod command
struct CartesianProdArgs {
    std::vector<std::string> tensors;
    
    bool IsValid() const {
        return !tensors.empty();
    }
};

// Parse dual syntax for cartesian_prod
CartesianProdArgs ParseCartesianProdArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    CartesianProdArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::cartesian_prod tensor1 tensor2 [tensor3...] | torch::cartesian_prod -tensors {tensor1 tensor2 ...}");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        for (int i = 1; i < objc; i++) {
            args.tensors.push_back(Tcl_GetString(objv[i]));
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-tensors") {
                // Support both list format and individual tensors
                std::string value = Tcl_GetString(objv[i + 1]);
                
                // Try to parse as a list first
                int list_length;
                Tcl_Obj** list_elements;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &list_length, &list_elements) == TCL_OK && list_length > 0) {
                    // It's a list
                    for (int j = 0; j < list_length; j++) {
                        args.tensors.push_back(Tcl_GetString(list_elements[j]));
                    }
                } else {
                    // It's a single tensor name
                    args.tensors.push_back(value);
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -tensors");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: at least one tensor required");
    }
    
    return args;
}

// torch::cartesian_prod - Cartesian product of tensors
int TensorCartesianProd_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        CartesianProdArgs args = ParseCartesianProdArgs(interp, objc, objv);
        
        std::vector<torch::Tensor> tensors;
        
        for (const auto& tensor_name : args.tensors) {
            if (tensor_storage.find(tensor_name) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>("Invalid tensor"), TCL_VOLATILE);
                return TCL_ERROR;
            }
            tensors.push_back(tensor_storage[tensor_name]);
        }

        auto output = torch::cartesian_prod(tensors);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensordot command
struct TensordotArgs {
    std::string a;
    std::string b;
    std::vector<int64_t> dims;
    
    bool IsValid() const {
        return !a.empty() && !b.empty() && !dims.empty();
    }
};

// Parse dual syntax for tensordot
static TensordotArgs ParseTensordotArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensordotArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: a b dims
        if (objc != 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "a b dims");
            throw std::runtime_error("Usage: torch::tensordot a b dims");
        }
        
        args.a = Tcl_GetString(objv[1]);
        args.b = Tcl_GetString(objv[2]);
        
        // Parse dims list
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, objv[3], &listLen, &listObjv) != TCL_OK) {
            throw std::runtime_error("Invalid dims list format");
        }
        
        for (int i = 0; i < listLen; i++) {
            int dim_val;
            if (Tcl_GetIntFromObj(interp, listObjv[i], &dim_val) != TCL_OK) {
                throw std::runtime_error("Invalid dimension value in dims list");
            }
            args.dims.push_back(dim_val);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-a") {
                args.a = Tcl_GetString(objv[i + 1]);
            } else if (param == "-b") {
                args.b = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dims") {
                // Parse dims list
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) != TCL_OK) {
                    throw std::runtime_error("Invalid dims list format");
                }
                
                for (int j = 0; j < listLen; j++) {
                    int dim_val;
                    if (Tcl_GetIntFromObj(interp, listObjv[j], &dim_val) != TCL_OK) {
                        throw std::runtime_error("Invalid dimension value in dims list");
                    }
                    args.dims.push_back(dim_val);
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -a, -b, -dims");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: a, b, and dims required");
    }
    
    return args;
}

// torch::tensordot - Tensor dot product
int TensorTensordot_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensordotArgs args = ParseTensordotArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.a) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor a"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.b) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor b"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto a = tensor_storage[args.a];
        auto b = tensor_storage[args.b];

        // tensordot requires dims_a and dims_b separately
        auto output = torch::tensordot(a, b, args.dims, args.dims);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for einsum command
struct EinsumArgs {
    std::string equation;
    std::vector<std::string> tensors;
    
    bool IsValid() const {
        return !equation.empty() && !tensors.empty();
    }
};

// Parse dual syntax for einsum
EinsumArgs ParseEinsumArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    EinsumArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::einsum equation tensor1 [tensor2...] | torch::einsum -equation str -tensors {tensor1 tensor2 ...}");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        args.equation = Tcl_GetString(objv[1]);
        
        for (int i = 2; i < objc; i++) {
            args.tensors.push_back(Tcl_GetString(objv[i]));
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-equation") {
                args.equation = Tcl_GetString(objv[i + 1]);
            } else if (param == "-tensors") {
                // Support both list format and individual tensors
                std::string value = Tcl_GetString(objv[i + 1]);
                
                // Try to parse as a list first
                int list_length;
                Tcl_Obj** list_elements;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &list_length, &list_elements) == TCL_OK && list_length > 0) {
                    // It's a list
                    for (int j = 0; j < list_length; j++) {
                        args.tensors.push_back(Tcl_GetString(list_elements[j]));
                    }
                } else {
                    // It's a single tensor name
                    args.tensors.push_back(value);
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -equation, -tensors");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: equation and at least one tensor required");
    }
    
    return args;
}

// torch::einsum - Einstein summation
int TensorEinsum_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        EinsumArgs args = ParseEinsumArgs(interp, objc, objv);
        
        std::vector<torch::Tensor> tensors;
        
        for (const auto& tensor_name : args.tensors) {
            if (tensor_storage.find(tensor_name) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>("Invalid tensor"), TCL_VOLATILE);
                return TCL_ERROR;
            }
            tensors.push_back(tensor_storage[tensor_name]);
        }

        auto output = torch::einsum(args.equation, tensors);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for torch::kron
// ============================================================================
struct KronArgs {
    std::string input;
    std::string other;
    
    bool IsValid() const {
        return !input.empty() && !other.empty();
    }
};

KronArgs ParseKronArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    KronArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): input other
        if (objc != 3) {
            throw std::runtime_error("Usage: kron input other");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.other = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        if (objc < 5) {
            throw std::runtime_error("Usage: kron -input tensor -other tensor");
        }
        
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter: " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-other") {
                args.other = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and other required");
    }
    
    return args;
}

// torch::kron - Kronecker product
int TensorKron_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        KronArgs args = ParseKronArgs(interp, objc, objv);
        
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

        auto output = torch::kron(input, other);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for broadcast_tensors command
struct BroadcastTensorsArgs {
    std::vector<std::string> tensors;
    
    bool IsValid() const {
        return !tensors.empty();
    }
};

// Parse dual syntax for broadcast_tensors
BroadcastTensorsArgs ParseBroadcastTensorsArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    BroadcastTensorsArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::broadcast_tensors tensor1 tensor2 [tensor3...] | torch::broadcast_tensors -tensors {tensor1 tensor2 ...}");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        for (int i = 1; i < objc; i++) {
            args.tensors.push_back(Tcl_GetString(objv[i]));
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-tensors") {
                // Support both list format and individual tensors
                std::string value = Tcl_GetString(objv[i + 1]);
                
                // Try to parse as a list first
                int list_length;
                Tcl_Obj** list_elements;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &list_length, &list_elements) == TCL_OK && list_length > 0) {
                    // It's a list
                    for (int j = 0; j < list_length; j++) {
                        args.tensors.push_back(Tcl_GetString(list_elements[j]));
                    }
                } else {
                    // It's a single tensor name
                    args.tensors.push_back(value);
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -tensors");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: at least one tensor required");
    }
    
    return args;
}

// torch::broadcast_tensors - Broadcast tensors to common shape
int TensorBroadcastTensors_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        BroadcastTensorsArgs args = ParseBroadcastTensorsArgs(interp, objc, objv);
        
        std::vector<torch::Tensor> tensors;
        
        for (const auto& tensor_name : args.tensors) {
            if (tensor_storage.find(tensor_name) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>("Invalid tensor"), TCL_VOLATILE);
                return TCL_ERROR;
            }
            tensors.push_back(tensor_storage[tensor_name]);
        }

        auto broadcasted = torch::broadcast_tensors(tensors);

        // Return list of tensor handles
        Tcl_Obj* result_list = Tcl_NewListObj(0, nullptr);
        
        for (const auto& tensor : broadcasted) {
            std::string handle = GetNextHandle("tensor");
            tensor_storage[handle] = tensor;
            
            Tcl_Obj* handle_obj = Tcl_NewStringObj(handle.c_str(), -1);
            Tcl_ListObjAppendElement(interp, result_list, handle_obj);
        }

        Tcl_SetObjResult(interp, result_list);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for atleast_1d command
struct TensorAtleast1dArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for atleast_1d
TensorAtleast1dArgs ParseTensorAtleast1dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorAtleast1dArgs args;
    
    // Provide immediate feedback if no additional arguments were supplied.
    // Tests expect the error message to contain the word "Usage" when the
    // command is invoked without the required tensor argument.
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::atleast_1d tensor | torch::atleast_1d -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::atleast_1d tensor");
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

// torch::atleast_1d - Ensure tensor is at least 1D
int TensorAtleast1d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        TensorAtleast1dArgs args = ParseTensorAtleast1dArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto output = torch::atleast_1d(input);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for atleast_2d command
struct TensorAtleast2dArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for atleast_2d
TensorAtleast2dArgs ParseTensorAtleast2dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorAtleast2dArgs args;
    
    // Provide immediate feedback if no additional arguments were supplied.
    // Tests expect the error message to contain the word "Usage" when the
    // command is invoked without the required tensor argument.
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::atleast_2d tensor | torch::atleast_2d -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::atleast_2d tensor");
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

// torch::atleast_2d - Ensure tensor is at least 2D
int TensorAtleast2d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        TensorAtleast2dArgs args = ParseTensorAtleast2dArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto output = torch::atleast_2d(input);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for atleast_3d command
struct TensorAtleast3dArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for atleast_3d
TensorAtleast3dArgs ParseTensorAtleast3dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorAtleast3dArgs args;
    
    // Provide immediate feedback if no additional arguments were supplied.
    // Tests expect the error message to contain the word "Usage" when the
    // command is invoked without the required tensor argument.
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::atleast_3d tensor | torch::atleast_3d -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::atleast_3d tensor");
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

// torch::atleast_3d - Ensure tensor is at least 3D
int TensorAtleast3d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        TensorAtleast3dArgs args = ParseTensorAtleast3dArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto output = torch::atleast_3d(input);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 

// Parameter structure for tensor_to_list command
struct TensorToListArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_to_list
TensorToListArgs ParseTensorToListArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorToListArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::tensor_to_list tensor | torch::tensor_to_list -input tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::tensor_to_list tensor");
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
        throw std::runtime_error("Required parameter missing: input tensor");
    }
    
    return args;
}

int TensorToList_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    try {
        TensorToListArgs args = ParseTensorToListArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        // Convert tensor to flat array
        auto flat_tensor = tensor.flatten();
        
        // Create TCL list
        Tcl_Obj* result_list = Tcl_NewListObj(0, nullptr);
        
        // Handle different data types
        if (tensor.dtype() == torch::kBool) {
            auto accessor = flat_tensor.accessor<bool, 1>();
            for (int64_t i = 0; i < flat_tensor.size(0); ++i) {
                Tcl_ListObjAppendElement(interp, result_list, Tcl_NewBooleanObj(accessor[i]));
            }
        } else if (tensor.dtype() == torch::kInt32 || tensor.dtype() == torch::kInt64) {
            auto accessor = flat_tensor.accessor<int64_t, 1>();
            for (int64_t i = 0; i < flat_tensor.size(0); ++i) {
                Tcl_ListObjAppendElement(interp, result_list, Tcl_NewLongObj(accessor[i]));
            }
        } else {
            // Default to float for all other types
            auto accessor = flat_tensor.accessor<float, 1>();
            for (int64_t i = 0; i < flat_tensor.size(0); ++i) {
                Tcl_ListObjAppendElement(interp, result_list, Tcl_NewDoubleObj(accessor[i]));
            }
        }
        
        Tcl_SetObjResult(interp, result_list);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 

// Parameter structure for tensor_select command
struct TensorSelectArgs {
    std::string input;
    int64_t dim;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_select
TensorSelectArgs ParseTensorSelectArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorSelectArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Error in tensor_select: Usage: torch::tensor_select tensor dim | torch::tensor_select -input tensor -dim index");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Error in tensor_select: Usage: torch::tensor_select tensor dim");
        }
        args.input = Tcl_GetString(objv[1]);
        if (Tcl_GetLongFromObj(interp, objv[2], &args.dim) != TCL_OK) {
            throw std::runtime_error("Error in tensor_select: Invalid dimension index");
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            std::string param = Tcl_GetString(objv[i]);
            if (i + 1 >= objc) {
                throw std::runtime_error("Error in tensor_select: Missing value for parameter: " + param);
            }
            
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dim") {
                if (Tcl_GetLongFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Error in tensor_select: Invalid dimension index");
                }
            } else {
                throw std::runtime_error("Error in tensor_select: Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Error in tensor_select: Required parameters missing");
    }
    
    return args;
}

// torch::tensor_select - Select a slice from a tensor along a dimension
int TensorSelect_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorSelectArgs args = ParseTensorSelectArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            throw std::runtime_error("Error in tensor_select: Invalid input tensor");
        }
        
        auto input = tensor_storage[args.input];
        
        if (args.dim < 0 || args.dim >= input.dim()) {
            throw std::runtime_error("Error in tensor_select: Dimension index out of range");
        }
        
        auto result = input.select(args.dim, 0);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 