#include "libtorchtcl.h"
#include <torch/torch.h>
#include <tcl.h>

// Parameter structure for reflection_pad1d command
struct ReflectionPad1dArgs {
    std::string input;
    std::vector<int64_t> padding;
    
    bool IsValid() const {
        return !input.empty() && padding.size() == 2;
    }
};

// Parse dual syntax for reflection_pad1d
ReflectionPad1dArgs ParseReflectionPad1dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ReflectionPad1dArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::reflection_pad1d tensor padding | torch::reflection_pad1d -input tensor -padding {left right}");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::reflection_pad1d tensor padding");
        }
        args.input = Tcl_GetString(objv[1]);
        
        // Parse padding values
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, objv[2], &listLen, &listObjv) != TCL_OK) {
            throw std::runtime_error("Invalid padding list format");
        }
        
        if (listLen != 2) {
            throw std::runtime_error("Padding must be a list of 2 values for 1D");
        }
        
        int pad_left, pad_right;
        if (Tcl_GetIntFromObj(interp, listObjv[0], &pad_left) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[1], &pad_right) != TCL_OK) {
            throw std::runtime_error("Invalid padding values");
        }
        
        args.padding = {pad_left, pad_right};
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-padding") {
                // Parse padding values
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) != TCL_OK) {
                    throw std::runtime_error("Invalid padding list format");
                }
                
                if (listLen != 2) {
                    throw std::runtime_error("Padding must be a list of 2 values for 1D");
                }
                
                int pad_left, pad_right;
                if (Tcl_GetIntFromObj(interp, listObjv[0], &pad_left) != TCL_OK ||
                    Tcl_GetIntFromObj(interp, listObjv[1], &pad_right) != TCL_OK) {
                    throw std::runtime_error("Invalid padding values");
                }
                
                args.padding = {pad_left, pad_right};
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and/or padding");
    }
    
    return args;
}

// Reflection padding operations
int ReflectionPad1D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        ReflectionPad1dArgs args = ParseReflectionPad1dArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result = torch::reflection_pad1d(tensor, args.padding);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for reflection_pad2d command
struct ReflectionPad2DArgs {
    std::string input;
    std::vector<int64_t> padding;
    
    bool IsValid() const {
        return !input.empty() && padding.size() == 4;
    }
};

// Parse dual syntax for reflection_pad2d
ReflectionPad2DArgs ParseReflectionPad2DArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ReflectionPad2DArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::reflection_pad2d tensor padding | torch::reflectionPad2d -input tensor -padding {left right top bottom}");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::reflection_pad2d tensor padding");
        }
        args.input = Tcl_GetString(objv[1]);
        
        // Parse padding values
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, objv[2], &listLen, &listObjv) != TCL_OK) {
            throw std::runtime_error("Invalid padding list format");
        }
        
        if (listLen != 4) {
            throw std::runtime_error("Padding must be a list of 4 values for 2D");
        }
        
        for (int i = 0; i < 4; i++) {
            int val;
            if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                throw std::runtime_error("Invalid padding value");
            }
            if (val < 0) {
                throw std::runtime_error("Invalid padding value: padding cannot be negative");
            }
            args.padding.push_back(val);
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
            } else if (param == "-padding" || param == "-pad") {
                // Parse padding values
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) != TCL_OK) {
                    throw std::runtime_error("Invalid padding list format");
                }
                
                if (listLen != 4) {
                    throw std::runtime_error("Padding must be a list of 4 values for 2D");
                }
                
                args.padding.clear();
                for (int j = 0; j < 4; j++) {
                    int val;
                    if (Tcl_GetIntFromObj(interp, listObjv[j], &val) != TCL_OK) {
                        throw std::runtime_error("Invalid padding value");
                    }
                    if (val < 0) {
                        throw std::runtime_error("Invalid padding value: padding cannot be negative");
                    }
                    args.padding.push_back(val);
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor, -padding/-pad");
            }
        }
    }
    
    if (!args.IsValid()) {
        if (args.input.empty()) {
            throw std::runtime_error("Required parameters missing: input tensor and padding values required");
        } else {
            throw std::runtime_error("Missing value for parameter");
        }
    }
    
    return args;
}

int ReflectionPad2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        ReflectionPad2DArgs args = ParseReflectionPad2DArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        // Check tensor dimensions
        if (tensor.dim() != 4) {
            throw std::runtime_error("Expected 4D tensor (batch_size, channels, height, width) for 2D padding, but got " + std::to_string(tensor.dim()) + "D tensor");
        }
        
        // Check for negative padding values
        for (const auto& pad : args.padding) {
            if (pad < 0) {
                throw std::runtime_error("Invalid padding value: padding cannot be negative");
            }
        }
        
        torch::Tensor result = torch::reflection_pad2d(tensor, args.padding);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for reflection_pad3d command
struct ReflectionPad3DArgs {
    std::string input;
    std::vector<int64_t> padding;
    
    bool IsValid() const {
        return !input.empty() && padding.size() == 6;
    }
};

// Parse dual syntax for reflection_pad3d
ReflectionPad3DArgs ParseReflectionPad3DArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ReflectionPad3DArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Required parameters missing: input tensor and padding values required");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::reflection_pad3d tensor padding");
        }
        args.input = Tcl_GetString(objv[1]);
        
        // Parse padding values
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, objv[2], &listLen, &listObjv) != TCL_OK) {
            throw std::runtime_error("Invalid padding list format");
        }
        
        if (listLen != 6) {
            throw std::runtime_error("Padding must be a list of 6 values for 3D");
        }
        
        for (int i = 0; i < 6; i++) {
            int val;
            if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                throw std::runtime_error("Invalid padding value");
            }
            if (val < 0) {
                throw std::runtime_error("Invalid padding value: padding cannot be negative");
            }
            args.padding.push_back(val);
        }
    } else {
        // Named parameter syntax
        bool has_input = false;
        bool has_padding = false;
        
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
                has_input = true;
            } else if (param == "-padding" || param == "-pad") {
                // Parse padding values
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) != TCL_OK) {
                    throw std::runtime_error("Invalid padding list format");
                }
                
                if (listLen != 6) {
                    throw std::runtime_error("Padding must be a list of 6 values for 3D");
                }
                
                args.padding.clear();
                for (int j = 0; j < 6; j++) {
                    int val;
                    if (Tcl_GetIntFromObj(interp, listObjv[j], &val) != TCL_OK) {
                        throw std::runtime_error("Invalid padding value");
                    }
                    if (val < 0) {
                        throw std::runtime_error("Invalid padding value: padding cannot be negative");
                    }
                    args.padding.push_back(val);
                }
                has_padding = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor, -padding/-pad");
            }
        }
        
        if (!has_input) {
            throw std::runtime_error("Required parameters missing: input tensor and padding values required");
        }
        if (!has_padding) {
            throw std::runtime_error("Missing value for parameter");
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and padding values required");
    }
    
    return args;
}

int ReflectionPad3D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        ReflectionPad3DArgs args = ParseReflectionPad3DArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        // Check tensor dimensions
        if (tensor.dim() != 5) {
            throw std::runtime_error("Expected 5D tensor for 3D padding, but got " + std::to_string(tensor.dim()) + "D tensor");
        }
        
        torch::Tensor result = torch::reflection_pad3d(tensor, args.padding);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for replication_pad1d command
struct ReplicationPad1DArgs {
    std::string input;
    std::vector<int64_t> padding;
    
    bool IsValid() const {
        return !input.empty() && padding.size() == 2;
    }
};

// Parse dual syntax for replication_pad1d
ReplicationPad1DArgs ParseReplicationPad1DArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ReplicationPad1DArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Required parameters missing: input tensor and padding values required");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::replication_pad1d tensor padding");
        }
        args.input = Tcl_GetString(objv[1]);
        
        // Parse padding values
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, objv[2], &listLen, &listObjv) != TCL_OK) {
            throw std::runtime_error("Invalid padding list format");
        }
        
        if (listLen != 2) {
            throw std::runtime_error("Padding must be a list of 2 values for 1D");
        }
        
        for (int i = 0; i < 2; i++) {
            int val;
            if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                throw std::runtime_error("Invalid padding value");
            }
            if (val < 0) {
                throw std::runtime_error("Invalid padding value: padding cannot be negative");
            }
            args.padding.push_back(val);
        }
    } else {
        // Named parameter syntax
        bool has_input = false;
        bool has_padding = false;
        
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
                has_input = true;
            } else if (param == "-padding" || param == "-pad") {
                // Parse padding values
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) != TCL_OK) {
                    throw std::runtime_error("Invalid padding list format");
                }
                
                if (listLen != 2) {
                    throw std::runtime_error("Padding must be a list of 2 values for 1D");
                }
                
                args.padding.clear();
                for (int j = 0; j < 2; j++) {
                    int val;
                    if (Tcl_GetIntFromObj(interp, listObjv[j], &val) != TCL_OK) {
                        throw std::runtime_error("Invalid padding value");
                    }
                    if (val < 0) {
                        throw std::runtime_error("Invalid padding value: padding cannot be negative");
                    }
                    args.padding.push_back(val);
                }
                has_padding = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor, -padding/-pad");
            }
        }
        
        if (!has_input) {
            throw std::runtime_error("Required parameters missing: input tensor and padding values required");
        }
        if (!has_padding) {
            throw std::runtime_error("Missing value for parameter");
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and padding values required");
    }
    
    return args;
}

// Replication padding operations
int ReplicationPad1D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        ReplicationPad1DArgs args = ParseReplicationPad1DArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        // Check tensor dimensions
        if (tensor.dim() != 3) {
            throw std::runtime_error("Expected 3D tensor (batch_size, channels, width) for 1D padding, but got " + std::to_string(tensor.dim()) + "D tensor");
        }
        
        torch::Tensor result = torch::replication_pad1d(tensor, args.padding);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for replication_pad2d command
struct ReplicationPad2DArgs {
    std::string input;
    std::vector<int64_t> padding;
    
    bool IsValid() const {
        return !input.empty() && padding.size() == 4;
    }
};

// Parse dual syntax for replication_pad2d
ReplicationPad2DArgs ParseReplicationPad2DArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ReplicationPad2DArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Required parameters missing: input tensor and padding values required");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::replication_pad2d tensor padding");
        }
        args.input = Tcl_GetString(objv[1]);
        
        // Parse padding values
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, objv[2], &listLen, &listObjv) != TCL_OK) {
            throw std::runtime_error("Invalid padding list format");
        }
        
        if (listLen != 4) {
            throw std::runtime_error("Padding must be a list of 4 values for 2D");
        }
        
        for (int i = 0; i < 4; i++) {
            int val;
            if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                throw std::runtime_error("Invalid padding value");
            }
            if (val < 0) {
                throw std::runtime_error("Invalid padding value: padding cannot be negative");
            }
            args.padding.push_back(val);
        }
    } else {
        // Named parameter syntax
        bool has_input = false;
        bool has_padding = false;
        
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
                has_input = true;
            } else if (param == "-padding" || param == "-pad") {
                // Parse padding values
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) != TCL_OK) {
                    throw std::runtime_error("Invalid padding list format");
                }
                
                if (listLen != 4) {
                    throw std::runtime_error("Padding must be a list of 4 values for 2D");
                }
                
                args.padding.clear();
                for (int j = 0; j < 4; j++) {
                    int val;
                    if (Tcl_GetIntFromObj(interp, listObjv[j], &val) != TCL_OK) {
                        throw std::runtime_error("Invalid padding value");
                    }
                    if (val < 0) {
                        throw std::runtime_error("Invalid padding value: padding cannot be negative");
                    }
                    args.padding.push_back(val);
                }
                has_padding = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor, -padding/-pad");
            }
        }
        
        if (!has_input) {
            throw std::runtime_error("Required parameters missing: input tensor and padding values required");
        }
        if (!has_padding) {
            throw std::runtime_error("Missing value for parameter");
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and padding values required");
    }
    
    return args;
}

// Replication padding operations
int ReplicationPad2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        ReplicationPad2DArgs args = ParseReplicationPad2DArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        // Check tensor dimensions
        if (tensor.dim() != 4) {
            throw std::runtime_error("Expected 4D tensor (batch_size, channels, height, width) for 2D padding, but got " + std::to_string(tensor.dim()) + "D tensor");
        }
        
        torch::Tensor result = torch::replication_pad2d(tensor, args.padding);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for replication_pad3d command
struct ReplicationPad3DArgs {
    std::string input;
    std::vector<int64_t> padding;
    
    bool IsValid() const {
        return !input.empty() && padding.size() == 6;
    }
};

// Parse dual syntax for replication_pad3d
ReplicationPad3DArgs ParseReplicationPad3DArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ReplicationPad3DArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Required parameters missing: input tensor and padding values required");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::replication_pad3d tensor padding");
        }
        args.input = Tcl_GetString(objv[1]);
        
        // Parse padding values
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, objv[2], &listLen, &listObjv) != TCL_OK) {
            throw std::runtime_error("Invalid padding list format");
        }
        
        if (listLen != 6) {
            throw std::runtime_error("Padding must be a list of 6 values for 3D");
        }
        
        for (int i = 0; i < 6; i++) {
            int val;
            if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                throw std::runtime_error("Invalid padding value");
            }
            if (val < 0) {
                throw std::runtime_error("Invalid padding value: padding cannot be negative");
            }
            args.padding.push_back(val);
        }
    } else {
        // Named parameter syntax
        bool has_input = false;
        bool has_padding = false;
        
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
                has_input = true;
            } else if (param == "-padding" || param == "-pad") {
                // Parse padding values
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) != TCL_OK) {
                    throw std::runtime_error("Invalid padding list format");
                }
                
                if (listLen != 6) {
                    throw std::runtime_error("Padding must be a list of 6 values for 3D");
                }
                
                args.padding.clear();
                for (int j = 0; j < 6; j++) {
                    int val;
                    if (Tcl_GetIntFromObj(interp, listObjv[j], &val) != TCL_OK) {
                        throw std::runtime_error("Invalid padding value");
                    }
                    if (val < 0) {
                        throw std::runtime_error("Invalid padding value: padding cannot be negative");
                    }
                    args.padding.push_back(val);
                }
                has_padding = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor, -padding/-pad");
            }
        }
        
        if (!has_input) {
            throw std::runtime_error("Required parameters missing: input tensor and padding values required");
        }
        if (!has_padding) {
            throw std::runtime_error("Missing value for parameter");
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and padding values required");
    }
    
    return args;
}

int ReplicationPad3D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        ReplicationPad3DArgs args = ParseReplicationPad3DArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        // Check tensor dimensions
        if (tensor.dim() != 5) {
            throw std::runtime_error("Expected 5D tensor for 3D padding, but got " + std::to_string(tensor.dim()) + "D tensor");
        }
        
        torch::Tensor result = torch::replication_pad3d(tensor, args.padding);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for constant_pad1d command
struct ConstantPad1DArgs {
    Tcl_Obj* input = nullptr;
    Tcl_Obj* padding = nullptr;
    double value = 0.0;
    
    bool IsValid() const {
        return input != nullptr && padding != nullptr;
    }
};

// Parse dual syntax for constant_pad1d
ConstantPad1DArgs ParseConstantPad1DArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ConstantPad1DArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::constant_pad1d tensor padding value | torch::constantPad1d -input tensor -padding {values} -value num");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 4) {
            throw std::runtime_error("Usage: torch::constant_pad1d tensor padding value");
        }
        args.input = objv[1];
        args.padding = objv[2];
        if (Tcl_GetDoubleFromObj(interp, objv[3], &args.value) != TCL_OK) {
            throw std::runtime_error("Invalid value parameter");
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = objv[i + 1];
            } else if (param == "-padding" || param == "-pad") {
                args.padding = objv[i + 1];
            } else if (param == "-value" || param == "-val") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.value) != TCL_OK) {
                    throw std::runtime_error("Invalid value parameter");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor, -padding/-pad, -value/-val");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and padding values required");
    }
    
    return args;
}

// Parameter structure for constant_pad2d command
struct ConstantPad2DArgs {
    Tcl_Obj* input = nullptr;
    Tcl_Obj* padding = nullptr;
    double value = 0.0;
    
    bool IsValid() const {
        return input != nullptr && padding != nullptr;
    }
};

// Parse dual syntax for constant_pad2d
ConstantPad2DArgs ParseConstantPad2DArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ConstantPad2DArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::constant_pad2d tensor padding value | torch::constantPad2d -input tensor -padding {values} -value num");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 4) {
            throw std::runtime_error("Usage: torch::constant_pad2d tensor padding value");
        }
        args.input = objv[1];
        args.padding = objv[2];
        if (Tcl_GetDoubleFromObj(interp, objv[3], &args.value) != TCL_OK) {
            throw std::runtime_error("Invalid value parameter");
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = objv[i + 1];
            } else if (param == "-padding" || param == "-pad") {
                args.padding = objv[i + 1];
            } else if (param == "-value" || param == "-val") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.value) != TCL_OK) {
                    throw std::runtime_error("Invalid value parameter");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor, -padding/-pad, -value/-val");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and padding values required");
    }
    
    return args;
}

// Parameter structure for constant_pad3d command
struct ConstantPad3DArgs {
    Tcl_Obj* input = nullptr;
    Tcl_Obj* padding = nullptr;
    double value = 0.0;
    
    bool IsValid() const {
        return input != nullptr && padding != nullptr;
    }
};

// Parse dual syntax for constant_pad3d
ConstantPad3DArgs ParseConstantPad3DArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ConstantPad3DArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::constant_pad3d tensor padding value | torch::constantPad3d -input tensor -padding {values} -value num");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 4) {
            throw std::runtime_error("Usage: torch::constant_pad3d tensor padding value");
        }
        args.input = objv[1];
        args.padding = objv[2];
        if (Tcl_GetDoubleFromObj(interp, objv[3], &args.value) != TCL_OK) {
            throw std::runtime_error("Invalid value parameter");
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = objv[i + 1];
            } else if (param == "-padding" || param == "-pad") {
                args.padding = objv[i + 1];
            } else if (param == "-value" || param == "-val") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.value) != TCL_OK) {
                    throw std::runtime_error("Invalid value parameter");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor, -padding/-pad, -value/-val");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and padding values required");
    }
    
    return args;
}

// Constant padding operations with dual syntax support
int ConstantPad1D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        ConstantPad1DArgs args = ParseConstantPad1DArgs(interp, objc, objv);
        
        torch::Tensor tensor = GetTensorFromObj(interp, args.input);
        if (tensor.numel() == 0) return TCL_ERROR;

        // Parse padding values
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, args.padding, &listLen, &listObjv) != TCL_OK) {
            return TCL_ERROR;
        }

        if (listLen != 2) {
            Tcl_SetResult(interp, const_cast<char*>("Padding must be a list of 2 values for 1D"), TCL_STATIC);
            return TCL_ERROR;
        }

        int pad_left, pad_right;
        if (Tcl_GetIntFromObj(interp, listObjv[0], &pad_left) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[1], &pad_right) != TCL_OK) {
            return TCL_ERROR;
        }

        torch::Tensor result = torch::constant_pad_nd(tensor, {pad_left, pad_right}, args.value);
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

int ConstantPad2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        ConstantPad2DArgs args = ParseConstantPad2DArgs(interp, objc, objv);
        
        torch::Tensor tensor = GetTensorFromObj(interp, args.input);
        if (tensor.numel() == 0) return TCL_ERROR;

        // Parse padding values
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, args.padding, &listLen, &listObjv) != TCL_OK) {
            return TCL_ERROR;
        }

        if (listLen != 4) {
            Tcl_SetResult(interp, const_cast<char*>("Padding must be a list of 4 values for 2D"), TCL_STATIC);
            return TCL_ERROR;
        }

        int pad_left, pad_right, pad_top, pad_bottom;
        if (Tcl_GetIntFromObj(interp, listObjv[0], &pad_left) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[1], &pad_right) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[2], &pad_top) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[3], &pad_bottom) != TCL_OK) {
            return TCL_ERROR;
        }

        torch::Tensor result = torch::constant_pad_nd(tensor, {pad_left, pad_right, pad_top, pad_bottom}, args.value);
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

int ConstantPad3D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        ConstantPad3DArgs args = ParseConstantPad3DArgs(interp, objc, objv);
        
        torch::Tensor tensor = GetTensorFromObj(interp, args.input);
        if (tensor.numel() == 0) return TCL_ERROR;

        // Parse padding values
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, args.padding, &listLen, &listObjv) != TCL_OK) {
            return TCL_ERROR;
        }

        if (listLen != 6) {
            Tcl_SetResult(interp, const_cast<char*>("Padding must be a list of 6 values for 3D"), TCL_STATIC);
            return TCL_ERROR;
        }

        int pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back;
        if (Tcl_GetIntFromObj(interp, listObjv[0], &pad_left) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[1], &pad_right) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[2], &pad_top) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[3], &pad_bottom) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[4], &pad_front) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[5], &pad_back) != TCL_OK) {
            return TCL_ERROR;
        }

        torch::Tensor result = torch::constant_pad_nd(tensor, {pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back}, args.value);
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for circular_pad1d command
struct CircularPad1DArgs {
    Tcl_Obj* input = nullptr;
    Tcl_Obj* padding = nullptr;
    
    bool IsValid() const {
        return input != nullptr && padding != nullptr;
    }
};

// Parse dual syntax for circular_pad1d
CircularPad1DArgs ParseCircularPad1DArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    CircularPad1DArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::circular_pad1d tensor padding | torch::circularPad1d -input tensor -padding {values}");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::circular_pad1d tensor padding");
        }
        args.input = objv[1];
        args.padding = objv[2];
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = objv[i + 1];
            } else if (param == "-padding" || param == "-pad") {
                args.padding = objv[i + 1];
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor, -padding, -pad");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and padding values required");
    }
    
    return args;
}

// Circular padding operations (using F::pad with circular mode)
int CircularPad1D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        CircularPad1DArgs args = ParseCircularPad1DArgs(interp, objc, objv);
        
        torch::Tensor tensor = GetTensorFromObj(interp, args.input);
        if (tensor.numel() == 0) return TCL_ERROR;

        // Parse padding values
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, args.padding, &listLen, &listObjv) != TCL_OK) {
            return TCL_ERROR;
        }

        if (listLen != 2) {
            Tcl_SetResult(interp, const_cast<char*>("Padding must be a list of 2 values for 1D"), TCL_STATIC);
            return TCL_ERROR;
        }

        int pad_left, pad_right;
        if (Tcl_GetIntFromObj(interp, listObjv[0], &pad_left) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[1], &pad_right) != TCL_OK) {
            return TCL_ERROR;
        }

        torch::Tensor result = torch::nn::functional::pad(tensor, 
            torch::nn::functional::PadFuncOptions({pad_left, pad_right}).mode(torch::kCircular));
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for circular_pad2d command
struct CircularPad2DArgs {
    Tcl_Obj* input = nullptr;
    Tcl_Obj* padding = nullptr;
    
    bool IsValid() const {
        return input != nullptr && padding != nullptr;
    }
};

// Parse dual syntax for circular_pad2d
CircularPad2DArgs ParseCircularPad2DArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    CircularPad2DArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::circular_pad2d tensor padding | torch::circularPad2d -input tensor -padding {values}");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::circular_pad2d tensor padding");
        }
        args.input = objv[1];
        args.padding = objv[2];
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = objv[i + 1];
            } else if (param == "-padding" || param == "-pad") {
                args.padding = objv[i + 1];
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor, -padding, -pad");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and padding values required");
    }
    
    return args;
}

int CircularPad2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        CircularPad2DArgs args = ParseCircularPad2DArgs(interp, objc, objv);
        
        torch::Tensor tensor = GetTensorFromObj(interp, args.input);
        if (tensor.numel() == 0) return TCL_ERROR;

        // Parse padding values
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, args.padding, &listLen, &listObjv) != TCL_OK) {
            return TCL_ERROR;
        }

        if (listLen != 4) {
            Tcl_SetResult(interp, const_cast<char*>("Padding must be a list of 4 values for 2D"), TCL_STATIC);
            return TCL_ERROR;
        }

        int pad_left, pad_right, pad_top, pad_bottom;
        if (Tcl_GetIntFromObj(interp, listObjv[0], &pad_left) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[1], &pad_right) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[2], &pad_top) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[3], &pad_bottom) != TCL_OK) {
            return TCL_ERROR;
        }

        torch::Tensor result = torch::nn::functional::pad(tensor, 
            torch::nn::functional::PadFuncOptions({pad_left, pad_right, pad_top, pad_bottom}).mode(torch::kCircular));
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for circular_pad3d command
struct CircularPad3DArgs {
    Tcl_Obj* input = nullptr;
    Tcl_Obj* padding = nullptr;
    
    bool IsValid() const {
        return input != nullptr && padding != nullptr;
    }
};

// Parse dual syntax for circular_pad3d
CircularPad3DArgs ParseCircularPad3DArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    CircularPad3DArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::circular_pad3d tensor padding | torch::circularPad3d -input tensor -padding {values}");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::circular_pad3d tensor padding");
        }
        args.input = objv[1];
        args.padding = objv[2];
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = objv[i + 1];
            } else if (param == "-padding" || param == "-pad") {
                args.padding = objv[i + 1];
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor, -padding, -pad");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and padding values required");
    }
    
    return args;
}

int CircularPad3D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        CircularPad3DArgs args = ParseCircularPad3DArgs(interp, objc, objv);
        
        torch::Tensor tensor = GetTensorFromObj(interp, args.input);
        if (tensor.numel() == 0) return TCL_ERROR;

        // Parse padding values
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, args.padding, &listLen, &listObjv) != TCL_OK) {
            return TCL_ERROR;
        }

        if (listLen != 6) {
            Tcl_SetResult(interp, const_cast<char*>("Padding must be a list of 6 values for 3D"), TCL_STATIC);
            return TCL_ERROR;
        }

        int pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back;
        if (Tcl_GetIntFromObj(interp, listObjv[0], &pad_left) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[1], &pad_right) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[2], &pad_top) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[3], &pad_bottom) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[4], &pad_front) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[5], &pad_back) != TCL_OK) {
            return TCL_ERROR;
        }

        torch::Tensor result = torch::nn::functional::pad(tensor, 
            torch::nn::functional::PadFuncOptions({pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back}).mode(torch::kCircular));
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for zero_pad1d
struct ZeroPad1DArgs {
    Tcl_Obj* input = nullptr;
    Tcl_Obj* padding = nullptr;
    bool IsValid() const {
        return input != nullptr && padding != nullptr;
    }
};

// Parse dual syntax for zero_pad1d
ZeroPad1DArgs ParseZeroPad1DArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ZeroPad1DArgs args;
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::zero_pad1d tensor padding | torch::zeroPad1d -input tensor -padding {values}");
    }
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::zero_pad1d tensor padding");
        }
        args.input = objv[1];
        args.padding = objv[2];
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input" || param == "-tensor") {
                args.input = objv[i + 1];
            } else if (param == "-padding" || param == "-pad") {
                args.padding = objv[i + 1];
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor, -padding, -pad");
            }
        }
    }
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and padding values required");
    }
    return args;
}

// Zero padding operations (aliases for constant padding with value 0)
int ZeroPad1D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        ZeroPad1DArgs args = ParseZeroPad1DArgs(interp, objc, objv);
        torch::Tensor tensor = GetTensorFromObj(interp, args.input);
        if (tensor.numel() == 0) {
            Tcl_SetResult(interp, const_cast<char*>("Input tensor is empty"), TCL_STATIC);
            return TCL_ERROR;
        }
        // Parse padding values
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, args.padding, &listLen, &listObjv) != TCL_OK) {
            return TCL_ERROR;
        }
        if (listLen != 2) {
            Tcl_SetResult(interp, const_cast<char*>("Padding must be a list of 2 values for 1D"), TCL_STATIC);
            return TCL_ERROR;
        }
        int pad_left, pad_right;
        if (Tcl_GetIntFromObj(interp, listObjv[0], &pad_left) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[1], &pad_right) != TCL_OK) {
            return TCL_ERROR;
        }
        torch::Tensor result = torch::constant_pad_nd(tensor, {pad_left, pad_right}, 0);
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for zero_pad2d
struct ZeroPad2DArgs {
    Tcl_Obj* input = nullptr;
    Tcl_Obj* padding = nullptr;
    bool IsValid() const {
        return input != nullptr && padding != nullptr;
    }
};

// Parse dual syntax for zero_pad2d
ZeroPad2DArgs ParseZeroPad2DArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ZeroPad2DArgs args;
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::zero_pad2d tensor padding | torch::zeroPad2d -input tensor -padding {values}");
    }
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::zero_pad2d tensor padding");
        }
        args.input = objv[1];
        args.padding = objv[2];
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input" || param == "-tensor") {
                args.input = objv[i + 1];
            } else if (param == "-padding" || param == "-pad") {
                args.padding = objv[i + 1];
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor, -padding, -pad");
            }
        }
    }
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and padding values required");
    }
    return args;
}

int ZeroPad2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        ZeroPad2DArgs args = ParseZeroPad2DArgs(interp, objc, objv);
        
        torch::Tensor tensor = GetTensorFromObj(interp, args.input);
        if (tensor.numel() == 0) {
            Tcl_SetResult(interp, const_cast<char*>("Input tensor is empty"), TCL_STATIC);
            return TCL_ERROR;
        }

        // Parse padding values
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, args.padding, &listLen, &listObjv) != TCL_OK) {
            return TCL_ERROR;
        }

        if (listLen != 4) {
            Tcl_SetResult(interp, const_cast<char*>("Padding must be a list of 4 values for 2D"), TCL_STATIC);
            return TCL_ERROR;
        }

        int pad_left, pad_right, pad_top, pad_bottom;
        if (Tcl_GetIntFromObj(interp, listObjv[0], &pad_left) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[1], &pad_right) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[2], &pad_top) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[3], &pad_bottom) != TCL_OK) {
            return TCL_ERROR;
        }

        torch::Tensor result = torch::constant_pad_nd(tensor, {pad_left, pad_right, pad_top, pad_bottom}, 0);
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for zero_pad3d
struct ZeroPad3DArgs {
    Tcl_Obj* input = nullptr;
    Tcl_Obj* padding = nullptr;
    bool IsValid() const {
        return input != nullptr && padding != nullptr;
    }
};

// Parse dual syntax for zero_pad3d
ZeroPad3DArgs ParseZeroPad3DArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ZeroPad3DArgs args;
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::zero_pad3d tensor padding | torch::zeroPad3d -input tensor -padding {values}");
    }
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::zero_pad3d tensor padding");
        }
        args.input = objv[1];
        args.padding = objv[2];
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input" || param == "-tensor") {
                args.input = objv[i + 1];
            } else if (param == "-padding" || param == "-pad") {
                args.padding = objv[i + 1];
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor, -padding, -pad");
            }
        }
    }
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and padding values required");
    }
    return args;
}

int ZeroPad3D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        ZeroPad3DArgs args = ParseZeroPad3DArgs(interp, objc, objv);
        torch::Tensor tensor = GetTensorFromObj(interp, args.input);
        if (tensor.numel() == 0) {
            Tcl_SetResult(interp, const_cast<char*>("Input tensor is empty"), TCL_STATIC);
            return TCL_ERROR;
        }
        // Parse padding values
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, args.padding, &listLen, &listObjv) != TCL_OK) {
            return TCL_ERROR;
        }
        if (listLen != 6) {
            Tcl_SetResult(interp, const_cast<char*>("Padding must be a list of 6 values for 3D"), TCL_STATIC);
            return TCL_ERROR;
        }
        int pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back;
        if (Tcl_GetIntFromObj(interp, listObjv[0], &pad_left) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[1], &pad_right) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[2], &pad_top) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[3], &pad_bottom) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[4], &pad_front) != TCL_OK ||
            Tcl_GetIntFromObj(interp, listObjv[5], &pad_back) != TCL_OK) {
            return TCL_ERROR;
        }
        torch::Tensor result = torch::constant_pad_nd(tensor, {pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back}, 0);
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 