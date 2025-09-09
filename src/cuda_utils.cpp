#include "libtorchtcl.h"
#include <cuda_runtime.h>

// ============================================================================
// CUDA Is Available Command - Dual Syntax Support
// ============================================================================

struct CudaIsAvailableArgs {
    // This command takes no parameters
    bool IsValid() const {
        return true; // Always valid as no parameters required
    }
};

CudaIsAvailableArgs ParseCudaIsAvailableArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    CudaIsAvailableArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::cuda_is_available
    if (objc != 1) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::cuda_is_available");
        }
    } else {
        // Named parameter syntax: torch::cuda_is_available (no parameters)
        if (objc > 1) {
            // Check for any provided parameters (should be none)
            for (int i = 1; i < objc; i += 2) {
                if (i + 1 >= objc) {
                    throw std::runtime_error("Missing value for parameter");
                }
                
                std::string param = Tcl_GetString(objv[i]);
                throw std::runtime_error("Unknown parameter: " + param + " (this command takes no parameters)");
            }
        }
    }
    
    return args;
    }

int CudaIsAvailable_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        CudaIsAvailableArgs args = ParseCudaIsAvailableArgs(interp, objc, objv);
        
        bool available = torch::cuda::is_available();
        Tcl_SetResult(interp, const_cast<char*>(available ? "1" : "0"), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// CUDA Device Count Command - Dual Syntax Support
// ============================================================================

struct CudaDeviceCountArgs {
    // This command takes no parameters
    bool IsValid() const {
        return true; // Always valid as no parameters required
    }
};

CudaDeviceCountArgs ParseCudaDeviceCountArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    CudaDeviceCountArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): torch::cuda_device_count
    if (objc != 1) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::cuda_device_count");
        }
    } else {
        // Named parameter syntax: torch::cuda_device_count (no parameters)
        if (objc > 1) {
            // Check for any provided parameters (should be none)
            for (int i = 1; i < objc; i += 2) {
                if (i + 1 >= objc) {
                    throw std::runtime_error("Missing value for parameter");
                }
                
                std::string param = Tcl_GetString(objv[i]);
                throw std::runtime_error("Unknown parameter: " + param + " (this command takes no parameters)");
            }
        }
    }
    
    return args;
    }

int CudaDeviceCount_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        CudaDeviceCountArgs args = ParseCudaDeviceCountArgs(interp, objc, objv);
        
        if (!torch::cuda::is_available()) {
            Tcl_SetResult(interp, const_cast<char*>("0"), TCL_VOLATILE);
            return TCL_OK;
        }
        
        int count = torch::cuda::device_count();
        std::string result = std::to_string(count);
        Tcl_SetResult(interp, const_cast<char*>(result.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// CUDA Device Info Command - Dual Syntax Support
// ============================================================================

struct CudaDeviceInfoArgs {
    int device_id = 0;
    
    bool IsValid() const {
        return device_id >= 0;
    }
};

CudaDeviceInfoArgs ParseCudaDeviceInfoArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    CudaDeviceInfoArgs args;
    
    // Check if this is positional syntax: either no dash, or a negative number
    bool is_positional = false;
    if (objc >= 2) {
        std::string first_arg = Tcl_GetString(objv[1]);
        // Positional if: doesn't start with '-', OR starts with '-' but is a valid number (negative number)
        if (first_arg[0] != '-') {
            is_positional = true;
        } else {
            // Check if it's a negative number by trying to parse it as int
            int temp_val;
            if (Tcl_GetIntFromObj(interp, objv[1], &temp_val) == TCL_OK) {
                is_positional = true;
            }
        }
    }
    
    if (objc < 2 || is_positional) {
        // Positional syntax (backward compatibility): torch::cuda_device_info [device_id]
        if (objc > 2) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::cuda_device_info [device_id]");
        }
        
        if (objc == 2) {
            if (Tcl_GetIntFromObj(interp, objv[1], &args.device_id) != TCL_OK) {
                throw std::runtime_error("Invalid device_id value. Expected integer.");
            }
        }
    } else {
        // Named parameter syntax: torch::cuda_device_info [-device_id device_id]
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-device_id") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.device_id) != TCL_OK) {
                    throw std::runtime_error("Invalid device_id value. Expected integer.");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Invalid device_id: must be non-negative");
    }
    
    return args;
}

int CudaDeviceInfo_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        CudaDeviceInfoArgs args = ParseCudaDeviceInfoArgs(interp, objc, objv);
        
        if (!torch::cuda::is_available()) {
            Tcl_SetResult(interp, const_cast<char*>("CUDA not available"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (static_cast<size_t>(args.device_id) >= torch::cuda::device_count()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid device ID"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get device properties using CUDA runtime API
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, args.device_id);
        
        std::ostringstream info;
        info << "Device " << args.device_id << ": " << props.name 
             << " (Compute " << props.major << "." << props.minor << ")";
        
        std::string result = info.str();
        Tcl_SetResult(interp, const_cast<char*>(result.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// CUDA Memory Info Command - Dual Syntax Support
// ============================================================================

struct CudaMemoryInfoArgs {
    int device_id = 0;
    
    bool IsValid() const {
        return device_id >= 0;
    }
};

CudaMemoryInfoArgs ParseCudaMemoryInfoArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    CudaMemoryInfoArgs args;
    
    // Check if this is positional syntax: either no dash, or a negative number
    bool is_positional = false;
    if (objc >= 2) {
        std::string first_arg = Tcl_GetString(objv[1]);
        // Positional if: doesn't start with '-', OR starts with '-' but is a valid number (negative number)
        if (first_arg[0] != '-') {
            is_positional = true;
        } else {
            // Check if it's a negative number by trying to parse it as int
            int temp_val;
            if (Tcl_GetIntFromObj(interp, objv[1], &temp_val) == TCL_OK) {
                is_positional = true;
            }
        }
    }
    
    if (objc < 2 || is_positional) {
        // Positional syntax (backward compatibility): torch::cuda_memory_info [device_id]
        if (objc > 2) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::cuda_memory_info [device_id]");
        }
        
        if (objc == 2) {
            if (Tcl_GetIntFromObj(interp, objv[1], &args.device_id) != TCL_OK) {
                throw std::runtime_error("Invalid device_id value. Expected integer.");
            }
        }
    } else {
        // Named parameter syntax: torch::cuda_memory_info [-device_id device_id]
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-device_id") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.device_id) != TCL_OK) {
                    throw std::runtime_error("Invalid device_id value. Expected integer.");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Invalid device_id: must be non-negative");
    }
    
    return args;
}

int CudaMemoryInfo_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        CudaMemoryInfoArgs args = ParseCudaMemoryInfoArgs(interp, objc, objv);
        
        if (!torch::cuda::is_available()) {
            Tcl_SetResult(interp, const_cast<char*>("CUDA not available"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (static_cast<size_t>(args.device_id) >= torch::cuda::device_count()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid device ID"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get memory info
        size_t free_bytes, total_bytes;
        torch::cuda::synchronize(args.device_id);
        
        // Use CUDA runtime API to get memory info
        cudaSetDevice(args.device_id);
        cudaMemGetInfo(&free_bytes, &total_bytes);
        
        size_t used_bytes = total_bytes - free_bytes;
        
        std::ostringstream info;
        info << "Device " << args.device_id << " Memory: "
             << "Used=" << (used_bytes / (1024*1024)) << "MB "
             << "Free=" << (free_bytes / (1024*1024)) << "MB "
             << "Total=" << (total_bytes / (1024*1024)) << "MB";
        
        std::string result = info.str();
        Tcl_SetResult(interp, const_cast<char*>(result.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
} 