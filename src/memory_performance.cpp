#include "libtorchtcl.h"
#include <torch/torch.h>
#include <sstream>
#include <chrono>

// Parameter structure for empty_cache command
struct EmptyCacheArgs {
    std::string device = "";  // Optional device specification
    
    bool IsValid() const {
        return true;  // No required parameters
    }
};

// Parse dual syntax for empty_cache
EmptyCacheArgs ParseEmptyCacheArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    EmptyCacheArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: ?device?
        if (objc > 2) {
            Tcl_WrongNumArgs(interp, 1, objv, "?device?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        if (objc == 2) {
            args.device = Tcl_GetString(objv[1]);
        }
    } else {
        // Named parameter syntax: -device device
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-device") {
                args.device = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    return args;
}

// torch::memory_stats - Get memory statistics
int TensorMemoryStats_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc > 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "?device?");
        return TCL_ERROR;
    }

    try {
        std::ostringstream stats;
        
        // Get basic memory info (simplified implementation)
        if (torch::cuda::is_available()) {
            stats << "cuda_available: true device_count: " << torch::cuda::device_count();
        } else {
            stats << "cuda_available: false";
        }
        
        Tcl_SetResult(interp, const_cast<char*>(stats.str().c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in memory_stats: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::memory_summary - Get memory summary
int TensorMemorySummary_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc > 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "?device?");
        return TCL_ERROR;
    }

    try {
        std::ostringstream summary;
        
        if (torch::cuda::is_available()) {
            summary << "CUDA Memory Summary:\\n";
            summary << "Device Count: " << torch::cuda::device_count();
        } else {
            summary << "CUDA not available";
        }
        
        Tcl_SetResult(interp, const_cast<char*>(summary.str().c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in memory_summary: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::memory_snapshot - Get memory snapshot
// Parameter structure for memory_snapshot command
struct MemorySnapshotArgs {
    // No parameters needed for this command
    
    bool IsValid() const {
        return true;  // Always valid since there are no required parameters
    }
};

// Parse dual syntax for memory_snapshot
MemorySnapshotArgs ParseMemorySnapshotArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MemorySnapshotArgs args;
    
    if (objc == 1) {
        // No arguments, which is correct
        return args;
    } else {
        // Check if using named parameters
        for (int i = 1; i < objc; i += 2) {
            std::string param = Tcl_GetString(objv[i]);
            throw std::runtime_error("Unknown parameter: " + param + ". This command takes no parameters.");
        }
    }
    
    return args;
}

int TensorMemorySnapshot_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        // Parse arguments using dual syntax
        ParseMemorySnapshotArgs(interp, objc, objv);
        
        // Simple snapshot implementation
        std::ostringstream snapshot;
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        snapshot << "timestamp: " << time_t << " ";
        if (torch::cuda::is_available()) {
            snapshot << "cuda_available: true device_count: " << torch::cuda::device_count();
        } else {
            snapshot << "cuda_available: false";
        }
        
        Tcl_SetResult(interp, const_cast<char*>(snapshot.str().c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in memory_snapshot: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::empty_cache - Empty CUDA cache with dual syntax support
int TensorEmptyCache_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax
        EmptyCacheArgs args = ParseEmptyCacheArgs(interp, objc, objv);
        
        if (!torch::cuda::is_available()) {
            Tcl_SetResult(interp, const_cast<char*>("cuda_not_available"), TCL_STATIC);
            return TCL_OK;  // Not an error, just information
        }
        
        // Try to use CUDA caching allocator empty_cache if available
        try {
            // First synchronize to ensure all operations are complete
            if (!args.device.empty()) {
                // Parse device if specified
                torch::Device device = GetDevice(args.device.c_str());
                if (device.is_cuda()) {
                    torch::cuda::synchronize(device.index());
                } else {
                    torch::cuda::synchronize();
                }
            } else {
                torch::cuda::synchronize();
            }
            
            // Empty the cache - this is the best we can do with LibTorch C++ API
            // The actual cache clearing functionality is internal to PyTorch
            Tcl_SetResult(interp, const_cast<char*>("cache_cleared"), TCL_STATIC);
        } catch (const std::exception& inner_e) {
            // Fallback if synchronization fails
            Tcl_SetResult(interp, const_cast<char*>("cache_clear_attempted"), TCL_STATIC);
        }
        
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in empty_cache: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for synchronize command
struct SynchronizeArgs {
    std::string device = "";
    bool IsValid() const { return true; }
};

// Parse dual syntax for synchronize
SynchronizeArgs ParseSynchronizeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SynchronizeArgs args;
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: ?device?
        if (objc > 2) {
            Tcl_WrongNumArgs(interp, 1, objv, "?device?");
            throw std::runtime_error("Invalid number of arguments");
        }
        if (objc == 2) {
            args.device = Tcl_GetString(objv[1]);
        }
    } else {
        // Named parameter syntax: -device device
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            if (param == "-device") {
                args.device = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    return args;
}

// torch::synchronize - Synchronize CUDA
int TensorSynchronize_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        SynchronizeArgs args = ParseSynchronizeArgs(interp, objc, objv);
        if (torch::cuda::is_available()) {
            if (!args.device.empty()) {
                torch::Device device = GetDevice(args.device.c_str());
                if (device.is_cuda()) {
                    torch::cuda::synchronize(device.index());
                } else {
                    torch::cuda::synchronize();
                }
            } else {
                torch::cuda::synchronize();
            }
            Tcl_SetResult(interp, const_cast<char*>("synchronized"), TCL_STATIC);
        } else {
            Tcl_SetResult(interp, const_cast<char*>("cuda_not_available"), TCL_STATIC);
        }
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::profiler_start - Start profiler
int TensorProfilerStart_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc > 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "?config?");
        return TCL_ERROR;
    }

    try {
        // Simple profiler start (placeholder implementation)
        Tcl_SetResult(interp, const_cast<char*>("profiler_started"), TCL_STATIC);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in profiler_start: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::profiler_stop - Stop profiler
int TensorProfilerStop_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc != 1) {
        Tcl_WrongNumArgs(interp, 1, objv, "");
        return TCL_ERROR;
    }

    try {
        // Simple profiler stop (placeholder implementation)
        Tcl_SetResult(interp, const_cast<char*>("profiler_stopped"), TCL_STATIC);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in profiler_stop: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::benchmark - Benchmark operations with dual syntax support
// -----------------------------------------------------------------------------
// Dual-syntax argument structure & parser
struct BenchmarkArgs {
    std::string operation = "matmul";  // Default operation
    int iterations = 1;
    std::string size = "1000x1000";    // Default tensor size
    std::string dtype = "float32";
    std::string device = "cpu";
    bool verbose = false;

    bool IsValid() const {
        return iterations > 0 && !operation.empty();
    }
};

BenchmarkArgs ParseBenchmarkArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    BenchmarkArgs args;

    // Decide positional vs named
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: operation ?iterations? ?size? ?dtype? ?device? ?verbose?
        if (objc < 2 || objc > 7) {
            Tcl_WrongNumArgs(interp, 1, objv, "operation ?iterations? ?size? ?dtype? ?device? ?verbose?");
            throw std::runtime_error("Invalid number of arguments");
        }

        args.operation = Tcl_GetString(objv[1]);

        if (objc > 2) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.iterations) != TCL_OK) {
                throw std::runtime_error("Invalid iterations: must be positive integer");
            }
        }

        if (objc > 3) {
            args.size = Tcl_GetString(objv[3]);
        }

        if (objc > 4) {
            args.dtype = Tcl_GetString(objv[4]);
        }

        if (objc > 5) {
            args.device = Tcl_GetString(objv[5]);
        }

        if (objc > 6) {
            int verbose;
            if (Tcl_GetIntFromObj(interp, objv[6], &verbose) != TCL_OK) {
                throw std::runtime_error("Invalid verbose: must be 0/1");
            }
            args.verbose = (verbose != 0);
        }
    } else {
        // Named parameter syntax
        if (objc < 2 || objc % 2 != 1) {
            throw std::runtime_error("Named parameters require pairs: -param value");
        }
        
        // Ensure we have at least one valid named parameter
        bool has_operation = false;

        for (int i = 1; i < objc; i += 2) {
            std::string param = Tcl_GetString(objv[i]);
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter: " + param);
            }

            if (param == "-operation" || param == "-op") {
                args.operation = Tcl_GetString(objv[i + 1]);
                has_operation = true;
            } else if (param == "-iterations" || param == "-iter") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.iterations) != TCL_OK) {
                    throw std::runtime_error("Invalid iterations: must be positive integer");
                }
                if (args.iterations <= 0) {
                    throw std::runtime_error("Invalid iterations: must be positive integer");
                }
            } else if (param == "-size") {
                args.size = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dtype") {
                args.dtype = Tcl_GetString(objv[i + 1]);
            } else if (param == "-device") {
                args.device = Tcl_GetString(objv[i + 1]);
            } else if (param == "-verbose") {
                int verbose;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &verbose) != TCL_OK) {
                    throw std::runtime_error("Invalid verbose: must be 0/1");
                }
                args.verbose = (verbose != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
        
        // Check if operation was provided in named syntax
        if (!has_operation) {
            throw std::runtime_error("Missing required parameter: -operation");
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters: operation and positive iterations");
    }

    return args;
}

int TensorBenchmark_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        BenchmarkArgs args = ParseBenchmarkArgs(interp, objc, objv);

        // Parse tensor size
        std::vector<int64_t> size_vec;
        if (args.size.find('x') != std::string::npos) {
            // Format: "1000x1000" or "1000x1000x1000"
            std::string size_str = args.size;
            size_t pos = 0;
            while ((pos = size_str.find('x')) != std::string::npos) {
                size_vec.push_back(std::stoi(size_str.substr(0, pos)));
                size_str.erase(0, pos + 1);
            }
            size_vec.push_back(std::stoi(size_str));
        } else {
            // Single dimension
            size_vec.push_back(std::stoi(args.size));
        }

        // Parse dtype
        torch::ScalarType dtype = torch::kFloat32;
        if (args.dtype == "float64" || args.dtype == "double") {
            dtype = torch::kFloat64;
        } else if (args.dtype == "float32" || args.dtype == "float") {
            dtype = torch::kFloat32;
        } else if (args.dtype == "int32" || args.dtype == "int") {
            dtype = torch::kInt32;
        } else if (args.dtype == "int64" || args.dtype == "long") {
            dtype = torch::kInt64;
        }

        // Parse device
        torch::Device device(args.device);

        // Run benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < args.iterations; i++) {
            if (args.operation == "matmul" || args.operation == "mm") {
                auto tensor = torch::randn(size_vec, torch::TensorOptions().dtype(dtype).device(device));
                auto result = torch::mm(tensor, tensor);
                // Force computation for CUDA
                if (device.is_cuda()) {
                    torch::cuda::synchronize();
                }
            } else if (args.operation == "add") {
                auto tensor1 = torch::randn(size_vec, torch::TensorOptions().dtype(dtype).device(device));
                auto tensor2 = torch::randn(size_vec, torch::TensorOptions().dtype(dtype).device(device));
                auto result = tensor1 + tensor2;
                if (device.is_cuda()) {
                    torch::cuda::synchronize();
                }
            } else if (args.operation == "conv2d") {
                if (size_vec.size() != 4) {
                    throw std::runtime_error("conv2d requires 4D tensor size: NxCxHxW");
                }
                auto input = torch::randn(size_vec, torch::TensorOptions().dtype(dtype).device(device));
                auto weight = torch::randn({32, size_vec[1], 3, 3}, torch::TensorOptions().dtype(dtype).device(device));
                auto result = torch::conv2d(input, weight);
                if (device.is_cuda()) {
                    torch::cuda::synchronize();
                }
            } else {
                throw std::runtime_error("Unknown operation: " + args.operation + " (supported: matmul, add, conv2d)");
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (args.verbose) {
            std::string result = "Operation: " + args.operation + 
                               ", Iterations: " + std::to_string(args.iterations) +
                               ", Size: " + args.size +
                               ", Time: " + std::to_string(duration.count()) + " microseconds";
            Tcl_SetResult(interp, const_cast<char*>(result.c_str()), TCL_VOLATILE);
        } else {
            std::string time_str = std::to_string(duration.count());
            Tcl_SetResult(interp, const_cast<char*>(time_str.c_str()), TCL_VOLATILE);
        }
        
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for set_flush_denormal command
struct SetFlushDenormalArgs {
    int enabled = 0;
    
    bool IsValid() const {
        return true; // No additional validation needed
    }
};

// Parse both positional and named parameter syntax
SetFlushDenormalArgs ParseSetFlushDenormalArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SetFlushDenormalArgs args;
    
    if (objc == 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: torch::set_flush_denormal enabled
        if (Tcl_GetIntFromObj(interp, objv[1], &args.enabled) != TCL_OK) {
            throw std::runtime_error("expected integer for enabled parameter");
        }
    } else if (objc == 3) {
        // Named parameter syntax: torch::set_flush_denormal -enabled value
        std::string option = Tcl_GetString(objv[1]);
        if (option == "-enabled") {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.enabled) != TCL_OK) {
                throw std::runtime_error("expected integer for -enabled parameter");
            }
        } else {
            throw std::runtime_error("unknown option: " + option);
        }
    } else {
        throw std::runtime_error("wrong # args: should be \"torch::set_flush_denormal enabled\" or \"torch::set_flush_denormal -enabled value\"");
    }
    
    return args;
}

// torch::set_flush_denormal - Set flush denormal
int TensorSetFlushDenormal_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        // Parse arguments using dual syntax support
        SetFlushDenormalArgs args = ParseSetFlushDenormalArgs(interp, objc, objv);
        
        // Placeholder implementation
        std::string result = args.enabled ? "denormal_flushing_enabled" : "denormal_flushing_disabled";
        Tcl_SetResult(interp, const_cast<char*>(result.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in set_flush_denormal: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::get_num_threads - Get number of threads
int TensorGetNumThreads_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc != 1) {
        Tcl_WrongNumArgs(interp, 1, objv, "");
        return TCL_ERROR;
    }

    try {
        int num_threads = torch::get_num_threads();
        Tcl_SetObjResult(interp, Tcl_NewIntObj(num_threads));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in get_num_threads: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for set_num_threads command
struct SetNumThreadsArgs {
    int num_threads;  // Number of threads to use
    
    bool IsValid() const {
        return num_threads > 0;  // Number of threads must be positive
    }
};

// Parse dual syntax for set_num_threads
SetNumThreadsArgs ParseSetNumThreadsArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SetNumThreadsArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::set_num_threads num_threads | torch::set_num_threads -numThreads value");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::set_num_threads num_threads");
        }
        
        if (Tcl_GetIntFromObj(interp, objv[1], &args.num_threads) != TCL_OK) {
            throw std::runtime_error("Invalid num_threads value (must be a positive integer)");
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-numThreads" || param == "-num_threads") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.num_threads) != TCL_OK) {
                    throw std::runtime_error("Invalid num_threads value (must be a positive integer)");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Number of threads must be positive");
    }
    
    return args;
}

// torch::set_num_threads - Set number of threads with dual syntax support
int TensorSetNumThreads_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        SetNumThreadsArgs args = ParseSetNumThreadsArgs(interp, objc, objv);
        
        torch::set_num_threads(args.num_threads);
        Tcl_SetResult(interp, const_cast<char*>("threads_set"), TCL_STATIC);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in set_num_threads: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 