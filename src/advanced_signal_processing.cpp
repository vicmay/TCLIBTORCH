#include "libtorchtcl.h"
#include <torch/torch.h>
#include <cmath>

// torch::fftshift - Shift zero-frequency component to center with dual syntax support
// -----------------------------------------------------------------------------
// Dual-syntax argument structure & parser
struct FFTShiftArgs {
    std::string input;
    c10::optional<int> dim = c10::nullopt;

    bool IsValid() const {
        return !input.empty();
    }
};

FFTShiftArgs ParseFFTShiftArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    FFTShiftArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::fftshift tensor ?dim? | torch::fftshift -input tensor [-dim dimension]");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        args.input = Tcl_GetString(objv[1]);
        if (objc >= 3) {
            int dim_val;
            if (Tcl_GetIntFromObj(interp, objv[2], &dim_val) != TCL_OK) {
                throw std::runtime_error("Invalid dimension value");
            }
            args.dim = dim_val;
        }
        if (objc > 3) {
            throw std::runtime_error("Too many positional arguments");
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
            } else if (param == "-dim" || param == "-dimension") {
                int dim_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &dim_val) != TCL_OK) {
                    throw std::runtime_error("Invalid dimension value");
                }
                args.dim = dim_val;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor, -dim, -dimension");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: input tensor required");
    }
    
    return args;
}

// torch::fftshift - Shift zero-frequency component to center
int TensorFFTShift_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        FFTShiftArgs args = ParseFFTShiftArgs(interp, objc, objv);
        torch::Tensor tensor = GetTensorFromObj(interp, Tcl_NewStringObj(args.input.c_str(), -1));
        
        torch::Tensor result;
        if (args.dim.has_value()) {
            // Simple implementation - roll by half the dimension size
            int64_t shift = tensor.size(args.dim.value()) / 2;
            result = torch::roll(tensor, shift, args.dim.value());
        } else {
            // Apply to all dimensions
            result = tensor.clone();
            for (int i = 0; i < tensor.dim(); i++) {
                int64_t shift = result.size(i) / 2;
                result = torch::roll(result, shift, i);
            }
        }
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in fftshift: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for torch::ifftshift
// ============================================================================
struct IFFTShiftArgs {
    std::string input;  // tensor handle for input
    c10::optional<int> dim = c10::nullopt;  // optional dimension
    
    bool IsValid() const {
        return !input.empty();
    }
};

IFFTShiftArgs ParseIFFTShiftArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    IFFTShiftArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor ?dim?
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("Usage: torch::ifftshift tensor ?dim?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc == 3) {
            int dim_val;
            if (Tcl_GetIntFromObj(interp, objv[2], &dim_val) != TCL_OK) {
                throw std::runtime_error("Invalid dimension: must be integer");
            }
            args.dim = dim_val;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = value;
            } else if (param == "-dim" || param == "-dimension") {
                int dim_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &dim_val) != TCL_OK) {
                    throw std::runtime_error("Invalid dimension: must be integer");
                }
                args.dim = dim_val;
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

// torch::ifftshift - Inverse FFT shift with dual syntax support
int TensorIFFTShift_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        IFFTShiftArgs args = ParseIFFTShiftArgs(interp, objc, objv);
        
        // Get input tensor from the parsed arguments
        torch::Tensor tensor = GetTensorFromObj(interp, Tcl_NewStringObj(args.input.c_str(), -1));
        
        torch::Tensor result;
        if (args.dim.has_value()) {
            int dim = args.dim.value();
            // Simple implementation - roll by negative half the dimension size
            int64_t shift = -(tensor.size(dim) / 2);
            result = torch::roll(tensor, shift, dim);
        } else {
            // Apply to all dimensions
            result = tensor.clone();
            for (int i = 0; i < tensor.dim(); i++) {
                int64_t shift = -(result.size(i) / 2);
                result = torch::roll(result, shift, i);
            }
        }
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in ifftshift: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for torch::hilbert
// ============================================================================
struct HilbertArgs {
    std::string input;  // tensor handle for input
    
    bool IsValid() const {
        return !input.empty();
    }
};

HilbertArgs ParseHilbertArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    HilbertArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::hilbert tensor");
        }
        
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = value;
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

// torch::hilbert - Hilbert transform with dual syntax support
int TensorHilbert_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        HilbertArgs args = ParseHilbertArgs(interp, objc, objv);
        
        // Get input tensor from the parsed arguments
        torch::Tensor tensor = GetTensorFromObj(interp, Tcl_NewStringObj(args.input.c_str(), -1));
        
        // Simplified Hilbert transform implementation using FFT
        auto fft_result = torch::fft::fft(tensor);
        auto n = tensor.size(-1);
        
        // Create Hilbert filter
        auto h = torch::zeros({n}, torch::kComplexFloat);
        h[0] = 1.0;
        if (n % 2 == 0) {
            h.index_put_({torch::arange(1, n/2)}, 2.0);
            h[n/2] = 1.0;
        } else {
            h.index_put_({torch::arange(1, (n+1)/2)}, 2.0);
        }
        
        auto result = torch::fft::ifft(fft_result * h);
        return SetTensorResult(interp, torch::real(result));
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in hilbert: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::bartlett_window - Bartlett window with dual syntax support
// -----------------------------------------------------------------------------
// Dual-syntax argument structure & parser
struct BartlettWindowArgs {
    int window_length = 0;
    std::string dtype = "float32";
    std::string device = "cpu";
    bool periodic = true;

    bool IsValid() const {
        return window_length > 0;
    }
};

BartlettWindowArgs ParseBartlettWindowArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    BartlettWindowArgs args;

    // Decide positional vs named
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: window_length ?dtype? ?device? ?periodic?
        if (objc < 2 || objc > 5) {
            Tcl_WrongNumArgs(interp, 1, objv, "window_length ?dtype? ?device? ?periodic?");
            throw std::runtime_error("Invalid number of arguments");
        }

        if (Tcl_GetIntFromObj(interp, objv[1], &args.window_length) != TCL_OK) {
            throw std::runtime_error("Invalid window_length: must be positive integer");
        }

        if (objc > 2) {
            args.dtype = Tcl_GetString(objv[2]);
        }

        if (objc > 3) {
            args.device = Tcl_GetString(objv[3]);
        }

        if (objc > 4) {
            int periodic;
            if (Tcl_GetIntFromObj(interp, objv[4], &periodic) != TCL_OK) {
                throw std::runtime_error("Invalid periodic: must be 0/1");
            }
            args.periodic = (periodic != 0);
        }
    } else {
        // Named parameter syntax
        if (objc < 2 || objc % 2 != 1) {
            throw std::runtime_error("Named parameters require pairs: -param value");
        }

        for (int i = 1; i < objc; i += 2) {
            std::string param = Tcl_GetString(objv[i]);
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter: " + param);
            }

            if (param == "-window_length" || param == "-windowLength" || param == "-length") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.window_length) != TCL_OK) {
                    throw std::runtime_error("Invalid window_length: must be positive integer");
                }
            } else if (param == "-dtype") {
                args.dtype = Tcl_GetString(objv[i + 1]);
            } else if (param == "-device") {
                args.device = Tcl_GetString(objv[i + 1]);
            } else if (param == "-periodic") {
                int periodic;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &periodic) != TCL_OK) {
                    throw std::runtime_error("Invalid periodic: must be 0/1");
                }
                args.periodic = (periodic != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters: window_length must be positive");
    }

    return args;
}

int TensorBartlettWindow_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        BartlettWindowArgs args = ParseBartlettWindowArgs(interp, objc, objv);

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
        } else if (args.dtype != "float32") {
            throw std::runtime_error("Unsupported dtype: " + args.dtype);
        }

        // Parse device
        torch::Device device(args.device);

        // Create bartlett window
        auto options = torch::TensorOptions().dtype(dtype).device(device);
        auto window = torch::bartlett_window(args.window_length, args.periodic, options);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = window;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for blackman_window command
struct BlackmanWindowArgs {
    int window_length = 0;
    std::string dtype = "float32";
    std::string device = "cpu";
    bool periodic = true;

    bool IsValid() const {
        return window_length > 0;
    }
};

// Parse dual syntax for blackman_window
BlackmanWindowArgs ParseBlackmanWindowArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    BlackmanWindowArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::blackman_window window_length | torch::blackman_window -window_length length");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 5) {
            throw std::runtime_error("Usage: torch::blackman_window window_length ?dtype? ?device? ?periodic?");
        }
        args.window_length = GetIntFromObj(interp, objv[1]);
        
        if (objc > 2) args.dtype = Tcl_GetString(objv[2]);
        if (objc > 3) args.device = Tcl_GetString(objv[3]);
        if (objc > 4) {
            int periodic;
            if (Tcl_GetBooleanFromObj(interp, objv[4], &periodic) != TCL_OK) {
                throw std::runtime_error("Invalid periodic parameter");
            }
            args.periodic = periodic != 0;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-window_length" || param == "-length") {
                args.window_length = std::stoi(value);
            } else if (param == "-dtype") {
                args.dtype = value;
            } else if (param == "-device") {
                args.device = value;
            } else if (param == "-periodic") {
                int periodic;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &periodic) != TCL_OK) {
                    throw std::runtime_error("Invalid periodic parameter");
                }
                args.periodic = periodic != 0;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -window_length, -length, -dtype, -device, -periodic");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: window_length must be positive");
    }
    
    return args;
}

// torch::blackman_window - Blackman window
int TensorBlackmanWindow_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        BlackmanWindowArgs args = ParseBlackmanWindowArgs(interp, objc, objv);
        
        if (args.window_length <= 0) {
            Tcl_SetResult(interp, const_cast<char*>("Window length must be positive"), TCL_STATIC);
            return TCL_ERROR;
        }
        
        auto window = torch::blackman_window(args.window_length);
        return SetTensorResult(interp, window);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in blackman_window: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for hamming_window command
struct HammingWindowArgs {
    int window_length = 0;
    std::string dtype = "float32";
    std::string device = "cpu";
    bool periodic = true;
    
    bool IsValid() const {
        return window_length > 0;
    }
};

// Parse dual syntax for hamming_window
HammingWindowArgs ParseHammingWindowArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    HammingWindowArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::hamming_window window_length | torch::hamming_window -length window_length [-dtype dtype] [-device device] [-periodic bool]");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::hamming_window window_length");
        }
        args.window_length = GetIntFromObj(interp, objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-length" || param == "-window_length") {
                args.window_length = GetIntFromObj(interp, objv[i + 1]);
            } else if (param == "-dtype") {
                args.dtype = Tcl_GetString(objv[i + 1]);
            } else if (param == "-device") {
                args.device = Tcl_GetString(objv[i + 1]);
            } else if (param == "-periodic") {
                int periodic;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &periodic) != TCL_OK) {
                    throw std::runtime_error("Invalid periodic parameter");
                }
                args.periodic = periodic != 0;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -length, -window_length, -dtype, -device, -periodic");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: window_length must be positive");
    }
    
    return args;
}

// torch::hamming_window - Hamming window
int TensorHammingWindow_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        HammingWindowArgs args = ParseHammingWindowArgs(interp, objc, objv);
        
        if (args.window_length <= 0) {
            Tcl_SetResult(interp, const_cast<char*>("Window length must be positive"), TCL_STATIC);
            return TCL_ERROR;
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
        } else if (args.dtype != "float32") {
            throw std::runtime_error("Unsupported dtype: " + args.dtype);
        }

        // Parse device
        torch::Device device(args.device);

        // Create hamming window
        auto options = torch::TensorOptions().dtype(dtype).device(device);
        auto window = torch::hamming_window(args.window_length, args.periodic, options);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = window;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in hamming_window: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for hann_window command
struct HannWindowArgs {
    int window_length = 0;
    std::string dtype = "float32";
    std::string device = "cpu";
    bool periodic = true;
    
    bool IsValid() const {
        return window_length > 0;
    }
};

// Parse dual syntax for hann_window
HannWindowArgs ParseHannWindowArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    HannWindowArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::hann_window window_length | torch::hann_window -length window_length [-dtype dtype] [-device device] [-periodic bool]");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::hann_window window_length");
        }
        args.window_length = GetIntFromObj(interp, objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-length" || param == "-window_length") {
                args.window_length = GetIntFromObj(interp, objv[i + 1]);
            } else if (param == "-dtype") {
                args.dtype = Tcl_GetString(objv[i + 1]);
            } else if (param == "-device") {
                args.device = Tcl_GetString(objv[i + 1]);
            } else if (param == "-periodic") {
                int periodic;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &periodic) != TCL_OK) {
                    throw std::runtime_error("Invalid periodic parameter");
                }
                args.periodic = periodic != 0;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -length, -window_length, -dtype, -device, -periodic");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: window_length must be positive");
    }
    
    return args;
}

// torch::hann_window - Hann window
int TensorHannWindow_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        HannWindowArgs args = ParseHannWindowArgs(interp, objc, objv);
        
        if (args.window_length <= 0) {
            Tcl_SetResult(interp, const_cast<char*>("Window length must be positive"), TCL_STATIC);
            return TCL_ERROR;
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
        } else if (args.dtype != "float32") {
            throw std::runtime_error("Unsupported dtype: " + args.dtype);
        }

        // Parse device
        torch::Device device(args.device);

        // Create hann window
        auto options = torch::TensorOptions().dtype(dtype).device(device);
        auto window = torch::hann_window(args.window_length, args.periodic, options);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = window;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in hann_window: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct KaiserWindowArgs {
    int window_length = 0;
    double beta = 12.0;
    std::string dtype = "float32";
    std::string device = "cpu";
    bool periodic = true;
    
    bool IsValid() const {
        return window_length > 0;
    }
};

KaiserWindowArgs ParseKaiserWindowArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    KaiserWindowArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 3) {
            throw std::runtime_error("Usage: kaiser_window window_length ?beta?");
        }
        
        args.window_length = GetIntFromObj(interp, objv[1]);
        
        if (objc == 3) {
            args.beta = GetDoubleFromObj(interp, objv[2]);
        }
    } else {
        // Named parameter syntax
        if (objc < 3) {
            throw std::runtime_error("Usage: kaiser_window -windowLength length [-beta value] [-dtype type] [-device device] [-periodic bool]");
        }
        
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter: " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-windowLength") {
                args.window_length = GetIntFromObj(interp, objv[i + 1]);
            } else if (param == "-beta") {
                args.beta = GetDoubleFromObj(interp, objv[i + 1]);
            } else if (param == "-dtype") {
                args.dtype = Tcl_GetString(objv[i + 1]);
            } else if (param == "-device") {
                args.device = Tcl_GetString(objv[i + 1]);
            } else if (param == "-periodic") {
                args.periodic = GetBoolFromObj(interp, objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Window length must be positive");
    }
    
    return args;
}

// torch::kaiser_window - Kaiser window
int TensorKaiserWindow_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        KaiserWindowArgs args = ParseKaiserWindowArgs(interp, objc, objv);
        
        // Create Kaiser window manually since PyTorch may not have this function
        auto n = torch::arange(args.window_length, torch::kFloat32);
        auto alpha = (args.window_length - 1) / 2.0;
        auto window = torch::special::i0(args.beta * torch::sqrt(1 - torch::pow((n - alpha) / alpha, 2))) / 
                     torch::special::i0(torch::tensor(args.beta));
        
        return SetTensorResult(interp, window);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in kaiser_window: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for spectrogram command
struct SpectrogramArgs {
    torch::Tensor input;
    int n_fft = 32;  // Changed from 400 to 32
    int hop_length = 16;  // Changed from 200 to 16
    int win_length = 32;  // Changed from 400 to 32
    torch::Tensor window;  // Optional, will be set to Hann window if not provided
    
    bool IsValid() const {
        return input.defined() && n_fft > 0 && hop_length > 0 && win_length > 0;
    }
};

// Parse dual syntax for spectrogram
SpectrogramArgs ParseSpectrogramArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SpectrogramArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Wrong number of arguments");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc > 6) {
            throw std::runtime_error("Wrong number of arguments");
        }
        
        args.input = GetTensorFromObj(interp, objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.n_fft) != TCL_OK) {
                throw std::runtime_error("Invalid n_fft value");
            }
        }
        
        if (objc > 3) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.hop_length) != TCL_OK) {
                throw std::runtime_error("Invalid hop_length value");
            }
        }
        
        if (objc > 4) {
            if (Tcl_GetIntFromObj(interp, objv[4], &args.win_length) != TCL_OK) {
                throw std::runtime_error("Invalid win_length value");
            }
        }
        
        if (objc > 5) {
            args.window = GetTensorFromObj(interp, objv[5]);
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
            } else if (param == "-nFft" || param == "-n_fft") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.n_fft) != TCL_OK) {
                    throw std::runtime_error("Invalid n_fft value");
                }
            } else if (param == "-hopLength" || param == "-hop_length") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.hop_length) != TCL_OK) {
                    throw std::runtime_error("Invalid hop_length value");
                }
            } else if (param == "-winLength" || param == "-win_length") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.win_length) != TCL_OK) {
                    throw std::runtime_error("Invalid win_length value");
                }
            } else if (param == "-window") {
                args.window = GetTensorFromObj(interp, objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    // Validate parameters
    if (!args.input.defined()) {
        throw std::runtime_error("Invalid tensor");
    }
    if (args.n_fft <= 0) {
        throw std::runtime_error("n_fft must be positive");
    }
    if (args.hop_length <= 0) {
        throw std::runtime_error("hop_length must be positive");
    }
    if (args.win_length <= 0) {
        throw std::runtime_error("win_length must be positive");
    }
    
    // Create default window if not provided
    if (!args.window.defined()) {
        args.window = torch::hann_window(args.win_length);
    }
    
    return args;
}

// torch::spectrogram - Compute spectrogram
int TensorSpectrogram_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        SpectrogramArgs args = ParseSpectrogramArgs(interp, objc, objv);
        
        // Compute STFT
        auto stft_result = torch::stft(args.input, args.n_fft, args.hop_length, args.win_length, args.window, 
                                     false, "reflect", false, c10::nullopt, true);
        
        // Compute magnitude spectrogram
        auto spectrogram = torch::pow(torch::abs(stft_result), 2);
        
        return SetTensorResult(interp, spectrogram);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in spectrogram: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for melscale_fbanks command
struct MelscaleFbanksArgs {
    int n_freqs = 0;
    int n_mels = 0;
    double sample_rate = 0.0;
    double f_min = 0.0;
    double f_max = 0.0;  // Will be set to sample_rate/2 if not specified
    
    bool IsValid() const {
        return n_freqs > 0 && n_mels > 0 && sample_rate > 0;
    }
};

// Parser function for dual syntax support
MelscaleFbanksArgs ParseMelscaleFbanksArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MelscaleFbanksArgs args;
    
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "n_freqs n_mels sample_rate ?f_min? ?f_max? OR -nFreqs int -nMels int -sampleRate double ?-fMin double? ?-fMax double?");
        throw std::runtime_error("Insufficient arguments");
    }
    
    // Check if using positional or named syntax
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 4 || objc > 6) {
            Tcl_WrongNumArgs(interp, 1, objv, "n_freqs n_mels sample_rate ?f_min? ?f_max?");
            throw std::runtime_error("Wrong number of arguments for positional syntax");
        }
        
        args.n_freqs = GetIntFromObj(interp, objv[1]);
        args.n_mels = GetIntFromObj(interp, objv[2]);
        args.sample_rate = GetDoubleFromObj(interp, objv[3]);
        
        // Set default f_max based on sample_rate
        args.f_max = args.sample_rate / 2.0;
        
        // Optional parameters
        if (objc > 4) args.f_min = GetDoubleFromObj(interp, objv[4]);
        if (objc > 5) args.f_max = GetDoubleFromObj(interp, objv[5]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-nFreqs" || param == "-n_freqs") {
                args.n_freqs = GetIntFromObj(interp, objv[i + 1]);
            } else if (param == "-nMels" || param == "-n_mels") {
                args.n_mels = GetIntFromObj(interp, objv[i + 1]);
            } else if (param == "-sampleRate" || param == "-sample_rate") {
                args.sample_rate = GetDoubleFromObj(interp, objv[i + 1]);
            } else if (param == "-fMin" || param == "-f_min") {
                args.f_min = GetDoubleFromObj(interp, objv[i + 1]);
            } else if (param == "-fMax" || param == "-f_max") {
                args.f_max = GetDoubleFromObj(interp, objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
        
        // Set default f_max if not specified
        if (args.f_max == 0.0 && args.sample_rate > 0) {
            args.f_max = args.sample_rate / 2.0;
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Invalid parameters: n_freqs, n_mels, and sample_rate must be positive");
    }
    
    return args;
}

// torch::melscale_fbanks - Mel-scale filter banks
int TensorMelscaleFbanks_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        // Parse arguments using dual syntax parser
        MelscaleFbanksArgs args = ParseMelscaleFbanksArgs(interp, objc, objv);
        
        // Extract parameters from the args structure
        int n_freqs = args.n_freqs;
        int n_mels = args.n_mels;
        double sample_rate = args.sample_rate;
        double f_min = args.f_min;
        double f_max = args.f_max;
        
        // Create mel filter bank
        auto mel_filters = torch::zeros({n_mels, n_freqs});
        
        // Convert Hz to mel scale
        auto hz_to_mel = [](double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); };
        auto mel_to_hz = [](double mel) { return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0); };
        
        double mel_min = hz_to_mel(f_min);
        double mel_max = hz_to_mel(f_max);
        
        // Create mel points
        std::vector<double> mel_points(n_mels + 2);
        for (int i = 0; i < n_mels + 2; i++) {
            mel_points[i] = mel_min + i * (mel_max - mel_min) / (n_mels + 1);
        }
        
        // Convert back to Hz
        std::vector<int> bin_points(n_mels + 2);
        for (int i = 0; i < n_mels + 2; i++) {
            double hz = mel_to_hz(mel_points[i]);
            bin_points[i] = static_cast<int>(std::floor((n_freqs + 1) * hz / sample_rate));
        }
        
        // Create triangular filters
        for (int m = 1; m <= n_mels; m++) {
            int left = bin_points[m - 1];
            int center = bin_points[m];
            int right = bin_points[m + 1];
            
            for (int k = left; k < center; k++) {
                if (center != left) {
                    mel_filters[m-1][k] = static_cast<double>(k - left) / (center - left);
                }
            }
            for (int k = center; k < right; k++) {
                if (right != center) {
                    mel_filters[m-1][k] = static_cast<double>(right - k) / (right - center);
                }
            }
        }
        
        return SetTensorResult(interp, mel_filters);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in melscale_fbanks: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for mfcc command
struct MFCCArgs {
    std::string spectrogram;
    int n_mfcc = 13;
    int dct_type = 2;
    
    bool IsValid() const {
        return !spectrogram.empty();
    }
};

// Parse dual syntax for mfcc
MFCCArgs ParseMFCCArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MFCCArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 2 || objc > 4) {
            throw std::runtime_error("Usage: torch::mfcc spectrogram ?n_mfcc? ?dct_type?");
        }
        
        args.spectrogram = Tcl_GetString(objv[1]);
        
        if (objc > 2) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.n_mfcc) != TCL_OK) {
                throw std::runtime_error("Invalid n_mfcc value");
            }
        }
        
        if (objc > 3) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.dct_type) != TCL_OK) {
                throw std::runtime_error("Invalid dct_type value");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-spectrogram") {
                args.spectrogram = Tcl_GetString(objv[i + 1]);
            } else if (param == "-nMfcc" || param == "-n_mfcc") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.n_mfcc) != TCL_OK) {
                    throw std::runtime_error("Invalid n_mfcc value");
                }
            } else if (param == "-dctType" || param == "-dct_type") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dct_type) != TCL_OK) {
                    throw std::runtime_error("Invalid dct_type value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Spectrogram tensor is required");
    }
    
    return args;
}

// torch::mfcc - Mel-frequency cepstral coefficients
int TensorMFCC_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        MFCCArgs args = ParseMFCCArgs(interp, objc, objv);
        
        // Get the spectrogram tensor
        if (tensor_storage.find(args.spectrogram) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid spectrogram tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        torch::Tensor spectrogram = tensor_storage[args.spectrogram];
        
        // Apply log to mel spectrogram
        auto log_mel = torch::log(torch::clamp(spectrogram, 1e-10));
        
        // Apply DCT
        int n_mels = log_mel.size(-2);
        auto dct_matrix = torch::zeros({args.n_mfcc, n_mels});
        
        for (int k = 0; k < args.n_mfcc; k++) {
            for (int n = 0; n < n_mels; n++) {
                dct_matrix[k][n] = std::cos(M_PI * k * (2 * n + 1) / (2 * n_mels));
            }
        }
        
        auto mfcc = torch::matmul(dct_matrix, log_mel);
        
        // Create a new tensor handle and store the result
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = mfcc;
        
        Tcl_SetResult(interp, const_cast<char*>(handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in mfcc: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for pitch_shift command
struct PitchShiftArgs {
    std::string waveform;
    double sample_rate = 0.0;
    double n_steps = 0.0;
    
    bool IsValid() const {
        return !waveform.empty() && sample_rate > 0.0;
    }
};

// Parse dual syntax for pitch_shift
PitchShiftArgs ParsePitchShiftArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    PitchShiftArgs args;
    
    if (objc < 4) {
        throw std::runtime_error("Usage: torch::pitch_shift waveform sample_rate n_steps | torch::pitch_shift -waveform tensor -sampleRate value -nSteps value");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 4) {
            throw std::runtime_error("Usage: torch::pitch_shift waveform sample_rate n_steps");
        }
        
        args.waveform = Tcl_GetString(objv[1]);
        
        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.sample_rate) != TCL_OK) {
            throw std::runtime_error("Invalid sample_rate parameter");
        }
        
        if (Tcl_GetDoubleFromObj(interp, objv[3], &args.n_steps) != TCL_OK) {
            throw std::runtime_error("Invalid n_steps parameter");
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-waveform" || param == "-input") {
                args.waveform = Tcl_GetString(objv[i + 1]);
            } else if (param == "-sampleRate" || param == "-sample_rate") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.sample_rate) != TCL_OK) {
                    throw std::runtime_error("Invalid sample_rate parameter");
                }
            } else if (param == "-nSteps" || param == "-n_steps") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.n_steps) != TCL_OK) {
                    throw std::runtime_error("Invalid n_steps parameter");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid (waveform and sample_rate required)");
    }
    
    return args;
}

// torch::pitch_shift - Pitch shifting with dual syntax support
int TensorPitchShift_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        PitchShiftArgs args = ParsePitchShiftArgs(interp, objc, objv);
        
        // Get input tensor
        torch::Tensor waveform = GetTensorFromObj(interp, Tcl_NewStringObj(args.waveform.c_str(), -1));
        
        // Simple pitch shift using time stretching and resampling
        double rate = std::pow(2.0, args.n_steps / 12.0);
        
        // Time stretch by 1/rate, then resample by rate
        // This is a simplified implementation
        auto stretched = waveform; // Placeholder - would need actual time stretching
        
        // Simple linear interpolation for resampling
        auto result = torch::nn::functional::interpolate(
            stretched.unsqueeze(0).unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{static_cast<int64_t>(waveform.size(0) / rate)})
                .mode(torch::kLinear)
                .align_corners(false)
        ).squeeze();
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in pitch_shift: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for time_stretch command
struct TimeStretchArgs {
    std::string input;
    double rate;
    
    bool IsValid() const {
        return !input.empty() && rate > 0.0;
    }
};

// Parse dual syntax for time_stretch
static TimeStretchArgs ParseTimeStretchArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TimeStretchArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: stft_matrix rate
        if (objc != 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "stft_matrix rate");
            throw std::runtime_error("Usage: torch::time_stretch stft_matrix rate");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.rate) != TCL_OK) {
            throw std::runtime_error("Invalid rate value");
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-stft_matrix") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-rate") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.rate) != TCL_OK) {
                    throw std::runtime_error("Invalid rate value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -rate");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor required, rate must be positive");
    }
    
    return args;
}

// torch::time_stretch - Time stretching
int TensorTimeStretch_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        TimeStretchArgs args = ParseTimeStretchArgs(interp, objc, objv);
        
        torch::Tensor stft_matrix = GetTensorFromObj(interp, Tcl_NewStringObj(args.input.c_str(), -1));
        
        // Phase vocoder time stretching
        auto magnitude = torch::abs(stft_matrix);
        auto phase = torch::angle(stft_matrix);
        
        // Simple time stretching by interpolating magnitude and adjusting phase
        int new_length = static_cast<int>(stft_matrix.size(-1) / args.rate);
        
        auto stretched_mag = torch::nn::functional::interpolate(
            magnitude.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{new_length})
                .mode(torch::kLinear)
                .align_corners(false)
        ).squeeze(0);
        
        // Phase adjustment (simplified)
        auto stretched_phase = torch::nn::functional::interpolate(
            phase.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{new_length})
                .mode(torch::kLinear)
                .align_corners(false)
        ).squeeze(0);
        
        // Simple reconstruction - just return magnitude (complex reconstruction is complex)
        auto result = stretched_mag;
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in time_stretch: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 