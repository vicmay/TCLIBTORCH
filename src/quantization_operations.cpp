#include "libtorchtcl.h"

// ============================================================================
// QUANTIZATION OPERATIONS - BATCH IMPLEMENTATION OF 20 OPERATIONS
// ============================================================================

// torch::quantize_per_tensor - Per-tensor quantization
int TensorQuantizePerTensor_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc != 5) {
        Tcl_WrongNumArgs(interp, 1, objv, "input scale zero_point dtype");
        return TCL_ERROR;
    }

    try {
        std::string input_name = Tcl_GetString(objv[1]);
        if (tensor_storage.find(input_name) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[input_name];
        
        double scale;
        if (Tcl_GetDoubleFromObj(interp, objv[2], &scale) != TCL_OK) {
            return TCL_ERROR;
        }
        
        int zero_point;
        if (Tcl_GetIntFromObj(interp, objv[3], &zero_point) != TCL_OK) {
            return TCL_ERROR;
        }
        
        c10::ScalarType dtype = GetScalarType(Tcl_GetString(objv[4]));
        auto output = torch::quantize_per_tensor(input, scale, zero_point, dtype);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::quantize_per_channel - Per-channel quantization
int TensorQuantizePerChannel_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc != 6) {
        Tcl_WrongNumArgs(interp, 1, objv, "input scales zero_points axis dtype");
        return TCL_ERROR;
    }

    try {
        std::string input_name = Tcl_GetString(objv[1]);
        if (tensor_storage.find(input_name) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[input_name];
        
        std::string scales_name = Tcl_GetString(objv[2]);
        if (tensor_storage.find(scales_name) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid scales tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        auto scales = tensor_storage[scales_name];
        
        std::string zero_points_name = Tcl_GetString(objv[3]);
        if (tensor_storage.find(zero_points_name) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid zero_points tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        auto zero_points = tensor_storage[zero_points_name];
        
        int axis;
        if (Tcl_GetIntFromObj(interp, objv[4], &axis) != TCL_OK) {
            return TCL_ERROR;
        }
        
        c10::ScalarType dtype = GetScalarType(Tcl_GetString(objv[5]));
        auto output = torch::quantize_per_channel(input, scales, zero_points, axis, dtype);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for dequantize command
struct DequantizeArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for dequantize
DequantizeArgs ParseDequantizeArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DequantizeArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::dequantize quantized_tensor | torch::dequantize -input quantized_tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::dequantize quantized_tensor");
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
        throw std::runtime_error("Required parameter missing: -input");
    }
    
    return args;
}

// torch::dequantize - Dequantization  
int TensorDequantize_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        DequantizeArgs args = ParseDequantizeArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid quantized tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto output = torch::dequantize(input);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for fake_quantize_per_tensor command
struct FakeQuantizePerTensorArgs {
    std::string input;
    double scale = 1.0;
    int zero_point = 0;
    int quant_min = -128;
    int quant_max = 127;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for fake_quantize_per_tensor
FakeQuantizePerTensorArgs ParseFakeQuantizePerTensorArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    FakeQuantizePerTensorArgs args;
    
    if (objc < 4) {
        throw std::runtime_error("Usage: torch::fake_quantize_per_tensor input scale zero_point ?quant_min? ?quant_max? | torch::fake_quantize_per_tensor -input input -scale scale -zero_point zero_point ?-quant_min min? ?-quant_max max?");
    }
    
    if (objc >= 4 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 4 || objc > 6) {
            throw std::runtime_error("Usage: torch::fake_quantize_per_tensor input scale zero_point ?quant_min? ?quant_max?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.scale) != TCL_OK) {
            throw std::runtime_error("Invalid scale value. Expected double.");
        }
        
        if (Tcl_GetIntFromObj(interp, objv[3], &args.zero_point) != TCL_OK) {
            throw std::runtime_error("Invalid zero_point value. Expected integer.");
        }
        
        if (objc >= 5) {
            if (Tcl_GetIntFromObj(interp, objv[4], &args.quant_min) != TCL_OK) {
                throw std::runtime_error("Invalid quant_min value. Expected integer.");
            }
        }
        
        if (objc >= 6) {
            if (Tcl_GetIntFromObj(interp, objv[5], &args.quant_max) != TCL_OK) {
                throw std::runtime_error("Invalid quant_max value. Expected integer.");
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
            } else if (param == "-scale") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.scale) != TCL_OK) {
                    throw std::runtime_error("Invalid scale value. Expected double.");
                }
            } else if (param == "-zero_point" || param == "-zeroPoint") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.zero_point) != TCL_OK) {
                    throw std::runtime_error("Invalid zero_point value. Expected integer.");
                }
            } else if (param == "-quant_min" || param == "-quantMin") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.quant_min) != TCL_OK) {
                    throw std::runtime_error("Invalid quant_min value. Expected integer.");
                }
            } else if (param == "-quant_max" || param == "-quantMax") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.quant_max) != TCL_OK) {
                    throw std::runtime_error("Invalid quant_max value. Expected integer.");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -scale, -zero_point/-zeroPoint, -quant_min/-quantMin, -quant_max/-quantMax");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -input, -scale, and -zero_point are required");
    }
    
    return args;
}

// torch::fake_quantize_per_tensor_affine - Fake quantization per tensor
int TensorFakeQuantizePerTensor_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        FakeQuantizePerTensorArgs args = ParseFakeQuantizePerTensorArgs(interp, objc, objv);
        
        // Validate input tensor
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        auto input = tensor_storage[args.input];

        auto output = torch::fake_quantize_per_tensor_affine(input, args.scale, args.zero_point, args.quant_min, args.quant_max);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::fake_quantize_per_channel_affine - Fake quantization per channel
struct FakeQuantizePerChannelArgs {
    std::string input;
    std::string scales;
    std::string zero_points;
    int axis = 0;
    int quant_min = -128;
    int quant_max = 127;
    
    bool IsValid() const {
        return !input.empty() && !scales.empty() && !zero_points.empty();
    }
};

FakeQuantizePerChannelArgs ParseFakeQuantizePerChannelArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    FakeQuantizePerChannelArgs args;
    
    if (objc < 5) {
        throw std::runtime_error("Usage: torch::fake_quantize_per_channel input scales zero_points axis ?quant_min? ?quant_max? OR torch::fake_quantize_per_channel -input tensor -scales tensor -zero_points tensor -axis int ?-quant_min int? ?-quant_max int? OR with camelCase parameters");
    }
    
    if (objc >= 5 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        args.input = Tcl_GetString(objv[1]);
        args.scales = Tcl_GetString(objv[2]);
        args.zero_points = Tcl_GetString(objv[3]);
        
        if (Tcl_GetIntFromObj(interp, objv[4], &args.axis) != TCL_OK) {
            throw std::runtime_error("Invalid axis value");
        }
        
        if (objc >= 6) {
            if (Tcl_GetIntFromObj(interp, objv[5], &args.quant_min) != TCL_OK) {
                throw std::runtime_error("Invalid quant_min value");
            }
        }
        
        if (objc >= 7) {
            if (Tcl_GetIntFromObj(interp, objv[6], &args.quant_max) != TCL_OK) {
                throw std::runtime_error("Invalid quant_max value");
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
            } else if (param == "-scales") {
                args.scales = Tcl_GetString(objv[i + 1]);
            } else if (param == "-zero_points" || param == "-zeroPoints") {
                args.zero_points = Tcl_GetString(objv[i + 1]);
            } else if (param == "-axis") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.axis) != TCL_OK) {
                    throw std::runtime_error("Invalid axis value");
                }
            } else if (param == "-quant_min" || param == "-quantMin") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.quant_min) != TCL_OK) {
                    throw std::runtime_error("Invalid quant_min value");
                }
            } else if (param == "-quant_max" || param == "-quantMax") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.quant_max) != TCL_OK) {
                    throw std::runtime_error("Invalid quant_max value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input, scales, and zero_points must be specified");
    }
    
    return args;
}
int TensorFakeQuantizePerChannel_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        FakeQuantizePerChannelArgs args = ParseFakeQuantizePerChannelArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            throw std::runtime_error("Invalid input tensor");
        }
        auto input = tensor_storage[args.input];
        
        if (tensor_storage.find(args.scales) == tensor_storage.end()) {
            throw std::runtime_error("Invalid scales tensor");
        }
        auto scales = tensor_storage[args.scales];
        
        if (tensor_storage.find(args.zero_points) == tensor_storage.end()) {
            throw std::runtime_error("Invalid zero_points tensor");
        }
        auto zero_points = tensor_storage[args.zero_points];
        
        int axis = args.axis;
        int quant_min = args.quant_min;
        int quant_max = args.quant_max;

        auto output = torch::fake_quantize_per_channel_affine(input, scales, zero_points, axis, quant_min, quant_max);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for torch::int_repr
// ============================================================================
struct IntReprArgs {
    std::string input;      // quantized tensor handle
    
    bool IsValid() const {
        return !input.empty();
    }
};

IntReprArgs ParseIntReprArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    IntReprArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::int_repr quantized_tensor");
        }
        
        args.input = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
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
        throw std::runtime_error("Required parameters missing: input quantized tensor required");
    }
    
    return args;
}

// torch::int_repr - Get integer representation of quantized tensor with dual syntax support
int TensorIntRepr_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        IntReprArgs args = ParseIntReprArgs(interp, objc, objv);
        
        // Get input tensor from the parsed arguments
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            throw std::runtime_error("Invalid quantized tensor: " + args.input);
        }

        auto input = tensor_storage[args.input];
        auto output = torch::int_repr(input);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in int_repr: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::q_scale - Get scale of quantized tensor
int TensorQScale_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc != 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "quantized_tensor");
        return TCL_ERROR;
    }

    try {
        std::string input_name = Tcl_GetString(objv[1]);
        if (tensor_storage.find(input_name) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid quantized tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[input_name];
        if (!input.is_quantized()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid quantized tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        double scale = input.q_scale();
        
        Tcl_SetObjResult(interp, Tcl_NewDoubleObj(scale));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::q_zero_point - Get zero point of quantized tensor
int TensorQZeroPoint_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc != 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "quantized_tensor");
        return TCL_ERROR;
    }

    try {
        std::string input_name = Tcl_GetString(objv[1]);
        if (tensor_storage.find(input_name) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid quantized tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[input_name];
        if (!input.is_quantized()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid quantized tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        int64_t zero_point = input.q_zero_point();
        
        Tcl_SetObjResult(interp, Tcl_NewLongObj(zero_point));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::q_per_channel_scales - Get per-channel scales
int TensorQPerChannelScales_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc != 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "quantized_tensor");
        return TCL_ERROR;
    }

    try {
        std::string input_name = Tcl_GetString(objv[1]);
        if (tensor_storage.find(input_name) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid quantized tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[input_name];
        if (!input.is_quantized()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid quantized tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto output = input.q_per_channel_scales();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>("Invalid quantized tensor"), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::q_per_channel_zero_points - Get per-channel zero points
int TensorQPerChannelZeroPoints_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc != 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "quantized_tensor");
        return TCL_ERROR;
    }

    try {
        std::string input_name = Tcl_GetString(objv[1]);
        if (tensor_storage.find(input_name) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid quantized tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[input_name];
        if (!input.is_quantized()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid quantized tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto output = input.q_per_channel_zero_points();
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>("Invalid quantized tensor"), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::q_per_channel_axis - Get per-channel quantization axis
int TensorQPerChannelAxis_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc != 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "quantized_tensor");
        return TCL_ERROR;
    }

    try {
        std::string input_name = Tcl_GetString(objv[1]);
        if (tensor_storage.find(input_name) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid quantized tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[input_name];
        if (!input.is_quantized()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid quantized tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        int64_t axis = input.q_per_channel_axis();
        
        Tcl_SetObjResult(interp, Tcl_NewLongObj(axis));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>("Invalid quantized tensor"), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for quantized_add
struct QuantizedAddArgs {
    std::string tensor1;
    std::string tensor2;
    double scale;
    int zero_point;
    double alpha = 1.0;  // Default alpha value
    
    bool IsValid() const {
        return !tensor1.empty() && !tensor2.empty();
    }
};

// Dual syntax parser for quantized_add
QuantizedAddArgs ParseQuantizedAddArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    QuantizedAddArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 5 || objc > 6) {
            throw std::runtime_error("Usage: torch::quantized_add tensor1 tensor2 scale zero_point ?alpha?");
        }
        args.tensor1 = Tcl_GetString(objv[1]);
        args.tensor2 = Tcl_GetString(objv[2]);
        
        if (Tcl_GetDoubleFromObj(interp, objv[3], &args.scale) != TCL_OK) {
            throw std::runtime_error("Invalid scale value");
        }
        if (Tcl_GetIntFromObj(interp, objv[4], &args.zero_point) != TCL_OK) {
            throw std::runtime_error("Invalid zero_point value");
        }
        if (objc > 5) {
            if (Tcl_GetDoubleFromObj(interp, objv[5], &args.alpha) != TCL_OK) {
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
            } else if (param == "-scale") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.scale) != TCL_OK) {
                    throw std::runtime_error("Invalid scale value");
                }
            } else if (param == "-zeroPoint") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.zero_point) != TCL_OK) {
                    throw std::runtime_error("Invalid zeroPoint value");
                }
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
        throw std::runtime_error("Required parameters missing: tensor1, tensor2, scale, zeroPoint");
    }
    
    return args;
}

// Quantized operations - Basic arithmetic
int TensorQuantizedAdd_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 5) {
        Tcl_SetResult(interp, (char*)"Usage: torch::quantized_add tensor1 tensor2 scale zero_point ?alpha?\n"
                      "   or: torch::quantized_add -tensor1 TENSOR -tensor2 TENSOR -scale DOUBLE -zeroPoint INT [-alpha DOUBLE]", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        QuantizedAddArgs args = ParseQuantizedAddArgs(interp, objc, objv);
        
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

        // Use standard add for quantized tensors - PyTorch handles the quantization internally
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

// Parameter structure for quantized_mul
struct QuantizedMulArgs {
    std::string tensor1;
    std::string tensor2;
    double scale;
    int zero_point;
    
    bool IsValid() const {
        return !tensor1.empty() && !tensor2.empty();
    }
};

// Dual syntax parser for quantized_mul
QuantizedMulArgs ParseQuantizedMulArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    QuantizedMulArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 5) {
            throw std::runtime_error("Usage: torch::quantized_mul tensor1 tensor2 scale zero_point");
        }
        args.tensor1 = Tcl_GetString(objv[1]);
        args.tensor2 = Tcl_GetString(objv[2]);
        
        if (Tcl_GetDoubleFromObj(interp, objv[3], &args.scale) != TCL_OK) {
            throw std::runtime_error("Invalid scale value");
        }
        if (Tcl_GetIntFromObj(interp, objv[4], &args.zero_point) != TCL_OK) {
            throw std::runtime_error("Invalid zero_point value");
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
            } else if (param == "-scale") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.scale) != TCL_OK) {
                    throw std::runtime_error("Invalid scale value");
                }
            } else if (param == "-zeroPoint") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.zero_point) != TCL_OK) {
                    throw std::runtime_error("Invalid zeroPoint value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: tensor1, tensor2, scale, zeroPoint");
    }
    
    return args;
}

// Additional quantization utility functions
int TensorQuantizedMul_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 5) {
        Tcl_SetResult(interp, (char*)"Usage: torch::quantized_mul tensor1 tensor2 scale zero_point\n"
                      "   or: torch::quantized_mul -tensor1 TENSOR -tensor2 TENSOR -scale DOUBLE -zeroPoint INT", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        QuantizedMulArgs args = ParseQuantizedMulArgs(interp, objc, objv);
        
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

        // Use standard mul for quantized tensors - PyTorch handles the quantization internally
        auto output = torch::mul(tensor1, tensor2);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for quantized_relu
struct QuantizedReluArgs {
    std::string input;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Dual syntax parser for quantized_relu
QuantizedReluArgs ParseQuantizedReluArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    QuantizedReluArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::quantized_relu quantized_tensor");
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
        throw std::runtime_error("Invalid input: input tensor name must be provided and non-empty");
    }
    
    return args;
}

// Continue with additional quantization functions...
int TensorQuantizedRelu_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc < 2) {
        Tcl_SetResult(interp, (char*)"Usage: torch::quantized_relu quantized_tensor\n"
                      "   or: torch::quantized_relu -input TENSOR", TCL_STATIC);
        return TCL_ERROR;
    }

    try {
        QuantizedReluArgs args = ParseQuantizedReluArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid quantized tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        // Use standard relu for quantized tensors - PyTorch handles the quantization internally  
        auto output = torch::relu(input);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 