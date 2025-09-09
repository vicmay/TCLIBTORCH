#include "libtorchtcl.h"

// Parameter structure for tensor_fft
typedef struct {
    std::string tensor;
    int dim = -1; // -1 means use default (last dimension)
    bool hasDim = false;
    
    bool IsValid() const { return !tensor.empty(); }
} TensorFFTArgs;

static TensorFFTArgs ParseTensorFFTArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorFFTArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor ?dim?
        if (objc < 2 || objc > 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor ?dim?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.tensor = Tcl_GetString(objv[1]);
        
        if (objc == 3) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
                throw std::runtime_error("Invalid dim parameter");
            }
            args.hasDim = true;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-tensor" || param == "-input") {
                args.tensor = value;
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dim parameter");
                }
                args.hasDim = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required tensor parameter missing");
    }
    
    return args;
}

int TensorFFT_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "tensor ?dim? OR -tensor tensor -dim dim");
        return TCL_ERROR;
    }

    try {
        TensorFFTArgs args = ParseTensorFFTArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.tensor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.tensor];
        torch::Tensor result;
        
        if (args.hasDim) {
            result = torch::fft::fft(tensor, c10::nullopt, args.dim);
        } else {
            // Default to last dimension
            result = torch::fft::fft(tensor);
        }
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

typedef struct {
    std::string tensor;
    int dim = -1; // -1 means use default (last dimension)
    bool hasDim = false;
    
    bool IsValid() const { return !tensor.empty(); }
} TensorIFFTArgs;

static TensorIFFTArgs ParseTensorIFFTArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorIFFTArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor ?dim?
        if (objc < 2 || objc > 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor ?dim?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.tensor = Tcl_GetString(objv[1]);
        
        if (objc == 3) {
            if (Tcl_GetIntFromObj(interp, objv[2], &args.dim) != TCL_OK) {
                throw std::runtime_error("Invalid dim parameter");
            }
            args.hasDim = true;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-tensor") {
                args.tensor = value;
            } else if (param == "-dim") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dim) != TCL_OK) {
                    throw std::runtime_error("Invalid dim parameter");
                }
                args.hasDim = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required tensor parameter missing");
    }
    
    return args;
}

int TensorIFFT_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "tensor ?dim? OR -tensor tensor -dim dim");
        return TCL_ERROR;
    }

    try {
        TensorIFFTArgs args = ParseTensorIFFTArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.tensor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.tensor];
        torch::Tensor result;
        
        if (args.hasDim) {
            result = torch::fft::ifft(tensor, c10::nullopt, args.dim);
        } else {
            // Default to last dimension
            result = torch::fft::ifft(tensor);
        }
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

typedef struct {
    std::string tensor;
    std::vector<int64_t> dims;
    bool hasDims = false;
    bool IsValid() const { return !tensor.empty(); }
} TensorFFT2DArgs;

static TensorFFT2DArgs ParseTensorFFT2DArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorFFT2DArgs args;
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor ?dims?
        if (objc < 2 || objc > 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor ?dims?");
            throw std::runtime_error("Invalid number of arguments");
        }
        args.tensor = Tcl_GetString(objv[1]);
        if (objc == 3) {
            args.dims = TclListToShape(interp, objv[2]);
            if (args.dims.size() != 2) {
                throw std::runtime_error("dims must be a list of 2 integers");
            }
            args.hasDims = true;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-tensor") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dims") {
                args.dims = TclListToShape(interp, objv[i + 1]);
                if (args.dims.size() != 2) {
                    throw std::runtime_error("dims must be a list of 2 integers");
                }
                args.hasDims = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    if (!args.IsValid()) {
        throw std::runtime_error("Required tensor parameter missing");
    }
    return args;
}

int TensorFFT2D_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "tensor ?dims? OR -tensor tensor -dims {d0 d1}");
        return TCL_ERROR;
    }
    try {
        TensorFFT2DArgs args = ParseTensorFFT2DArgs(interp, objc, objv);
        if (tensor_storage.find(args.tensor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        auto& tensor = tensor_storage[args.tensor];
        torch::Tensor result;
        if (args.hasDims) {
            result = torch::fft::fft2(tensor, c10::nullopt, args.dims);
        } else {
            // Default to last two dimensions
            result = torch::fft::fft2(tensor);
        }
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

typedef struct {
    std::string tensor;
    std::vector<int64_t> dims;
    bool hasDims = false;
    bool IsValid() const { return !tensor.empty(); }
} TensorIFFT2DArgs;

static TensorIFFT2DArgs ParseTensorIFFT2DArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorIFFT2DArgs args;
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor ?dims?
        if (objc < 2 || objc > 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor ?dims?");
            throw std::runtime_error("Invalid number of arguments");
        }
        args.tensor = Tcl_GetString(objv[1]);
        if (objc == 3) {
            args.dims = TclListToShape(interp, objv[2]);
            if (args.dims.size() != 2) {
                throw std::runtime_error("dims must be a list of 2 integers");
            }
            args.hasDims = true;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-tensor") {
                args.tensor = Tcl_GetString(objv[i + 1]);
            } else if (param == "-dims") {
                args.dims = TclListToShape(interp, objv[i + 1]);
                if (args.dims.size() != 2) {
                    throw std::runtime_error("dims must be a list of 2 integers");
                }
                args.hasDims = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    if (!args.IsValid()) {
        throw std::runtime_error("Required tensor parameter missing");
    }
    return args;
}

int TensorIFFT2D_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "tensor ?dims? OR -tensor tensor -dims {d0 d1}");
        return TCL_ERROR;
    }
    try {
        TensorIFFT2DArgs args = ParseTensorIFFT2DArgs(interp, objc, objv);
        if (tensor_storage.find(args.tensor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        auto& tensor = tensor_storage[args.tensor];
        torch::Tensor result;
        if (args.hasDims) {
            result = torch::fft::ifft2(tensor, c10::nullopt, args.dims);
        } else {
            // Default to last two dimensions
            result = torch::fft::ifft2(tensor);
        }
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

//torch::tensor_conv1d - 1D convolution with dual syntax support
typedef struct {
    torch::Tensor input;
    torch::Tensor weight;
    torch::Tensor bias; // may be undefined
    bool hasBias = false;
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    int groups = 1;

    bool IsValid() const { return input.defined() && weight.defined(); }
} TensorConv1dArgs;

static TensorConv1dArgs ParseTensorConv1dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorConv1dArgs args;

    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: input weight ?bias? ?stride? ?padding? ?dilation? ?groups?
        if (objc < 3 || objc > 8) {
            Tcl_WrongNumArgs(interp, 1, objv, "input weight ?bias? ?stride? ?padding? ?dilation? ?groups?");
            throw std::runtime_error("Invalid number of arguments");
        }
        std::string inName = Tcl_GetString(objv[1]);
        std::string wName  = Tcl_GetString(objv[2]);
        if (tensor_storage.find(inName) == tensor_storage.end()) throw std::runtime_error("Invalid input tensor name");
        if (tensor_storage.find(wName) == tensor_storage.end()) throw std::runtime_error("Invalid weight tensor name");
        args.input  = tensor_storage[inName];
        args.weight = tensor_storage[wName];

        int index = 3;
        if (objc > index) {
            std::string biasName = Tcl_GetString(objv[index]);
            if (biasName != "none" && !biasName.empty()) {
                if (tensor_storage.find(biasName) == tensor_storage.end()) throw std::runtime_error("Invalid bias tensor name");
                args.bias = tensor_storage[biasName];
                args.hasBias = true;
            }
            ++index;
        }
        if (objc > index) { if (Tcl_GetIntFromObj(interp, objv[index++], &args.stride) != TCL_OK) throw std::runtime_error("Invalid stride"); }
        if (objc > index) { if (Tcl_GetIntFromObj(interp, objv[index++], &args.padding) != TCL_OK) throw std::runtime_error("Invalid padding"); }
        if (objc > index) { if (Tcl_GetIntFromObj(interp, objv[index++], &args.dilation) != TCL_OK) throw std::runtime_error("Invalid dilation"); }
        if (objc > index) { if (Tcl_GetIntFromObj(interp, objv[index++], &args.groups) != TCL_OK) throw std::runtime_error("Invalid groups"); }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) throw std::runtime_error("Missing value for parameter");
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* valObj = objv[i+1];
            if (param == "-input") {
                std::string name = Tcl_GetString(valObj);
                if (tensor_storage.find(name) == tensor_storage.end()) throw std::runtime_error("Invalid input tensor name");
                args.input = tensor_storage[name];
            } else if (param == "-weight") {
                std::string name = Tcl_GetString(valObj);
                if (tensor_storage.find(name) == tensor_storage.end()) throw std::runtime_error("Invalid weight tensor name");
                args.weight = tensor_storage[name];
            } else if (param == "-bias") {
                std::string name = Tcl_GetString(valObj);
                if (name != "none" && !name.empty()) {
                    if (tensor_storage.find(name) == tensor_storage.end()) throw std::runtime_error("Invalid bias tensor name");
                    args.bias = tensor_storage[name];
                    args.hasBias = true;
                }
            } else if (param == "-stride") {
                if (Tcl_GetIntFromObj(interp, valObj, &args.stride) != TCL_OK) throw std::runtime_error("Invalid stride");
            } else if (param == "-padding") {
                if (Tcl_GetIntFromObj(interp, valObj, &args.padding) != TCL_OK) throw std::runtime_error("Invalid padding");
            } else if (param == "-dilation") {
                if (Tcl_GetIntFromObj(interp, valObj, &args.dilation) != TCL_OK) throw std::runtime_error("Invalid dilation");
            } else if (param == "-groups") {
                if (Tcl_GetIntFromObj(interp, valObj, &args.groups) != TCL_OK) throw std::runtime_error("Invalid groups");
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) throw std::runtime_error("Parameters -input and -weight are required");
    return args;
}

int TensorConv1D_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorConv1dArgs args = ParseTensorConv1dArgs(interp, objc, objv);
        torch::Tensor result;
        if (args.hasBias)
            result = torch::conv1d(args.input, args.weight, args.bias, args.stride, args.padding, args.dilation, args.groups);
        else
            result = torch::conv1d(args.input, args.weight, torch::Tensor(), args.stride, args.padding, args.dilation, args.groups);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

//torch::tensor_conv_transpose1d - 1D transposed convolution with dual syntax support
typedef struct {
    torch::Tensor input;
    torch::Tensor weight;
    torch::Tensor bias; // may be undefined
    bool hasBias = false;
    int stride = 1;
    int padding = 0;
    int output_padding = 0;
    int groups = 1;
    int dilation = 1;

    bool IsValid() const { return input.defined() && weight.defined(); }
} TensorConvTranspose1dArgs;

static TensorConvTranspose1dArgs ParseTensorConvTranspose1dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorConvTranspose1dArgs args;

    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: input weight ?bias? ?stride? ?padding? ?output_padding? ?groups? ?dilation?
        if (objc < 3 || objc > 9) {
            Tcl_WrongNumArgs(interp, 1, objv, "input weight ?bias? ?stride? ?padding? ?output_padding? ?groups? ?dilation?");
            throw std::runtime_error("Invalid number of arguments");
        }
        std::string inName = Tcl_GetString(objv[1]);
        std::string wName  = Tcl_GetString(objv[2]);
        if (tensor_storage.find(inName) == tensor_storage.end()) throw std::runtime_error("Invalid input tensor name");
        if (tensor_storage.find(wName) == tensor_storage.end()) throw std::runtime_error("Invalid weight tensor name");
        args.input  = tensor_storage[inName];
        args.weight = tensor_storage[wName];

        int index = 3;
        if (objc > index) {
            std::string biasName = Tcl_GetString(objv[index]);
            if (biasName != "none" && !biasName.empty()) {
                if (tensor_storage.find(biasName) == tensor_storage.end()) throw std::runtime_error("Invalid bias tensor name");
                args.bias = tensor_storage[biasName];
                args.hasBias = true;
            }
            ++index;
        }
        if (objc > index) { if (Tcl_GetIntFromObj(interp, objv[index++], &args.stride) != TCL_OK) throw std::runtime_error("Invalid stride"); }
        if (objc > index) { if (Tcl_GetIntFromObj(interp, objv[index++], &args.padding) != TCL_OK) throw std::runtime_error("Invalid padding"); }
        if (objc > index) { if (Tcl_GetIntFromObj(interp, objv[index++], &args.output_padding) != TCL_OK) throw std::runtime_error("Invalid output_padding"); }
        if (objc > index) { if (Tcl_GetIntFromObj(interp, objv[index++], &args.groups) != TCL_OK) throw std::runtime_error("Invalid groups"); }
        if (objc > index) { if (Tcl_GetIntFromObj(interp, objv[index++], &args.dilation) != TCL_OK) throw std::runtime_error("Invalid dilation"); }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) throw std::runtime_error("Missing value for parameter");
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* valObj = objv[i+1];
            if (param == "-input") {
                std::string name = Tcl_GetString(valObj);
                if (tensor_storage.find(name) == tensor_storage.end()) throw std::runtime_error("Invalid input tensor name");
                args.input = tensor_storage[name];
            } else if (param == "-weight") {
                std::string name = Tcl_GetString(valObj);
                if (tensor_storage.find(name) == tensor_storage.end()) throw std::runtime_error("Invalid weight tensor name");
                args.weight = tensor_storage[name];
            } else if (param == "-bias") {
                std::string name = Tcl_GetString(valObj);
                if (name != "none" && !name.empty()) {
                    if (tensor_storage.find(name) == tensor_storage.end()) throw std::runtime_error("Invalid bias tensor name");
                    args.bias = tensor_storage[name];
                    args.hasBias = true;
                }
            } else if (param == "-stride") {
                if (Tcl_GetIntFromObj(interp, valObj, &args.stride) != TCL_OK) throw std::runtime_error("Invalid stride");
            } else if (param == "-padding") {
                if (Tcl_GetIntFromObj(interp, valObj, &args.padding) != TCL_OK) throw std::runtime_error("Invalid padding");
            } else if (param == "-output_padding") {
                if (Tcl_GetIntFromObj(interp, valObj, &args.output_padding) != TCL_OK) throw std::runtime_error("Invalid output_padding");
            } else if (param == "-groups") {
                if (Tcl_GetIntFromObj(interp, valObj, &args.groups) != TCL_OK) throw std::runtime_error("Invalid groups");
            } else if (param == "-dilation") {
                if (Tcl_GetIntFromObj(interp, valObj, &args.dilation) != TCL_OK) throw std::runtime_error("Invalid dilation");
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) throw std::runtime_error("Parameters -input and -weight are required");
    return args;
}

int TensorConvTranspose1D_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorConvTranspose1dArgs args = ParseTensorConvTranspose1dArgs(interp, objc, objv);
        torch::Tensor result;
        if (args.hasBias)
            result = torch::conv_transpose1d(args.input, args.weight, args.bias, args.stride, args.padding, args.output_padding, args.groups, args.dilation);
        else
            result = torch::conv_transpose1d(args.input, args.weight, torch::Tensor(), args.stride, args.padding, args.output_padding, args.groups, args.dilation);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

//torch::tensor_conv_transpose2d - 2D transposed convolution with dual syntax support
typedef struct {
    torch::Tensor input;
    torch::Tensor weight;
    torch::Tensor bias; // may be undefined
    bool hasBias = false;
    std::vector<int64_t> stride = {1, 1};
    std::vector<int64_t> padding = {0, 0};
    std::vector<int64_t> output_padding = {0, 0};
    int groups = 1;
    std::vector<int64_t> dilation = {1, 1};

    bool IsValid() const { return input.defined() && weight.defined(); }
} TensorConvTranspose2dArgs;

// Helper function to parse int or list for 2D parameters
static std::vector<int64_t> ParseIntOrPair(Tcl_Interp* interp, Tcl_Obj* obj, const std::string& paramName) {
    int listLen;
    if (Tcl_ListObjLength(interp, obj, &listLen) == TCL_OK && listLen > 1) {
        // It's a list with multiple elements
        if (listLen != 2) {
            throw std::runtime_error(paramName + " must be an integer or a list of 2 integers");
        }
        std::vector<int64_t> result(2);
        for (int i = 0; i < 2; i++) {
            Tcl_Obj* elemObj;
            if (Tcl_ListObjIndex(interp, obj, i, &elemObj) != TCL_OK) {
                throw std::runtime_error("Failed to parse " + paramName + " list element");
            }
            int val;
            if (Tcl_GetIntFromObj(interp, elemObj, &val) != TCL_OK) {
                throw std::runtime_error("Invalid " + paramName + " list element");
            }
            result[i] = val;
        }
        return result;
    } else {
        // It's a single integer
        int val;
        if (Tcl_GetIntFromObj(interp, obj, &val) != TCL_OK) {
            throw std::runtime_error("Invalid " + paramName + " value");
        }
        return {val, val};
    }
}

static TensorConvTranspose2dArgs ParseTensorConvTranspose2dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorConvTranspose2dArgs args;

    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: input weight ?bias? ?stride? ?padding? ?output_padding? ?groups? ?dilation?
        if (objc < 3 || objc > 9) {
            Tcl_WrongNumArgs(interp, 1, objv, "input weight ?bias? ?stride? ?padding? ?output_padding? ?groups? ?dilation?");
            throw std::runtime_error("Invalid number of arguments");
        }
        std::string inName = Tcl_GetString(objv[1]);
        std::string wName  = Tcl_GetString(objv[2]);
        if (tensor_storage.find(inName) == tensor_storage.end()) throw std::runtime_error("Invalid input tensor name");
        if (tensor_storage.find(wName) == tensor_storage.end()) throw std::runtime_error("Invalid weight tensor name");
        args.input  = tensor_storage[inName];
        args.weight = tensor_storage[wName];

        int index = 3;
        if (objc > index) {
            std::string biasName = Tcl_GetString(objv[index]);
            if (biasName != "none" && !biasName.empty()) {
                if (tensor_storage.find(biasName) == tensor_storage.end()) throw std::runtime_error("Invalid bias tensor name");
                args.bias = tensor_storage[biasName];
                args.hasBias = true;
            }
            ++index;
        }
        if (objc > index) { args.stride = ParseIntOrPair(interp, objv[index++], "stride"); }
        if (objc > index) { args.padding = ParseIntOrPair(interp, objv[index++], "padding"); }
        if (objc > index) { args.output_padding = ParseIntOrPair(interp, objv[index++], "output_padding"); }
        if (objc > index) { if (Tcl_GetIntFromObj(interp, objv[index++], &args.groups) != TCL_OK) throw std::runtime_error("Invalid groups"); }
        if (objc > index) { args.dilation = ParseIntOrPair(interp, objv[index++], "dilation"); }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) throw std::runtime_error("Missing value for parameter");
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* valObj = objv[i+1];
            if (param == "-input") {
                std::string name = Tcl_GetString(valObj);
                if (tensor_storage.find(name) == tensor_storage.end()) throw std::runtime_error("Invalid input tensor name");
                args.input = tensor_storage[name];
            } else if (param == "-weight") {
                std::string name = Tcl_GetString(valObj);
                if (tensor_storage.find(name) == tensor_storage.end()) throw std::runtime_error("Invalid weight tensor name");
                args.weight = tensor_storage[name];
            } else if (param == "-bias") {
                std::string name = Tcl_GetString(valObj);
                if (name != "none" && !name.empty()) {
                    if (tensor_storage.find(name) == tensor_storage.end()) throw std::runtime_error("Invalid bias tensor name");
                    args.bias = tensor_storage[name];
                    args.hasBias = true;
                }
            } else if (param == "-stride") {
                args.stride = ParseIntOrPair(interp, valObj, "stride");
            } else if (param == "-padding") {
                args.padding = ParseIntOrPair(interp, valObj, "padding");
            } else if (param == "-output_padding") {
                args.output_padding = ParseIntOrPair(interp, valObj, "output_padding");
            } else if (param == "-groups") {
                if (Tcl_GetIntFromObj(interp, valObj, &args.groups) != TCL_OK) throw std::runtime_error("Invalid groups");
            } else if (param == "-dilation") {
                args.dilation = ParseIntOrPair(interp, valObj, "dilation");
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) throw std::runtime_error("Parameters -input and -weight are required");
    return args;
}

int TensorConvTranspose2D_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorConvTranspose2dArgs args = ParseTensorConvTranspose2dArgs(interp, objc, objv);
        torch::Tensor result;
        if (args.hasBias)
            result = torch::conv_transpose2d(args.input, args.weight, args.bias, args.stride, args.padding, args.output_padding, args.groups, args.dilation);
        else
            result = torch::conv_transpose2d(args.input, args.weight, torch::Tensor(), args.stride, args.padding, args.output_padding, args.groups, args.dilation);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Real FFT Operations
// ============================================================================

// Parameter structure for tensor_rfft command
struct TensorRFFTArgs {
    std::string input;
    c10::optional<int64_t> n = c10::nullopt;
    int64_t dim = -1;  // Default to last dimension
    bool has_n = false;
    bool has_dim = false;
    
    bool IsValid() const {
        return !input.empty();
    }
};

// Parse dual syntax for tensor_rfft
TensorRFFTArgs ParseTensorRFFTArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorRFFTArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor ?n? ?dim?
        if (objc < 2 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor ?n? ?dim?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (objc >= 3) {
            int n_val;
            if (Tcl_GetIntFromObj(interp, objv[2], &n_val) != TCL_OK) {
                throw std::runtime_error("Invalid n parameter");
            }
            args.n = n_val;
            args.has_n = true;
        }
        
        if (objc >= 4) {
            int dim_val;
            if (Tcl_GetIntFromObj(interp, objv[3], &dim_val) != TCL_OK) {
                throw std::runtime_error("Invalid dim parameter");
            }
            args.dim = dim_val;
            args.has_dim = true;
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
            } else if (param == "-n") {
                int n_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &n_val) != TCL_OK) {
                    throw std::runtime_error("Invalid n parameter");
                }
                args.n = n_val;
                args.has_n = true;
            } else if (param == "-dim") {
                int dim_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &dim_val) != TCL_OK) {
                    throw std::runtime_error("Invalid dim parameter");
                }
                args.dim = dim_val;
                args.has_dim = true;
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

int TensorRFFT_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorRFFTArgs args = ParseTensorRFFTArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        torch::Tensor result;
        
        result = torch::fft::rfft(tensor, args.n, args.dim);
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

typedef struct {
    std::string tensor;
    c10::optional<int64_t> n = c10::nullopt;
    int64_t dim = -1;  // Default to last dimension
    bool hasN = false;
    bool hasDim = false;
    
    bool IsValid() const { return !tensor.empty(); }
} TensorIRFFTArgs;

static TensorIRFFTArgs ParseTensorIRFFTArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorIRFFTArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: tensor ?n? ?dim?
        if (objc < 2 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor ?n? ?dim?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.tensor = Tcl_GetString(objv[1]);
        
        if (objc >= 3) {
            int n_val;
            if (Tcl_GetIntFromObj(interp, objv[2], &n_val) != TCL_OK) {
                throw std::runtime_error("Invalid n parameter");
            }
            args.n = n_val;
            args.hasN = true;
        }
        
        if (objc >= 4) {
            int dim_val;
            if (Tcl_GetIntFromObj(interp, objv[3], &dim_val) != TCL_OK) {
                throw std::runtime_error("Invalid dim parameter");
            }
            args.dim = dim_val;
            args.hasDim = true;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-tensor" || param == "-input") {
                args.tensor = value;
            } else if (param == "-n") {
                int n_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &n_val) != TCL_OK) {
                    throw std::runtime_error("Invalid n parameter");
                }
                args.n = n_val;
                args.hasN = true;
            } else if (param == "-dim") {
                int dim_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &dim_val) != TCL_OK) {
                    throw std::runtime_error("Invalid dim parameter");
                }
                args.dim = dim_val;
                args.hasDim = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required tensor parameter missing");
    }
    
    return args;
}

int TensorIRFFT_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorIRFFTArgs args = ParseTensorIRFFTArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.tensor) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.tensor];
        torch::Tensor result;
        
        result = torch::fft::irfft(tensor, args.n, args.dim);
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Short-Time Fourier Transform (STFT)
// ============================================================================

// Parameter structure for tensor_stft command
struct TensorSTFTArgs {
    std::string input;
    int n_fft;
    c10::optional<int64_t> hop_length = c10::nullopt;
    c10::optional<int64_t> win_length = c10::nullopt;
    c10::optional<torch::Tensor> window = c10::nullopt;
    bool has_hop_length = false;
    bool has_win_length = false;
    bool has_window = false;
    
    bool IsValid() const {
        return !input.empty() && n_fft > 0;
    }
};

// Parse dual syntax for tensor_stft
TensorSTFTArgs ParseTensorSTFTArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorSTFTArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor n_fft ?hop_length? ?win_length? ?window?
        if (objc < 3 || objc > 6) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor n_fft ?hop_length? ?win_length? ?window?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.n_fft) != TCL_OK) {
            throw std::runtime_error("Invalid n_fft value");
        }
        
        if (objc >= 4) {
            int hop_val;
            if (Tcl_GetIntFromObj(interp, objv[3], &hop_val) != TCL_OK) {
                throw std::runtime_error("Invalid hop_length value");
            }
            args.hop_length = hop_val;
            args.has_hop_length = true;
        }
        
        if (objc >= 5) {
            int win_val;
            if (Tcl_GetIntFromObj(interp, objv[4], &win_val) != TCL_OK) {
                throw std::runtime_error("Invalid win_length value");
            }
            args.win_length = win_val;
            args.has_win_length = true;
        }
        
        if (objc >= 6) {
            std::string window_name = Tcl_GetString(objv[5]);
            if (tensor_storage.find(window_name) != tensor_storage.end()) {
                args.window = tensor_storage[window_name];
                args.has_window = true;
            }
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
            } else if (param == "-n_fft" || param == "-nfft") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.n_fft) != TCL_OK) {
                    throw std::runtime_error("Invalid n_fft value");
                }
            } else if (param == "-hop_length" || param == "-hopLength") {
                int hop_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &hop_val) != TCL_OK) {
                    throw std::runtime_error("Invalid hop_length value");
                }
                args.hop_length = hop_val;
                args.has_hop_length = true;
            } else if (param == "-win_length" || param == "-winLength") {
                int win_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &win_val) != TCL_OK) {
                    throw std::runtime_error("Invalid win_length value");
                }
                args.win_length = win_val;
                args.has_win_length = true;
            } else if (param == "-window") {
                if (tensor_storage.find(value) != tensor_storage.end()) {
                    args.window = tensor_storage[value];
                    args.has_window = true;
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required input and n_fft parameters missing");
    }
    
    return args;
}

int TensorSTFT_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorSTFTArgs args = ParseTensorSTFTArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        // Create Hann window if no window provided
        if (!args.has_window && args.has_win_length) {
            args.window = torch::hann_window(args.win_length.value());
        } else if (!args.has_window) {
            args.window = torch::hann_window(args.n_fft);
        }
        
        auto result = torch::stft(tensor, args.n_fft, args.hop_length, args.win_length, args.window, 
                                  true, // normalized 
                                  true, // onesided
                                  true); // return_complex
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for tensor_istft command
struct TensorISTFTArgs {
    std::string input;
    int n_fft;
    c10::optional<int64_t> hop_length = c10::nullopt;
    c10::optional<int64_t> win_length = c10::nullopt;
    c10::optional<torch::Tensor> window = c10::nullopt;
    bool center = true;
    bool normalized = true;
    bool onesided = true;
    c10::optional<int64_t> length = c10::nullopt;
    
    bool IsValid() const {
        return !input.empty() && n_fft > 0;
    }
};

// Parse dual syntax for tensor_istft
TensorISTFTArgs ParseTensorISTFTArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorISTFTArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): tensor n_fft ?hop_length? ?win_length? ?window?
        if (objc < 3 || objc > 6) {
            Tcl_WrongNumArgs(interp, 1, objv, "tensor n_fft ?hop_length? ?win_length? ?window?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.n_fft) != TCL_OK) {
            throw std::runtime_error("Invalid n_fft value");
        }
        
        if (objc >= 4) {
            int hop_val;
            if (Tcl_GetIntFromObj(interp, objv[3], &hop_val) != TCL_OK) {
                throw std::runtime_error("Invalid hop_length value");
            }
            args.hop_length = hop_val;
        }
        
        if (objc >= 5) {
            int win_val;
            if (Tcl_GetIntFromObj(interp, objv[4], &win_val) != TCL_OK) {
                throw std::runtime_error("Invalid win_length value");
            }
            args.win_length = win_val;
        }
        
        if (objc >= 6) {
            std::string window_name = Tcl_GetString(objv[5]);
            if (tensor_storage.find(window_name) != tensor_storage.end()) {
                args.window = tensor_storage[window_name];
            }
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
            } else if (param == "-n_fft" || param == "-nfft") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.n_fft) != TCL_OK) {
                    throw std::runtime_error("Invalid n_fft value");
                }
            } else if (param == "-hop_length" || param == "-hopLength") {
                int hop_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &hop_val) != TCL_OK) {
                    throw std::runtime_error("Invalid hop_length value");
                }
                args.hop_length = hop_val;
            } else if (param == "-win_length" || param == "-winLength") {
                int win_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &win_val) != TCL_OK) {
                    throw std::runtime_error("Invalid win_length value");
                }
                args.win_length = win_val;
            } else if (param == "-window") {
                if (tensor_storage.find(value) != tensor_storage.end()) {
                    args.window = tensor_storage[value];
                }
            } else if (param == "-center") {
                if (value == "true" || value == "1") {
                    args.center = true;
                } else if (value == "false" || value == "0") {
                    args.center = false;
                } else {
                    throw std::runtime_error("Invalid center value (use true/false or 1/0)");
                }
            } else if (param == "-normalized") {
                if (value == "true" || value == "1") {
                    args.normalized = true;
                } else if (value == "false" || value == "0") {
                    args.normalized = false;
                } else {
                    throw std::runtime_error("Invalid normalized value (use true/false or 1/0)");
                }
            } else if (param == "-onesided") {
                if (value == "true" || value == "1") {
                    args.onesided = true;
                } else if (value == "false" || value == "0") {
                    args.onesided = false;
                } else {
                    throw std::runtime_error("Invalid onesided value (use true/false or 1/0)");
                }
            } else if (param == "-length") {
                int length_val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &length_val) != TCL_OK) {
                    throw std::runtime_error("Invalid length value");
                }
                args.length = length_val;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required input and n_fft parameters missing");
    }
    
    return args;
}

int TensorISTFT_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        TensorISTFTArgs args = ParseTensorISTFTArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& tensor = tensor_storage[args.input];
        
        // Create Hann window if no window provided
        if (!args.window.has_value() && args.win_length.has_value()) {
            args.window = torch::hann_window(args.win_length.value());
        } else if (!args.window.has_value()) {
            args.window = torch::hann_window(args.n_fft);
        }
        
        auto result = torch::istft(tensor, args.n_fft, args.hop_length, args.win_length, args.window,
                                   args.center, args.normalized, args.onesided, args.length);
        
        std::string result_handle = GetNextHandle("tensor");
        tensor_storage[result_handle] = result;
        
        Tcl_SetResult(interp, const_cast<char*>(result_handle.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 