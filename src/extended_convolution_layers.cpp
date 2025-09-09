#include "libtorchtcl.h"

// torch::conv1d - 1D convolution
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
} Conv1dArgs;

static Conv1dArgs ParseConv1dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    Conv1dArgs args;

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
            if (biasName != "none") {
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
                if (name != "none") {
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

int TensorConv1d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        Conv1dArgs args = ParseConv1dArgs(interp, objc, objv);
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

struct Conv3dArgs {
    std::string input;
    std::string weight;
    std::string bias;  // Optional, can be "none" or empty
    std::vector<int64_t> stride = {1, 1, 1};
    std::vector<int64_t> padding = {0, 0, 0};
    std::vector<int64_t> dilation = {1, 1, 1};
    int groups = 1;
    
    bool IsValid() const {
        return !input.empty() && !weight.empty();
    }
};

// Helper function to parse int or list of 3 ints
std::vector<int64_t> ParseIntOrList2(Tcl_Interp* interp, Tcl_Obj* obj, const std::vector<int64_t>& defaultVal) {
    int single_val;
    if (Tcl_GetIntFromObj(interp, obj, &single_val) == TCL_OK) {
        return {single_val, single_val};
            } else {
                // Try to parse as list
                int listLen;
                Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, obj, &listLen, &listObjv) == TCL_OK && listLen == 2) {
            std::vector<int64_t> result(2);
            for (int i = 0; i < 2; i++) {
                        int val;
                        if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                    throw std::runtime_error("Invalid integer in list");
                        }
                result[i] = val;
                    }
            return result;
                } else {
            throw std::runtime_error("Value must be int or list of 2 ints");
                }
            }
        }
        
std::vector<int64_t> ParseIntOrList3(Tcl_Interp* interp, Tcl_Obj* obj, const std::vector<int64_t>& defaultVal) {
    int single_val;
    if (Tcl_GetIntFromObj(interp, obj, &single_val) == TCL_OK) {
        return {single_val, single_val, single_val};
            } else {
                // Try to parse as list
                int listLen;
                Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, obj, &listLen, &listObjv) == TCL_OK && listLen == 3) {
            std::vector<int64_t> result(3);
                    for (int i = 0; i < 3; i++) {
                        int val;
                        if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                    throw std::runtime_error("Invalid integer in list");
                }
                result[i] = val;
                    }
            return result;
                } else {
            throw std::runtime_error("Value must be int or list of 3 ints");
        }
    }
}

Conv3dArgs ParseConv3dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    Conv3dArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 8) {
            throw std::runtime_error("Usage: input weight ?bias? ?stride? ?padding? ?dilation? ?groups?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.weight = Tcl_GetString(objv[2]);
        
        if (objc > 3) {
            args.bias = Tcl_GetString(objv[3]);
        }
        if (objc > 4) {
            args.stride = ParseIntOrList3(interp, objv[4], {1, 1, 1});
        }
        if (objc > 5) {
            args.padding = ParseIntOrList3(interp, objv[5], {0, 0, 0});
        }
        if (objc > 6) {
            args.dilation = ParseIntOrList3(interp, objv[6], {1, 1, 1});
        }
        if (objc > 7) {
            int groups;
            if (Tcl_GetIntFromObj(interp, objv[7], &groups) != TCL_OK) {
                throw std::runtime_error("Invalid groups value");
                        }
            args.groups = groups;
                    }
                } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-weight") {
                args.weight = Tcl_GetString(objv[i + 1]);
            } else if (param == "-bias") {
                args.bias = Tcl_GetString(objv[i + 1]);
            } else if (param == "-stride") {
                args.stride = ParseIntOrList3(interp, objv[i + 1], {1, 1, 1});
            } else if (param == "-padding") {
                args.padding = ParseIntOrList3(interp, objv[i + 1], {0, 0, 0});
            } else if (param == "-dilation") {
                args.dilation = ParseIntOrList3(interp, objv[i + 1], {1, 1, 1});
            } else if (param == "-groups") {
                int groups;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &groups) != TCL_OK) {
                    throw std::runtime_error("Invalid groups value");
                }
                args.groups = groups;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters: input and weight");
    }
    
    return args;
}

// torch::conv3d - 3D convolution
int TensorConv3d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        Conv3dArgs args = ParseConv3dArgs(interp, objc, objv);
        
        // Validate input tensor
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
                    return TCL_ERROR;
                }
        
        // Validate weight tensor
        if (tensor_storage.find(args.weight) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid weight tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto& weight = tensor_storage[args.weight];
        
        // Handle bias
        torch::Tensor bias;
        bool has_bias = false;
        if (!args.bias.empty() && args.bias != "none") {
            if (tensor_storage.find(args.bias) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>("Invalid bias tensor name"), TCL_VOLATILE);
                return TCL_ERROR;
            }
            bias = tensor_storage[args.bias];
            has_bias = true;
        }
        
        // Perform convolution
        torch::Tensor result;
        if (has_bias) {
            result = torch::conv3d(input, weight, bias, args.stride, args.padding, args.dilation, args.groups);
        } else {
            result = torch::conv3d(input, weight, torch::Tensor(), args.stride, args.padding, args.dilation, args.groups);
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::conv_transpose1d - 1D transposed convolution
struct ConvTranspose1dArgs {
    std::string input;
    std::string weight;
    std::string bias;  // Optional, can be "none" or empty
    int stride = 1;
    int padding = 0;
    int output_padding = 0;
    int groups = 1;
    int dilation = 1;
    
    bool IsValid() const {
        return !input.empty() && !weight.empty();
    }
};

ConvTranspose1dArgs ParseConvTranspose1dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ConvTranspose1dArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 9) {
            throw std::runtime_error("Usage: conv_transpose1d input weight ?bias? ?stride? ?padding? ?output_padding? ?groups? ?dilation?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.weight = Tcl_GetString(objv[2]);
        
        if (objc > 3) {
            std::string bias_str = Tcl_GetString(objv[3]);
            if (bias_str != "none" && !bias_str.empty()) {
                args.bias = bias_str;
            }
        }
        
        if (objc > 4) {
            if (Tcl_GetIntFromObj(interp, objv[4], &args.stride) != TCL_OK) {
                throw std::runtime_error("Invalid stride parameter");
            }
        }
        
        if (objc > 5) {
            if (Tcl_GetIntFromObj(interp, objv[5], &args.padding) != TCL_OK) {
                throw std::runtime_error("Invalid padding parameter");
            }
        }
        
        if (objc > 6) {
            if (Tcl_GetIntFromObj(interp, objv[6], &args.output_padding) != TCL_OK) {
                throw std::runtime_error("Invalid output_padding parameter");
            }
        }
        
        if (objc > 7) {
            if (Tcl_GetIntFromObj(interp, objv[7], &args.groups) != TCL_OK) {
                throw std::runtime_error("Invalid groups parameter");
            }
        }
        
        if (objc > 8) {
            if (Tcl_GetIntFromObj(interp, objv[8], &args.dilation) != TCL_OK) {
                throw std::runtime_error("Invalid dilation parameter");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string key = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (key == "-input") {
                args.input = value;
            } else if (key == "-weight") {
                args.weight = value;
            } else if (key == "-bias") {
                if (value != "none" && !value.empty()) {
                    args.bias = value;
                }
            } else if (key == "-stride") {
                int val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &val) != TCL_OK) {
                    throw std::runtime_error("Invalid stride value");
                }
                args.stride = val;
            } else if (key == "-padding") {
                int val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &val) != TCL_OK) {
                    throw std::runtime_error("Invalid padding value");
                }
                args.padding = val;
            } else if (key == "-output_padding") {
                int val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &val) != TCL_OK) {
                    throw std::runtime_error("Invalid output_padding value");
                }
                args.output_padding = val;
            } else if (key == "-groups") {
                int val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &val) != TCL_OK) {
                    throw std::runtime_error("Invalid groups value");
                }
                args.groups = val;
            } else if (key == "-dilation") {
                int val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &val) != TCL_OK) {
                    throw std::runtime_error("Invalid dilation value");
                }
                args.dilation = val;
            } else {
                throw std::runtime_error("Unknown parameter: " + key);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and weight must be specified");
    }
    
    return args;
}

int TensorConvTranspose1d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        ConvTranspose1dArgs args = ParseConvTranspose1dArgs(interp, objc, objv);
        
        // Validate input tensor
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Validate weight tensor
        if (tensor_storage.find(args.weight) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid weight tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto& weight = tensor_storage[args.weight];
        
        // Handle bias
        torch::Tensor bias;
        bool has_bias = false;
        if (!args.bias.empty() && args.bias != "none") {
            if (tensor_storage.find(args.bias) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>("Invalid bias tensor name"), TCL_VOLATILE);
                return TCL_ERROR;
            }
            bias = tensor_storage[args.bias];
            has_bias = true;
        }
        
        // Perform transposed convolution
        torch::Tensor result;
        if (has_bias) {
            result = torch::conv_transpose1d(input, weight, bias, args.stride, args.padding, 
                                           args.output_padding, args.groups, args.dilation);
        } else {
            result = torch::conv_transpose1d(input, weight, torch::Tensor(), args.stride, args.padding, 
                                           args.output_padding, args.groups, args.dilation);
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::conv_transpose3d - 3D transposed convolution
struct ConvTranspose3dArgs {
    std::string input;
    std::string weight;
    std::string bias;  // Optional, can be "none" or empty
    std::vector<int64_t> stride = {1, 1, 1};
    std::vector<int64_t> padding = {0, 0, 0};
    std::vector<int64_t> output_padding = {0, 0, 0};
    int groups = 1;
    std::vector<int64_t> dilation = {1, 1, 1};
    
    bool IsValid() const {
        return !input.empty() && !weight.empty();
    }
};

ConvTranspose3dArgs ParseConvTranspose3dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ConvTranspose3dArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 9) {
            throw std::runtime_error("Usage: conv_transpose3d input weight ?bias? ?stride? ?padding? ?output_padding? ?groups? ?dilation?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.weight = Tcl_GetString(objv[2]);
        
        if (objc > 3) {
            std::string bias_str = Tcl_GetString(objv[3]);
            if (bias_str != "none" && !bias_str.empty()) {
                args.bias = bias_str;
            }
        }
        
        if (objc > 4) {
            args.stride = ParseIntOrList3(interp, objv[4], {1, 1, 1});
                    }
        
        if (objc > 5) {
            args.padding = ParseIntOrList3(interp, objv[5], {0, 0, 0});
                    }
        
        if (objc > 6) {
            args.output_padding = ParseIntOrList3(interp, objv[6], {0, 0, 0});
        }
        
        if (objc > 7) {
            if (Tcl_GetIntFromObj(interp, objv[7], &args.groups) != TCL_OK) {
                throw std::runtime_error("Invalid groups parameter");
            }
        }
        
        if (objc > 8) {
            args.dilation = ParseIntOrList3(interp, objv[8], {1, 1, 1});
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-weight") {
                args.weight = Tcl_GetString(objv[i + 1]);
            } else if (param == "-bias") {
                std::string bias_str = Tcl_GetString(objv[i + 1]);
                if (bias_str != "none" && !bias_str.empty()) {
                    args.bias = bias_str;
                }
            } else if (param == "-stride") {
                args.stride = ParseIntOrList3(interp, objv[i + 1], {1, 1, 1});
            } else if (param == "-padding") {
                args.padding = ParseIntOrList3(interp, objv[i + 1], {0, 0, 0});
            } else if (param == "-output_padding" || param == "-outputPadding") {
                args.output_padding = ParseIntOrList3(interp, objv[i + 1], {0, 0, 0});
            } else if (param == "-groups") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.groups) != TCL_OK) {
                    throw std::runtime_error("Invalid groups parameter");
                }
            } else if (param == "-dilation") {
                args.dilation = ParseIntOrList3(interp, objv[i + 1], {1, 1, 1});
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters 'input' and 'weight' are missing");
    }
    
    return args;
}

int TensorConvTranspose3d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        ConvTranspose3dArgs args = ParseConvTranspose3dArgs(interp, objc, objv);
        
        // Validate input tensor
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Validate weight tensor
        if (tensor_storage.find(args.weight) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid weight tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto& weight = tensor_storage[args.weight];
        
        // Handle bias
        torch::Tensor bias;
        bool has_bias = false;
        if (!args.bias.empty() && args.bias != "none") {
            if (tensor_storage.find(args.bias) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>("Invalid bias tensor name"), TCL_VOLATILE);
                return TCL_ERROR;
            }
            bias = tensor_storage[args.bias];
            has_bias = true;
                    }
        
        // Perform transposed convolution
        torch::Tensor result;
        if (has_bias) {
            result = torch::conv_transpose3d(input, weight, bias, args.stride, args.padding, 
                                           args.output_padding, args.groups, args.dilation);
        } else {
            result = torch::conv_transpose3d(input, weight, torch::Tensor(), args.stride, args.padding, 
                                           args.output_padding, args.groups, args.dilation);
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for unfold command
struct UnfoldArgs {
    std::string input;
    int dimension;
    int size;
    int step;
    
    bool IsValid() const {
        return !input.empty() && size > 0 && step > 0;
    }
};

// Parse dual syntax for unfold
UnfoldArgs ParseUnfoldArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    UnfoldArgs args;
    
    if (objc < 5) {
        throw std::runtime_error("Usage: torch::unfold input dimension size step | torch::unfold -input tensor -dimension int -size int -step int");
    }
    
    if (objc >= 5 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 5) {
            throw std::runtime_error("Usage: torch::unfold input dimension size step");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.dimension) != TCL_OK) {
            throw std::runtime_error("Invalid dimension parameter: must be an integer");
        }
        if (Tcl_GetIntFromObj(interp, objv[3], &args.size) != TCL_OK) {
            throw std::runtime_error("Invalid size parameter: must be an integer");
        }
        if (Tcl_GetIntFromObj(interp, objv[4], &args.step) != TCL_OK) {
            throw std::runtime_error("Invalid step parameter: must be an integer");
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
            } else if (param == "-dimension") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.dimension) != TCL_OK) {
                    throw std::runtime_error("Invalid dimension parameter: must be an integer");
                }
            } else if (param == "-size") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.size) != TCL_OK) {
                    throw std::runtime_error("Invalid size parameter: must be an integer");
                }
            } else if (param == "-step") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.step) != TCL_OK) {
                    throw std::runtime_error("Invalid step parameter: must be an integer");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor, -dimension, -size, -step");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid");
    }
    
    return args;
}

// torch::unfold - Extract sliding blocks
int TensorUnfold_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        UnfoldArgs args = ParseUnfoldArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        torch::Tensor result = input.unfold(args.dimension, args.size, args.step);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// -----------------------------------------------------------------------------
// Dual-syntax argument structure & parser
struct FoldArgs {
    std::string input;
    std::vector<int64_t> output_size;
    std::vector<int64_t> kernel_size;
    std::vector<int64_t> dilation = {1, 1};
    std::vector<int64_t> padding = {0, 0};
    std::vector<int64_t> stride = {1, 1};

    bool IsValid() const {
        return !input.empty() && output_size.size() == 2 && kernel_size.size() == 2;
    }
};

FoldArgs ParseFoldArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    FoldArgs args;
    
    if (objc < 4) {
        throw std::runtime_error("Usage: torch::fold input output_size kernel_size ?dilation? ?padding? ?stride? | torch::fold -input tensor -outputSize {h w} -kernelSize {h w} [-dilation {h w}] [-padding {h w}] [-stride {h w}]");
    }
    
    if (objc >= 4 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc > 7) {
            throw std::runtime_error("Too many positional arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        // Parse output_size (list of 2 ints)
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, objv[2], &listLen, &listObjv) != TCL_OK || listLen != 2) {
            throw std::runtime_error("Output size must be list of 2 ints");
        }
        for (int i = 0; i < 2; i++) {
            int val;
            if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                throw std::runtime_error("Invalid output size value");
            }
            args.output_size.push_back(val);
        }
        
        // Parse kernel_size (list of 2 ints)
        if (Tcl_ListObjGetElements(interp, objv[3], &listLen, &listObjv) != TCL_OK || listLen != 2) {
            throw std::runtime_error("Kernel size must be list of 2 ints");
        }
        for (int i = 0; i < 2; i++) {
            int val;
            if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                throw std::runtime_error("Invalid kernel size value");
            }
            args.kernel_size.push_back(val);
        }
        
        // Parse optional parameters
        if (objc > 4) {
            if (Tcl_ListObjGetElements(interp, objv[4], &listLen, &listObjv) == TCL_OK && listLen == 2) {
                args.dilation.clear();
                for (int i = 0; i < 2; i++) {
                    int val;
                    if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                        throw std::runtime_error("Invalid dilation value");
                    }
                    args.dilation.push_back(val);
                }
            }
        }
        
        if (objc > 5) {
            if (Tcl_ListObjGetElements(interp, objv[5], &listLen, &listObjv) == TCL_OK && listLen == 2) {
                args.padding.clear();
                for (int i = 0; i < 2; i++) {
                    int val;
                    if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                        throw std::runtime_error("Invalid padding value");
                    }
                    args.padding.push_back(val);
                }
            }
        }
        
        if (objc > 6) {
            if (Tcl_ListObjGetElements(interp, objv[6], &listLen, &listObjv) == TCL_OK && listLen == 2) {
                args.stride.clear();
                for (int i = 0; i < 2; i++) {
                    int val;
                    if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                        throw std::runtime_error("Invalid stride value");
                    }
                    args.stride.push_back(val);
                }
            }
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
            } else if (param == "-output_size" || param == "-outputSize") {
                // Parse output_size
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) != TCL_OK || listLen != 2) {
                    throw std::runtime_error("Output size must be list of 2 ints");
                }
                args.output_size.clear();
                for (int j = 0; j < 2; j++) {
                    int val;
                    if (Tcl_GetIntFromObj(interp, listObjv[j], &val) != TCL_OK) {
                        throw std::runtime_error("Invalid output size value");
                    }
                    args.output_size.push_back(val);
                }
            } else if (param == "-kernel_size" || param == "-kernelSize") {
                // Parse kernel_size
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) != TCL_OK || listLen != 2) {
                    throw std::runtime_error("Kernel size must be list of 2 ints");
                }
                args.kernel_size.clear();
                for (int j = 0; j < 2; j++) {
                    int val;
                    if (Tcl_GetIntFromObj(interp, listObjv[j], &val) != TCL_OK) {
                        throw std::runtime_error("Invalid kernel size value");
                    }
                    args.kernel_size.push_back(val);
                }
            } else if (param == "-dilation") {
                // Parse dilation
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) != TCL_OK || listLen != 2) {
                    throw std::runtime_error("Dilation must be list of 2 ints");
                }
                args.dilation.clear();
                for (int j = 0; j < 2; j++) {
                    int val;
                    if (Tcl_GetIntFromObj(interp, listObjv[j], &val) != TCL_OK) {
                        throw std::runtime_error("Invalid dilation value");
                    }
                    args.dilation.push_back(val);
                }
            } else if (param == "-padding") {
                // Parse padding
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) != TCL_OK || listLen != 2) {
                    throw std::runtime_error("Padding must be list of 2 ints");
                }
                args.padding.clear();
                for (int j = 0; j < 2; j++) {
                    int val;
                    if (Tcl_GetIntFromObj(interp, listObjv[j], &val) != TCL_OK) {
                        throw std::runtime_error("Invalid padding value");
                    }
                    args.padding.push_back(val);
                }
            } else if (param == "-stride") {
                // Parse stride
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) != TCL_OK || listLen != 2) {
                    throw std::runtime_error("Stride must be list of 2 ints");
                }
                args.stride.clear();
                for (int j = 0; j < 2; j++) {
                    int val;
                    if (Tcl_GetIntFromObj(interp, listObjv[j], &val) != TCL_OK) {
                        throw std::runtime_error("Invalid stride value");
                    }
                    args.stride.push_back(val);
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor, -output_size, -outputSize, -kernel_size, -kernelSize, -dilation, -padding, -stride");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor, output_size, and kernel_size required");
    }
    
    return args;
}

// torch::fold - Combine sliding blocks
int TensorFold_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        FoldArgs args = ParseFoldArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        
        torch::Tensor result = torch::nn::functional::fold(input, 
            torch::nn::functional::FoldFuncOptions(args.output_size, args.kernel_size)
                .dilation(args.dilation).padding(args.padding).stride(args.stride));
        
        // Preserve the input tensor's options (dtype, device, etc.)
        result = result.to(input.options());
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in fold: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::conv_transpose2d - 2D transposed convolution
struct ConvTranspose2dArgs {
    std::string input;
    std::string weight;
    std::string bias;  // Optional, can be "none" or empty
    std::vector<int64_t> stride = {1, 1};
    std::vector<int64_t> padding = {0, 0};
    std::vector<int64_t> output_padding = {0, 0};
    int groups = 1;
    std::vector<int64_t> dilation = {1, 1};
    
    bool IsValid() const {
        return !input.empty() && !weight.empty();
    }
};

ConvTranspose2dArgs ParseConvTranspose2dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ConvTranspose2dArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 9) {
            throw std::runtime_error("Usage: conv_transpose2d input weight ?bias? ?stride? ?padding? ?output_padding? ?groups? ?dilation?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.weight = Tcl_GetString(objv[2]);
        
        if (objc > 3) {
            std::string bias_str = Tcl_GetString(objv[3]);
            if (bias_str != "none" && !bias_str.empty()) {
                args.bias = bias_str;
            }
        }
        
        if (objc > 4) {
            args.stride = ParseIntOrList2(interp, objv[4], {1, 1});
        }
        
        if (objc > 5) {
            args.padding = ParseIntOrList2(interp, objv[5], {0, 0});
        }
        
        if (objc > 6) {
            args.output_padding = ParseIntOrList2(interp, objv[6], {0, 0});
        }
        
        if (objc > 7) {
            if (Tcl_GetIntFromObj(interp, objv[7], &args.groups) != TCL_OK) {
                throw std::runtime_error("Invalid groups parameter");
            }
        }
        
        if (objc > 8) {
            args.dilation = ParseIntOrList2(interp, objv[8], {1, 1});
        }
        
    } else {
        // Named parameter syntax
        if (objc % 2 == 0) {
            throw std::runtime_error("Named parameters must come in pairs");
        }
        
        for (int i = 1; i < objc; i += 2) {
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-input") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-weight") {
                args.weight = Tcl_GetString(objv[i + 1]);
            } else if (param == "-bias") {
                std::string bias_str = Tcl_GetString(objv[i + 1]);
                if (bias_str != "none" && !bias_str.empty()) {
                    args.bias = bias_str;
                }
            } else if (param == "-stride") {
                args.stride = ParseIntOrList2(interp, objv[i + 1], {1, 1});
            } else if (param == "-padding") {
                args.padding = ParseIntOrList2(interp, objv[i + 1], {0, 0});
            } else if (param == "-output_padding" || param == "-outputPadding") {
                args.output_padding = ParseIntOrList2(interp, objv[i + 1], {0, 0});
            } else if (param == "-groups") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.groups) != TCL_OK) {
                    throw std::runtime_error("Invalid groups parameter");
                }
            } else if (param == "-dilation") {
                args.dilation = ParseIntOrList2(interp, objv[i + 1], {1, 1});
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters 'input' and 'weight' are missing");
    }
    
    return args;
}

int TensorConvTranspose2d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        ConvTranspose2dArgs args = ParseConvTranspose2dArgs(interp, objc, objv);
        
        // Validate input tensor
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Validate weight tensor
        if (tensor_storage.find(args.weight) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid weight tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto& weight = tensor_storage[args.weight];
        
        // Handle bias
        torch::Tensor bias;
        bool has_bias = false;
        if (!args.bias.empty() && args.bias != "none") {
            if (tensor_storage.find(args.bias) == tensor_storage.end()) {
                Tcl_SetResult(interp, const_cast<char*>("Invalid bias tensor name"), TCL_VOLATILE);
                return TCL_ERROR;
            }
            bias = tensor_storage[args.bias];
            has_bias = true;
        }
        
        // Perform transposed convolution
        torch::Tensor result;
        if (has_bias) {
            result = torch::conv_transpose2d(input, weight, bias, args.stride, args.padding, 
                                           args.output_padding, args.groups, args.dilation);
        } else {
            result = torch::conv_transpose2d(input, weight, torch::Tensor(), args.stride, args.padding, 
                                           args.output_padding, args.groups, args.dilation);
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 