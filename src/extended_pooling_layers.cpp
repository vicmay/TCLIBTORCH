#include "libtorchtcl.h"

// Start with torch::avgpool1d

// torch::avgpool1d - 1D average pooling
// -----------------------------------------------------------------------------
// Dual-syntax argument structure & parser
struct AvgPool1dArgs {
    std::string input;
    int kernel_size = 0;
    int stride = -1;  // -1 means not set; will default to kernel_size
    int padding = 0;
    bool count_include_pad = true;

    bool IsValid() const {
        return !input.empty() && kernel_size > 0;
    }
};

AvgPool1dArgs ParseAvgPool1dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    AvgPool1dArgs args;

    // Decide positional vs named
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: input kernel_size ?stride? ?padding? ?count_include_pad?
        if (objc < 3 || objc > 6) {
            Tcl_WrongNumArgs(interp, 1, objv, "input kernel_size ?stride? ?padding? ?count_include_pad?");
            throw std::runtime_error("Invalid number of arguments");
        }

        args.input = Tcl_GetString(objv[1]);

        if (Tcl_GetIntFromObj(interp, objv[2], &args.kernel_size) != TCL_OK) {
            throw std::runtime_error("Invalid kernel_size: must be integer");
        }

        if (objc > 3) {
            if (Tcl_GetIntFromObj(interp, objv[3], &args.stride) != TCL_OK) {
                throw std::runtime_error("Invalid stride: must be integer");
            }
        }

        if (objc > 4) {
            if (Tcl_GetIntFromObj(interp, objv[4], &args.padding) != TCL_OK) {
                throw std::runtime_error("Invalid padding: must be integer");
            }
        }

        if (objc > 5) {
            int include_pad;
            if (Tcl_GetIntFromObj(interp, objv[5], &include_pad) != TCL_OK) {
                throw std::runtime_error("Invalid count_include_pad: must be 0/1");
            }
            args.count_include_pad = (include_pad != 0);
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

            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-kernel_size" || param == "-kernelSize") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.kernel_size) != TCL_OK) {
                    throw std::runtime_error("Invalid kernel_size: must be integer");
                }
            } else if (param == "-stride") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.stride) != TCL_OK) {
                    throw std::runtime_error("Invalid stride: must be integer");
                }
            } else if (param == "-padding") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.padding) != TCL_OK) {
                    throw std::runtime_error("Invalid padding: must be integer");
                }
            } else if (param == "-count_include_pad" || param == "-countIncludePad") {
                int cip;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &cip) != TCL_OK) {
                    throw std::runtime_error("Invalid count_include_pad: must be 0/1");
                }
                args.count_include_pad = (cip != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (args.stride == -1) args.stride = args.kernel_size;  // default

    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters: input tensor and positive kernel_size");
    }

    return args;
}

int TensorAvgPool1d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        AvgPool1dArgs args = ParseAvgPool1dArgs(interp, objc, objv);

        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto& input = tensor_storage[args.input];

        torch::Tensor result = torch::avg_pool1d(input, args.kernel_size, args.stride, args.padding, false, args.count_include_pad);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::avgpool2d - 2D average pooling (direct tensor operation)
// -----------------------------------------------------------------------------
// Dual-syntax argument structure & parser
struct AvgPool2dTensorArgs {
    std::string input;
    std::vector<int64_t> kernel_size;
    std::vector<int64_t> stride;  // empty means default to kernel_size
    std::vector<int64_t> padding = {0, 0};
    bool count_include_pad = true;

    bool IsValid() const {
        return !input.empty() && !kernel_size.empty() && 
               (kernel_size.size() == 1 || kernel_size.size() == 2);
    }
};

// Helper function to parse int or 2-element list
std::vector<int64_t> ParseIntOrList2(Tcl_Interp* interp, Tcl_Obj* obj) {
    int single_val;
    if (Tcl_GetIntFromObj(interp, obj, &single_val) == TCL_OK) {
        return {single_val, single_val};
    } else {
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, obj, &listLen, &listObjv) == TCL_OK && listLen == 2) {
            std::vector<int64_t> result;
            for (int i = 0; i < 2; i++) {
                int val;
                if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                    throw std::runtime_error("List elements must be integers");
                }
                result.push_back(val);
            }
            return result;
        } else {
            throw std::runtime_error("Parameter must be int or list of 2 ints");
        }
    }
}

AvgPool2dTensorArgs ParseAvgPool2dTensorArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    AvgPool2dTensorArgs args;

    // Decide positional vs named
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: input kernel_size ?stride? ?padding? ?count_include_pad?
        if (objc < 3 || objc > 6) {
            Tcl_WrongNumArgs(interp, 1, objv, "input kernel_size ?stride? ?padding? ?count_include_pad?");
            throw std::runtime_error("Invalid number of arguments");
        }

        args.input = Tcl_GetString(objv[1]);

        // Parse kernel_size
        args.kernel_size = ParseIntOrList2(interp, objv[2]);

        // Default stride to kernel_size
        args.stride = args.kernel_size;

        // Parse stride (optional)
        if (objc > 3) {
            args.stride = ParseIntOrList2(interp, objv[3]);
        }

        // Parse padding (optional)
        if (objc > 4) {
            args.padding = ParseIntOrList2(interp, objv[4]);
        }

        // Parse count_include_pad (optional)
        if (objc > 5) {
            int include_pad;
            if (Tcl_GetIntFromObj(interp, objv[5], &include_pad) != TCL_OK) {
                throw std::runtime_error("Invalid count_include_pad: must be 0/1");
            }
            args.count_include_pad = (include_pad != 0);
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

            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-kernel_size" || param == "-kernelSize") {
                args.kernel_size = ParseIntOrList2(interp, objv[i + 1]);
            } else if (param == "-stride") {
                args.stride = ParseIntOrList2(interp, objv[i + 1]);
            } else if (param == "-padding") {
                args.padding = ParseIntOrList2(interp, objv[i + 1]);
            } else if (param == "-count_include_pad" || param == "-countIncludePad") {
                int cip;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &cip) != TCL_OK) {
                    throw std::runtime_error("Invalid count_include_pad: must be 0/1");
                }
                args.count_include_pad = (cip != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    // Set default stride if not specified
    if (args.stride.empty()) {
        args.stride = args.kernel_size;
    }

    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters: input tensor and valid kernel_size");
    }

    return args;
}

int TensorAvgPool2d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        AvgPool2dTensorArgs args = ParseAvgPool2dTensorArgs(interp, objc, objv);

        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto& input = tensor_storage[args.input];

        torch::Tensor result = torch::avg_pool2d(
            input, 
            args.kernel_size, 
            args.stride, 
            args.padding, 
            false,  // ceil_mode
            args.count_include_pad);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::maxpool2d - 2D max pooling (direct tensor operation)
// -----------------------------------------------------------------------------
// Dual-syntax argument structure & parser
struct MaxPool2dTensorArgs {
    std::string input;
    std::vector<int64_t> kernel_size;
    std::vector<int64_t> stride;  // empty means default to kernel_size
    std::vector<int64_t> padding = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    bool ceil_mode = false;
    bool return_indices = false;

    bool IsValid() const {
        return !input.empty() && !kernel_size.empty() && 
               (kernel_size.size() == 1 || kernel_size.size() == 2);
    }
};

MaxPool2dTensorArgs ParseMaxPool2dTensorArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MaxPool2dTensorArgs args;

    // Decide positional vs named
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: input kernel_size ?stride? ?padding? ?dilation? ?ceil_mode?
        if (objc < 3 || objc > 7) {
            Tcl_WrongNumArgs(interp, 1, objv, "input kernel_size ?stride? ?padding? ?dilation? ?ceil_mode?");
            throw std::runtime_error("Invalid number of arguments");
        }

        args.input = Tcl_GetString(objv[1]);

        // Parse kernel_size
        args.kernel_size = ParseIntOrList2(interp, objv[2]);

        // Default stride to kernel_size
        args.stride = args.kernel_size;

        // Parse stride (optional)
        if (objc > 3) {
            args.stride = ParseIntOrList2(interp, objv[3]);
        }

        // Parse padding (optional)
        if (objc > 4) {
            args.padding = ParseIntOrList2(interp, objv[4]);
        }

        // Parse dilation (optional)
        if (objc > 5) {
            args.dilation = ParseIntOrList2(interp, objv[5]);
        }

        // Parse ceil_mode (optional)
        if (objc > 6) {
            int ceil_mode_int;
            if (Tcl_GetBooleanFromObj(interp, objv[6], &ceil_mode_int) != TCL_OK) {
                throw std::runtime_error("Invalid ceil_mode: must be boolean");
            }
            args.ceil_mode = ceil_mode_int != 0;
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

            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-kernel_size" || param == "-kernelSize") {
                args.kernel_size = ParseIntOrList2(interp, objv[i + 1]);
            } else if (param == "-stride") {
                args.stride = ParseIntOrList2(interp, objv[i + 1]);
            } else if (param == "-padding") {
                args.padding = ParseIntOrList2(interp, objv[i + 1]);
            } else if (param == "-dilation") {
                args.dilation = ParseIntOrList2(interp, objv[i + 1]);
            } else if (param == "-ceil_mode" || param == "-ceilMode") {
                int ceil_mode_int;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &ceil_mode_int) != TCL_OK) {
                    throw std::runtime_error("Invalid ceil_mode: must be boolean");
                }
                args.ceil_mode = ceil_mode_int != 0;
            } else if (param == "-return_indices" || param == "-returnIndices") {
                int return_indices_int;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &return_indices_int) != TCL_OK) {
                    throw std::runtime_error("Invalid return_indices: must be boolean");
                }
                args.return_indices = return_indices_int != 0;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    // Set default stride if not specified
    if (args.stride.empty()) {
        args.stride = args.kernel_size;
    }

    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters: input tensor and valid kernel_size");
    }

    return args;
}

int TensorMaxPool2d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        MaxPool2dTensorArgs args = ParseMaxPool2dTensorArgs(interp, objc, objv);

        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto& input = tensor_storage[args.input];

        torch::Tensor result = torch::max_pool2d(
            input, 
            args.kernel_size, 
            args.stride, 
            args.padding, 
            args.dilation,
            args.ceil_mode);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

#if 0  // Legacy positional-only implementation retained for reference but disabled
// Parameter structure for avgpool3d command
struct AvgPool3dArgs {
    std::string input;
    std::vector<int64_t> kernelSize;
    std::vector<int64_t> stride;
    std::vector<int64_t> padding = {0, 0, 0};
    bool countIncludePad = true;
    
    bool IsValid() const {
        return !input.empty() && !kernelSize.empty();
    }
};

// Parse dual syntax for avgpool3d command
AvgPool3dArgs ParseAvgPool3dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    AvgPool3dArgs args;
    
    // Determine syntax style
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: input kernel_size ?stride? ?padding? ?count_include_pad?
        if (objc < 3 || objc > 6) {
            Tcl_WrongNumArgs(interp, 1, objv, "input kernel_size ?stride? ?padding? ?count_include_pad?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        // Parse input tensor name
        args.input = Tcl_GetString(objv[1]);
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            throw std::runtime_error("Invalid input tensor name");
        }
        
        // Parse kernel_size
        int single_kernel;
        if (Tcl_GetIntFromObj(interp, objv[2], &single_kernel) == TCL_OK) {
            args.kernelSize = {single_kernel, single_kernel, single_kernel};
        } else {
            int listLen;
            Tcl_Obj** listObjv;
            if (Tcl_ListObjGetElements(interp, objv[2], &listLen, &listObjv) == TCL_OK && listLen == 3) {
                for (int i = 0; i < 3; i++) {
                    int val;
                    if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                        throw std::runtime_error("Kernel size list must contain integers");
                    }
                    args.kernelSize.push_back(val);
                }
            } else {
                throw std::runtime_error("Kernel size must be int or list of 3 ints");
            }
        }
        
        // Default stride to kernel_size
        args.stride = args.kernelSize;
        
        // Parse stride (optional)
        if (objc > 3) {
            int single_stride;
            if (Tcl_GetIntFromObj(interp, objv[3], &single_stride) == TCL_OK) {
                args.stride = {single_stride, single_stride, single_stride};
            } else {
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[3], &listLen, &listObjv) == TCL_OK && listLen == 3) {
                    args.stride.clear();
                    for (int i = 0; i < 3; i++) {
                        int val;
                        if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                            throw std::runtime_error("Stride list must contain integers");
                        }
                        args.stride.push_back(val);
                    }
                } else {
                    throw std::runtime_error("Stride must be int or list of 3 ints");
                }
            }
        }
        
        // Parse padding (optional)
        if (objc > 4) {
            int single_padding;
            if (Tcl_GetIntFromObj(interp, objv[4], &single_padding) == TCL_OK) {
                args.padding = {single_padding, single_padding, single_padding};
            } else {
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[4], &listLen, &listObjv) == TCL_OK && listLen == 3) {
                    args.padding.clear();
                    for (int i = 0; i < 3; i++) {
                        int val;
                        if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                            throw std::runtime_error("Padding list must contain integers");
                        }
                        args.padding.push_back(val);
                    }
                } else {
                    throw std::runtime_error("Padding must be int or list of 3 ints");
                }
            }
        }
        
        // Parse count_include_pad (optional)
        if (objc > 5) {
            int include_pad;
            if (Tcl_GetIntFromObj(interp, objv[5], &include_pad) != TCL_OK) {
                throw std::runtime_error("Count include pad must be an integer");
            }
            args.countIncludePad = (include_pad != 0);
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
                if (tensor_storage.find(args.input) == tensor_storage.end()) {
                    throw std::runtime_error("Invalid input tensor name");
                }
            } else if (param == "-kernelSize") {
                int single_kernel;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &single_kernel) == TCL_OK) {
                    args.kernelSize = {single_kernel, single_kernel, single_kernel};
                } else {
                    int listLen;
                    Tcl_Obj** listObjv;
                    if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) == TCL_OK && listLen == 3) {
                        args.kernelSize.clear();
                        for (int j = 0; j < 3; j++) {
                            int val;
                            if (Tcl_GetIntFromObj(interp, listObjv[j], &val) != TCL_OK) {
                                throw std::runtime_error("Kernel size list must contain integers");
                            }
                            args.kernelSize.push_back(val);
                        }
                    } else {
                        throw std::runtime_error("Kernel size must be int or list of 3 ints");
                    }
                }
                // Default stride to kernel_size if not already set
                if (args.stride.empty()) {
                    args.stride = args.kernelSize;
                }
            } else if (param == "-stride") {
                int single_stride;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &single_stride) == TCL_OK) {
                    args.stride = {single_stride, single_stride, single_stride};
                } else {
                    int listLen;
                    Tcl_Obj** listObjv;
                    if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) == TCL_OK && listLen == 3) {
                        args.stride.clear();
                        for (int j = 0; j < 3; j++) {
                            int val;
                            if (Tcl_GetIntFromObj(interp, listObjv[j], &val) != TCL_OK) {
                                throw std::runtime_error("Stride list must contain integers");
                            }
                            args.stride.push_back(val);
                        }
                    } else {
                        throw std::runtime_error("Stride must be int or list of 3 ints");
                    }
                }
            } else if (param == "-padding") {
                int single_padding;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &single_padding) == TCL_OK) {
                    args.padding = {single_padding, single_padding, single_padding};
                } else {
                    int listLen;
                    Tcl_Obj** listObjv;
                    if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) == TCL_OK && listLen == 3) {
                        args.padding.clear();
                        for (int j = 0; j < 3; j++) {
                            int val;
                            if (Tcl_GetIntFromObj(interp, listObjv[j], &val) != TCL_OK) {
                                throw std::runtime_error("Padding list must contain integers");
                            }
                            args.padding.push_back(val);
                        }
                    } else {
                        throw std::runtime_error("Padding must be int or list of 3 ints");
                    }
                }
            } else if (param == "-countIncludePad") {
                int include_pad;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &include_pad) != TCL_OK) {
                    throw std::runtime_error("Count include pad must be an integer");
                }
                args.countIncludePad = (include_pad != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and kernelSize must be specified");
    }
    
    return args;
}

// torch::avgpool3d - 3D average pooling with dual syntax support
int TensorAvgPool3d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        AvgPool3dArgs args = ParseAvgPool3dArgs(interp, objc, objv);
        
        auto& input = tensor_storage[args.input];
        
        torch::Tensor result = torch::avg_pool3d(
            input, 
            args.kernelSize, 
            args.stride, 
            args.padding, 
            false,  // ceil_mode is always false
            args.countIncludePad
        );
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}
#endif // legacy TensorAvgPool3d_Cmd

// Parameter structure for adaptive_avgpool1d command
struct AdaptiveAvgpool1dArgs {
    std::string input;
    int output_size;
    
    bool IsValid() const {
        return !input.empty() && output_size > 0;
    }
};

// Parse dual syntax for adaptive_avgpool1d
AdaptiveAvgpool1dArgs ParseAdaptiveAvgpool1dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    AdaptiveAvgpool1dArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): input output_size
        if (objc != 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "input output_size");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        if (Tcl_GetIntFromObj(interp, objv[2], &args.output_size) != TCL_OK) {
            throw std::runtime_error("Invalid output_size value");
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
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.output_size) != TCL_OK) {
                    throw std::runtime_error("Invalid output_size value: " + std::string(Tcl_GetString(objv[i + 1])));
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -input and -output_size");
    }
    
    return args;
}

// torch::adaptive_avgpool1d - 1D adaptive average pooling
int TensorAdaptiveAvgPool1d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        AdaptiveAvgpool1dArgs args = ParseAdaptiveAvgpool1dArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        
        torch::Tensor result = torch::adaptive_avg_pool1d(input, args.output_size);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for adaptive_avgpool3d command
struct AdaptiveAvgpool3dArgs {
    std::string input;
    std::vector<int64_t> output_size;
    
    bool IsValid() const {
        return !input.empty() && !output_size.empty() && (output_size.size() == 1 || output_size.size() == 3);
    }
};

// Parse dual syntax for adaptive_avgpool3d
AdaptiveAvgpool3dArgs ParseAdaptiveAvgpool3dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    AdaptiveAvgpool3dArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): input output_size
        if (objc != 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "input output_size");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        // Parse output_size (can be int or list of 3 ints)
        int single_size;
        if (Tcl_GetIntFromObj(interp, objv[2], &single_size) == TCL_OK) {
            args.output_size = {single_size, single_size, single_size};
        } else {
            int listLen;
            Tcl_Obj** listObjv;
            if (Tcl_ListObjGetElements(interp, objv[2], &listLen, &listObjv) == TCL_OK && listLen == 3) {
                for (int i = 0; i < 3; i++) {
                    int val;
                    if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                        throw std::runtime_error("Invalid output_size value in list");
                    }
                    args.output_size.push_back(val);
                }
            } else {
                throw std::runtime_error("Output size must be int or list of 3 ints");
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
                // Parse output_size (can be int or list of 3 ints)
                int single_size;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &single_size) == TCL_OK) {
                    args.output_size = {single_size, single_size, single_size};
                } else {
                    int listLen;
                    Tcl_Obj** listObjv;
                    if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) == TCL_OK && listLen == 3) {
                        for (int j = 0; j < 3; j++) {
                            int val;
                            if (Tcl_GetIntFromObj(interp, listObjv[j], &val) != TCL_OK) {
                                throw std::runtime_error("Invalid output_size value in list");
                            }
                            args.output_size.push_back(val);
                        }
                    } else {
                        throw std::runtime_error("Output size must be an int or list of 3 ints");
                    }
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -input and -output_size");
    }
    
    return args;
}

// torch::adaptive_avgpool3d - 3D adaptive average pooling
int TensorAdaptiveAvgPool3d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        AdaptiveAvgpool3dArgs args = ParseAdaptiveAvgpool3dArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        
        torch::Tensor result = torch::adaptive_avg_pool3d(input, args.output_size);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::adaptive_maxpool1d - 1D adaptive max pooling
struct AdaptiveMaxpool1dArgs {
    std::string input;
    int output_size;
    
    bool IsValid() const {
        return !input.empty() && output_size > 0;
    }
};

AdaptiveMaxpool1dArgs ParseAdaptiveMaxpool1dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    AdaptiveMaxpool1dArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Wrong number of arguments: expected 'input output_size'");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.output_size) != TCL_OK) {
            throw std::runtime_error("Invalid output_size: must be an integer");
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
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-output_size" || param == "-outputSize") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.output_size) != TCL_OK) {
                    throw std::runtime_error("Invalid " + param + ": must be an integer");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters: input tensor and positive output_size");
    }
    
    return args;
}

int TensorAdaptiveMaxPool1d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        auto args = ParseAdaptiveMaxpool1dArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        
        auto result_tuple = torch::adaptive_max_pool1d(input, args.output_size);
        torch::Tensor result = std::get<0>(result_tuple);  // Get values, ignore indices
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::adaptive_maxpool3d - 3D adaptive max pooling
// -----------------------------------------------------------------------------
// Dual-syntax argument structure & parser
struct AdaptiveMaxpool3dArgs {
    std::string input;
    std::vector<int64_t> output_size;  // Accept either 1 or 3 dims

    bool IsValid() const {
        return !input.empty() && !output_size.empty() &&
               (output_size.size() == 1 || output_size.size() == 3);
    }
};

AdaptiveMaxpool3dArgs ParseAdaptiveMaxpool3dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    AdaptiveMaxpool3dArgs args;

    // Determine syntax: positional if second token doesn't start with '-'
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: expect exactly 3 arguments: command, input, output_size
        if (objc != 3) {
            throw std::runtime_error("Wrong number of arguments: expected 'input output_size'");
        }

        args.input = Tcl_GetString(objv[1]);

        // Parse output_size which can be int or list of 3 ints
        int single_size;
        if (Tcl_GetIntFromObj(interp, objv[2], &single_size) == TCL_OK) {
            args.output_size = {single_size, single_size, single_size};
        } else {
            int listLen;
            Tcl_Obj** listObjv;
            if (Tcl_ListObjGetElements(interp, objv[2], &listLen, &listObjv) == TCL_OK && listLen == 3) {
                for (int i = 0; i < 3; i++) {
                    int val;
                    if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                        throw std::runtime_error("Invalid element in output_size list: must be int");
                    }
                    args.output_size.push_back(val);
                }
            } else {
                throw std::runtime_error("Output size must be an int or list of 3 ints");
            }
        }
    } else {
        // Named parameter syntax: expect odd number of objc (command + pairs)
        if (objc < 2 || objc % 2 != 1) {
            throw std::runtime_error("Named parameters require pairs: -param value");
        }

        for (int i = 1; i < objc; i += 2) {
            std::string param = Tcl_GetString(objv[i]);
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter: " + param);
            }

            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-output_size" || param == "-outputSize") {
                // Accept int or list of 3 ints
                int single_size;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &single_size) == TCL_OK) {
                    args.output_size = {single_size, single_size, single_size};
                } else {
                    int listLen;
                    Tcl_Obj** listObjv;
                    if (Tcl_ListObjGetElements(interp, objv[i + 1], &listLen, &listObjv) == TCL_OK && listLen == 3) {
                        for (int j = 0; j < 3; j++) {
                            int val;
                            if (Tcl_GetIntFromObj(interp, listObjv[j], &val) != TCL_OK) {
                                throw std::runtime_error("Invalid element in output_size list: must be int");
                            }
                            args.output_size.push_back(val);
                        }
                    } else {
                        throw std::runtime_error("Output size must be an int or list of 3 ints");
                    }
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters: input tensor and positive output_size");
    }

    return args;
}

int TensorAdaptiveMaxPool3d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        auto args = ParseAdaptiveMaxpool3dArgs(interp, objc, objv);

        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto& input = tensor_storage[args.input];

        auto result_tuple = torch::adaptive_max_pool3d(input, args.output_size);
        torch::Tensor result = std::get<0>(result_tuple);  // Get values, ignore indices

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::fractional_maxpool2d - 2D fractional max pooling
struct FractionalMaxPool2dArgs {
    std::string input;
    std::vector<int64_t> kernel_size;
    std::vector<double> output_ratio = {0.5, 0.5};  // Default to half size
    
    bool IsValid() const {
        return !input.empty() && kernel_size.size() == 2 && 
               output_ratio.size() == 2 && 
               kernel_size[0] > 0 && kernel_size[1] > 0 &&
               output_ratio[0] > 0 && output_ratio[1] > 0;
    }
};

std::vector<int64_t> ParseIntList2(Tcl_Interp* interp, Tcl_Obj* obj) {
    std::vector<int64_t> result;
    int listLen;
    Tcl_Obj** listObjv;
    
    if (Tcl_ListObjGetElements(interp, obj, &listLen, &listObjv) == TCL_OK && listLen == 2) {
        for (int i = 0; i < 2; i++) {
            int val;
            if (Tcl_GetIntFromObj(interp, listObjv[i], &val) == TCL_OK) {
                result.push_back(val);
            } else {
                throw std::runtime_error("Invalid integer in list");
            }
        }
    } else {
        throw std::runtime_error("Expected list of 2 integers");
    }
    return result;
}

std::vector<double> ParseDoubleList2(Tcl_Interp* interp, Tcl_Obj* obj) {
    std::vector<double> result;
    int listLen;
    Tcl_Obj** listObjv;
    
    if (Tcl_ListObjGetElements(interp, obj, &listLen, &listObjv) == TCL_OK && listLen == 2) {
        for (int i = 0; i < 2; i++) {
            double val;
            if (Tcl_GetDoubleFromObj(interp, listObjv[i], &val) == TCL_OK) {
                result.push_back(val);
            } else {
                throw std::runtime_error("Invalid double in list");
            }
        }
    } else {
        throw std::runtime_error("Expected list of 2 doubles");
    }
    return result;
}

FractionalMaxPool2dArgs ParseFractionalMaxPool2dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    FractionalMaxPool2dArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): input kernel_size ?output_ratio?
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: fractional_maxpool2d input kernel_size ?output_ratio?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.kernel_size = ParseIntList2(interp, objv[2]);
        
        if (objc > 3) {
            args.output_ratio = ParseDoubleList2(interp, objv[3]);
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
            } else if (param == "-kernel_size" || param == "-kernelSize") {
                args.kernel_size = ParseIntList2(interp, objv[i + 1]);
            } else if (param == "-output_ratio" || param == "-outputRatio") {
                args.output_ratio = ParseDoubleList2(interp, objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters 'input' and 'kernel_size' are missing or invalid");
    }
    
    return args;
}

int TensorFractionalMaxPool2d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        FractionalMaxPool2dArgs args = ParseFractionalMaxPool2dArgs(interp, objc, objv);
        
        // Validate input tensor
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        
        // Calculate output size based on input size and ratio
        auto input_size = input.sizes();
        std::vector<int64_t> output_size = {
            static_cast<int64_t>(input_size[2] * args.output_ratio[0]), 
            static_cast<int64_t>(input_size[3] * args.output_ratio[1])
        };
        
        // Create random samples tensor - for 2D, should be (batch, channels, 2) for x,y sampling
        torch::Tensor random_samples = torch::rand({input.size(0), input.size(1), 2});
        
        auto result_tuple = torch::fractional_max_pool2d(input, args.kernel_size, output_size, random_samples);
        torch::Tensor result = std::get<0>(result_tuple);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::fractional_maxpool3d - 3D fractional max pooling
struct FractionalMaxPool3dArgs {
    std::string input;
    std::vector<int64_t> kernel_size;
    std::vector<double> output_ratio = {0.5, 0.5, 0.5};  // Default to half size
    
    bool IsValid() const {
        return !input.empty() && kernel_size.size() == 3 && 
               output_ratio.size() == 3 && 
               kernel_size[0] > 0 && kernel_size[1] > 0 && kernel_size[2] > 0 &&
               output_ratio[0] > 0 && output_ratio[1] > 0 && output_ratio[2] > 0;
    }
};

std::vector<int64_t> ParseIntList3(Tcl_Interp* interp, Tcl_Obj* obj) {
    std::vector<int64_t> result;
    int listLen;
    Tcl_Obj** listObjv;
    
    if (Tcl_ListObjGetElements(interp, obj, &listLen, &listObjv) == TCL_OK && listLen == 3) {
        for (int i = 0; i < 3; i++) {
            int val;
            if (Tcl_GetIntFromObj(interp, listObjv[i], &val) == TCL_OK) {
                result.push_back(val);
            } else {
                throw std::runtime_error("Invalid integer in list");
            }
        }
    } else {
        throw std::runtime_error("Expected list of 3 integers");
    }
    return result;
}

std::vector<double> ParseDoubleList3(Tcl_Interp* interp, Tcl_Obj* obj) {
    std::vector<double> result;
    int listLen;
    Tcl_Obj** listObjv;
    
    if (Tcl_ListObjGetElements(interp, obj, &listLen, &listObjv) == TCL_OK && listLen == 3) {
        for (int i = 0; i < 3; i++) {
            double val;
            if (Tcl_GetDoubleFromObj(interp, listObjv[i], &val) == TCL_OK) {
                result.push_back(val);
            } else {
                throw std::runtime_error("Invalid double in list");
            }
        }
    } else {
        throw std::runtime_error("Expected list of 3 doubles");
    }
    return result;
}

FractionalMaxPool3dArgs ParseFractionalMaxPool3dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    FractionalMaxPool3dArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): input kernel_size ?output_ratio?
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: fractional_maxpool3d input kernel_size ?output_ratio?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.kernel_size = ParseIntList3(interp, objv[2]);
        
        if (objc > 3) {
            args.output_ratio = ParseDoubleList3(interp, objv[3]);
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
            } else if (param == "-kernel_size" || param == "-kernelSize") {
                args.kernel_size = ParseIntList3(interp, objv[i + 1]);
            } else if (param == "-output_ratio" || param == "-outputRatio") {
                args.output_ratio = ParseDoubleList3(interp, objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters 'input' and 'kernel_size' are missing or invalid");
    }
    
    return args;
}

int TensorFractionalMaxPool3d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        FractionalMaxPool3dArgs args = ParseFractionalMaxPool3dArgs(interp, objc, objv);
        
        // Validate input tensor
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        
        // Calculate output size based on input size and ratio
        auto input_size = input.sizes();
        std::vector<int64_t> output_size = {
            static_cast<int64_t>(input_size[2] * args.output_ratio[0]), 
            static_cast<int64_t>(input_size[3] * args.output_ratio[1]),
            static_cast<int64_t>(input_size[4] * args.output_ratio[2])
        };
        
        // Create random samples tensor - for 3D, should be (batch, channels, 3) for x,y,z sampling
        torch::Tensor random_samples = torch::rand({input.size(0), input.size(1), 3});
        
        auto result_tuple = torch::fractional_max_pool3d(input, args.kernel_size, output_size, random_samples);
        torch::Tensor result = std::get<0>(result_tuple);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct LpPool1dArgs {
    std::string input;
    double normType = 2.0;  // Default L2 norm
    int kernelSize = 0;
    int stride = -1;  // -1 means not set; will default to kernelSize
    bool ceilMode = false;
    
    bool IsValid() const {
        return !input.empty() && kernelSize > 0 && normType > 0;
    }
};

// Parse dual syntax for lppool1d command
LpPool1dArgs ParseLpPool1dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LpPool1dArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 4 || objc > 6) {
            throw std::runtime_error("Usage: torch::lppool1d input norm_type kernel_size ?stride? ?ceil_mode?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.normType) != TCL_OK) {
            throw std::runtime_error("Invalid norm_type parameter");
        }
        
        if (Tcl_GetIntFromObj(interp, objv[3], &args.kernelSize) != TCL_OK) {
            throw std::runtime_error("Invalid kernel_size parameter");
        }
        
        if (objc >= 5) {
            if (Tcl_GetIntFromObj(interp, objv[4], &args.stride) != TCL_OK) {
                throw std::runtime_error("Invalid stride parameter");
            }
        }
        
        if (objc >= 6) {
            int ceilFlag;
            if (Tcl_GetBooleanFromObj(interp, objv[5], &ceilFlag) != TCL_OK) {
                throw std::runtime_error("Invalid ceil_mode parameter (should be boolean)");
            }
            args.ceilMode = (ceilFlag != 0);
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
            } else if (param == "-normType") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.normType) != TCL_OK) {
                    throw std::runtime_error("Invalid value for -normType parameter");
                }
            } else if (param == "-kernelSize") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.kernelSize) != TCL_OK) {
                    throw std::runtime_error("Invalid value for -kernelSize parameter");
                }
            } else if (param == "-stride") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.stride) != TCL_OK) {
                    throw std::runtime_error("Invalid value for -stride parameter");
                }
            } else if (param == "-ceilMode") {
                int ceilFlag;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &ceilFlag) != TCL_OK) {
                    std::string value = Tcl_GetString(objv[i + 1]);
                    if (value == "true") ceilFlag = 1;
                    else if (value == "false") ceilFlag = 0;
                    else throw std::runtime_error("Invalid ceilMode value");
                }
                args.ceilMode = (ceilFlag != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid: input must be specified, kernelSize must be positive, and normType must be positive");
    }
    
    // Set default stride if not specified
    if (args.stride == -1) {
        args.stride = args.kernelSize;
    }
    
    return args;
}

// torch::lppool1d - 1D LP pooling
int TensorLpPool1d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        LpPool1dArgs args = ParseLpPool1dArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        
        torch::Tensor result = torch::nn::functional::lp_pool1d(input, 
            torch::nn::functional::LPPool1dFuncOptions(args.normType, args.kernelSize)
                .stride(args.stride).ceil_mode(args.ceilMode));
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

struct LpPool2dArgs {
    std::string input;
    double normType = 2.0;  // Default L2 norm
    std::vector<int64_t> kernelSize;
    std::vector<int64_t> stride;  // empty means default to kernelSize
    bool ceilMode = false;
    
    bool IsValid() const {
        return !input.empty() && !kernelSize.empty() && 
               kernelSize.size() == 2 && normType > 0 &&
               kernelSize[0] > 0 && kernelSize[1] > 0;
    }
};

// Parse dual syntax for lppool2d command
LpPool2dArgs ParseLpPool2dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LpPool2dArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 4 || objc > 6) {
            throw std::runtime_error("Usage: torch::lppool2d input norm_type kernel_size ?stride? ?ceil_mode?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        double norm_type;
        if (Tcl_GetDoubleFromObj(interp, objv[2], &norm_type) != TCL_OK) {
            throw std::runtime_error("Invalid norm_type value");
        }
        args.normType = norm_type;
        
        // Parse kernel_size
        int single_kernel;
        if (Tcl_GetIntFromObj(interp, objv[3], &single_kernel) == TCL_OK) {
            args.kernelSize = {single_kernel, single_kernel};
        } else {
            int listLen;
            Tcl_Obj** listObjv;
            if (Tcl_ListObjGetElements(interp, objv[3], &listLen, &listObjv) == TCL_OK && listLen == 2) {
                for (int i = 0; i < 2; i++) {
                    int val;
                    if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                        throw std::runtime_error("Invalid kernel_size value");
                    }
                    args.kernelSize.push_back(val);
                }
            } else {
                throw std::runtime_error("Kernel size must be int or list of 2 ints");
            }
        }
        
        // Parse stride (optional)
        if (objc > 4) {
            int single_stride;
            if (Tcl_GetIntFromObj(interp, objv[4], &single_stride) == TCL_OK) {
                args.stride = {single_stride, single_stride};
            } else {
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, objv[4], &listLen, &listObjv) == TCL_OK && listLen == 2) {
                    for (int i = 0; i < 2; i++) {
                        int val;
                        if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
                            throw std::runtime_error("Invalid stride value");
                        }
                        args.stride.push_back(val);
                    }
                } else {
                    throw std::runtime_error("Stride must be int or list of 2 ints");
                }
            }
        }
        
        // Parse ceil_mode (optional)
        if (objc > 5) {
            int ceil_flag;
            if (Tcl_GetIntFromObj(interp, objv[5], &ceil_flag) != TCL_OK) {
                std::string value = Tcl_GetString(objv[5]);
                if (value == "true") ceil_flag = 1;
                else if (value == "false") ceil_flag = 0;
                else throw std::runtime_error("Invalid ceil_mode value");
            }
            args.ceilMode = (ceil_flag != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* valueObj = objv[i + 1];
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(valueObj);
            } else if (param == "-normType" || param == "-norm_type") {
                double norm_type;
                if (Tcl_GetDoubleFromObj(interp, valueObj, &norm_type) != TCL_OK) {
                    throw std::runtime_error("Invalid normType value");
                }
                args.normType = norm_type;
            } else if (param == "-kernelSize" || param == "-kernel_size") {
                int single_kernel;
                if (Tcl_GetIntFromObj(interp, valueObj, &single_kernel) == TCL_OK) {
                    args.kernelSize = {single_kernel, single_kernel};
                } else {
                    args.kernelSize = ParseIntList2(interp, valueObj);
                }
            } else if (param == "-stride") {
                int single_stride;
                if (Tcl_GetIntFromObj(interp, valueObj, &single_stride) == TCL_OK) {
                    args.stride = {single_stride, single_stride};
                } else {
                    args.stride = ParseIntList2(interp, valueObj);
                }
            } else if (param == "-ceilMode" || param == "-ceil_mode") {
                int ceil_flag;
                if (Tcl_GetIntFromObj(interp, valueObj, &ceil_flag) != TCL_OK) {
                    std::string value = Tcl_GetString(valueObj);
                    if (value == "true") ceil_flag = 1;
                    else if (value == "false") ceil_flag = 0;
                    else throw std::runtime_error("Invalid ceilMode value");
                }
                args.ceilMode = (ceil_flag != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid: input must be specified, kernelSize must be positive, and normType must be positive");
    }
    
    // Set default stride if not specified
    if (args.stride.empty()) {
        args.stride = args.kernelSize;
    }
    
    return args;
}

// torch::lppool2d - 2D LP pooling
int TensorLpPool2d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        LpPool2dArgs args = ParseLpPool2dArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        
        torch::Tensor result = torch::nn::functional::lp_pool2d(input, 
            torch::nn::functional::LPPool2dFuncOptions(args.normType, args.kernelSize)
                .stride(args.stride).ceil_mode(args.ceilMode));
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::lppool3d args structure
struct LpPool3dArgs {
    std::string input;
    double normType = 2.0;  // Default L2 norm
    std::vector<int64_t> kernelSize;
    std::vector<int64_t> stride;  // empty means default to kernelSize
    bool ceilMode = false;
    
    bool IsValid() const {
        return !input.empty() && !kernelSize.empty() && 
               kernelSize.size() == 3 && normType > 0 &&
               kernelSize[0] > 0 && kernelSize[1] > 0 && kernelSize[2] > 0;
    }
};

LpPool3dArgs ParseLpPool3dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LpPool3dArgs args;

    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: input norm_type kernel_size ?stride? ?ceil_mode?
        if (objc < 4 || objc > 6) {
            Tcl_WrongNumArgs(interp, 1, objv, "input norm_type kernel_size ?stride? ?ceil_mode?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        args.input = Tcl_GetString(objv[1]);

        // norm_type
        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.normType) != TCL_OK) {
            throw std::runtime_error("Invalid norm type");
        }

        // kernel_size
        int single_kernel;
        if (Tcl_GetIntFromObj(interp, objv[3], &single_kernel) == TCL_OK) {
            args.kernelSize = {single_kernel, single_kernel, single_kernel};
        } else {
            args.kernelSize = ParseIntList3(interp, objv[3]);
        }

        // stride (optional)
        if (objc >= 5) {
            int single_stride;
            if (Tcl_GetIntFromObj(interp, objv[4], &single_stride) == TCL_OK) {
                args.stride = {single_stride, single_stride, single_stride};
            } else {
                args.stride = ParseIntList3(interp, objv[4]);
            }
        }

        // ceil_mode (optional)
        if (objc >= 6) {
            int ceilFlag;
            if (Tcl_GetIntFromObj(interp, objv[5], &ceilFlag) != TCL_OK) {
                throw std::runtime_error("Invalid ceil_mode value");
            }
            args.ceilMode = (ceilFlag != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* valueObj = objv[i + 1];

            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(valueObj);
            } else if (param == "-normType" || param == "-norm_type") {
                if (Tcl_GetDoubleFromObj(interp, valueObj, &args.normType) != TCL_OK) {
                    throw std::runtime_error("Invalid norm type");
                }
            } else if (param == "-kernelSize" || param == "-kernel_size") {
                int single_kernel;
                if (Tcl_GetIntFromObj(interp, valueObj, &single_kernel) == TCL_OK) {
                    args.kernelSize = {single_kernel, single_kernel, single_kernel};
                } else {
                    args.kernelSize = ParseIntList3(interp, valueObj);
                }
            } else if (param == "-stride") {
                int single_stride;
                if (Tcl_GetIntFromObj(interp, valueObj, &single_stride) == TCL_OK) {
                    args.stride = {single_stride, single_stride, single_stride};
                } else {
                    args.stride = ParseIntList3(interp, valueObj);
                }
            } else if (param == "-ceilMode" || param == "-ceil_mode") {
                int ceilFlag;
                if (Tcl_GetIntFromObj(interp, valueObj, &ceilFlag) != TCL_OK) {
                    throw std::runtime_error("Invalid ceil_mode value");
                }
                args.ceilMode = (ceilFlag != 0);
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

// torch::lppool3d - 3D LP pooling
int TensorLpPool3d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        LpPool3dArgs args = ParseLpPool3dArgs(interp, objc, objv);

        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        
        // Default stride to kernel_size if not specified
        std::vector<int64_t> stride = args.stride.empty() ? args.kernelSize : args.stride;
        
        torch::Tensor result = torch::nn::functional::lp_pool3d(input, 
            torch::nn::functional::LPPool3dFuncOptions(args.normType, args.kernelSize)
                .stride(stride).ceil_mode(args.ceilMode));
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Add before TensorAvgPool3d_Cmd definition
struct AvgPool3dArgs {
    std::string input;
    std::vector<int64_t> kernelSize;
    std::vector<int64_t> stride;   // empty means default
    std::vector<int64_t> padding = {0, 0, 0};
    bool countIncludePad = true;

    bool IsValid() const {
        return !input.empty() && !kernelSize.empty() &&
               (kernelSize.size() == 1 || kernelSize.size() == 3);
    }
};

static std::vector<int64_t> TclObjToIntVector3(Tcl_Interp* interp, Tcl_Obj* obj) {
    std::vector<int64_t> vec;
    int listLen;
    Tcl_Obj** listObjv;
    if (Tcl_ListObjGetElements(interp, obj, &listLen, &listObjv) != TCL_OK) {
        throw std::runtime_error("Expected list of ints");
    }
    if (listLen != 3) {
        throw std::runtime_error("List must have length 3");
    }
    for (int i = 0; i < 3; ++i) {
        int val;
        if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
            throw std::runtime_error("Invalid int in list");
        }
        vec.push_back(val);
    }
    return vec;
}

AvgPool3dArgs ParseAvgPool3dArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    AvgPool3dArgs args;

    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional: input kernel_size ?stride? ?padding? ?count_include_pad?
        if (objc < 3 || objc > 6) {
            Tcl_WrongNumArgs(interp, 1, objv, "input kernel_size ?stride? ?padding? ?count_include_pad?");
            throw std::runtime_error("Invalid number of arguments");
        }
        args.input = Tcl_GetString(objv[1]);

        // kernel_size parsing
        int single;
        if (Tcl_GetIntFromObj(interp, objv[2], &single) == TCL_OK) {
            args.kernelSize = {single, single, single};
        } else {
            args.kernelSize = TclObjToIntVector3(interp, objv[2]);
        }

        // stride
        if (objc >= 4) {
            if (Tcl_GetIntFromObj(interp, objv[3], &single) == TCL_OK) {
                args.stride = {single, single, single};
            } else {
                args.stride = TclObjToIntVector3(interp, objv[3]);
            }
        }

        // padding
        if (objc >= 5) {
            if (Tcl_GetIntFromObj(interp, objv[4], &single) == TCL_OK) {
                args.padding = {single, single, single};
            } else {
                args.padding = TclObjToIntVector3(interp, objv[4]);
            }
        }

        // count_include_pad
        if (objc >= 6) {
            int cip;
            if (Tcl_GetIntFromObj(interp, objv[5], &cip) != TCL_OK) {
                throw std::runtime_error("Invalid count_include_pad value");
            }
            args.countIncludePad = (cip != 0);
        }
    } else {
        // Named parameters
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* valueObj = objv[i + 1];
            std::string valueStr = Tcl_GetString(valueObj);
            int single;

            if (param == "-input" || param == "-tensor") {
                args.input = valueStr;
            } else if (param == "-kernelSize" || param == "-kernel_size") {
                if (Tcl_GetIntFromObj(interp, valueObj, &single) == TCL_OK) {
                    args.kernelSize = {single, single, single};
                } else {
                    args.kernelSize = TclObjToIntVector3(interp, valueObj);
                }
            } else if (param == "-stride") {
                if (Tcl_GetIntFromObj(interp, valueObj, &single) == TCL_OK) {
                    args.stride = {single, single, single};
                } else {
                    args.stride = TclObjToIntVector3(interp, valueObj);
                }
            } else if (param == "-padding") {
                if (Tcl_GetIntFromObj(interp, valueObj, &single) == TCL_OK) {
                    args.padding = {single, single, single};
                } else {
                    args.padding = TclObjToIntVector3(interp, valueObj);
                }
            } else if (param == "-countIncludePad" || param == "-count_include_pad") {
                int cip;
                if (Tcl_GetIntFromObj(interp, valueObj, &cip) != TCL_OK) {
                    throw std::runtime_error("Invalid countIncludePad value");
                }
                args.countIncludePad = (cip != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters: -input and -kernelSize");
    }

    if (args.stride.empty()) {
        args.stride = args.kernelSize;
    }

    return args;
}

// Replace existing implementation of TensorAvgPool3d_Cmd
int TensorAvgPool3d_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        AvgPool3dArgs args = ParseAvgPool3dArgs(interp, objc, objv);

        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        auto& input = tensor_storage[args.input];

        torch::Tensor result = torch::avg_pool3d(
            input,
            args.kernelSize,
            args.stride,
            args.padding,
            /*ceil_mode=*/false,
            args.countIncludePad);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 