#include "libtorchtcl.h"
#include <torch/torch.h>
#include <torch/script.h>

// Parameter structure for pixel_shuffle command
struct PixelShuffleArgs {
    std::string input;
    int upscale_factor = 2;  // Default value
    
    bool IsValid() const {
        return !input.empty() && upscale_factor > 0;
    }
};

// Parse dual syntax for pixel_shuffle
PixelShuffleArgs ParsePixelShuffleArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    PixelShuffleArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::pixel_shuffle input upscale_factor | torch::pixel_shuffle -input tensor -upscaleFactor int");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::pixel_shuffle input upscale_factor");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.upscale_factor) != TCL_OK) {
            throw std::runtime_error("Invalid upscale_factor parameter");
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
            } else if (param == "-upscaleFactor" || param == "-upscale_factor" || param == "-factor") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.upscale_factor) != TCL_OK) {
                    throw std::runtime_error("Invalid upscale_factor parameter");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -input and -upscaleFactor");
    }
    
    return args;
}

// torch::pixel_shuffle - Pixel shuffle for upsampling with dual syntax support
int PixelShuffle_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        PixelShuffleArgs args = ParsePixelShuffleArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto output = torch::nn::functional::pixel_shuffle(input, args.upscale_factor);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for pixel_unshuffle command
struct PixelUnshuffleArgs {
    std::string input;
    int downscale_factor = 2;  // Default value
    
    bool IsValid() const {
        return !input.empty() && downscale_factor > 0;
    }
};

// Parse dual syntax for pixel_unshuffle
PixelUnshuffleArgs ParsePixelUnshuffleArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    PixelUnshuffleArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::pixel_unshuffle input downscale_factor | torch::pixel_unshuffle -input tensor -downscaleFactor int");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::pixel_unshuffle input downscale_factor | torch::pixel_unshuffle -input tensor -downscaleFactor int");
        }
        
        args.input = Tcl_GetString(objv[1]);
        
        if (Tcl_GetIntFromObj(interp, objv[2], &args.downscale_factor) != TCL_OK) {
            throw std::runtime_error("Invalid downscale_factor parameter");
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
            } else if (param == "-downscaleFactor" || param == "-downscale_factor" || param == "-factor") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.downscale_factor) != TCL_OK) {
                    throw std::runtime_error("Invalid downscale_factor parameter");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -input and -downscaleFactor");
    }
    
    return args;
}

// torch::pixel_unshuffle - Pixel unshuffle for downsampling with dual syntax support
int PixelUnshuffle_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        PixelUnshuffleArgs args = ParsePixelUnshuffleArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto output = torch::nn::functional::pixel_unshuffle(input, args.downscale_factor);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::upsample_nearest - Nearest neighbor upsampling with dual syntax support
typedef struct {
    torch::Tensor input;
    std::vector<int64_t> size;
    std::optional<std::vector<double>> scale_factor;
    
    bool IsValid() const { 
        return input.defined() && (!size.empty() || scale_factor.has_value()); 
    }
} UpsampleNearestArgs;

static UpsampleNearestArgs ParseUpsampleNearestArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    UpsampleNearestArgs args;

    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: input size ?scale_factor?
        if (objc < 3 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "input size ?scale_factor?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        std::string inputName = Tcl_GetString(objv[1]);
        if (tensor_storage.find(inputName) == tensor_storage.end()) 
            throw std::runtime_error("Invalid input tensor name");
        args.input = tensor_storage[inputName];

        // Parse size list
        int listLen;
        Tcl_Obj** listObjv;
        if (Tcl_ListObjGetElements(interp, objv[2], &listLen, &listObjv) != TCL_OK) {
            throw std::runtime_error("Failed to parse size list");
        }

        for (int i = 0; i < listLen; i++) {
            int size_val;
            if (Tcl_GetIntFromObj(interp, listObjv[i], &size_val) != TCL_OK) {
                throw std::runtime_error("Invalid size list element");
            }
            args.size.push_back(size_val);
        }

        // Optional scale_factor (not used in current implementation but kept for compatibility)
        if (objc > 3) {
            // Parse scale_factor list
            if (Tcl_ListObjGetElements(interp, objv[3], &listLen, &listObjv) != TCL_OK) {
                throw std::runtime_error("Failed to parse scale_factor list");
            }

            std::vector<double> scale_factors;
            for (int i = 0; i < listLen; i++) {
                double scale_val;
                if (Tcl_GetDoubleFromObj(interp, listObjv[i], &scale_val) != TCL_OK) {
                    throw std::runtime_error("Invalid scale_factor list element");
                }
                scale_factors.push_back(scale_val);
            }
            args.scale_factor = scale_factors;
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) throw std::runtime_error("Missing value for parameter");
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* valObj = objv[i+1];
            
            if (param == "-input") {
                std::string name = Tcl_GetString(valObj);
                if (tensor_storage.find(name) == tensor_storage.end()) 
                    throw std::runtime_error("Invalid input tensor name");
                args.input = tensor_storage[name];
            } else if (param == "-size") {
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, valObj, &listLen, &listObjv) != TCL_OK) {
                    throw std::runtime_error("Failed to parse size list");
                }

                for (int j = 0; j < listLen; j++) {
                    int size_val;
                    if (Tcl_GetIntFromObj(interp, listObjv[j], &size_val) != TCL_OK) {
                        throw std::runtime_error("Invalid size list element");
                    }
                    args.size.push_back(size_val);
                }
            } else if (param == "-scale_factor") {
                int listLen;
                Tcl_Obj** listObjv;
                if (Tcl_ListObjGetElements(interp, valObj, &listLen, &listObjv) != TCL_OK) {
                    throw std::runtime_error("Failed to parse scale_factor list");
                }

                std::vector<double> scale_factors;
                for (int j = 0; j < listLen; j++) {
                    double scale_val;
                    if (Tcl_GetDoubleFromObj(interp, listObjv[j], &scale_val) != TCL_OK) {
                        throw std::runtime_error("Invalid scale_factor list element");
                    }
                    scale_factors.push_back(scale_val);
                }
                args.scale_factor = scale_factors;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) 
        throw std::runtime_error("Parameter -input and either -size or -scale_factor are required");
    return args;
}

int UpsampleNearest_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        UpsampleNearestArgs args = ParseUpsampleNearestArgs(interp, objc, objv);
        
        torch::nn::functional::InterpolateFuncOptions options;
        options = options.mode(torch::kNearest);
        
        if (!args.size.empty()) {
            options = options.size(args.size);
        } else if (args.scale_factor.has_value()) {
            options = options.scale_factor(args.scale_factor.value());
        }

        auto output = torch::nn::functional::interpolate(args.input, options);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::upsample_bilinear - Bilinear upsampling with dual syntax support
typedef struct {
    torch::Tensor input;
    std::optional<std::vector<int64_t>> output_size;
    std::optional<std::vector<double>> scale_factor;
    bool align_corners = false;
    bool antialias = false;

    bool IsValid() const { 
        return input.defined() && (output_size.has_value() || scale_factor.has_value()); 
    }
} UpsampleBilinearArgs;

// Helper function to parse size or scale factor list
static std::vector<int64_t> ParseSizeList(Tcl_Interp* interp, Tcl_Obj* obj) {
    int listLen;
    Tcl_Obj** listObjv;
    if (Tcl_ListObjGetElements(interp, obj, &listLen, &listObjv) != TCL_OK) {
        throw std::runtime_error("Failed to parse size list");
    }

    std::vector<int64_t> result;
    for (int i = 0; i < listLen; i++) {
        int val;
        if (Tcl_GetIntFromObj(interp, listObjv[i], &val) != TCL_OK) {
            throw std::runtime_error("Invalid size list element");
        }
        result.push_back(val);
    }
    return result;
}

// Helper function to parse scale factor list (doubles)
static std::vector<double> ParseScaleFactorList(Tcl_Interp* interp, Tcl_Obj* obj) {
    int listLen;
    Tcl_Obj** listObjv;
    if (Tcl_ListObjGetElements(interp, obj, &listLen, &listObjv) != TCL_OK) {
        throw std::runtime_error("Failed to parse scale factor list");
    }

    std::vector<double> result;
    for (int i = 0; i < listLen; i++) {
        double val;
        if (Tcl_GetDoubleFromObj(interp, listObjv[i], &val) != TCL_OK) {
            throw std::runtime_error("Invalid scale factor list element");
        }
        result.push_back(val);
    }
    return result;
}

static UpsampleBilinearArgs ParseUpsampleBilinearArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    UpsampleBilinearArgs args;

    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: input size|scale_factor ?align_corners? ?antialias?
        if (objc < 3 || objc > 5) {
            Tcl_WrongNumArgs(interp, 1, objv, "input size|scale_factor ?align_corners? ?antialias?");
            throw std::runtime_error("Invalid number of arguments");
        }
        
        std::string inputName = Tcl_GetString(objv[1]);
        if (tensor_storage.find(inputName) == tensor_storage.end()) 
            throw std::runtime_error("Invalid input tensor name");
        args.input = tensor_storage[inputName];

        // Try to parse as size (integers) first, then as scale factor (doubles)
        std::string sizeOrScale = Tcl_GetString(objv[2]);
        try {
            args.output_size = ParseSizeList(interp, objv[2]);
        } catch (...) {
            try {
                args.scale_factor = ParseScaleFactorList(interp, objv[2]);
            } catch (...) {
                throw std::runtime_error("Invalid size or scale_factor parameter");
            }
        }

        if (objc > 3) {
            int align_int;
            if (Tcl_GetIntFromObj(interp, objv[3], &align_int) != TCL_OK) 
                throw std::runtime_error("Invalid align_corners");
            args.align_corners = (align_int != 0);
        }
        
        if (objc > 4) {
            int antialias_int;
            if (Tcl_GetIntFromObj(interp, objv[4], &antialias_int) != TCL_OK) 
                throw std::runtime_error("Invalid antialias");
            args.antialias = (antialias_int != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) throw std::runtime_error("Missing value for parameter");
            std::string param = Tcl_GetString(objv[i]);
            Tcl_Obj* valObj = objv[i+1];
            
            if (param == "-input") {
                std::string name = Tcl_GetString(valObj);
                if (tensor_storage.find(name) == tensor_storage.end()) 
                    throw std::runtime_error("Invalid input tensor name");
                args.input = tensor_storage[name];
            } else if (param == "-output_size" || param == "-size") {
                args.output_size = ParseSizeList(interp, valObj);
            } else if (param == "-scale_factor") {
                args.scale_factor = ParseScaleFactorList(interp, valObj);
            } else if (param == "-align_corners") {
                int val;
                if (Tcl_GetIntFromObj(interp, valObj, &val) != TCL_OK) 
                    throw std::runtime_error("Invalid align_corners");
                args.align_corners = (val != 0);
            } else if (param == "-antialias") {
                int val;
                if (Tcl_GetIntFromObj(interp, valObj, &val) != TCL_OK) 
                    throw std::runtime_error("Invalid antialias");
                args.antialias = (val != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) 
        throw std::runtime_error("Parameter -input and either -output_size or -scale_factor are required");
    return args;
}

int UpsampleBilinear_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        UpsampleBilinearArgs args = ParseUpsampleBilinearArgs(interp, objc, objv);
        
        torch::nn::functional::InterpolateFuncOptions options;
        options = options.mode(torch::kBilinear).align_corners(args.align_corners);
        
        if (args.output_size.has_value()) {
            options = options.size(args.output_size.value());
        }
        
        if (args.scale_factor.has_value()) {
            options = options.scale_factor(args.scale_factor.value());
        }
        
        // Set antialias if supported (newer PyTorch versions)
        // Note: antialias is available in some PyTorch versions for interpolate
        if (args.antialias) {
            try {
                options = options.antialias(args.antialias);
            } catch (...) {
                // Silently ignore if antialias is not supported in this PyTorch version
            }
        }

        torch::Tensor output = torch::nn::functional::interpolate(args.input, options);

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
// Dual Syntax Parser for torch::interpolate
// ============================================================================
struct InterpolateArgs {
    std::string input;
    std::vector<int64_t> size;
    std::string mode = "nearest";      // Default mode
    bool align_corners = false;        // Default align_corners
    std::optional<std::vector<double>> scale_factor;  // Optional scale factor
    
    bool IsValid() const {
        return !input.empty() && (!size.empty() || scale_factor.has_value());
    }
};



InterpolateArgs ParseInterpolateArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    InterpolateArgs args;
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): input size ?mode? ?align_corners? ?scale_factor?
        if (objc < 3 || objc > 6) {
            throw std::runtime_error("Usage: torch::interpolate input size ?mode? ?align_corners? ?scale_factor?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.size = ParseSizeList(interp, objv[2]);
        
        if (objc > 3) {
            args.mode = Tcl_GetString(objv[3]);
        }
        
        if (objc > 4) {
            int align_int;
            if (Tcl_GetIntFromObj(interp, objv[4], &align_int) != TCL_OK) {
                throw std::runtime_error("Invalid align_corners parameter: must be integer (0 or 1)");
            }
            args.align_corners = (align_int != 0);
        }
        
        if (objc > 5) {
            args.scale_factor = ParseScaleFactorList(interp, objv[5]);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-size") {
                args.size = ParseSizeList(interp, objv[i + 1]);
            } else if (param == "-mode") {
                args.mode = Tcl_GetString(objv[i + 1]);
            } else if (param == "-align_corners" || param == "-alignCorners") {
                int align_int;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &align_int) != TCL_OK) {
                    throw std::runtime_error("Invalid align_corners parameter: must be integer (0 or 1)");
                }
                args.align_corners = (align_int != 0);
            } else if (param == "-scale_factor" || param == "-scaleFactor") {
                args.scale_factor = ParseScaleFactorList(interp, objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor, -size, -mode, -align_corners/-alignCorners, -scale_factor/-scaleFactor");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and either size or scale_factor required");
    }
    
    return args;
}

// torch::interpolate - General interpolation with dual syntax support
int Interpolate_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        InterpolateArgs args = ParseInterpolateArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            throw std::runtime_error("Invalid input tensor: " + args.input);
        }

        auto input = tensor_storage[args.input];

        // Parse mode
        torch::nn::functional::InterpolateFuncOptions::mode_t mode = torch::kNearest;
        if (args.mode == "nearest") {
            mode = torch::kNearest;
        } else if (args.mode == "linear") {
            mode = torch::kLinear;
        } else if (args.mode == "bilinear") {
            mode = torch::kBilinear;
        } else if (args.mode == "bicubic") {
            mode = torch::kBicubic;
        } else if (args.mode == "trilinear") {
            mode = torch::kTrilinear;
        } else if (args.mode == "area") {
            mode = torch::kArea;
        } else {
            throw std::runtime_error("Invalid mode: " + args.mode + ". Valid modes are: nearest, linear, bilinear, bicubic, trilinear, area");
        }

        auto options = torch::nn::functional::InterpolateFuncOptions()
                          .mode(mode);

        // Only set align_corners for interpolating modes that support it
        if (args.mode == "linear" || args.mode == "bilinear" || args.mode == "bicubic" || args.mode == "trilinear") {
            options = options.align_corners(args.align_corners);
        }

        // Set either size or scale_factor
        if (!args.size.empty()) {
            options = options.size(args.size);
        } else if (args.scale_factor.has_value()) {
            options = options.scale_factor(args.scale_factor.value());
        }

        auto output = torch::nn::functional::interpolate(input, options);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in interpolate: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for grid_sample command
struct GridSampleArgs {
    std::string input;
    std::string grid;
    std::string mode = "bilinear";        // Default interpolation mode
    std::string padding_mode = "zeros";   // Default padding mode
    bool align_corners = false;           // Default align corners
    
    bool IsValid() const {
        return !input.empty() && !grid.empty();
    }
};

// Parse dual syntax for grid_sample command
GridSampleArgs ParseGridSampleArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    GridSampleArgs args;
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility): input grid ?mode? ?padding_mode? ?align_corners?
        if (objc < 3 || objc > 6) {
            throw std::runtime_error("Usage: torch::grid_sample input grid ?mode? ?padding_mode? ?align_corners?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.grid = Tcl_GetString(objv[2]);
        
        if (objc > 3) {
            args.mode = Tcl_GetString(objv[3]);
        }
        
        if (objc > 4) {
            args.padding_mode = Tcl_GetString(objv[4]);
        }
        
        if (objc > 5) {
            int align_int;
            if (Tcl_GetIntFromObj(interp, objv[5], &align_int) != TCL_OK) {
                throw std::runtime_error("Invalid align_corners parameter: must be integer");
            }
            args.align_corners = (align_int != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-grid") {
                args.grid = Tcl_GetString(objv[i + 1]);
            } else if (param == "-mode") {
                args.mode = Tcl_GetString(objv[i + 1]);
            } else if (param == "-padding_mode" || param == "-paddingMode") {
                args.padding_mode = Tcl_GetString(objv[i + 1]);
            } else if (param == "-align_corners" || param == "-alignCorners") {
                int align_int;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &align_int) != TCL_OK) {
                    throw std::runtime_error("Invalid align_corners parameter: must be integer");
                }
                args.align_corners = (align_int != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor, -grid, -mode, -padding_mode/-paddingMode, -align_corners/-alignCorners");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input and grid tensors required");
    }
    
    return args;
}

// torch::grid_sample - Grid sampling with dual syntax support
int GridSample_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        GridSampleArgs args = ParseGridSampleArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.grid) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid grid tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto grid = tensor_storage[args.grid];

        // Parse mode
        torch::nn::functional::GridSampleFuncOptions::mode_t mode = torch::kBilinear;
        if (args.mode == "nearest") {
            mode = torch::kNearest;
        } else if (args.mode == "bilinear") {
            mode = torch::kBilinear;
        } else if (args.mode != "bilinear") {
            throw std::runtime_error("Invalid mode: " + args.mode + ". Valid modes are: bilinear, nearest");
        }

        // Parse padding mode
        torch::nn::functional::GridSampleFuncOptions::padding_mode_t padding_mode = torch::kZeros;
        if (args.padding_mode == "zeros") {
            padding_mode = torch::kZeros;
        } else if (args.padding_mode == "border") {
            padding_mode = torch::kBorder;
        } else if (args.padding_mode == "reflection") {
            padding_mode = torch::kReflection;
        } else {
            throw std::runtime_error("Invalid padding_mode: " + args.padding_mode + ". Valid modes are: zeros, border, reflection");
        }

        auto options = torch::nn::functional::GridSampleFuncOptions()
                          .mode(mode)
                          .padding_mode(padding_mode)
                          .align_corners(args.align_corners);

        auto output = torch::nn::functional::grid_sample(input, grid, options);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::affine_grid - Affine grid generation
struct AffineGridArgs {
    std::string theta;
    std::vector<int64_t> size;
    bool alignCorners = false;

    bool IsValid() const {
        return !theta.empty() && !size.empty();
    }
};

static AffineGridArgs ParseAffineGridArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    AffineGridArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 4) {
            throw std::runtime_error("Usage: torch::affine_grid theta size ?align_corners?");
        }
        
        args.theta = Tcl_GetString(objv[1]);
        args.size = ParseSizeList(interp, objv[2]);
        
        if (objc > 3) {
            int align_int;
            if (Tcl_GetIntFromObj(interp, objv[3], &align_int) == TCL_OK) {
                args.alignCorners = (align_int != 0);
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for option " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string option = Tcl_GetString(objv[i]);
            
            if (option == "-theta") {
                args.theta = Tcl_GetString(objv[i + 1]);
            } else if (option == "-size") {
                args.size = ParseSizeList(interp, objv[i + 1]);
            } else if (option == "-alignCorners" || option == "-align_corners") {
                int align_int;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &align_int) == TCL_OK) {
                    args.alignCorners = (align_int != 0);
                } else {
                    throw std::runtime_error("Invalid align_corners value: " + std::string(Tcl_GetString(objv[i + 1])));
                }
            } else {
                throw std::runtime_error("Unknown option: " + option);
            }
        }
    }
    
    return args;
}

int AffineGrid_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        AffineGridArgs args = ParseAffineGridArgs(interp, objc, objv);
        
        if (!args.IsValid()) {
            Tcl_SetResult(interp, const_cast<char*>("Required parameters: theta, size"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        if (tensor_storage.find(args.theta) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid theta tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto theta = tensor_storage[args.theta];
        auto output = torch::nn::functional::affine_grid(theta, args.size, args.alignCorners);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for channel_shuffle command
struct ChannelShuffleArgs {
    std::string input;
    int groups = 2;  // Default value
    
    bool IsValid() const {
        return !input.empty() && groups > 0;
    }
};

// Parse dual syntax for channel_shuffle
ChannelShuffleArgs ParseChannelShuffleArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ChannelShuffleArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::channel_shuffle input groups | torch::channelShuffle -input tensor -groups num_groups");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::channel_shuffle input groups");
        }
        args.input = Tcl_GetString(objv[1]);
        if (Tcl_GetIntFromObj(interp, objv[2], &args.groups) != TCL_OK) {
            throw std::runtime_error("Invalid groups parameter");
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
            } else if (param == "-groups") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.groups) != TCL_OK) {
                    throw std::runtime_error("Invalid groups parameter value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input, -tensor, -groups");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and groups (> 0) required");
    }
    
    return args;
}

// torch::channel_shuffle - Channel shuffle
int ChannelShuffle_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        ChannelShuffleArgs args = ParseChannelShuffleArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];

        // Use tensor operations since channel_shuffle may not be available
        auto output = input.view({input.size(0), args.groups, input.size(1)/args.groups, input.size(2), input.size(3)})
                          .transpose(1, 2).contiguous()
                          .view({input.size(0), input.size(1), input.size(2), input.size(3)});

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for nms command
struct NmsArgs {
    std::string boxes;
    std::string scores;
    double iouThreshold;
    double scoreThreshold = 0.0;  // Optional, default 0.0
    
    bool IsValid() const {
        return !boxes.empty() && !scores.empty();
    }
};

// Parse dual syntax for nms
NmsArgs ParseNmsArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    NmsArgs args;
    
    // Minimum arguments check
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::nms boxes scores iou_threshold ?score_threshold? | torch::nms -boxes boxes -scores scores -iouThreshold value ?-scoreThreshold value?");
    }
    
    // Safely check if this is named parameter syntax
    bool isNamedSyntax = false;
    if (objc >= 2) {
        const char* firstArg = Tcl_GetString(objv[1]);
        if (firstArg && firstArg[0] == '-') {
            isNamedSyntax = true;
        }
    }
    
    if (!isNamedSyntax) {
        // Positional syntax (backward compatibility)
        if (objc < 4) {
            throw std::runtime_error("Usage: torch::nms boxes scores iou_threshold ?score_threshold?");
        }
        if (objc > 5) {
            throw std::runtime_error("Usage: torch::nms boxes scores iou_threshold ?score_threshold?");
        }
        
        // Safe access to positional arguments
        args.boxes = Tcl_GetString(objv[1]);
        args.scores = Tcl_GetString(objv[2]);
        
        // Parse iou_threshold
        if (Tcl_GetDoubleFromObj(interp, objv[3], &args.iouThreshold) != TCL_OK) {
            throw std::runtime_error("Invalid iou_threshold value - expected floating-point number");
        }
        
        // Parse optional score_threshold
        if (objc > 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &args.scoreThreshold) != TCL_OK) {
                throw std::runtime_error("Invalid score_threshold value - expected floating-point number");
            }
        }
    } else {
        // Named parameter syntax
        bool boxes_set = false;
        bool scores_set = false;
        bool iou_threshold_set = false;
        
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-boxes") {
                args.boxes = Tcl_GetString(objv[i + 1]);
                boxes_set = true;
            } else if (param == "-scores") {
                args.scores = Tcl_GetString(objv[i + 1]);
                scores_set = true;
            } else if (param == "-iouThreshold") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.iouThreshold) != TCL_OK) {
                    throw std::runtime_error("Invalid iouThreshold value - expected floating-point number");
                }
                iou_threshold_set = true;
            } else if (param == "-scoreThreshold") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.scoreThreshold) != TCL_OK) {
                    throw std::runtime_error("Invalid scoreThreshold value - expected floating-point number");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -boxes, -scores, -iouThreshold, -scoreThreshold");
            }
        }
        
        // Check required parameters
        if (!boxes_set || !scores_set || !iou_threshold_set) {
            throw std::runtime_error("Named syntax requires at least -boxes, -scores, and -iouThreshold parameters");
        }
    }
    
    // Validate required parameters
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: boxes and scores tensors required");
    }
    
    return args;
}

// torch::nms - Non-maximum suppression (simplified implementation)
int NMS_Cmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)cd; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        NmsArgs args = ParseNmsArgs(interp, objc, objv);
        
        // Validate tensors exist
        if (tensor_storage.find(args.boxes) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid boxes tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.scores) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid scores tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Get tensors from storage
        auto boxes = tensor_storage[args.boxes];
        auto scores = tensor_storage[args.scores];
        
        // Validate tensor dimensions
        if (boxes.dim() != 2 || boxes.size(1) != 4) {
            Tcl_SetResult(interp, const_cast<char*>("Boxes tensor must be 2D with shape [N, 4]"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (scores.dim() != 1 || scores.size(0) != boxes.size(0)) {
            Tcl_SetResult(interp, const_cast<char*>("Scores tensor must be 1D with same length as boxes"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Validate threshold values
        if (args.iouThreshold < 0.0 || args.iouThreshold > 1.0) {
            Tcl_SetResult(interp, const_cast<char*>("iouThreshold must be between 0.0 and 1.0"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        // Apply score threshold if specified
        torch::Tensor keep_scores;
        torch::Tensor keep_boxes;
        if (args.scoreThreshold > 0.0) {
            auto mask = scores > args.scoreThreshold;
            keep_scores = scores.masked_select(mask);
            keep_boxes = boxes.index_select(0, mask.nonzero().squeeze());
        } else {
            keep_scores = scores;
            keep_boxes = boxes;
        }

        // Get sorted indices by score
        auto sorted_indices = std::get<1>(keep_scores.sort(0, true));
        keep_boxes = keep_boxes.index_select(0, sorted_indices);
        keep_scores = keep_scores.index_select(0, sorted_indices);

        // Calculate areas for all boxes
        auto x1 = keep_boxes.select(1, 0);
        auto y1 = keep_boxes.select(1, 1);
        auto x2 = keep_boxes.select(1, 2);
        auto y2 = keep_boxes.select(1, 3);
        auto areas = (x2 - x1) * (y2 - y1);

        std::vector<int64_t> keep_indices;
        auto num_boxes = keep_boxes.size(0);

        // NMS main loop
        for (int64_t i = 0; i < num_boxes; i++) {
            bool keep = true;
            
            // Compare with all previously kept boxes
            for (int64_t j : keep_indices) {
                // Calculate IoU between boxes[i] and boxes[j]
                auto xx1 = std::max(x1[i].item<float>(), x1[j].item<float>());
                auto yy1 = std::max(y1[i].item<float>(), y1[j].item<float>());
                auto xx2 = std::min(x2[i].item<float>(), x2[j].item<float>());
                auto yy2 = std::min(y2[i].item<float>(), y2[j].item<float>());

                auto w = std::max(0.0f, xx2 - xx1);
                auto h = std::max(0.0f, yy2 - yy1);
                auto inter = w * h;
                auto ovr = inter / (areas[i].item<float>() + areas[j].item<float>() - inter);

                if (ovr > args.iouThreshold) {
                    keep = false;
                    break;
                }
            }

            if (keep) {
                keep_indices.push_back(i);
            }
        }

        // Convert kept indices back to original indices
        auto output = torch::tensor(keep_indices, torch::kLong);
        output = sorted_indices.index_select(0, output);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for box_iou command
struct BoxIouArgs {
    std::string boxes1;
    std::string boxes2;
    
    bool IsValid() const {
        return !boxes1.empty() && !boxes2.empty();
    }
};

// Parse dual syntax for box_iou
BoxIouArgs ParseBoxIouArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    BoxIouArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::box_iou boxes1 boxes2 | torch::box_iou -boxes1 tensor1 -boxes2 tensor2");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            throw std::runtime_error("Usage: torch::box_iou boxes1 boxes2");
        }
        args.boxes1 = Tcl_GetString(objv[1]);
        args.boxes2 = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-boxes1" || param == "-input1") {
                args.boxes1 = value;
            } else if (param == "-boxes2" || param == "-input2") {
                args.boxes2 = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -boxes1, -boxes2, -input1, -input2");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: boxes1 and boxes2 tensors required");
    }
    
    return args;
}

// torch::box_iou - Bounding box IoU calculation
int BoxIoU_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        BoxIouArgs args = ParseBoxIouArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.boxes1) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid boxes1 tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.boxes2) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid boxes2 tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto boxes1 = tensor_storage[args.boxes1];
        auto boxes2 = tensor_storage[args.boxes2];

        // Calculate IoU: intersection / union
        auto area1 = (boxes1.select(1, 2) - boxes1.select(1, 0)) * (boxes1.select(1, 3) - boxes1.select(1, 1));
        auto area2 = (boxes2.select(1, 2) - boxes2.select(1, 0)) * (boxes2.select(1, 3) - boxes2.select(1, 1));
        
        auto lt = torch::max(boxes1.unsqueeze(1).select(2, 0).unsqueeze(2), 
                           boxes2.unsqueeze(0).select(1, 0).unsqueeze(1));
        auto rb = torch::min(boxes1.unsqueeze(1).select(2, 2).unsqueeze(2), 
                           boxes2.unsqueeze(0).select(1, 2).unsqueeze(1));
        
        auto wh = (rb - lt).clamp(0);
        auto inter = wh.select(2, 0) * wh.select(2, 1);
        auto union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter;
        auto iou = inter / union_area;

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = iou;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for roi_align command
struct RoIAlignArgs {
    std::string input;
    std::string boxes;
    std::vector<int64_t> outputSize;
    double spatialScale = 1.0;
    int samplingRatio = -1;
    bool aligned = true;
    
    bool IsValid() const {
        return !input.empty() && !boxes.empty() && !outputSize.empty();
    }
};

// Parse dual syntax for roi_align
RoIAlignArgs ParseRoIAlignArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    RoIAlignArgs args;
    
    if (objc < 4) {
        throw std::runtime_error("Usage: torch::roi_align input boxes output_size ?spatial_scale? ?sampling_ratio? ?aligned? | torch::roi_align -input tensor -boxes tensor -outputSize {size...} ?-spatialScale double? ?-samplingRatio int? ?-aligned bool?");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 4 || objc > 7) {
            throw std::runtime_error("Usage: torch::roi_align input boxes output_size ?spatial_scale? ?sampling_ratio? ?aligned?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.boxes = Tcl_GetString(objv[2]);
        args.outputSize = ParseSizeList(interp, objv[3]);
        
        if (objc > 4) {
            if (Tcl_GetDoubleFromObj(interp, objv[4], &args.spatialScale) != TCL_OK) {
                throw std::runtime_error("Invalid spatial_scale value");
            }
        }
        
        if (objc > 5) {
            if (Tcl_GetIntFromObj(interp, objv[5], &args.samplingRatio) != TCL_OK) {
                throw std::runtime_error("Invalid sampling_ratio value");
            }
        }
        
        if (objc > 6) {
            int aligned_int;
            if (Tcl_GetBooleanFromObj(interp, objv[6], &aligned_int) != TCL_OK) {
                throw std::runtime_error("Invalid aligned value");
            }
            args.aligned = (aligned_int != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            std::string param = Tcl_GetString(objv[i]);
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter: " + param);
            }
            
            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-boxes") {
                args.boxes = Tcl_GetString(objv[i + 1]);
            } else if (param == "-outputSize" || param == "-output_size") {
                args.outputSize = ParseSizeList(interp, objv[i + 1]);
            } else if (param == "-spatialScale" || param == "-spatial_scale") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.spatialScale) != TCL_OK) {
                    throw std::runtime_error("Invalid spatialScale value");
                }
            } else if (param == "-samplingRatio" || param == "-sampling_ratio") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.samplingRatio) != TCL_OK) {
                    throw std::runtime_error("Invalid samplingRatio value");
                }
            } else if (param == "-aligned") {
                int aligned_int;
                if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &aligned_int) != TCL_OK) {
                    throw std::runtime_error("Invalid aligned value");
                }
                args.aligned = (aligned_int != 0);
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

// torch::roi_align - ROI Align (simplified implementation)
int RoIAlign_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        RoIAlignArgs args = ParseRoIAlignArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.boxes) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid boxes tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto input = tensor_storage[args.input];
        auto boxes = tensor_storage[args.boxes];

        // Simplified ROI align using adaptive pooling
        auto options = torch::nn::functional::AdaptiveAvgPool2dFuncOptions(args.outputSize);
        auto output = torch::nn::functional::adaptive_avg_pool2d(input, options);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Dual-syntax support for ROI Pool
struct RoiPoolArgs {
    std::string input;
    std::string boxes;
    std::vector<int64_t> outputSize; // length 2
    double spatialScale = 1.0;

    bool IsValid() const {
        return !input.empty() && !boxes.empty() && !outputSize.empty() && (outputSize.size()==2);
    }
};

static RoiPoolArgs ParseRoiPoolArgs(Tcl_Interp* interp,int objc,Tcl_Obj* const objv[]){
    RoiPoolArgs a;
    if(objc>=2 && Tcl_GetString(objv[1])[0] != '-'){
        // positional: input boxes output_size ?spatial_scale?
        if(objc<4 || objc>5){ Tcl_WrongNumArgs(interp,1,objv,"input boxes output_size ?spatial_scale?"); throw std::runtime_error("Invalid arg count"); }
        a.input = Tcl_GetString(objv[1]);
        a.boxes = Tcl_GetString(objv[2]);
        a.outputSize = ParseSizeList(interp,objv[3]);
        if(objc>4){ if(Tcl_GetDoubleFromObj(interp,objv[4],&a.spatialScale)!=TCL_OK) throw std::runtime_error("Invalid spatial_scale"); }
    } else {
        for(int i=1;i<objc;i+=2){ if(i+1>=objc) throw std::runtime_error("Missing value for parameter"); std::string p=Tcl_GetString(objv[i]);
            if(p=="-input"||p=="-tensor") a.input = Tcl_GetString(objv[i+1]);
            else if(p=="-boxes") a.boxes = Tcl_GetString(objv[i+1]);
            else if(p=="-outputSize"||p=="-output_size") a.outputSize = ParseSizeList(interp,objv[i+1]);
            else if(p=="-spatialScale"||p=="-spatial_scale") {if(Tcl_GetDoubleFromObj(interp,objv[i+1],&a.spatialScale)!=TCL_OK) throw std::runtime_error("Invalid spatialScale");}
            else throw std::runtime_error("Unknown parameter: "+p);
        }
    }
    if(!a.IsValid()) throw std::runtime_error("Required parameters missing or invalid");
    return a;
}

// Refactor RoIPool_Cmd using dual syntax
int RoIPool_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        RoiPoolArgs args = ParseRoiPoolArgs(interp,objc,objv);
        if(tensor_storage.find(args.input)==tensor_storage.end()) { Tcl_SetResult(interp,const_cast<char*>("Invalid input tensor"),TCL_VOLATILE); return TCL_ERROR; }
        if(tensor_storage.find(args.boxes)==tensor_storage.end()) { Tcl_SetResult(interp,const_cast<char*>("Invalid boxes tensor"),TCL_VOLATILE); return TCL_ERROR; }
        auto input = tensor_storage[args.input];
        auto boxes = tensor_storage[args.boxes];

        // Simplified ROI pooling using adaptive max pooling on cropped regions (placeholder)
        auto output = torch::nn::functional::adaptive_max_pool2d(input, torch::nn::functional::AdaptiveMaxPool2dFuncOptions(args.outputSize));

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;
        Tcl_SetObjResult(interp,Tcl_NewStringObj(handle.c_str(),-1));
        return TCL_OK;
    } catch(const std::exception &e) {
        Tcl_SetResult(interp,const_cast<char*>(e.what()),TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for normalize_image command
struct NormalizeImageArgs {
    std::string image;
    std::string mean;
    std::string std;
    bool inplace = false;
    
    bool IsValid() const {
        return !image.empty() && !mean.empty() && !std.empty();
    }
};

// Parse dual syntax for normalize_image
NormalizeImageArgs ParseNormalizeImageArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    NormalizeImageArgs args;
    
    if (objc < 4) {
        throw std::runtime_error("Usage: torch::normalize_image image mean std ?inplace? | torch::normalize_image -image tensor -mean tensor -std tensor ?-inplace bool?");
    }
    
    if (objc >= 4 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 4 || objc > 5) {
            throw std::runtime_error("Usage: torch::normalize_image image mean std ?inplace?");
        }
        
        args.image = Tcl_GetString(objv[1]);
        args.mean = Tcl_GetString(objv[2]);
        args.std = Tcl_GetString(objv[3]);
        
        if (objc > 4) {
            int inplace_int;
            if (Tcl_GetIntFromObj(interp, objv[4], &inplace_int) != TCL_OK) {
                throw std::runtime_error("Invalid inplace value");
            }
            args.inplace = (inplace_int != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            std::string param = Tcl_GetString(objv[i]);
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter: " + param);
            }
            
            if (param == "-image") {
                args.image = Tcl_GetString(objv[i + 1]);
            } else if (param == "-mean") {
                args.mean = Tcl_GetString(objv[i + 1]);
            } else if (param == "-std") {
                args.std = Tcl_GetString(objv[i + 1]);
            } else if (param == "-inplace") {
                int inplace_int;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &inplace_int) != TCL_OK) {
                    throw std::runtime_error("Invalid inplace value");
                }
                args.inplace = (inplace_int != 0);
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

// torch::normalize_image - Image normalization with dual syntax support
int NormalizeImage_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        NormalizeImageArgs args = ParseNormalizeImageArgs(interp, objc, objv);
        
        // Validate tensor existence
        if (tensor_storage.find(args.image) == tensor_storage.end()) {
            throw std::runtime_error("Invalid image tensor");
        }
        if (tensor_storage.find(args.mean) == tensor_storage.end()) {
            throw std::runtime_error("Invalid mean tensor");
        }
        if (tensor_storage.find(args.std) == tensor_storage.end()) {
            throw std::runtime_error("Invalid std tensor");
        }
        
        auto image = tensor_storage[args.image];
        auto mean = tensor_storage[args.mean];
        auto std = tensor_storage[args.std];
        
        if (args.inplace) {
            // For inplace operation, modify the input tensor directly
            image.sub_(mean).div_(std);
            // Return the original tensor handle
            Tcl_SetObjResult(interp, Tcl_NewStringObj(args.image.c_str(), -1));
        } else {
            // For non-inplace operation, create a new tensor
            auto output = (image - mean) / std;
            std::string handle = GetNextHandle("tensor");
            tensor_storage[handle] = output;
            Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        }
        
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in normalize_image: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for denormalize_image command
struct DenormalizeImageArgs {
    std::string image;
    std::string mean;
    std::string std;
    bool inplace = false;
    
    bool IsValid() const {
        return !image.empty() && !mean.empty() && !std.empty();
    }
};

// Parse dual syntax for denormalize_image
DenormalizeImageArgs ParseDenormalizeImageArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DenormalizeImageArgs args;
    
    if (objc < 4) {
        throw std::runtime_error("Usage: torch::denormalize_image image mean std ?inplace? | torch::denormalize_image -image image -mean mean -std std ?-inplace inplace?");
    }
    
    if (objc >= 4 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 4 || objc > 5) {
            throw std::runtime_error("Usage: torch::denormalize_image image mean std ?inplace?");
        }
        
        args.image = Tcl_GetString(objv[1]);
        args.mean = Tcl_GetString(objv[2]);
        args.std = Tcl_GetString(objv[3]);
        
        if (objc > 4) {
            int inplace_int;
            if (Tcl_GetIntFromObj(interp, objv[4], &inplace_int) != TCL_OK) {
                throw std::runtime_error("Invalid inplace value");
            }
            args.inplace = (inplace_int != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            if (param == "-image") {
                args.image = Tcl_GetString(objv[i + 1]);
            } else if (param == "-mean") {
                args.mean = Tcl_GetString(objv[i + 1]);
            } else if (param == "-std") {
                args.std = Tcl_GetString(objv[i + 1]);
            } else if (param == "-inplace") {
                int inplace_int;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &inplace_int) != TCL_OK) {
                    throw std::runtime_error("Invalid inplace value");
                }
                args.inplace = (inplace_int != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -image, -mean, -std");
    }
    
    return args;
}

// torch::denormalize_image - Image denormalization
int DenormalizeImage_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        DenormalizeImageArgs args = ParseDenormalizeImageArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.image) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid image tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.mean) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid mean tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        if (tensor_storage.find(args.std) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid std tensor"), TCL_VOLATILE);
            return TCL_ERROR;
        }

        auto image = tensor_storage[args.image];
        auto mean = tensor_storage[args.mean];
        auto std = tensor_storage[args.std];

        torch::Tensor output;
        if (args.inplace) {
            image.mul_(std).add_(mean);
            output = image;
        } else {
            output = (image * std) + mean;
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

// Parameter structure for resize_image command
struct ResizeImageArgs {
    std::string input;
    std::vector<int64_t> size;
    std::string mode = "bilinear";  // Default mode
    bool align_corners = false;      // Default align_corners
    
    bool IsValid() const {
        return !input.empty() && !size.empty();
    }
};

// Parse dual syntax for resize_image
ResizeImageArgs ParseResizeImageArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ResizeImageArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::resize_image image size ?mode? ?align_corners? | torch::resizeImage -input tensor -size {height width} ?-mode mode? ?-alignCorners bool?");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 5) {
            throw std::runtime_error("Usage: torch::resize_image image size ?mode? ?align_corners?");
        }
        
        args.input = Tcl_GetString(objv[1]);
        args.size = ParseSizeList(interp, objv[2]);
        
        if (objc > 3) {
            args.mode = Tcl_GetString(objv[3]);
        }
        
        if (objc > 4) {
            int align_int;
            if (Tcl_GetIntFromObj(interp, objv[4], &align_int) != TCL_OK) {
                throw std::runtime_error("Invalid align_corners value");
            }
            args.align_corners = (align_int != 0);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-input" || param == "-tensor" || param == "-image") {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (param == "-size") {
                args.size = ParseSizeList(interp, objv[i + 1]);
            } else if (param == "-mode") {
                args.mode = Tcl_GetString(objv[i + 1]);
            } else if (param == "-align_corners" || param == "-alignCorners") {
                int val;
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &val) != TCL_OK) {
                    throw std::runtime_error("Invalid align_corners value");
                }
                args.align_corners = (val != 0);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -input/-tensor/-image, -size, -mode, -align_corners/-alignCorners");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: input tensor and size required");
    }
    
    return args;
}

// torch::resize_image - Image resizing with dual syntax support
int ResizeImage_Cmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        ResizeImageArgs args = ParseResizeImageArgs(interp, objc, objv);
        
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            throw std::runtime_error("Invalid input tensor");
        }

        auto input = tensor_storage[args.input];

        torch::nn::functional::InterpolateFuncOptions::mode_t mode = torch::kBilinear;
        if (args.mode == "nearest") {
            mode = torch::kNearest;
            if (args.align_corners) {
                throw std::runtime_error("align_corners option can only be used with bilinear or bicubic mode");
            }
        } else if (args.mode == "bicubic") {
            mode = torch::kBicubic;
        } else if (args.mode != "bilinear") {
            throw std::runtime_error("Invalid mode: " + args.mode + ". Valid modes are: nearest, bilinear, bicubic");
        }

        auto options = torch::nn::functional::InterpolateFuncOptions()
                          .size(args.size)
                          .mode(mode);

        // Only set align_corners for interpolating modes
        if (args.mode == "bilinear" || args.mode == "bicubic") {
            options = options.align_corners(args.align_corners);
        }

        auto output = torch::nn::functional::interpolate(input, options);

        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = output;

        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        std::string error = e.what();
        // Extract just the error message without the stack trace
        size_t pos = error.find("\nException raised from");
        if (pos != std::string::npos) {
            error = error.substr(0, pos);
        }
        Tcl_SetResult(interp, const_cast<char*>(error.c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 