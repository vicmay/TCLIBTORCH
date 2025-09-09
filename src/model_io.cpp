#include "libtorchtcl.h"

// Parameter structure for save_state command
struct SaveStateArgs {
    std::string module;
    std::string filename;
    
    bool IsValid() const {
        return !module.empty() && !filename.empty();
    }
};

// Parse dual syntax for save_state
SaveStateArgs ParseSaveStateArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SaveStateArgs args;
    
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "module filename");
        throw std::runtime_error("");  // Empty error message since we use Tcl_WrongNumArgs
    }
    
    if (Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "module filename");
            throw std::runtime_error("");  // Empty error message since we use Tcl_WrongNumArgs
        }
        args.module = Tcl_GetString(objv[1]);
        args.filename = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-module") {
                args.module = value;
            } else if (param == "-filename" || param == "-file") {
                args.filename = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -module and -filename");
    }
    
    return args;
}

int SaveState_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        SaveStateArgs args = ParseSaveStateArgs(interp, objc, objv);
        
        if (module_storage.find(args.module) == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid module name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& module = module_storage[args.module];
        torch::save(module, args.filename);
        
        Tcl_SetResult(interp, const_cast<char*>("OK"), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        if (std::string(e.what()).empty()) {
            return TCL_ERROR;  // Error message already set by Tcl_WrongNumArgs
        }
        Tcl_SetResult(interp, const_cast<char*>(("Error in save_state: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for load_state command
struct LoadStateArgs {
    std::string module;
    std::string filename;
    
    bool IsValid() const {
        return !module.empty() && !filename.empty();
    }
};

// Parse dual syntax for load_state
LoadStateArgs ParseLoadStateArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LoadStateArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "module filename");
            throw std::runtime_error("Usage: torch::load_state module filename");
        }
        args.module = Tcl_GetString(objv[1]);
        args.filename = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-module") {
                args.module = value;
            } else if (param == "-filename" || param == "-file") {
                args.filename = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -module and -filename");
    }
    
    return args;
}

int LoadState_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        LoadStateArgs args = ParseLoadStateArgs(interp, objc, objv);
        
        if (module_storage.find(args.module) == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid module name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& module = module_storage[args.module];
        torch::load(module, args.filename);
        
        Tcl_SetResult(interp, const_cast<char*>("OK"), TCL_VOLATILE);
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 