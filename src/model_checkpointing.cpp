#include "libtorchtcl.h"
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <chrono>

// Forward declarations of global variables
extern std::unordered_map<std::string, torch::Tensor> tensor_storage;
extern std::unordered_map<std::string, std::shared_ptr<torch::nn::Module>> module_storage;
extern std::unordered_map<std::string, std::shared_ptr<torch::optim::Optimizer>> optimizer_storage;

// Storage for checkpoint metadata
struct CheckpointMetadata {
    std::string model_name;
    std::string optimizer_name;
    int epoch;
    double loss;
    double learning_rate;
    std::string timestamp;
    std::unordered_map<std::string, double> metrics;
};

std::unordered_map<std::string, CheckpointMetadata> checkpoint_metadata;

extern "C" {

// ============================================================================
// Advanced Model Checkpointing Functions
// ============================================================================

// Parameter structure for save_checkpoint command
struct SaveCheckpointArgs {
    std::string model;
    std::string optimizer;
    std::string filename;
    int epoch = 0;
    double loss = 0.0;
    double lr = 0.0;
    
    bool IsValid() const {
        return !model.empty() && !optimizer.empty() && !filename.empty();
    }
};

// Parse dual syntax for save_checkpoint
SaveCheckpointArgs ParseSaveCheckpointArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SaveCheckpointArgs args;
    
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "model optimizer filename ?epoch? ?loss? ?lr?");
        throw std::runtime_error("");  // Empty error message since we use Tcl_WrongNumArgs
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 4) {
            throw std::runtime_error("wrong # args: should be \"torch::save_checkpoint model optimizer filename ?epoch? ?loss? ?lr?\"");
        }
        args.model = Tcl_GetString(objv[1]);
        args.optimizer = Tcl_GetString(objv[2]);
        args.filename = Tcl_GetString(objv[3]);
        
        if (objc >= 5) {
            if (Tcl_GetIntFromObj(interp, objv[4], &args.epoch) != TCL_OK) {
                throw std::runtime_error("Invalid epoch value");
            }
        }
        
        if (objc >= 6) {
            if (Tcl_GetDoubleFromObj(interp, objv[5], &args.loss) != TCL_OK) {
                throw std::runtime_error("Invalid loss value");
            }
        }
        
        if (objc >= 7) {
            if (Tcl_GetDoubleFromObj(interp, objv[6], &args.lr) != TCL_OK) {
                throw std::runtime_error("Invalid learning rate value");
            }
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter: " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-model") {
                args.model = Tcl_GetString(objv[i + 1]);
            } else if (param == "-optimizer") {
                args.optimizer = Tcl_GetString(objv[i + 1]);
            } else if (param == "-filename" || param == "-file") {
                args.filename = Tcl_GetString(objv[i + 1]);
            } else if (param == "-epoch") {
                if (Tcl_GetIntFromObj(interp, objv[i + 1], &args.epoch) != TCL_OK) {
                    throw std::runtime_error("Invalid epoch value");
                }
            } else if (param == "-loss") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.loss) != TCL_OK) {
                    throw std::runtime_error("Invalid loss value");
                }
            } else if (param == "-lr") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.lr) != TCL_OK) {
                    throw std::runtime_error("Invalid learning rate value");
                }
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: model, optimizer, and filename are required");
    }
    
    return args;
}

// torch::save_checkpoint - Save model checkpoint
int Torch_SaveCheckpoint_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        SaveCheckpointArgs args = ParseSaveCheckpointArgs(interp, objc, objv);
        
        // Find model and optimizer
        auto model_it = module_storage.find(args.model);
        auto optimizer_it = optimizer_storage.find(args.optimizer);
        
        if (model_it == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Model not found"), TCL_STATIC);
            return TCL_ERROR;
        }
        
        if (optimizer_it == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Optimizer not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        // Create checkpoint dictionary
        torch::serialize::OutputArchive archive;
        
        // Save model state dict
        model_it->second->save(archive);
        
        // Save optimizer state dict
        optimizer_it->second->save(archive);
        
        // Save additional metadata
        archive.write("epoch", torch::tensor(args.epoch));
        archive.write("loss", torch::tensor(args.loss));
        archive.write("learning_rate", torch::tensor(args.lr));
        
        // Create timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::string timestamp = std::ctime(&time_t);
        timestamp.pop_back(); // Remove newline
        
        // Save checkpoint metadata
        CheckpointMetadata metadata;
        metadata.model_name = args.model;
        metadata.optimizer_name = args.optimizer;
        metadata.epoch = args.epoch;
        metadata.loss = args.loss;
        metadata.learning_rate = args.lr;
        metadata.timestamp = timestamp;
        
        checkpoint_metadata[args.filename] = metadata;
        
        // Save to file
        archive.save_to(args.filename);
        
        std::string result = "Checkpoint saved: " + args.filename + " (epoch=" + std::to_string(args.epoch) + 
                           ", loss=" + std::to_string(args.loss) + ")";
        Tcl_SetResult(interp, const_cast<char*>(result.c_str()), TCL_VOLATILE);
        return TCL_OK;

    } catch (const std::exception& e) {
        if (std::string(e.what()).empty()) {
            return TCL_ERROR;  // Error message already set by Tcl_WrongNumArgs
        }
        Tcl_SetResult(interp, const_cast<char*>(("Error in save_checkpoint: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for load_checkpoint command
struct LoadCheckpointArgs {
    std::string filename;
    std::string model;
    std::string optimizer;
    
    bool IsValid() const {
        return !filename.empty() && !model.empty() && !optimizer.empty();
    }
};

// Parse dual syntax for load_checkpoint
LoadCheckpointArgs ParseLoadCheckpointArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LoadCheckpointArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 4) {
            throw std::runtime_error("Usage: torch::load_checkpoint filename model optimizer");
        }
        args.filename = Tcl_GetString(objv[1]);
        args.model = Tcl_GetString(objv[2]);
        args.optimizer = Tcl_GetString(objv[3]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-filename" || param == "-file") {
                args.filename = Tcl_GetString(objv[i + 1]);
            } else if (param == "-model") {
                args.model = Tcl_GetString(objv[i + 1]);
            } else if (param == "-optimizer") {
                args.optimizer = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -filename/-file, -model, -optimizer");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: filename, model, and optimizer are required");
    }
    
    return args;
}

int Torch_LoadCheckpoint_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        // Parse arguments using dual syntax
        LoadCheckpointArgs args = ParseLoadCheckpointArgs(interp, objc, objv);
        
        std::string filename = args.filename;
        std::string model_name = args.model;
        std::string optimizer_name = args.optimizer;
        
        // Find model and optimizer
        auto model_it = module_storage.find(model_name);
        auto optimizer_it = optimizer_storage.find(optimizer_name);
        
        if (model_it == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Model not found"), TCL_STATIC);
            return TCL_ERROR;
        }
        
        if (optimizer_it == optimizer_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Optimizer not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        // Load checkpoint
        torch::serialize::InputArchive archive;
        archive.load_from(filename);
        
        // Load model state dict
        model_it->second->load(archive);
        
        // Load optimizer state dict
        optimizer_it->second->load(archive);
        
        // Try to load metadata
        torch::Tensor epoch_tensor, loss_tensor, lr_tensor;
        int epoch = 0;
        double loss = 0.0;
        double lr = 0.0;
        
        try {
            archive.read("epoch", epoch_tensor);
            epoch = epoch_tensor.item<int>();
        } catch (...) {
            // Metadata not found, use defaults
        }
        
        try {
            archive.read("loss", loss_tensor);
            loss = loss_tensor.item<double>();
        } catch (...) {
            // Metadata not found, use defaults
        }
        
        try {
            archive.read("learning_rate", lr_tensor);
            lr = lr_tensor.item<double>();
        } catch (...) {
            // Metadata not found, use defaults
        }
        
        std::string result = "Checkpoint loaded: " + filename + " (epoch=" + std::to_string(epoch) + 
                           ", loss=" + std::to_string(loss) + ", lr=" + std::to_string(lr) + ")";
        Tcl_SetResult(interp, const_cast<char*>(result.c_str()), TCL_VOLATILE);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ---------------------------------------------------------------------------
// Dual syntax structure & parser for get_checkpoint_info
// ---------------------------------------------------------------------------

struct GetCheckpointInfoArgs {
    std::string filename;
    bool IsValid() const { return !filename.empty(); }
};

static GetCheckpointInfoArgs ParseGetCheckpointInfoArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    GetCheckpointInfoArgs args;

    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: filename
        if (objc != 2) {
            Tcl_WrongNumArgs(interp, 1, objv, "filename");
            throw std::runtime_error("Wrong # args: expected filename");
        }
        args.filename = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax: -file|-filename value
        if (objc < 3 || (objc % 2) == 0) {
            if (objc == 2) {
                // Exactly one option provided but no value
                Tcl_SetResult(interp, const_cast<char*>("Missing value for option -file"), TCL_VOLATILE);
                throw std::runtime_error("Missing value for option -file");
            }
            Tcl_WrongNumArgs(interp, 1, objv, "-file filename");
            throw std::runtime_error("wrong # args: should be \"torch::getCheckpointInfo -file filename\"");
        }
        for (int i = 1; i < objc; i += 2) {
            std::string key = Tcl_GetString(objv[i]);
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for option " + key);
            }
            std::string value = Tcl_GetString(objv[i + 1]);

            if (key == "-file" || key == "-filename") {
                args.filename = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + key);
            }
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("Filename is required");
    }

    return args;
}

int Torch_GetCheckpointInfo_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        GetCheckpointInfoArgs args = ParseGetCheckpointInfoArgs(interp, objc, objv);

        const std::string& filename = args.filename;
         
        auto metadata_it = checkpoint_metadata.find(filename);
        if (metadata_it != checkpoint_metadata.end()) {
            // Return metadata from memory
            const auto& metadata = metadata_it->second;
            std::string result = "{";
            result += "epoch " + std::to_string(metadata.epoch) + " ";
            result += "loss " + std::to_string(metadata.loss) + " ";
            result += "learning_rate " + std::to_string(metadata.learning_rate) + " ";
            result += "timestamp {" + metadata.timestamp + "} ";
            result += "model_name " + metadata.model_name + " ";
            result += "optimizer_name " + metadata.optimizer_name;
            result += "}";
            
            Tcl_SetResult(interp, const_cast<char*>(result.c_str()), TCL_VOLATILE);
            return TCL_OK;
        }
        
        // Try to load metadata from file
        torch::serialize::InputArchive archive;
        archive.load_from(filename);
        
        torch::Tensor epoch_tensor, loss_tensor, lr_tensor;
        int epoch = 0;
        double loss = 0.0;
        double lr = 0.0;
        
        try {
            archive.read("epoch", epoch_tensor);
            epoch = epoch_tensor.item<int>();
        } catch (...) {
            // Metadata not found
        }
        
        try {
            archive.read("loss", loss_tensor);
            loss = loss_tensor.item<double>();
        } catch (...) {
            // Metadata not found
        }
        
        try {
            archive.read("learning_rate", lr_tensor);
            lr = lr_tensor.item<double>();
        } catch (...) {
            // Metadata not found
        }
        
        std::string result = "{";
        result += "epoch " + std::to_string(epoch) + " ";
        result += "loss " + std::to_string(loss) + " ";
        result += "learning_rate " + std::to_string(lr);
        result += "}";
        
        Tcl_SetResult(interp, const_cast<char*>(result.c_str()), TCL_VOLATILE);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for save_state_dict
// ============================================================================

struct SaveStateDictArgs {
    std::string model;
    std::string filename;
    
    bool IsValid() const {
        return !model.empty() && !filename.empty();
    }
};

SaveStateDictArgs ParseSaveStateDictArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SaveStateDictArgs args;
    
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "model filename");
        throw std::runtime_error("");  // Empty error message since we use Tcl_WrongNumArgs
    }
    
    if (Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "model filename");
            throw std::runtime_error("");  // Empty error message since we use Tcl_WrongNumArgs
        }
        args.model = Tcl_GetString(objv[1]);
        args.filename = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-model") {
                args.model = value;
            } else if (param == "-filename" || param == "-file") {
                args.filename = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -model and -filename");
    }
    
    return args;
}

int Torch_SaveStateDict_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        SaveStateDictArgs args = ParseSaveStateDictArgs(interp, objc, objv);
        
        auto model_it = module_storage.find(args.model);
        if (model_it == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Model not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        // Save only the model state dict (parameters)
        torch::serialize::OutputArchive archive;
        model_it->second->save(archive);
        archive.save_to(args.filename);
        
        std::string result = "Model state dict saved to: " + args.filename;
        Tcl_SetResult(interp, const_cast<char*>(result.c_str()), TCL_VOLATILE);
        return TCL_OK;

    } catch (const std::exception& e) {
        if (std::string(e.what()).empty()) {
            return TCL_ERROR;  // Error message already set by Tcl_WrongNumArgs
        }
        Tcl_SetResult(interp, const_cast<char*>(("Error in save_state_dict: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for load_state_dict
// ============================================================================

struct LoadStateDictArgs {
    std::string model;
    std::string filename;
    
    bool IsValid() const {
        return !model.empty() && !filename.empty();
    }
};

LoadStateDictArgs ParseLoadStateDictArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    LoadStateDictArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 3) {
            Tcl_WrongNumArgs(interp, 1, objv, "model filename");
            throw std::runtime_error("Usage: torch::load_state_dict model filename");
        }
        args.model = Tcl_GetString(objv[1]);
        args.filename = Tcl_GetString(objv[2]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-model") {
                args.model = value;
            } else if (param == "-filename" || param == "-file") {
                args.filename = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: -model and -filename");
    }
    
    return args;
}

int Torch_LoadStateDict_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        LoadStateDictArgs args = ParseLoadStateDictArgs(interp, objc, objv);
        
        auto model_it = module_storage.find(args.model);
        if (model_it == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Model not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        // Load only the model state dict (parameters)
        torch::serialize::InputArchive archive;
        archive.load_from(args.filename);
        model_it->second->load(archive);
        
        std::string result = "Model state dict loaded from: " + args.filename;
        Tcl_SetResult(interp, const_cast<char*>(result.c_str()), TCL_VOLATILE);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// Model Freezing and Unfreezing
// ============================================================================

// ============================================================================
// Dual Syntax Parser for freeze_model
// ============================================================================

struct FreezeModelArgs {
    std::string model;
    
    bool IsValid() const {
        return !model.empty();
    }
};

FreezeModelArgs ParseFreezeModelArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    FreezeModelArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: torch::freeze_model model
        if (objc != 2) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::freeze_model model");
        }
        args.model = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax: torch::freeze_model -model model
        if (objc < 2) {
            throw std::runtime_error("Wrong number of arguments for named syntax. Expected: torch::freeze_model -model model");
        }
        
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter: " + std::string(Tcl_GetString(objv[i])));
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-model") {
                args.model = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: -model");
    }
    
    return args;
}

int Torch_FreezeModel_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        FreezeModelArgs args = ParseFreezeModelArgs(interp, objc, objv);
        
        auto model_it = module_storage.find(args.model);
        if (model_it == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Model not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        // Freeze all parameters
        for (auto& param : model_it->second->parameters()) {
            param.set_requires_grad(false);
        }
        
        Tcl_SetResult(interp, const_cast<char*>("Model parameters frozen"), TCL_STATIC);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ============================================================================
// Dual Syntax Parser for unfreeze_model
// ============================================================================

struct UnfreezeModelArgs {
    std::string model;

    bool IsValid() const {
        return !model.empty();
    }
};

UnfreezeModelArgs ParseUnfreezeModelArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    UnfreezeModelArgs args;

    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: torch::unfreeze_model model
        if (objc != 2) {
            throw std::runtime_error("Wrong number of arguments for positional syntax. Expected: torch::unfreeze_model model");
        }
        args.model = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax: torch::unfreeze_model -model model
        if (objc < 2) {
            throw std::runtime_error("Wrong number of arguments for named syntax. Expected: torch::unfreeze_model -model model");
        }

        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error(std::string("Missing value for parameter: ") + Tcl_GetString(objv[i]));
            }

            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);

            if (param == "-model") {
                args.model = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("Required parameter missing: -model");
    }

    return args;
}

int Torch_UnfreezeModel_Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        UnfreezeModelArgs args = ParseUnfreezeModelArgs(interp, objc, objv);

        auto model_it = module_storage.find(args.model);
        if (model_it == module_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Model not found"), TCL_STATIC);
            return TCL_ERROR;
        }

        // Unfreeze all parameters
        for (auto& param : model_it->second->parameters()) {
            param.set_requires_grad(true);
        }

        Tcl_SetResult(interp, const_cast<char*>("Model parameters unfrozen"), TCL_STATIC);
        return TCL_OK;

    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_STATIC);
        return TCL_ERROR;
    }
}

} // extern "C" 