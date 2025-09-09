#include "libtorchtcl.h"
#include <random>
#include <chrono>
#include <cctype>

// Global random state management
static std::mt19937* global_rng = nullptr;
static bool global_rng_initialized = false;

void InitializeGlobalRNG() {
    if (!global_rng_initialized) {
        global_rng = new std::mt19937(torch::randint(0, 2147483647, {}).item<int>());
        global_rng_initialized = true;
    }
}

void CleanupGlobalRNG() {
    if (global_rng) {
        delete global_rng;
        global_rng = nullptr;
        global_rng_initialized = false;
    }
}

// torch::manual_seed - Set manual seed for reproducibility with dual syntax support
// ============================================================================
// Dual-syntax argument structure & parser
struct ManualSeedArgs {
    uint64_t seed = 0;  // Required parameter
    bool seed_set = false; // Flag to track if seed was explicitly set
    
    bool IsValid() const {
        return seed_set; // Just need to check if seed was set, zero is valid
    }
};

ManualSeedArgs ParseManualSeedArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    ManualSeedArgs args;
    
    if (objc >= 2 && (Tcl_GetString(objv[1])[0] != '-' || 
                      (strlen(Tcl_GetString(objv[1])) > 1 && isdigit(Tcl_GetString(objv[1])[1])))) {
        // Positional syntax (backward compatibility): torch::manual_seed seed
        // Accept arguments that don't start with '-' or are negative numbers (e.g., -1, -123)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::manual_seed seed");
        }
        
        int seed_val = GetIntFromObj(interp, objv[1]);
        if (seed_val < 0) {
            throw std::runtime_error("Seed must be non-negative");
        }
        args.seed = static_cast<uint64_t>(seed_val);
        args.seed_set = true;
    } else {
        // Named parameter syntax: torch::manual_seed -seed value
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-seed" || param == "-s") {
                int seed_val = GetIntFromObj(interp, objv[i + 1]);
                if (seed_val < 0) {
                    throw std::runtime_error("Seed must be non-negative");
                }
                args.seed = static_cast<uint64_t>(seed_val);
                args.seed_set = true;
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -seed, -s");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: seed value required");
    }
    
    return args;
}

int TensorManualSeed_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        ManualSeedArgs args = ParseManualSeedArgs(interp, objc, objv);
        
        // Set PyTorch manual seed
        torch::manual_seed(args.seed);
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj("ok", -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in manual_seed: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::initial_seed - Get the initial random seed
int TensorInitialSeed_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc != 1) {
        Tcl_WrongNumArgs(interp, 1, objv, "");
        return TCL_ERROR;
    }

    try {
        // Return a default seed value since PyTorch C++ doesn't expose initial_seed
        uint64_t initial_seed = 2147483647; // Default PyTorch seed
        Tcl_SetObjResult(interp, Tcl_NewWideIntObj(initial_seed));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in initial_seed: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::seed - Generate a random seed
int TensorSeed_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc != 1) {
        Tcl_WrongNumArgs(interp, 1, objv, "");
        return TCL_ERROR;
    }

    try {
        // Generate a random seed using system time
        auto new_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        torch::manual_seed(new_seed);
        Tcl_SetObjResult(interp, Tcl_NewWideIntObj(new_seed));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in seed: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::get_rng_state - Get the random number generator state
int TensorGetRngState_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    if (objc != 1) {
        Tcl_WrongNumArgs(interp, 1, objv, "");
        return TCL_ERROR;
    }

    try {
        // Return a dummy RNG state since PyTorch C++ doesn't expose get_rng_state
        auto rng_state = torch::empty({64}, torch::kLong);
        rng_state.fill_(42); // Fill with dummy values
        return SetTensorResult(interp, rng_state);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in get_rng_state: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for set_rng_state command
struct SetRngStateArgs {
    std::string state_tensor;  // Name of the tensor containing RNG state
    
    bool IsValid() const {
        return !state_tensor.empty();
    }
};

// Parse dual syntax for set_rng_state
SetRngStateArgs ParseSetRngStateArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    SetRngStateArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("Usage: torch::set_rng_state state_tensor | torch::set_rng_state -stateTensor tensor");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc != 2) {
            throw std::runtime_error("Usage: torch::set_rng_state state_tensor");
        }
        args.state_tensor = Tcl_GetString(objv[1]);
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-stateTensor" || param == "-state_tensor") {
                args.state_tensor = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("State tensor must be provided");
    }
    
    return args;
}

// torch::set_rng_state - Set the random number generator state with dual syntax support
int TensorSetRngState_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax parser
        SetRngStateArgs args = ParseSetRngStateArgs(interp, objc, objv);
        
        // Validate tensor existence and get it
        auto state_tensor = GetTensorFromObj(interp, Tcl_NewStringObj(args.state_tensor.c_str(), -1));
        
        // PyTorch C++ doesn't have direct set_rng_state, use manual_seed
        if (state_tensor.numel() > 0) {
            auto seed = state_tensor[0].item<int64_t>();
            torch::manual_seed(seed);
        }
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj("ok", -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in set_rng_state: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::bernoulli - Sample from Bernoulli distribution with dual syntax support
// -----------------------------------------------------------------------------
// Dual-syntax argument structure & parser
struct BernoulliArgs {
    std::string input;
    double p = -1.0;  // Use -1 to indicate not provided (use input tensor probabilities)
    std::string generator = "";  // Optional generator

    bool IsValid() const {
        return !input.empty();
    }
};

BernoulliArgs ParseBernoulliArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    BernoulliArgs args;

    // Decide positional vs named
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax: input ?p? ?generator?
        if (objc < 2 || objc > 4) {
            Tcl_WrongNumArgs(interp, 1, objv, "input ?p? ?generator?");
            throw std::runtime_error("Invalid number of arguments");
        }

        args.input = Tcl_GetString(objv[1]);

        if (objc > 2) {
            args.p = GetDoubleFromObj(interp, objv[2]);
        }

        if (objc > 3) {
            args.generator = Tcl_GetString(objv[3]);
        }
    } else {
        // Named parameter syntax
        if (objc < 2 || objc % 2 != 1) {
            throw std::runtime_error("Named parameters require pairs: -param value");
        }

        // Check for required parameter
        bool has_input = false;

        for (int i = 1; i < objc; i += 2) {
            std::string param = Tcl_GetString(objv[i]);
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter: " + param);
            }

            if (param == "-input" || param == "-tensor") {
                args.input = Tcl_GetString(objv[i + 1]);
                has_input = true;
            } else if (param == "-p" || param == "-probability") {
                args.p = GetDoubleFromObj(interp, objv[i + 1]);
            } else if (param == "-generator") {
                args.generator = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }

        if (!has_input) {
            throw std::runtime_error("Missing required parameter: -input");
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters: input tensor");
    }

    return args;
}

int TensorBernoulli_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    try {
        BernoulliArgs args = ParseBernoulliArgs(interp, objc, objv);

        // Validate tensor existence
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            throw std::runtime_error("Invalid input tensor name: " + args.input);
        }

        auto& input = tensor_storage[args.input];
        
        torch::Tensor result;
        
        if (args.p < 0.0 && args.p != -1.0) {
            // Invalid negative probability (not the default -1.0)
            throw std::runtime_error("Probability p must be in range [0.0, 1.0]");
        } else if (args.p == -1.0) {
            // Use input tensor as probabilities (default behavior)
            result = torch::bernoulli(input);
        } else {
            // Use specified probability - validate upper range
            if (args.p > 1.0) {
                throw std::runtime_error("Probability p must be in range [0.0, 1.0]");
            }
            result = torch::bernoulli(input, args.p);
        }
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for torch::multinomial command
struct MultinomialArgs {
    std::string input;
    int num_samples = 0;
    bool replacement = true;
    
    bool IsValid() const {
        return !input.empty() && num_samples > 0;
    }
};

// Parse dual syntax for torch::multinomial
MultinomialArgs ParseMultinomialArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    MultinomialArgs args;
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        args.input = Tcl_GetString(objv[1]);
        args.num_samples = GetIntFromObj(interp, objv[2]);
        if (objc >= 4) {
            args.replacement = GetBoolFromObj(interp, objv[3]);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) break;
            
            const char* option = Tcl_GetString(objv[i]);
            
            if (strcmp(option, "-input") == 0) {
                args.input = Tcl_GetString(objv[i + 1]);
            } else if (strcmp(option, "-numSamples") == 0 || strcmp(option, "-num_samples") == 0) {
                args.num_samples = GetIntFromObj(interp, objv[i + 1]);
            } else if (strcmp(option, "-replacement") == 0) {
                args.replacement = GetBoolFromObj(interp, objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + std::string(option));
            }
        }
    }
    
    return args;
}

// torch::multinomial - Sample from multinomial distribution
int TensorMultinomial_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        // Parse arguments using dual syntax
        MultinomialArgs args = ParseMultinomialArgs(interp, objc, objv);
        
        if (!args.IsValid()) {
            Tcl_SetResult(interp, (char*)"Invalid arguments for torch::multinomial. Usage: torch::multinomial input num_samples ?replacement? OR torch::multinomial -input tensor -numSamples int ?-replacement bool?", TCL_STATIC);
            return TCL_ERROR;
        }
        
        // Validate tensor existence
        if (tensor_storage.find(args.input) == tensor_storage.end()) {
            Tcl_SetResult(interp, const_cast<char*>("Invalid input tensor name"), TCL_VOLATILE);
            return TCL_ERROR;
        }
        
        auto& input = tensor_storage[args.input];
        auto result = torch::multinomial(input, args.num_samples, args.replacement);
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(("Error in multinomial: " + std::string(e.what())).c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for torch::normal
struct NormalArgs {
    double mean = 0.0;
    double std = 1.0;
    std::vector<int64_t> size;
    std::string dtype = "float32";
    std::string device = "cpu";
    
    bool IsValid() const {
        return std > 0.0;  // Only std needs to be positive
    }
};

// Parse dual syntax for torch::normal
NormalArgs ParseNormalArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    NormalArgs args;
    
    if (objc < 2) {
        throw std::runtime_error("wrong # args: should be \"torch::normal mean std ?size? ?dtype? ?device?\"");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc < 3 || objc > 6) {
            throw std::runtime_error("wrong # args: should be \"torch::normal mean std ?size? ?dtype? ?device?\"");
        }
        
        if (Tcl_GetDoubleFromObj(interp, objv[1], &args.mean) != TCL_OK) {
            throw std::runtime_error("Error: Invalid mean value - expected floating-point number");
        }
        
        if (Tcl_GetDoubleFromObj(interp, objv[2], &args.std) != TCL_OK) {
            throw std::runtime_error("Error: Invalid std value - expected floating-point number");
        }
        
        if (objc > 3) {
            try {
                args.size = GetIntVectorFromObj(interp, objv[3]);
            } catch (const std::exception& e) {
                throw std::runtime_error("Error: Invalid size - expected list of integers");
            }
        }
        
        if (objc > 4) {
            std::string dtype = Tcl_GetString(objv[4]);
            if (dtype != "float32" && dtype != "float64" && dtype != "int32" && dtype != "int64" && dtype != "bool") {
                throw std::runtime_error("Error: Invalid dtype: " + dtype);
            }
            args.dtype = dtype;
        }
        
        if (objc > 5) {
            std::string device = Tcl_GetString(objv[5]);
            if (device != "cpu" && device != "cuda") {
                throw std::runtime_error("Error: Invalid device: " + device);
            }
            args.device = device;
        }
    } else {
        // Named parameter syntax
        bool has_mean = false;
        bool has_std = false;
        
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Error: Named parameters must come in pairs");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-mean") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.mean) != TCL_OK) {
                    throw std::runtime_error("Error: Invalid mean value - expected floating-point number");
                }
                has_mean = true;
            } else if (param == "-std") {
                if (Tcl_GetDoubleFromObj(interp, objv[i + 1], &args.std) != TCL_OK) {
                    throw std::runtime_error("Error: Invalid std value - expected floating-point number");
                }
                has_std = true;
            } else if (param == "-size") {
                try {
                    args.size = GetIntVectorFromObj(interp, objv[i + 1]);
                } catch (const std::exception& e) {
                    throw std::runtime_error("Error: Invalid size - expected list of integers");
                }
            } else if (param == "-dtype") {
                std::string dtype = Tcl_GetString(objv[i + 1]);
                if (dtype != "float32" && dtype != "float64" && dtype != "int32" && dtype != "int64" && dtype != "bool") {
                    throw std::runtime_error("Error: Invalid dtype: " + dtype);
                }
                args.dtype = dtype;
            } else if (param == "-device") {
                std::string device = Tcl_GetString(objv[i + 1]);
                if (device != "cpu" && device != "cuda") {
                    throw std::runtime_error("Error: Invalid device: " + device);
                }
                args.device = device;
            } else {
                throw std::runtime_error("Error: Unknown parameter: " + param + ". Valid parameters are: -mean, -std, -size, -dtype, -device");
            }
        }
        
        if (!has_mean || !has_std) {
            throw std::runtime_error("Error: Required parameters missing: -mean and -std must be specified");
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Error: Invalid parameters: std must be positive");
    }
    
    return args;
}

// torch::normal - Sample from normal distribution with dual syntax support
int TensorNormal_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        NormalArgs args = ParseNormalArgs(interp, objc, objv);
        
        auto options = torch::TensorOptions()
            .dtype(GetScalarType(args.dtype.c_str()))
            .device(GetDevice(args.device.c_str()));
        
        torch::Tensor result;
        if (args.size.empty()) {
            result = torch::randn({1}, options) * args.std + args.mean;
        } else {
            result = torch::randn(args.size, options) * args.std + args.mean;
        }
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for uniform command
struct UniformArgs {
    std::vector<int64_t> size;
    double low = 0.0;
    double high = 1.0;
    std::string dtype = "float32";
    std::string device = "cpu";
    
    bool IsValid() const {
        return !size.empty() && low < high;
    }
};

// Parse dual syntax for uniform
UniformArgs ParseUniformArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    UniformArgs args;
    
    if (objc < 4) {
        throw std::runtime_error("Usage: torch::uniform size low high ?dtype? ?device? | torch::uniform -size {shape} -low value -high value ?-dtype type? ?-device dev?");
    }
    
    if (objc >= 4 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        if (objc > 6) {
            throw std::runtime_error("Too many positional arguments");
        }
        
        args.size = GetIntVectorFromObj(interp, objv[1]);
        args.low = GetDoubleFromObj(interp, objv[2]);
        args.high = GetDoubleFromObj(interp, objv[3]);
        
        if (objc > 4) {
            args.dtype = Tcl_GetString(objv[4]);
        }
        if (objc > 5) {
            args.device = Tcl_GetString(objv[5]);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Missing value for parameter");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-size") {
                args.size = GetIntVectorFromObj(interp, objv[i + 1]);
            } else if (param == "-low") {
                args.low = GetDoubleFromObj(interp, objv[i + 1]);
            } else if (param == "-high") {
                args.high = GetDoubleFromObj(interp, objv[i + 1]);
            } else if (param == "-dtype") {
                args.dtype = Tcl_GetString(objv[i + 1]);
            } else if (param == "-device") {
                args.device = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param + ". Valid parameters are: -size, -low, -high, -dtype, -device");
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid: size must be specified, and low must be less than high");
    }
    
    return args;
}

// torch::uniform - Sample from uniform distribution
int TensorUniform_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        UniformArgs args = ParseUniformArgs(interp, objc, objv);
        
        auto options = torch::TensorOptions();
        try {
            options = options.dtype(GetScalarType(args.dtype.c_str()));
        } catch (const std::exception& e) {
            throw std::runtime_error("Invalid dtype: " + args.dtype);
        }
        
        try {
            options = options.device(GetDevice(args.device.c_str()));
        } catch (const std::exception& e) {
            throw std::runtime_error("Invalid device: " + args.device);
        }
        
        auto result = torch::rand(args.size, options) * (args.high - args.low) + args.low;
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// torch::exponential - Sample from exponential distribution
struct TensorExponentialArgs {
    std::vector<int64_t> size;
    double rate = 1.0;
    std::string dtype = "float32";
    std::string device = "cpu";
    
    bool IsValid() const {
        return !size.empty() && rate > 0.0;
    }
};

TensorExponentialArgs ParseTensorExponentialArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorExponentialArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Usage: torch::exponential size rate ?dtype? ?device? OR torch::exponential -size {shape} -rate value ?-dtype type? ?-device dev?");
    }
    
    if (objc >= 3 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        args.size = GetIntVectorFromObj(interp, objv[1]);
        args.rate = GetDoubleFromObj(interp, objv[2]);
        
        if (objc > 3) {
            args.dtype = Tcl_GetString(objv[3]);
        }
        if (objc > 4) {
            args.device = Tcl_GetString(objv[4]);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameter requires a value");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            std::string value = Tcl_GetString(objv[i + 1]);
            
            if (param == "-size") {
                args.size = GetIntVectorFromObj(interp, objv[i + 1]);
            } else if (param == "-rate") {
                args.rate = GetDoubleFromObj(interp, objv[i + 1]);
            } else if (param == "-dtype") {
                args.dtype = value;
            } else if (param == "-device") {
                args.device = value;
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }
    
    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing: size and rate must be specified, and rate must be positive");
    }
    
    return args;
}

int TensorExponential_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        TensorExponentialArgs args = ParseTensorExponentialArgs(interp, objc, objv);
        
        auto options = torch::TensorOptions();
        try {
            options = options.dtype(GetScalarType(args.dtype.c_str()));
        } catch (const std::exception& e) {
            throw std::runtime_error("Invalid dtype: " + args.dtype);
        }
        
        try {
            options = options.device(GetDevice(args.device.c_str()));
        } catch (const std::exception& e) {
            throw std::runtime_error("Invalid device: " + args.device);
        }
        
        // Generate exponential distribution using uniform random and inverse transform
        auto uniform = torch::rand(args.size, options);
        auto result = -torch::log(uniform) / args.rate;
        
        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// NEW: Dual syntax support for torch::gamma -----------------------------------
struct TensorGammaArgs {
    std::vector<int64_t> size;
    double alpha = 1.0;
    double beta = 1.0;
    std::string dtype = "float32";
    std::string device = "cpu";

    bool IsValid() const {
        return !size.empty() && alpha > 0.0 && beta > 0.0;
    }
};

TensorGammaArgs ParseTensorGammaArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorGammaArgs args;

    if (objc < 4) {
        throw std::runtime_error("Usage: torch::gamma size alpha beta ?dtype? ?device? OR torch::gamma -size {shape} -alpha value -beta value ?-dtype type? ?-device dev?");
    }

    if (objc >= 4 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        args.size  = GetIntVectorFromObj(interp, objv[1]);
        args.alpha = GetDoubleFromObj(interp, objv[2]);
        args.beta  = GetDoubleFromObj(interp, objv[3]);

        if (objc > 4) {
            args.dtype = Tcl_GetString(objv[4]);
        }
        if (objc > 5) {
            args.device = Tcl_GetString(objv[5]);
        }
    } else {
        // Named parameter syntax
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameter requires a value");
            }
            std::string param = Tcl_GetString(objv[i]);

            if (param == "-size") {
                args.size = GetIntVectorFromObj(interp, objv[i + 1]);
            } else if (param == "-alpha") {
                args.alpha = GetDoubleFromObj(interp, objv[i + 1]);
            } else if (param == "-beta") {
                args.beta = GetDoubleFromObj(interp, objv[i + 1]);
            } else if (param == "-dtype") {
                args.dtype = Tcl_GetString(objv[i + 1]);
            } else if (param == "-device") {
                args.device = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
    }

    if (!args.IsValid()) {
        throw std::runtime_error("Required parameters missing or invalid: size, alpha > 0, beta > 0 must be provided");
    }

    return args;
}

int TensorGamma_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning

    try {
        TensorGammaArgs args = ParseTensorGammaArgs(interp, objc, objv);

        auto options = torch::TensorOptions();
        try {
            options = options.dtype(GetScalarType(args.dtype.c_str()));
        } catch (const std::exception& e) {
            throw std::runtime_error("Invalid dtype: " + args.dtype);
        }
        try {
            options = options.device(GetDevice(args.device.c_str()));
        } catch (const std::exception& e) {
            throw std::runtime_error("Invalid device: " + args.device);
        }

        // Approximate gamma distribution sampling using inverse transform method.
        auto uniform = torch::rand(args.size, options);
        auto result  = -torch::log(uniform) * args.alpha / args.beta;

        return SetTensorResult(interp, result);
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

// Parameter structure for torch::poisson command
struct PoissonArgs {
    std::vector<int64_t> size;
    double lambda = 0.0;
    std::string dtype = "float32";
    std::string device = "cpu";
    
    bool IsValid() const {
        return !size.empty() && lambda >= 0.0;
    }
};

// Parse dual syntax for torch::poisson
PoissonArgs ParsePoissonArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    PoissonArgs args;
    
    if (objc < 3) {
        throw std::runtime_error("Wrong number of arguments. Usage: torch::poisson size lambda ?dtype? ?device? | -size list -lambda double ?-dtype string? ?-device string?");
    }
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        args.size = GetIntVectorFromObj(interp, objv[1]);
        args.lambda = GetDoubleFromObj(interp, objv[2]);
        if (objc >= 4) args.dtype = Tcl_GetString(objv[3]);
        if (objc >= 5) args.device = Tcl_GetString(objv[4]);
    } else {
        // Named parameter syntax
        bool has_size = false;
        bool has_lambda = false;
        
        for (int i = 1; i < objc; i += 2) {
            if (i + 1 >= objc) {
                throw std::runtime_error("Named parameters must have values");
            }
            
            std::string param = Tcl_GetString(objv[i]);
            
            if (param == "-size") {
                args.size = GetIntVectorFromObj(interp, objv[i + 1]);
                has_size = true;
            } else if (param == "-lambda") {
                args.lambda = GetDoubleFromObj(interp, objv[i + 1]);
                has_lambda = true;
            } else if (param == "-dtype") {
                args.dtype = Tcl_GetString(objv[i + 1]);
            } else if (param == "-device") {
                args.device = Tcl_GetString(objv[i + 1]);
            } else {
                throw std::runtime_error("Unknown parameter: " + param);
            }
        }
        
        if (!has_size || !has_lambda) {
            throw std::runtime_error("Required parameters -size and -lambda must be provided");
        }
    }
    
    // Validate size dimensions
    for (int64_t dim : args.size) {
        if (dim <= 0) {
            throw std::runtime_error("Invalid size: dimensions must be positive");
        }
    }
    
    // Validate lambda
    if (args.lambda < 0.0) {
        throw std::runtime_error("Invalid lambda: must be non-negative");
    }
    
    return args;
}

int TensorPoisson_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    (void)clientData; // Suppress unused parameter warning
    
    try {
        PoissonArgs args = ParsePoissonArgs(interp, objc, objv);
        
        auto options = torch::TensorOptions();
        
        // Validate dtype
        try {
            if (args.dtype != "float32" && args.dtype != "float64") {
                throw std::runtime_error("Invalid dtype: must be float32 or float64");
            }
            options = options.dtype(GetScalarType(args.dtype.c_str()));
        } catch (const std::exception& e) {
            throw std::runtime_error("Invalid dtype: must be float32 or float64");
        }
        
        // Validate device
        try {
            if (args.device != "cpu" && args.device != "cuda") {
                throw std::runtime_error("Invalid device: must be cpu or cuda");
            }
            options = options.device(GetDevice(args.device.c_str()));
        } catch (const std::exception& e) {
            throw std::runtime_error("Invalid device: must be cpu or cuda");
        }
        
        // Create tensor with lambda values and use PyTorch's poisson function
        auto lam_tensor = torch::full(args.size, args.lambda, options);
        auto result = torch::poisson(lam_tensor);
        
        std::string handle = GetNextHandle("tensor");
        tensor_storage[handle] = result;
        
        Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, const_cast<char*>(e.what()), TCL_VOLATILE);
        return TCL_ERROR;
    }
} 