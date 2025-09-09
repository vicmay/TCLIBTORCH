#ifndef PARAMETER_PARSING_H
#define PARAMETER_PARSING_H

#include <tcl.h>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <torch/torch.h>

// Forward declarations
struct TensorCreationArgs;

// Base parameter parser class
class DualSyntaxParser {
protected:
    std::map<std::string, std::function<void(TensorCreationArgs&, Tcl_Obj*)>> param_setters;
    std::vector<std::string> positional_order;
    
public:
    DualSyntaxParser(const std::map<std::string, std::function<void(TensorCreationArgs&, Tcl_Obj*)>>& setters,
                     const std::vector<std::string>& order) 
        : param_setters(setters), positional_order(order) {}
    
    virtual ~DualSyntaxParser() = default;
};

// Tensor creation arguments structure
struct TensorCreationArgs {
    std::vector<int64_t> shape;
    std::string dtype = "float32";
    std::string device = "cpu";
    bool requires_grad = false;
    
    // Validation
    bool IsValid() const;
    
    // Parse function
    static TensorCreationArgs Parse(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
    
private:
    static TensorCreationArgs ParsePositionalArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
    static TensorCreationArgs ParseMixedArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
    static TensorCreationArgs ParseNamedArgs(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
};

// Helper functions (declared in libtorchtcl.h)
c10::ScalarType GetScalarType(const std::string& type_str);
torch::Device GetDevice(const std::string& device_str);

#endif // PARAMETER_PARSING_H 