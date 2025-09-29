#include "libtorchtcl.h"
#include <atomic>

// Global storage definitions
std::unordered_map<std::string, torch::Tensor> tensor_storage;
std::unordered_map<std::string, std::shared_ptr<torch::optim::Optimizer>> optimizer_storage;
std::unordered_map<std::string, std::shared_ptr<torch::nn::Module>> module_storage;

// Helper function to get scalar type from string
c10::ScalarType GetScalarType(const char* type_str) {
    if (strcmp(type_str, "float32") == 0 || strcmp(type_str, "Float32") == 0 || strcmp(type_str, "float") == 0) return torch::kFloat32;
    if (strcmp(type_str, "float64") == 0 || strcmp(type_str, "Float64") == 0 || strcmp(type_str, "double") == 0) return torch::kFloat64;
    if (strcmp(type_str, "int32") == 0 || strcmp(type_str, "Int32") == 0 || strcmp(type_str, "int") == 0) return torch::kInt32;
    if (strcmp(type_str, "int64") == 0 || strcmp(type_str, "Int64") == 0 || strcmp(type_str, "long") == 0) return torch::kInt64;
    if (strcmp(type_str, "bool") == 0 || strcmp(type_str, "Bool") == 0) return torch::kBool;
    throw std::runtime_error(std::string("Unknown scalar type: ") + type_str);
}

// Helper function to get device from string
torch::Device GetDevice(const char* device_str) {
    if (strcmp(device_str, "cuda") == 0) {
        // Try to use CUDA, but fall back to CPU if it fails
        try {
            if (torch::cuda::is_available()) {
                return torch::kCUDA;
            }
        } catch (const std::exception& e) {
            // CUDA not available or error, fall back to CPU
        }
    }
    return torch::kCPU;
}

// Helper function to convert TCL list (supports 1-D or 2-D rectangular lists) to tensor with type
torch::Tensor TclListToTensor(Tcl_Interp* interp, Tcl_Obj* list,
                             const char* type_str,
                             const char* device_str,
                             bool requires_grad) {
    int outerLen;
    if (Tcl_ListObjLength(interp, list, &outerLen) != TCL_OK) {
        throw std::runtime_error("Invalid list object");
    }

    if (outerLen == 0) {
        // Create an empty tensor with the specified options
        auto options = torch::TensorOptions()
            .dtype(GetScalarType(type_str))
            .device(GetDevice(device_str))
            .requires_grad(requires_grad);
        
        return torch::empty({0}, options);
    }

    // Detect if first element is a list (2-D case)
    Tcl_Obj* firstElem;
    Tcl_ListObjIndex(interp, list, 0, &firstElem);
    int innerLen = 0;
    bool firstElemIsList = (Tcl_ListObjLength(interp, firstElem, &innerLen) == TCL_OK);
    double dummyNumeric;
    bool firstElemIsNumeric = (Tcl_GetDoubleFromObj(interp, firstElem, &dummyNumeric) == TCL_OK);
    bool is2D = firstElemIsList && !firstElemIsNumeric;

    std::vector<double> flat;
    std::vector<int64_t> shape;

    if (!is2D) {
        // ---------------- 1-D ----------------
        shape.push_back(outerLen);
        flat.reserve(outerLen);
        for (int i = 0; i < outerLen; ++i) {
            Tcl_Obj* elem;
            Tcl_ListObjIndex(interp, list, i, &elem);
            double v;
            if (Tcl_GetDoubleFromObj(interp, elem, &v) != TCL_OK) {
                throw std::runtime_error("Invalid numeric value in list");
            }
            flat.push_back(v);
        }
    } else {
        // ---------------- 2-D ----------------
        shape.push_back(outerLen);       // rows
        shape.push_back(innerLen);       // cols
        flat.reserve(static_cast<size_t>(outerLen) * innerLen);

        // Validate rectangular matrix and collect values
        for (int r = 0; r < outerLen; ++r) {
            Tcl_Obj* rowObj;
            Tcl_ListObjIndex(interp, list, r, &rowObj);
            int thisLen;
            if (Tcl_ListObjLength(interp, rowObj, &thisLen) != TCL_OK) {
                throw std::runtime_error("Invalid sub-list in 2-D tensor data");
            }
            if (thisLen != innerLen) {
                throw std::runtime_error("Jagged lists are not supported – each row must have equal length");
            }
            for (int c = 0; c < innerLen; ++c) {
                Tcl_Obj* elem;
                Tcl_ListObjIndex(interp, rowObj, c, &elem);
                double v;
                if (Tcl_GetDoubleFromObj(interp, elem, &v) != TCL_OK) {
                    throw std::runtime_error("Invalid numeric value in list");
                }
                flat.push_back(v);
            }
        }
    }

    auto options = torch::TensorOptions()
        .dtype(GetScalarType(type_str))
        .device(GetDevice(device_str))
        .requires_grad(requires_grad);

    torch::Tensor t = torch::tensor(flat, options);
    if (shape.size() > 1) {
        t = t.reshape(shape);
    }
    return t;
}

// Helper function to convert TCL list to shape vector
std::vector<int64_t> TclListToShape(Tcl_Interp* interp, Tcl_Obj* list) {
    int length;
    Tcl_ListObjLength(interp, list, &length);
    
    std::vector<int64_t> shape;
    shape.reserve(length);
    
    for (int i = 0; i < length; i++) {
        Tcl_Obj* element;
        Tcl_ListObjIndex(interp, list, i, &element);
        
        int value;
        if (Tcl_GetIntFromObj(interp, element, &value) != TCL_OK) {
            if (length == 1) {
                std::string txt = Tcl_GetString(list);
                throw std::runtime_error("expected list but got \"" + txt + "\"");
            }
            throw std::runtime_error("Invalid integer in shape list");
        }
        shape.push_back(static_cast<int64_t>(value));
    }
    
    return shape;
}

// Helper function to generate unique handles
std::string GetNextHandle(const std::string& prefix) {
    static std::atomic<int> counter{0};
    return prefix + std::to_string(counter.fetch_add(1));
}

// Helper function to get tensor from Tcl object
torch::Tensor GetTensorFromObj(Tcl_Interp* interp, Tcl_Obj* obj) {
    (void)interp; // Suppress unused parameter warning
    std::string name = Tcl_GetString(obj);
    if (tensor_storage.find(name) == tensor_storage.end()) {
        throw std::runtime_error("Invalid tensor");
    }
    return tensor_storage[name];
}

// Helper function to get integer from Tcl object
int GetIntFromObj(Tcl_Interp* interp, Tcl_Obj* obj) {
    int value;
    if (Tcl_GetIntFromObj(interp, obj, &value) != TCL_OK) {
        throw std::runtime_error("Invalid integer value");
    }
    return value;
}

// Helper function to get double from Tcl object
double GetDoubleFromObj(Tcl_Interp* interp, Tcl_Obj* obj) {
    double value;
    if (Tcl_GetDoubleFromObj(interp, obj, &value) != TCL_OK) {
        throw std::runtime_error("Invalid double value");
    }
    return value;
}

// Helper function to get boolean from Tcl object
bool GetBoolFromObj(Tcl_Interp* interp, Tcl_Obj* obj) {
    int value;
    if (Tcl_GetBooleanFromObj(interp, obj, &value) != TCL_OK) {
        throw std::runtime_error("Invalid boolean value");
    }
    return value != 0;
}

// Helper function to get integer vector from Tcl object
std::vector<int64_t> GetIntVectorFromObj(Tcl_Interp* interp, Tcl_Obj* obj) {
    int length;
    if (Tcl_ListObjLength(interp, obj, &length) != TCL_OK) {
        throw std::runtime_error("Invalid list object");
    }
    
    std::vector<int64_t> result;
    result.reserve(length);
    
    for (int i = 0; i < length; i++) {
        Tcl_Obj* element;
        if (Tcl_ListObjIndex(interp, obj, i, &element) != TCL_OK) {
            throw std::runtime_error("Invalid list element");
        }
        
        int value;
        if (Tcl_GetIntFromObj(interp, element, &value) != TCL_OK) {
            throw std::runtime_error("Invalid integer in list");
        }
        result.push_back(static_cast<int64_t>(value));
    }
    
    return result;
}

// Helper function to set tensor result
int SetTensorResult(Tcl_Interp* interp, const torch::Tensor& tensor) {
    std::string handle = GetNextHandle("tensor");
    tensor_storage[handle] = tensor;
    Tcl_SetObjResult(interp, Tcl_NewStringObj(handle.c_str(), -1));
    return TCL_OK;
}

 