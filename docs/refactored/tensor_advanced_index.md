# torch::tensor_advanced_index / torch::tensorAdvancedIndex

Performs advanced indexing on a tensor using tensor indices for flexible element selection.

---

## 📝 **Syntax**

### **Positional (Backward Compatible)**
```tcl
torch::tensor_advanced_index tensor indices_list
```

### **Named Parameters (Modern)**
```tcl
torch::tensor_advanced_index -tensor tensor -indices indices_list
```

---

## 📋 **Parameters**
| Name         | Type     | Required | Description                    |
|--------------|----------|----------|--------------------------------|
| tensor       | handle   | Yes      | Input tensor to index          |
| indices_list | list     | Yes      | List of tensor handles for indexing |

---

## 🔄 **Return Value**
A new tensor handle containing the indexed elements from the input tensor.

---

## 🚦 **Examples**

### **Positional Syntax**
```tcl
# Create a 2x3 tensor
set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]

# Create index tensors
set idx1 [torch::tensor_create {0 1} int64]
set idx2 [torch::tensor_create {1 2} int64]

# Perform advanced indexing
set result [torch::tensor_advanced_index $t [list $idx1 $idx2]]
```

### **Named Parameter Syntax**
```tcl
# Create a 2x3 tensor
set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]

# Create index tensors
set idx1 [torch::tensor_create {0 1} int64]
set idx2 [torch::tensor_create {1 2} int64]

# Perform advanced indexing with named parameters
set result [torch::tensor_advanced_index -tensor $t -indices [list $idx1 $idx2]]
```

### **CamelCase Alias**
```tcl
# Using the camelCase alias
set result [torch::tensorAdvancedIndex -tensor $t -indices [list $idx1 $idx2]]
```

---

## 🛠️ **Migration Guide**
| Old (positional)                           | New (named)                                    |
|--------------------------------------------|------------------------------------------------|
| torch::tensor_advanced_index $t [list $idx] | torch::tensor_advanced_index -tensor $t -indices [list $idx] |
| torch::tensor_advanced_index $t [list $idx] | torch::tensorAdvancedIndex -tensor $t -indices [list $idx] |

---

## ⚠️ **Error Handling**
- If the tensor handle is missing or invalid, returns `Tensor not found`.
- If an index tensor handle is invalid, returns `Index tensor not found`.
- If no indices are provided, returns `Required parameters missing: tensor and indices list required`.
- If required parameters are missing, returns `Usage: torch::tensor_advanced_index tensor indices_list | torch::tensor_advanced_index -tensor tensor -indices indices_list`.
- If an unknown parameter is provided, returns `Unknown parameter: -param. Valid parameters are: -tensor, -indices`.
- If a parameter value is missing, returns `Missing value for parameter`.

---

## 🧪 **Testing**
- Both syntaxes are tested for correctness and error handling.
- Multiple index scenarios are tested.
- Edge cases and error conditions are validated.

---

## 📝 **Notes**
- Advanced indexing allows for complex tensor selection using tensor indices.
- The indices_list must contain valid tensor handles.
- The result tensor shape depends on the input tensor and the provided indices.
- 100% backward compatibility is maintained.
- Both `torch::tensor_advanced_index` and `torch::tensorAdvancedIndex` are available. 