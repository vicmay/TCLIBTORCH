# torch::tensor_clamp / torch::tensorClamp

Clamps all elements in the input tensor to be within the specified range [min, max].

---

## 📝 **Syntax**

### **Positional (Backward Compatible)**
```tcl
torch::tensor_clamp tensor ?min? ?max?
```

### **Named Parameters (Modern)**
```tcl
torch::tensor_clamp -tensor tensor ?-min value? ?-max value?
```

---

## 📋 **Parameters**
| Name   | Type   | Required | Description                |
|--------|--------|----------|----------------------------|
| tensor | handle | Yes      | Input tensor to clamp      |
| min    | double | No       | Lower bound (default: no limit) |
| max    | double | No       | Upper bound (default: no limit) |

---

## 🔄 **Return Value**
A new tensor handle containing the clamped values.

---

## 🚦 **Examples**

### **Positional Syntax**
```tcl
# Create a tensor
set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0}]

# No clamping (just clone)
set result [torch::tensor_clamp $t]

# Clamp with minimum only
set result [torch::tensor_clamp $t 2.5]

# Clamp with both minimum and maximum
set result [torch::tensor_clamp $t 2.0 4.0]
```

### **Named Parameter Syntax**
```tcl
# Create a tensor
set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0}]

# No clamping (just clone)
set result [torch::tensor_clamp -tensor $t]

# Clamp with minimum only
set result [torch::tensor_clamp -tensor $t -min 2.5]

# Clamp with maximum only
set result [torch::tensor_clamp -tensor $t -max 3.5]

# Clamp with both minimum and maximum
set result [torch::tensor_clamp -tensor $t -min 2.0 -max 4.0]
```

### **CamelCase Alias**
```tcl
# Using the camelCase alias
set result [torch::tensorClamp -tensor $t -min 2.0 -max 4.0]
```

---

## 🛠️ **Migration Guide**
| Old (positional)                    | New (named)                                    |
|-------------------------------------|------------------------------------------------|
| torch::tensor_clamp $t              | torch::tensor_clamp -tensor $t                 |
| torch::tensor_clamp $t 2.5          | torch::tensor_clamp -tensor $t -min 2.5        |
| torch::tensor_clamp $t 2.0 4.0      | torch::tensor_clamp -tensor $t -min 2.0 -max 4.0 |
| torch::tensor_clamp $t 2.0 4.0      | torch::tensorClamp -tensor $t -min 2.0 -max 4.0 |

---

## ⚠️ **Error Handling**
- If the tensor handle is missing or invalid, returns `Tensor not found`.
- If required parameters are missing, returns `Usage: torch::tensor_clamp tensor ?min? ?max? | torch::tensor_clamp -tensor tensor ?-min value? ?-max value?`.
- If an unknown parameter is provided, returns `Unknown parameter: -param. Valid parameters are: -tensor, -min, -max`.
- If a parameter value is missing, returns `Missing value for parameter`.
- If min or max values are invalid numbers, returns `Invalid min value` or `Invalid max value`.

---

## 🧪 **Testing**
- Both syntaxes are tested for correctness and error handling.
- Various clamping scenarios are tested (no bounds, min only, max only, both bounds).
- Error conditions and edge cases are validated.

---

## 📝 **Notes**
- If no min or max is specified, the tensor is cloned without modification.
- If only min is specified, values below min are set to min.
- If only max is specified, values above max are set to max.
- If both min and max are specified, values are clamped to the range [min, max].
- The operation is element-wise and preserves the tensor shape.
- 100% backward compatibility is maintained.
- Both `torch::tensor_clamp` and `torch::tensorClamp` are available.
- This operation is commonly used for gradient clipping, activation function implementation, and data normalization. 