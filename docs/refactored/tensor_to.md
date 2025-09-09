# `torch::tensor_to` / `torch::tensorTo`

Converts a tensor to a specified device and/or data type.

---

## 🆕 Dual Syntax Support

### **Positional Syntax (backward compatible)**
```tcl
torch::tensor_to <tensor> <device> ?<dtype>?
```

### **Named Parameter Syntax (recommended)**
```tcl
torch::tensor_to -input <tensor> -device <device> ?-dtype <dtype>?
```

### **CamelCase Alias**
```tcl
torch::tensorTo -input <tensor> -device <device> ?-dtype <dtype>?
```

---

## Parameters
| Name   | Type    | Required | Description                       |
|--------|---------|----------|-----------------------------------|
| input  | string  | Yes      | Handle of the input tensor        |
| device | string  | Yes      | Target device (cpu, cuda, etc.)   |
| dtype  | string  | No       | Target data type (float32, int64, etc.) |

---

## Return Value
- Returns a tensor handle for the converted tensor.
- Returns an error message if the tensor is invalid, device is unsupported, or dtype is invalid.

---

## Examples

### **Positional Syntax**
```tcl
# Device conversion only
set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set result [torch::tensor_to $x cpu]

# Device and dtype conversion
set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set result [torch::tensor_to $x cpu float64]
```

### **Named Parameter Syntax**
```tcl
# Device conversion only
set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set result [torch::tensor_to -input $x -device cpu]

# Device and dtype conversion
set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set result [torch::tensor_to -input $x -device cpu -dtype float64]
```

### **CamelCase Alias**
```tcl
# Device conversion only
set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set result [torch::tensorTo -input $x -device cpu]

# Device and dtype conversion
set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set result [torch::tensorTo -input $x -device cpu -dtype float64]
```

### **CUDA Examples**
```tcl
# Convert to CUDA (if available)
if {[torch::cuda_is_available]} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_to -input $x -device cuda]
}

# Convert from CUDA to CPU
if {[torch::cuda_is_available]} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cuda]
    set result [torch::tensor_to -input $x -device cpu]
}
```

---

## Error Handling
- **Invalid tensor name**: Returns `Invalid tensor name`.
- **Missing required parameter**: Returns `Required parameters missing: input and device`.
- **Invalid device**: Returns device-specific error message.
- **Invalid dtype**: Returns dtype-specific error message.
- **Unknown parameter**: Returns `Unknown parameter: ...`.

---

## Migration Guide
- **Old usage:**
  ```tcl
  torch::tensor_to $x cpu
  torch::tensor_to $x cpu float64
  ```
- **New usage (recommended):**
  ```tcl
  torch::tensor_to -input $x -device cpu
  torch::tensor_to -input $x -device cpu -dtype float64
  # or
  torch::tensorTo -input $x -device cpu -dtype float64
  ```

Both syntaxes are fully supported for backward compatibility.

---

## Supported Devices
- `cpu` - CPU device
- `cuda` - CUDA GPU device (if available)
- `cuda:0`, `cuda:1`, etc. - Specific CUDA devices

## Supported Data Types
- `float32`, `float64` - Floating point types
- `int32`, `int64` - Integer types
- `bool` - Boolean type

---

## See Also
- [`torch::tensor_device`](tensor_device.md)
- [`torch::tensor_dtype`](tensor_dtype.md)
- [`torch::cuda_is_available`](../cuda_commands.md) 