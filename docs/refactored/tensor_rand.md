# `torch::tensor_rand` / `torch::tensorRand`

Creates a tensor filled with random numbers from a uniform distribution on [0, 1). Supports both positional and named parameter syntax, as well as a camelCase alias.

---

## 📝 **Usage**

### **Positional Syntax (Backward Compatible)**
```tcl
# Basic usage
set tensor [torch::tensor_rand {2 3}]

# With device and dtype
set tensor [torch::tensor_rand {2 3} cpu float32]

# Scalar tensor
set tensor [torch::tensor_rand {}]
```

### **Named Parameter Syntax (New)**
```tcl
# Basic usage
set tensor [torch::tensor_rand -shape {2 3}]

# With device and dtype
set tensor [torch::tensor_rand -shape {2 3} -device cpu -dtype float64]

# Scalar tensor
set tensor [torch::tensor_rand -shape {}]
```

### **CamelCase Alias**
```tcl
# Basic usage
set tensor [torch::tensorRand {2 3}]

# With named parameters
set tensor [torch::tensorRand -shape {2 3} -dtype float32]
```

---

## 🧾 **Parameters**

| Name   | Type      | Required | Default | Description                |
|--------|-----------|----------|---------|----------------------------|
| shape  | list      | Yes      | -       | Tensor shape (empty for scalar) |
| device | string    | No       | "cpu"   | Device ("cpu", "cuda", etc.) |
| dtype  | string    | No       | "float32" | Data type (float types only) |

- **Positional:** `torch::tensor_rand <shape> ?device? ?dtype?`
- **Named:** `torch::tensor_rand -shape <shape> ?-device <device>? ?-dtype <dtype>?`

---

## ✅ **Return Value**
A tensor handle (string) containing random values from uniform distribution [0, 1).

---

## ⚠️ **Error Handling**
- **Missing shape parameter:**
  - Returns: `Required parameter missing: shape`
- **Invalid shape:**
  - Returns: `Invalid shape parameter`
- **Unknown parameter:**
  - Returns: `Unknown parameter: -param`
- **Too many arguments (positional):**
  - Returns: `Invalid number of arguments`
- **Missing value for parameter:**
  - Returns: `Missing value for parameter`

---

## 🔄 **Migration Guide**
- **Old (positional):**
  - `torch::tensor_rand {2 3}`
- **New (named):**
  - `torch::tensor_rand -shape {2 3}`
- **CamelCase:**
  - `torch::tensorRand {2 3}`

All syntaxes are fully supported and produce identical results.

---

## 🧪 **Examples**

### Create a 2x3 tensor with default settings
```tcl
set t [torch::tensor_rand {2 3}]
puts [torch::tensor_print $t]
```

### Create a tensor with specific device and dtype
```tcl
set t [torch::tensor_rand -shape {3 3} -device cpu -dtype float64]
puts [torch::tensor_print $t]
```

### Create a scalar tensor
```tcl
set t [torch::tensor_rand {}]
puts [torch::tensor_print $t]
```

### Using camelCase alias
```tcl
set t [torch::tensorRand {2 2}]
puts [torch::tensor_print $t]
```

---

## 🧩 **Edge Cases**
- **Scalar tensors:** Use empty shape `{}`
- **Large tensors:** Supported but may be memory intensive
- **Float types only:** PyTorch's `rand` only supports float32, float64

---

## 🛡️ **Test Coverage**
- Both syntaxes, camelCase alias
- Error handling (missing shape, unknown parameters, too many args)
- Edge cases (scalar, large tensors)
- Device and dtype support
- Syntax consistency
- Mathematical correctness (values in [0,1))

---

## 📚 **See Also**
- [`tensor_randn`](tensor_randn.md) - Normal distribution random tensors
- [`tensor_create`](tensor_create.md) - Create tensors with specific values
- [`tensor_shape`](tensor_shape.md) - Get tensor shape

---

## ⚠️ **Limitations**
- **Float types only:** PyTorch's `rand` function only supports float32 and float64 dtypes
- **Uniform distribution:** Values are uniformly distributed on [0, 1)
- **Device support:** Depends on PyTorch installation (CPU always available) 