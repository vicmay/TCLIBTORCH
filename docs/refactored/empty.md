# torch::empty / torch::Empty

Create an uninitialized tensor with specified shape and properties. Supports both positional and named parameter syntax, as well as a camelCase alias.

---

## 📝 **Syntax**

### **Positional (backward compatible)**
```
torch::empty shape ?dtype? ?device? ?requires_grad?
```

### **Named parameters (modern)**
```
torch::empty -shape shape ?-dtype dtype? ?-device device? ?-requiresGrad bool?
torch::Empty   -shape shape ?-dtype dtype? ?-device device? ?-requiresGrad bool?
```

### **Mixed syntax**
```
torch::empty shape -dtype dtype -device device -requiresGrad bool
```

---

## 🧩 **Parameters**
| Name          | Type    | Required | Default    | Description                           |
|---------------|---------|----------|------------|---------------------------------------|
| shape         | list    | Yes      | -          | Tensor shape as list of integers      |
| dtype         | string  | No       | "float32"  | Data type (float32, float64, int32, int64, bool) |
| device        | string  | No       | "cpu"      | Device ("cpu" or "cuda")              |
| requires_grad | bool    | No       | false      | Whether tensor requires gradients     |

---

## 🏷️ **Aliases**
- `torch::empty` (snake_case)
- `torch::Empty` (camelCase)

---

## 📋 **Examples**

### **Positional Syntax**
```tcl
;# Basic usage
set tensor [torch::empty {2 3}]

;# With dtype
set tensor [torch::empty {2 3} int64]

;# With device
set tensor [torch::empty {2 3} float32 cpu]

;# With requires_grad
set tensor [torch::empty {2 3} float32 cpu true]
```

### **Named Parameter Syntax**
```tcl
;# Basic usage
set tensor [torch::empty -shape {2 3}]

;# With all parameters
set tensor [torch::empty -shape {2 3} -dtype float64 -device cpu -requiresGrad true]

;# Parameters in different order
set tensor [torch::empty -dtype int32 -shape {3 4} -requiresGrad false -device cpu]
```

### **Mixed Syntax**
```tcl
;# Positional shape with named parameters
set tensor [torch::empty {2 3} -dtype float32 -requiresGrad true]
```

### **CamelCase Alias**
```tcl
;# Using camelCase alias
set tensor [torch::Empty {2 3}]
set tensor [torch::Empty -shape {2 3} -dtype float32]
```

---

## ⚠️ **Error Handling**
- Missing arguments: returns "Invalid arguments for torch::empty"
- Invalid shape: returns "expected list but got ..."
- Invalid dtype: returns "Unknown scalar type: ..."
- Invalid device: returns "Invalid device string: ..."
- Non-floating point tensors with requires_grad=true: raises exception

---

## 🔄 **Migration Guide**
- **Old (positional):**
  ```tcl
  torch::empty {2 3} float32 cpu true
  ```
- **New (named):**
  ```tcl
  torch::empty -shape {2 3} -dtype float32 -device cpu -requiresGrad true
  torch::Empty -shape {2 3} -dtype float32 -device cpu -requiresGrad true
  ```

Both syntaxes are fully supported. Migration is optional but recommended for clarity and future compatibility.

---

## 📊 **Data Types**
| Type     | Description                    | Requires Grad |
|----------|--------------------------------|---------------|
| float32  | 32-bit floating point          | Yes           |
| float64  | 64-bit floating point          | Yes           |
| int32    | 32-bit integer                 | No            |
| int64    | 64-bit integer                 | No            |
| bool     | Boolean                        | No            |

---

## ✅ **Test Coverage**
- Positional and named syntax
- Mixed syntax (positional + named)
- CamelCase alias
- Error handling
- Different shapes and data types
- Edge cases
- Syntax consistency

---

## 🔗 **See Also**
- [torch::zeros](zeros.md)
- [torch::ones](ones.md)
- [torch::full](full.md)
- [torch::eye](eye.md) 