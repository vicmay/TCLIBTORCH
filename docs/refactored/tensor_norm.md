# torch::tensor_norm / torch::tensorNorm

Computes the norm of a tensor. Supports Lp norm, dimension selection, and both positional and named parameter syntax.

---

## 🆕 Dual Syntax Support

### **Positional Syntax (Backward Compatible)**
```tcl
torch::tensor_norm tensor ?p? ?dim?
```

### **Named Parameter Syntax (Modern)**
```tcl
torch::tensor_norm -tensor tensor ?-p value? ?-dim value?
```

### **CamelCase Alias**
```tcl
torch::tensorNorm ...
```

---

## Parameters
| Name    | Type    | Required | Description                        |
|---------|---------|----------|------------------------------------|
| tensor  | handle  | Yes      | Input tensor                       |
| p       | double  | No       | Norm order (default: 2.0, L2 norm) |
| dim     | int     | No       | Dimension to reduce (optional)     |

---

## Return Value
- Returns a new tensor with the computed norm.

---

## Examples

### **Positional Syntax**
```tcl
set t [torch::tensor_create {3.0 4.0} float32 cpu true]
set n [torch::tensor_norm $t]         ;# L2 norm, result: 5.0
set n1 [torch::tensor_norm $t 1.0]    ;# L1 norm, result: 7.0
```

### **Named Parameter Syntax**
```tcl
set t [torch::tensor_create {3.0 4.0} float32 cpu true]
set n [torch::tensor_norm -tensor $t -p 1.0]
```

### **CamelCase Alias**
```tcl
set n [torch::tensorNorm -tensor $t -p 1.0]
```

---

## Error Handling
- **Missing tensor**: Returns error "Tensor not found".
- **Invalid p or dim value**: Returns error "Invalid p value" or "Invalid dim value".
- **Unknown named parameter**: Returns error listing valid parameters.
- **Missing required parameter**: Returns usage message.

---

## Migration Guide
- **Old code** using positional syntax will continue to work.
- **New code** can use named parameters and camelCase alias for clarity and modern style.

---

## Test Coverage
- Both syntaxes, camelCase alias, error handling, edge cases, and various p/dim values are tested in `tests/refactored/tensor_norm_test.tcl`. 