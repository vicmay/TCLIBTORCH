# torch::tensor_masked_fill / torch::tensorMaskedFill

Fills elements of a tensor with a scalar value where a boolean mask is true.

---

## 🆕 Dual Syntax Support

### **Positional Syntax (Backward Compatible)**
```tcl
torch::tensor_masked_fill tensor mask value
```

### **Named Parameter Syntax (Modern)**
```tcl
torch::tensor_masked_fill -tensor tensor -mask mask -value value
```

### **CamelCase Alias**
```tcl
torch::tensorMaskedFill ...
```

---

## Parameters
| Name    | Type    | Required | Description                        |
|---------|---------|----------|------------------------------------|
| tensor  | handle  | Yes      | Input tensor to fill               |
| mask    | handle  | Yes      | Boolean tensor mask (same shape)   |
| value   | double  | Yes      | Scalar value to fill where mask=1  |

---

## Return Value
- Returns a new tensor with elements filled where mask is true.

---

## Examples

### **Positional Syntax**
```tcl
set t [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
set m [torch::tensor_create {1 0 1 0} bool cpu false]
set r [torch::tensor_masked_fill $t $m 0.0]
# r is [0.0 2.0 0.0 4.0]
```

### **Named Parameter Syntax**
```tcl
set t [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
set m [torch::tensor_create {1 0 1 0} bool cpu false]
set r [torch::tensor_masked_fill -tensor $t -mask $m -value 0.0]
```

### **CamelCase Alias**
```tcl
set r [torch::tensorMaskedFill -tensor $t -mask $m -value 0.0]
```

---

## Error Handling
- **Missing parameters**: Returns error with usage message.
- **Invalid tensor/mask handle**: Returns "Tensor not found".
- **Invalid value type**: Returns "Invalid value parameter".
- **Unknown named parameter**: Returns error listing valid parameters.

---

## Migration Guide
- **Old code** using positional syntax will continue to work.
- **New code** can use named parameters and camelCase alias for clarity and modern style.

---

## Test Coverage
- Both syntaxes, camelCase alias, error handling, edge cases, and 2D/negative values are tested in `tests/refactored/tensor_masked_fill_test.tcl`. 