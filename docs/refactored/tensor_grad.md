# `torch::tensor_grad` / `torch::tensorGrad`

Returns the gradient tensor for a given input tensor (if computed).

---

## 🆕 Dual Syntax Support

### **Positional Syntax (backward compatible)**
```tcl
torch::tensor_grad <tensor>
```

### **Named Parameter Syntax (recommended)**
```tcl
torch::tensor_grad -input <tensor>
```

### **CamelCase Alias**
```tcl
torch::tensorGrad -input <tensor>
```

---

## Parameters
| Name   | Type    | Required | Description                       |
|--------|---------|----------|-----------------------------------|
| input  | string  | Yes      | Handle of the input tensor        |

---

## Return Value
- Returns a tensor handle for the gradient tensor if available.
- Returns an error message if the tensor does not require gradients, no gradient has been computed, or the tensor is invalid.

---

## Examples

### **Positional Syntax**
```tcl
set x [torch::tensor_create -data {2.0} -dtype float32 -requiresGrad true]
set y [torch::tensor_mul $x $x]
torch::tensor_backward $y
set grad [torch::tensor_grad $x]
```

### **Named Parameter Syntax**
```tcl
set x [torch::tensor_create -data {3.0} -dtype float32 -requiresGrad true]
set y [torch::tensor_mul $x $x]
torch::tensor_backward $y
set grad [torch::tensor_grad -input $x]
```

### **CamelCase Alias**
```tcl
set x [torch::tensor_create -data {4.0} -dtype float32 -requiresGrad true]
set y [torch::tensor_mul $x $x]
torch::tensor_backward $y
set grad [torch::tensorGrad -input $x]
```

---

## Error Handling
- **Invalid tensor name**: Returns `Invalid tensor name`.
- **Tensor does not require gradients**: Returns `Tensor does not require gradients`.
- **No gradient computed yet**: Returns `No gradient computed yet`.
- **Missing required parameter**: Returns `Input tensor is required`.
- **Unknown parameter**: Returns `Unknown parameter: ...`.

---

## Migration Guide
- **Old usage:**
  ```tcl
  torch::tensor_grad $x
  ```
- **New usage (recommended):**
  ```tcl
  torch::tensor_grad -input $x
  # or
  torch::tensorGrad -input $x
  ```

Both syntaxes are fully supported for backward compatibility.

---

## See Also
- [`torch::tensor_requires_grad`](tensor_requires_grad.md)
- [`torch::tensor_backward`](tensor_backward.md) 