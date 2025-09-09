# `torch::tensor_backward` / `torch::tensorBackward`

Performs a backward pass (autograd) on the given tensor, computing gradients for all tensors in the computation graph that have `requiresGrad=true`.

---

## 🆕 Dual Syntax Support

### **Positional Syntax (backward compatible)**
```tcl
torch::tensor_backward <tensor>
```

### **Named Parameter Syntax (recommended)**
```tcl
torch::tensor_backward -input <tensor>
```

### **CamelCase Alias**
```tcl
torch::tensorBackward -input <tensor>
```

---

## Parameters
| Name   | Type    | Required | Description                       |
|--------|---------|----------|-----------------------------------|
| input  | string  | Yes      | Handle of the tensor to backward  |

---

## Return Value
- Returns `OK` if gradients are computed successfully.
- Returns an error message if the tensor is invalid, does not require gradients, or if backward is called multiple times on the same graph.

---

## Examples

### **Positional Syntax**
```tcl
set x [torch::tensor_create -data {2.0} -dtype float32 -requiresGrad true]
set y [torch::tensor_mul $x $x]
torch::tensor_backward $y
```

### **Named Parameter Syntax**
```tcl
set x [torch::tensor_create -data {3.0} -dtype float32 -requiresGrad true]
set y [torch::tensor_mul $x $x]
torch::tensor_backward -input $y
```

### **CamelCase Alias**
```tcl
set x [torch::tensor_create -data {4.0} -dtype float32 -requiresGrad true]
set y [torch::tensor_mul $x $x]
torch::tensorBackward -input $y
```

---

## Error Handling
- **Invalid tensor name**: Returns `Invalid tensor name`.
- **Tensor does not require gradients**: Returns `Tensor does not require gradients`.
- **Missing required parameter**: Returns `Input tensor is required`.
- **Unknown parameter**: Returns `Unknown parameter: ...`.
- **Multiple backward passes**: Returns an error if `.backward()` is called twice on the same computation graph (see below).

---

## PyTorch Limitation: Multiple Backward Passes
- PyTorch does **not** allow calling `.backward()` twice on the same computation graph unless `retain_graph=True` is specified.
- If you attempt to call `torch::tensor_backward` multiple times on the same tensor or computation, you will get an error like:
  > Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed)...
- This is a limitation of PyTorch, not the TCL extension. See the test file for details.

---

## Migration Guide
- **Old usage:**
  ```tcl
  torch::tensor_backward $y
  ```
- **New usage (recommended):**
  ```tcl
  torch::tensor_backward -input $y
  # or
  torch::tensorBackward -input $y
  ```

Both syntaxes are fully supported for backward compatibility.

---

## See Also
- [`torch::tensor_grad`](tensor_grad.md)
- [`torch::tensor_requires_grad`](tensor_requires_grad.md) 