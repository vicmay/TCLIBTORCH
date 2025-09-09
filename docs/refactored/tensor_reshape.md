# torch::tensor_reshape / torch::tensorReshape

**Description:**
Reshapes a tensor to the specified shape. Supports both positional and named parameter syntax. Maintains backward compatibility.

---

## Syntax

**Positional:**
```tcl
set result [torch::tensor_reshape tensor_handle shape]
```

**Named:**
```tcl
set result [torch::tensor_reshape -input tensor_handle -shape shape]
set result [torch::tensor_reshape -shape shape -input tensor_handle]
```

**CamelCase alias:**
```tcl
set result [torch::tensorReshape $tensor {2 3}]
set result [torch::tensorReshape -input $tensor -shape {2 3}]
```

---

## Parameters
| Name   | Type   | Required | Description                       |
|--------|--------|----------|-----------------------------------|
| input  | str    | Yes      | Tensor handle to reshape          |
| shape  | list   | Yes      | New shape (may include -1)        |

- `shape` can include -1 for one inferred dimension.
- The total number of elements must remain the same.

---

## Returns
- (tensor handle) Reshaped tensor (view if possible, otherwise copy)

---

## Examples

**Reshape 1D to 2D:**
```tcl
set t [torch::tensor_create {1 2 3 4 5 6}]
set t2d [torch::tensor_reshape $t {2 3}]
# $t2d shape: [2, 3]
```

**Named parameter syntax:**
```tcl
set t [torch::tensor_create {1 2 3 4 5 6}]
set t2d [torch::tensor_reshape -input $t -shape {2 3}]
```

**CamelCase alias:**
```tcl
set t [torch::tensor_create {1 2 3 4 5 6}]
set t2d [torch::tensorReshape $t {2 3}]
```

**Inferred dimension:**
```tcl
set t [torch::tensor_create {1 2 3 4 5 6}]
set t2d [torch::tensor_reshape $t {-1 2}]  # shape: [3, 2]
```

---

## Error Handling
- Invalid tensor handle: returns "Invalid tensor name"
- Shape mismatch: returns error about size
- Multiple inferred dimensions: error about inferred dimension
- Missing parameters: error about required parameters
- Unknown parameter: error about unknown parameter

---

## Migration Guide
- **Old:** `torch::tensor_reshape $t {2 3}`
- **New:** `torch::tensor_reshape -input $t -shape {2 3}` or `torch::tensorReshape $t {2 3}`
- Both syntaxes are fully supported and produce identical results.

---

## See Also
- [torch::tensor_permute](tensor_permute.md)
- [torch::tensor_stack](tensor_stack.md)
- [torch::tensor_cat](tensor_cat.md) 