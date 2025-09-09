# torch::tensor_stack / torch::tensorStack

**Description:**
Stacks a sequence of tensors along a new dimension. Supports both positional and named parameter syntax. Maintains backward compatibility.

---

## Syntax

**Positional:**
```tcl
set result [torch::tensor_stack tensor_list dim]
```

**Named:**
```tcl
set result [torch::tensor_stack -tensors tensor_list -dim dim]
set result [torch::tensor_stack -dim dim -tensors tensor_list]
```

**CamelCase alias:**
```tcl
set result [torch::tensorStack $tensorList $dim]
set result [torch::tensorStack -tensors $tensorList -dim $dim]
```

---

## Parameters
| Name    | Type   | Required | Description                        |
|---------|--------|----------|------------------------------------|
| tensors | list   | Yes      | List of tensor handles to stack    |
| dim     | int    | Yes      | Dimension to insert (default: 0)   |

- All tensors must have the same shape.
- The output tensor has one more dimension than the inputs.

---

## Returns
- (tensor handle) Stacked tensor with an added dimension

---

## Examples

**Stack 1D tensors:**
```tcl
set t1 [torch::tensor_create {1 2 3}]
set t2 [torch::tensor_create {4 5 6}]
set stacked [torch::tensor_stack [list $t1 $t2] 0]  # Shape: [2, 3]
```

**Named parameter syntax:**
```tcl
set t1 [torch::tensor_create {1 2 3}]
set t2 [torch::tensor_create {4 5 6}]
set stacked [torch::tensor_stack -tensors [list $t1 $t2] -dim 1]  # Shape: [3, 2]
```

**CamelCase alias:**
```tcl
set t1 [torch::tensor_create {1 2 3}]
set t2 [torch::tensor_create {4 5 6}]
set stacked [torch::tensorStack [list $t1 $t2] 0]
```

---

## Error Handling
- Invalid tensor handle: returns "Invalid tensor name"
- Shape mismatch: returns error about size
- Missing parameters: error about required parameters
- Unknown parameter: error about unknown parameter

---

## Migration Guide
- **Old:** `torch::tensor_stack [list $t1 $t2] 0`
- **New:** `torch::tensor_stack -tensors [list $t1 $t2] -dim 0` or `torch::tensorStack [list $t1 $t2] 0`
- Both syntaxes are fully supported and produce identical results.

---

## See Also
- [torch::tensor_cat](tensor_cat.md)
- [torch::tensor_reshape](tensor_reshape.md)
- [torch::tensor_permute](tensor_permute.md) 