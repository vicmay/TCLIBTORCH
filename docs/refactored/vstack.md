# torch::vstack

**Category:** Tensor Manipulation

---

## Overview

Stacks tensors vertically (along dimension 0) to create a new tensor. All input tensors must have the same number of columns. Supports both positional and named parameter syntax, as well as a camelCase alias.

---

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::vstack tensor_list
# or
torch::vstack tensor1 tensor2 tensor3...
```

### Named Parameter Syntax (Modern)
```tcl
torch::vstack -tensors tensor_list
# or
torch::vstack -inputs tensor_list
# or
# camelCase alias
torch::vStack -tensors tensor_list
```

---

## Parameters

| Name      | Alias   | Type         | Required | Description                                 |
|-----------|---------|--------------|----------|---------------------------------------------|
| tensors   | inputs  | tensor list  | Yes      | List of tensors to stack vertically         |

---

## Examples

### Stack two tensors (positional)
```tcl
set tensor1 [torch::ones -shape {2 3}]
set tensor2 [torch::ones -shape {2 3}]
set result [torch::vstack [list $tensor1 $tensor2]]
# result: tensor with shape {4 3}
```

### Stack multiple tensors (named)
```tcl
set tensor1 [torch::ones -shape {1 4}]
set tensor2 [torch::ones -shape {1 4}]
set tensor3 [torch::ones -shape {1 4}]
set result [torch::vstack -tensors [list $tensor1 $tensor2 $tensor3]]
# result: tensor with shape {3 4}
```

### Using camelCase alias
```tcl
set tensor1 [torch::ones -shape {2 3}]
set tensor2 [torch::ones -shape {2 3}]
set result [torch::vStack -tensors [list $tensor1 $tensor2]]
```

### Stack with different shapes
```tcl
set tensor1 [torch::ones -shape {1 3}]
set tensor2 [torch::ones -shape {2 3}]
set result [torch::vstack $tensor1 $tensor2]
# result: tensor with shape {3 3}
```

---

## Error Handling
- All input tensors must have the same number of columns
- At least one tensor must be provided
- Clear error messages for missing/invalid parameters
- Invalid tensor handles result in descriptive error messages

---

## Migration Guide

| Old Syntax                        | New Syntax (Recommended)                  |
|-----------------------------------|-------------------------------------------|
| torch::vstack {tensor1 tensor2}   | torch::vstack -tensors {tensor1 tensor2} |
| torch::vstack tensor1 tensor2     | torch::vstack -tensors {tensor1 tensor2} |
| torch::vstack {tensor1 tensor2}   | torch::vStack -tensors {tensor1 tensor2} |

Both syntaxes are fully supported for backward compatibility.

---

## See Also
- [torch::hstack](hstack.md) - Stack tensors horizontally
- [torch::dstack](dstack.md) - Stack tensors depth-wise
- [torch::column_stack](column_stack.md) - Stack tensors column-wise
- [torch::row_stack](row_stack.md) - Alias for vstack 