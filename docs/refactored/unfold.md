# torch::unfold

Extract Sliding Blocks from Tensor

---

## Overview

Returns a view of the original tensor which contains all slices of size `size` from `self` tensor in the dimension `dimension`. Step between two slices is given by `step`.

---

## Syntax

### Positional (Backward Compatible)
```tcl
torch::unfold input dimension size step
```

### Named Parameters (Modern)
```tcl
torch::unfold -input INPUT -dimension DIMENSION -size SIZE -step STEP
# or
torch::unfold -tensor INPUT -dimension DIMENSION -size SIZE -step STEP
```

---

## Parameters
| Name      | Type    | Required | Description                                    |
|-----------|---------|----------|------------------------------------------------|
| input     | Tensor  | Yes      | Input tensor                                   |
| tensor    | Tensor  | Yes      | Alias for input (optional)                     |
| dimension | Integer | Yes      | Dimension in which to unfold                   |
| size      | Integer | Yes      | Size of each slice (must be > 0)               |
| step      | Integer | Yes      | Step between slices (must be > 0)              |

---

## Return Value
A tensor containing all slices of size `size` from the input tensor in the specified dimension.

---

## Examples

### Positional Syntax
```tcl
# 1D tensor example
set t [torch::tensor_create {1 2 3 4 5 6 7 8} float32 cpu true]
set unfold [torch::unfold $t 0 3 2]

# 2D tensor example
set t2 [torch::tensor_create {{1 2 3 4} {5 6 7 8} {9 10 11 12}} float32 cpu true]
set unfold2 [torch::unfold $t2 1 2 1]
```

### Named Parameter Syntax
```tcl
# 1D tensor example
set t [torch::tensor_create {1 2 3 4 5 6 7 8} float32 cpu true]
set unfold [torch::unfold -input $t -dimension 0 -size 3 -step 2]

# 2D tensor example
set t2 [torch::tensor_create {{1 2 3 4} {5 6 7 8} {9 10 11 12}} float32 cpu true]
set unfold2 [torch::unfold -input $t2 -dimension 1 -size 2 -step 1]
```

---

## Migration Guide
- **Old code using positional syntax will continue to work.**
- **New code should use named parameters for clarity and maintainability.**

| Old (positional)           | New (named)                                    |
|----------------------------|------------------------------------------------|
| torch::unfold $t 0 3 2    | torch::unfold -input $t -dimension 0 -size 3 -step 2 |
| torch::unfold $t 0 3 2    | torch::unfold -tensor $t -dimension 0 -size 3 -step 2 |

---

## Error Handling
- Missing or invalid parameters will produce a clear error message.
- Both syntaxes are validated for required arguments.
- Size and step parameters must be positive integers.

---

## See Also
- [torch::fold](fold.md)
- [torch::view](view.md)
- [torch::reshape](reshape.md) 