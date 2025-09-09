# torch::vsplit

**Category:** Tensor Manipulation

---

## Overview

Splits a tensor vertically (along dimension 0) into multiple sub-tensors, either by number of sections or by indices. Supports both positional and named parameter syntax, as well as a camelCase alias.

---

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::vsplit tensor sections_or_indices
```

### Named Parameter Syntax (Modern)
```tcl
torch::vsplit -tensor tensor -sections sections
# or
torch::vsplit -tensor tensor -indices indices
# or
# camelCase alias
torch::vSplit -tensor tensor -sections sections
```

---

## Parameters

| Name      | Alias   | Type         | Required | Description                                 |
|-----------|---------|--------------|----------|---------------------------------------------|
| tensor    | input   | tensor handle| Yes      | The tensor to split (must be at least 2D)   |
| sections  |         | int          | Yes*     | Number of sections to split into            |
| indices   |         | list of int  | Yes*     | Indices at which to split                   |

*Either `sections` or `indices` must be provided, not both.

---

## Examples

### Split into equal sections (positional)
```tcl
set tensor [torch::ones -shape {6 4}]
set result [torch::vsplit $tensor 3]
# result: list of 3 tensors, each shape {2 4}
```

### Split at indices (named)
```tcl
set tensor [torch::arange -start 0 -end 24 -dtype float32]
set reshaped [torch::tensor_reshape $tensor {6 4}]
set result [torch::vsplit -tensor $reshaped -indices {2 4}]
# result: 3 tensors, shapes {2 4}, {2 4}, {2 4}
```

### Using camelCase alias
```tcl
set tensor [torch::ones -shape {6 4}]
set result [torch::vSplit -tensor $tensor -sections 2]
```

---

## Error Handling
- Tensor must have at least 2 dimensions
- If using `sections`, the size of dimension 0 must be divisible by `sections`
- If using `indices`, indices must be valid split points
- Clear error messages for missing/invalid parameters

---

## Migration Guide

| Old Syntax                        | New Syntax (Recommended)                  |
|-----------------------------------|-------------------------------------------|
| torch::vsplit $tensor 2           | torch::vsplit -tensor $tensor -sections 2 |
| torch::vsplit $tensor {1 3}       | torch::vsplit -tensor $tensor -indices {1 3} |
| torch::vsplit $tensor 2           | torch::vSplit -tensor $tensor -sections 2 |

Both syntaxes are fully supported for backward compatibility.

---

## See Also
- [torch::hsplit](hsplit.md)
- [torch::dsplit](dsplit.md)
- [torch::tensor_split](tensor_split.md) 