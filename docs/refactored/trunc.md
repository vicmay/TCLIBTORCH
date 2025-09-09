# torch::trunc

Truncate Tensor Values

---

## Overview

Returns a new tensor with the truncated integer values of the elements of the input tensor (i.e., rounds each element toward zero).

---

## Syntax

### Positional (Backward Compatible)
```tcl
torch::trunc tensor
```

### Named Parameters (Modern)
```tcl
torch::trunc -input TENSOR
# or
torch::trunc -tensor TENSOR
```

---

## Parameters
| Name    | Type   | Required | Description                |
|---------|--------|----------|----------------------------|
| input   | Tensor | Yes      | Input tensor to truncate   |
| tensor  | Tensor | Yes      | Alias for input (optional) |

---

## Return Value
A tensor of the same shape as `input`, with each element truncated toward zero.

---

## Examples

### Positional Syntax
```tcl
set t [torch::tensor_create {1.7 -2.3 3.8 -4.1 5.9} float32 cpu true]
set trunc [torch::trunc $t]
```

### Named Parameter Syntax
```tcl
set t [torch::tensor_create {1.7 -2.3 3.8 -4.1 5.9} float32 cpu true]
set trunc [torch::trunc -input $t]
set trunc2 [torch::trunc -tensor $t]
```

---

## Migration Guide
- **Old code using positional syntax will continue to work.**
- **New code should use named parameters for clarity and maintainability.**

| Old (positional)   | New (named)                |
|--------------------|---------------------------|
| torch::trunc $t    | torch::trunc -input $t     |
| torch::trunc $t    | torch::trunc -tensor $t    |

---

## Error Handling
- Missing or invalid parameters will produce a clear error message.
- Both syntaxes are validated for required arguments.

---

## See Also
- [torch::floor](floor.md)
- [torch::ceil](ceil.md)
- [torch::round](round.md) 