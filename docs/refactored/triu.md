# torch::triu

Upper Triangular Matrix Extraction

---

## Overview

Returns the upper triangular part of a matrix (or batch of matrices), the other elements are set to zero.

---

## Syntax

### Positional (Backward Compatible)
```tcl
torch::triu input ?diagonal?
```

### Named Parameters (Modern)
```tcl
torch::triu -input INPUT
# or with diagonal offset
torch::triu -input INPUT -diagonal DIAGONAL
```

---

## Parameters
| Name      | Type    | Required | Description                                 |
|-----------|---------|----------|---------------------------------------------|
| input     | Tensor  | Yes      | Input tensor (2D or batch of 2D)            |
| diagonal  | Integer | No       | Diagonal offset (default: 0, main diagonal) |

---

## Return Value
A tensor of the same shape as `input`, with elements below the specified diagonal set to zero.

---

## Examples

### Positional Syntax
```tcl
set t [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9}} float32 cpu true]
set triu [torch::triu $t]
# With diagonal offset
set triu1 [torch::triu $t 1]
```

### Named Parameter Syntax
```tcl
set t [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9}} float32 cpu true]
set triu [torch::triu -input $t]
set triu1 [torch::triu -input $t -diagonal 1]
```

---

## Migration Guide
- **Old code using positional syntax will continue to work.**
- **New code should use named parameters for clarity and maintainability.**

| Old (positional) | New (named) |
|------------------|-------------|
| torch::triu $t 1 | torch::triu -input $t -diagonal 1 |

---

## Error Handling
- Missing or invalid parameters will produce a clear error message.
- Both syntaxes are validated for required arguments.

---

## See Also
- [torch::tril](tril.md)
- [torch::diag](diag.md) 