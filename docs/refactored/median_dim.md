# torch::median_dim / torch::medianDim

**Category:** Reduction Operations  
**Aliases:** `torch::median_dim`, `torch::medianDim`

---

## Description

Computes the median value of elements of a tensor along a specified dimension, with optional retention of the reduced dimension.

---

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::median_dim tensor dim ?keepdim?
```
- `tensor`: Name/handle of the input tensor
- `dim`: Integer dimension to reduce
- `keepdim` (optional): Boolean, whether to retain reduced dimension (default: false)

### Named Parameter Syntax (Modern)
```tcl
torch::median_dim -input tensor -dim dim ?-keepdim bool?
```
- `-input`: Name/handle of the input tensor
- `-dim`: Integer dimension to reduce
- `-keepdim` (optional): Boolean, whether to retain reduced dimension (default: false)

### CamelCase Alias
```tcl
torch::medianDim ...
```
Supports both syntaxes above.

---

## Parameters

| Name      | Type    | Required | Default | Description                                 |
|-----------|---------|----------|---------|---------------------------------------------|
| input     | string  | Yes      |         | Input tensor handle/name                    |
| dim       | int     | Yes      |         | Dimension to reduce                         |
| keepdim   | bool    | No       | false   | Retain reduced dimension in output shape     |

---

## Examples

### 1. Positional Syntax
```tcl
set t [torch::tensor_create {{1.0 2.0 3.0} {4.0 5.0 6.0}}]
set result [torch::median_dim $t 1]
```

### 2. Named Parameter Syntax
```tcl
set t [torch::tensor_create {{1.0 2.0 3.0} {4.0 5.0 6.0}}]
set result [torch::median_dim -input $t -dim 1]
```

### 3. CamelCase Alias
```tcl
set t [torch::tensor_create {{1.0 2.0 3.0} {4.0 5.0 6.0}}]
set result [torch::medianDim -input $t -dim 1]
```

### 4. With keepdim
```tcl
set t [torch::tensor_create {{1.0 2.0 3.0} {4.0 5.0 6.0}}]
set result [torch::median_dim -input $t -dim 1 -keepdim 1]
```

---

## Error Handling

- Invalid tensor name: returns error "Invalid tensor name"
- Missing required parameters: returns usage error
- Invalid dim or keepdim type: returns error "Invalid dim value" or "Invalid keepdim value"

---

## Migration Guide

| Old (Positional)                        | New (Named)                                 |
|-----------------------------------------|---------------------------------------------|
| `torch::median_dim $t 1`                | `torch::median_dim -input $t -dim 1`        |
| `torch::median_dim $t 0 1`              | `torch::median_dim -input $t -dim 0 -keepdim 1`|
| `torch::medianDim $t 1`                 | `torch::medianDim -input $t -dim 1`         |

Both syntaxes are fully supported for backward compatibility.

---

## Return Value

Returns a new tensor handle containing the median-reduced tensor.
