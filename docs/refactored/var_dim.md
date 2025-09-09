# torch::var_dim

Computes the variance of elements along a specified dimension of a tensor. Supports both positional and named parameter syntax, and includes a camelCase alias.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::var_dim tensor dim ?unbiased? ?keepdim?
```

### Named Parameter Syntax (Recommended)
```tcl
torch::var_dim -input tensor_handle -dim dim ?-unbiased bool? ?-keepdim bool?
```

### CamelCase Alias
```tcl
torch::varDim ...
```

## Parameters

| Parameter   | Type    | Description                                 | Required | Default |
|-------------|---------|---------------------------------------------|----------|---------|
| input       | tensor  | Input tensor handle                         | Yes      |         |
| dim         | int     | Dimension to reduce                         | Yes      |         |
| unbiased    | bool    | Use unbiased estimator (divide by n-1)      | No       | true    |
| keepdim     | bool    | Retain reduced dimension in result          | No       | false   |

## Examples

### Positional Syntax
```tcl
set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu false]
set v [torch::var_dim $t 0]
```

### Named Parameter Syntax
```tcl
set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu false]
set v [torch::var_dim -input $t -dim 0 -unbiased 0 -keepdim 1]
```

### CamelCase Alias
```tcl
set v [torch::varDim $t 0]
set v [torch::varDim -input $t -dim 0]
```

## Return Value
A new tensor handle containing the variance along the specified dimension.

## Error Handling
- Returns an error if the input tensor does not exist.
- Returns an error if required parameters are missing or invalid.
- Returns an error for unknown named parameters.

## Migration Guide
- **Old syntax:** `torch::var_dim tensor dim ?unbiased? ?keepdim?`
- **New syntax:** `torch::var_dim -input tensor -dim dim ?-unbiased bool? ?-keepdim bool?`
- Both syntaxes are supported for backward compatibility.

## See Also
- [torch::std_dim](std_dim.md)
- [torch::mean_dim](mean_dim.md) 