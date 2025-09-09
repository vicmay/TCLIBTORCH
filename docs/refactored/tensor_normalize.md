# torch::tensor_normalize

Normalizes a tensor using the specified norm type and optional dimension.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_normalize tensor ?p? ?dim?
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_normalize -tensor tensor ?-p value? ?-dim value?
```

### CamelCase Alias
```tcl
torch::tensorNormalize tensor ?p? ?dim?
torch::tensorNormalize -tensor tensor ?-p value? ?-dim value?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor` | string | required | Name of the input tensor |
| `p` | double | 2.0 | Norm type (1.0 for L1, 2.0 for L2, etc.) |
| `dim` | int | nullopt | Dimension along which to normalize (optional) |

## Description

The `torch::tensor_normalize` command normalizes a tensor using the specified norm type. If no dimension is specified, the entire tensor is normalized by flattening it first. If a dimension is specified, normalization is performed along that dimension.

The normalization formula is:
- For L2 norm (p=2): `result = tensor / (||tensor||_2 + ε)`
- For L1 norm (p=1): `result = tensor / (||tensor||_1 + ε)`
- For other p values: `result = tensor / (||tensor||_p + ε)`

Where ε is a small epsilon (1e-8) to prevent division by zero.

## Examples

### Basic L2 Normalization (Default)
```tcl
# Create a tensor
set t [torch::tensor_create {3.0 4.0} float32 cpu true]

# Normalize using L2 norm (default)
set result [torch::tensor_normalize $t]
puts [torch::tensor_to_list $result]
# Output: 0.6000000238418579 0.800000011920929
```

### L1 Normalization
```tcl
# Create a tensor
set t [torch::tensor_create {3.0 4.0} float32 cpu true]

# Normalize using L1 norm
set result [torch::tensor_normalize $t 1.0]
puts [torch::tensor_to_list $result]
# Output: 0.4285714328289032 0.5714285969734192
```

### Normalization Along Specific Dimension
```tcl
# Create a 2D tensor
set t [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]

# Normalize along dimension 0
set result [torch::tensor_normalize $t 2.0 0]
puts [torch::tensor_to_list $result]
# Output: 0.3162277638912201 0.4472135901451111 0.9486832618713379 0.8944271802902222
```

### Using Named Parameters
```tcl
# Create a tensor
set t [torch::tensor_create {3.0 4.0} float32 cpu true]

# Normalize using named parameters
set result [torch::tensor_normalize -tensor $t -p 1.0]
puts [torch::tensor_to_list $result]
# Output: 0.4285714328289032 0.5714285969734192
```

### Using CamelCase Alias
```tcl
# Create a tensor
set t [torch::tensor_create {3.0 4.0} float32 cpu true]

# Normalize using camelCase alias
set result [torch::tensorNormalize $t]
puts [torch::tensor_to_list $result]
# Output: 0.6000000238418579 0.800000011920929
```

### Different Norm Types
```tcl
# Create a tensor
set t [torch::tensor_create {1.0 2.0} float32 cpu true]

# L0.5 norm
set result [torch::tensor_normalize $t 0.5]
puts [torch::tensor_to_list $result]
# Output: 0.17157284915447235 0.3431456983089447

# L3 norm
set result [torch::tensor_normalize $t 3.0]
puts [torch::tensor_to_list $result]
# Output: 0.48074984550476074 0.9614996910095215
```

### Negative Dimension
```tcl
# Create a 2D tensor
set t [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]

# Normalize along last dimension (-1)
set result [torch::tensor_normalize $t 2.0 -1]
puts [torch::tensor_to_list $result]
# Output: 0.4472135901451111 0.8944271802902222 0.6000000238418579 0.800000011920929
```

## Error Handling

### Missing Tensor
```tcl
set result [catch {torch::tensor_normalize nonexistent_tensor} error]
puts $error
# Output: Tensor not found
```

### Invalid Arguments
```tcl
# No arguments
set result [catch {torch::tensor_normalize} error]
puts $error
# Output: Usage: torch::tensor_normalize tensor ?p? ?dim? | torch::tensor_normalize -tensor tensor ?-p value? ?-dim value?

# Invalid p value
set t [torch::tensor_create {1.0 2.0} float32 cpu true]
set result [catch {torch::tensor_normalize $t "invalid"} error]
puts $error
# Output: Invalid p value

# Invalid dim value
set result [catch {torch::tensor_normalize $t 2.0 "invalid"} error]
puts $error
# Output: Invalid dim value

# Unknown named parameter
set result [catch {torch::tensor_normalize -tensor $t -unknown param} error]
puts $error
# Output: Unknown parameter: -unknown. Valid parameters are: -tensor, -p, -dim
```

## Migration Guide

### From Old Positional Syntax
The old syntax is still fully supported for backward compatibility:

```tcl
# Old syntax (still works)
set result [torch::tensor_normalize $tensor 2.0 0]

# New named parameter syntax (recommended)
set result [torch::tensor_normalize -tensor $tensor -p 2.0 -dim 0]

# CamelCase alias (also available)
set result [torch::tensorNormalize -tensor $tensor -p 2.0 -dim 0]
```

### Benefits of Named Parameters
1. **Clarity**: Parameter names make the code more readable
2. **Flexibility**: Parameters can be specified in any order
3. **Maintainability**: Easier to understand and modify
4. **Documentation**: Self-documenting code

## Notes

- The command adds a small epsilon (1e-8) to prevent division by zero
- When no dimension is specified, the tensor is flattened before normalization
- The result is a new tensor with the same shape as the input
- All numeric types are supported
- The command works on both CPU and CUDA tensors

## See Also

- `torch::tensor_norm` - Compute the norm of a tensor
- `torch::tensor_masked_fill` - Fill masked elements of a tensor
- `torch::tensor_create` - Create a new tensor 