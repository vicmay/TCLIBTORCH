# torch::sinh

Computes the hyperbolic sine (sinh) of the input tensor element-wise.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::sinh -input <tensor>
torch::sinh -tensor <tensor>
```

### Positional Syntax (Legacy)
```tcl
torch::sinh <tensor>
```

### CamelCase Alias
```tcl
torch::siNh <tensor>
torch::siNh -input <tensor>
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `-input` or `-tensor` | string | Yes | Name of the input tensor |

**Positional Parameters (Legacy):**
1. `tensor` (string): Name of the input tensor

## Returns

Returns a new tensor handle containing the hyperbolic sine of each element in the input tensor.

## Mathematical Definition

The hyperbolic sine function is defined as:
```
sinh(x) = (e^x - e^(-x)) / 2
```

## Examples

### Using Named Parameters (Recommended)

```tcl
# Basic usage
set input [torch::tensorCreate -data {0.0 1.0 -1.0 2.0} -dtype float32]
set result [torch::sinh -input $input]

# Alternative parameter name
set result [torch::sinh -tensor $input]

# Using camelCase alias
set result [torch::siNh -input $input]
```

### Using Positional Syntax (Legacy)

```tcl
# Basic usage
set input [torch::tensorCreate -data {0.0 1.0 -1.0 2.0} -dtype float32]
set result [torch::sinh $input]

# Using camelCase alias
set result [torch::siNh $input]
```

### Complete Example

```tcl
# Load the extension
load libtorchtcl.so

# Create input tensor
set input [torch::tensorCreate -data {0.0 1.0 -1.0 2.0} -dtype float32]

# Compute hyperbolic sine
set result [torch::sinh -input $input]

# Display result
torch::tensorPrint $result
```

### Mathematical Properties Examples

```tcl
# sinh(0) = 0
set zero [torch::tensorCreate -data {0.0} -dtype float32]
set sinh_zero [torch::sinh -input $zero]
# Result: ~0.0

# sinh(1) ≈ 1.175201
set one [torch::tensorCreate -data {1.0} -dtype float32]
set sinh_one [torch::sinh -input $one]
# Result: ~1.175201

# sinh(-x) = -sinh(x) (odd function)
set neg_one [torch::tensorCreate -data {-1.0} -dtype float32]
set sinh_neg_one [torch::sinh -input $neg_one]
# Result: ~-1.175201
```

## Error Handling

The command will throw an error in the following cases:

### Named Parameter Syntax Errors
```tcl
# Missing required parameter
torch::sinh
# Error: Usage: torch::sinh tensor | torch::sinh -input tensor

# Invalid parameter name
torch::sinh -invalid_param tensor1
# Error: Unknown parameter: -invalid_param. Valid parameters are: -input, -tensor

# Missing parameter value
torch::sinh -input
# Error: Missing value for parameter
```

### Common Errors
```tcl
# Invalid tensor name
torch::sinh -input invalid_tensor
# Error: Invalid tensor name

# Too many arguments in positional syntax
torch::sinh tensor1 extra_argument
# Error: Usage: torch::sinh tensor
```

## Data Type Support

The `torch::sinh` command supports all floating-point data types:

- `float32` (single precision)
- `float64` (double precision)

```tcl
# Float32 example
set input_f32 [torch::tensorCreate -data {1.0} -dtype float32]
set result_f32 [torch::sinh -input $input_f32]

# Float64 example  
set input_f64 [torch::tensorCreate -data {1.0} -dtype float64]
set result_f64 [torch::sinh -input $input_f64]
```

## Performance Notes

- The operation is computed element-wise on the entire tensor
- GPU tensors are supported when CUDA is available
- The computation preserves the shape and device of the input tensor

## Migration Guide

### From Positional to Named Parameters

**Old (Positional) Syntax:**
```tcl
set result [torch::sinh $input]
```

**New (Named Parameter) Syntax:**
```tcl
set result [torch::sinh -input $input]
```

### Advantages of Named Parameters

1. **Clarity**: Parameter purpose is explicit
2. **Flexibility**: Parameters can be provided in any order
3. **Maintainability**: Code is self-documenting
4. **Consistency**: Matches other refactored commands

## Backward Compatibility

The original positional syntax continues to work unchanged:

```tcl
# This still works exactly as before
set result [torch::sinh $input]

# Both syntaxes can be used in the same program
set result1 [torch::sinh $input]           # positional
set result2 [torch::sinh -input $input]    # named parameters
```

## See Also

- [torch::cosh](cosh.md) - Hyperbolic cosine
- [torch::tanh](tanh.md) - Hyperbolic tangent  
- [torch::asinh](asinh.md) - Inverse hyperbolic sine
- [torch::sin](sin.md) - Trigonometric sine
- [torch::exp](exp.md) - Exponential function

## Mathematical Functions Family

| Function | Description | Domain | Range |
|----------|-------------|---------|-------|
| `sinh(x)` | Hyperbolic sine | ℝ (all real numbers) | ℝ (all real numbers) |
| `cosh(x)` | Hyperbolic cosine | ℝ (all real numbers) | [1, +∞) |
| `tanh(x)` | Hyperbolic tangent | ℝ (all real numbers) | (-1, 1) |
| `asinh(x)` | Inverse hyperbolic sine | ℝ (all real numbers) | ℝ (all real numbers) | 