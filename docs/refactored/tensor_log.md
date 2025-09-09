# torch::tensor_log

Computes the natural logarithm of each element in a tensor.

## Description

The `torch::tensor_log` command computes the natural logarithm (ln(x)) of each element in the input tensor. This operation is commonly used in neural networks for loss functions, activation functions, and mathematical computations.

**Note**: The natural logarithm is defined as ln(x) where the base is Euler's number (e ≈ 2.71828).

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_log tensor
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_log -input tensor
```

### CamelCase Alias
```tcl
torch::tensorLog -input tensor
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | string | Yes | Name of the input tensor |

## Return Value

Returns a string containing the handle name of the resulting tensor.

## Examples

### Basic Usage

```tcl
# Create a tensor
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]

# Using positional syntax
set result1 [torch::tensor_log $a]

# Using named parameter syntax
set result2 [torch::tensor_log -input $a]

# Using camelCase alias
set result3 [torch::tensorLog -input $a]
```

### Different Value Types

```tcl
# Positive values
set positive [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
set result_pos [torch::tensor_log -input $positive]

# Values greater than 1
set large [torch::tensor_create -data {2.0 4.0 8.0 16.0} -dtype float32 -device cpu]
set result_large [torch::tensor_log -input $large]

# Value of 1 (log(1) = 0)
set one [torch::tensor_create -data {1.0 1.0 1.0 1.0} -dtype float32 -device cpu]
set result_one [torch::tensor_log -input $one]
```

### Multi-dimensional Tensors

```tcl
# Create a 2D tensor
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
set a2d [torch::tensor_reshape $a {2 4}]
set result [torch::tensor_log -input $a2d]
```

### Different Data Types

```tcl
# Float32 tensor
set a32 [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
set result32 [torch::tensor_log -input $a32]

# Float64 tensor
set a64 [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float64 -device cpu]
set result64 [torch::tensor_log -input $a64]
```

## Error Handling

### Invalid Tensor Names
```tcl
# This will throw an error
catch {torch::tensor_log invalid_tensor} result
puts $result
# Output: Invalid tensor name
```

### Missing Parameters
```tcl
# This will throw an error
catch {torch::tensor_log} result
puts $result
# Output: Required parameter missing: input
```

### Unknown Parameters
```tcl
# This will throw an error
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
catch {torch::tensor_log -input $a -unknown_param value} result
puts $result
# Output: Unknown parameter: -unknown_param
```

## Migration Guide

### From Old Syntax to New Syntax

**Before (Positional Only):**
```tcl
set result [torch::tensor_log $a]
```

**After (Named Parameters):**
```tcl
set result [torch::tensor_log -input $a]
```

**After (CamelCase):**
```tcl
set result [torch::tensorLog -input $a]
```

### Benefits of New Syntax

1. **Clarity**: Parameter names make the code more readable
2. **Maintainability**: Easier to understand and modify
3. **Consistency**: Follows modern API design patterns
4. **Backward Compatibility**: Old syntax still works

## Technical Notes

- **Mathematical Operation**: Computes ln(x) for each element
- **Domain**: Works with positive real numbers (x > 0)
- **Range**: Produces real numbers
- **Special Cases**: 
  - log(1) = 0
  - log(e) = 1
  - log(0) = -∞ (undefined)
  - log(-x) = undefined for real numbers
- **Numerical Stability**: May produce -∞ for values very close to 0
- **Memory Usage**: Creates a new tensor for the result

## Related Commands

- `torch::tensor_exp` - Exponential function (inverse of log)
- `torch::tensor_sqrt` - Square root
- `torch::tensor_pow` - Power function
- `torch::log2` - Base-2 logarithm
- `torch::log10` - Base-10 logarithm 