# torch::tensor_exp

Computes the exponential of each element in a tensor.

## Description

The `torch::tensor_exp` command computes the exponential (e^x) of each element in the input tensor. This operation is commonly used in neural networks for activation functions and mathematical computations.

**Note**: The exponential function is defined as e^x where e is Euler's number (approximately 2.71828).

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_exp tensor
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_exp -input tensor
```

### CamelCase Alias
```tcl
torch::tensorExp -input tensor
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
set result1 [torch::tensor_exp $a]

# Using named parameter syntax
set result2 [torch::tensor_exp -input $a]

# Using camelCase alias
set result3 [torch::tensorExp -input $a]
```

### Different Value Types

```tcl
# Positive values
set positive [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
set result_pos [torch::tensor_exp -input $positive]

# Negative values
set negative [torch::tensor_create -data {-1.0 -2.0 -3.0 -4.0} -dtype float32 -device cpu]
set result_neg [torch::tensor_exp -input $negative]

# Zero values
set zero [torch::tensor_create -data {0.0 0.0 0.0 0.0} -dtype float32 -device cpu]
set result_zero [torch::tensor_exp -input $zero]
```

### Multi-dimensional Tensors

```tcl
# Create a 2D tensor
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
set a2d [torch::tensor_reshape $a {2 4}]
set result [torch::tensor_exp -input $a2d]
```

### Different Data Types

```tcl
# Float32 tensor
set a32 [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
set result32 [torch::tensor_exp -input $a32]

# Float64 tensor
set a64 [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float64 -device cpu]
set result64 [torch::tensor_exp -input $a64]
```

## Error Handling

### Invalid Tensor Names
```tcl
# This will throw an error
catch {torch::tensor_exp invalid_tensor} result
puts $result
# Output: Invalid tensor name
```

### Missing Parameters
```tcl
# This will throw an error
catch {torch::tensor_exp} result
puts $result
# Output: Required parameter missing: input
```

### Unknown Parameters
```tcl
# This will throw an error
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
catch {torch::tensor_exp -input $a -unknown_param value} result
puts $result
# Output: Unknown parameter: -unknown_param
```

## Migration Guide

### From Old Syntax to New Syntax

**Before (Positional Only):**
```tcl
set result [torch::tensor_exp $a]
```

**After (Named Parameters):**
```tcl
set result [torch::tensor_exp -input $a]
```

**After (CamelCase):**
```tcl
set result [torch::tensorExp -input $a]
```

### Benefits of New Syntax

1. **Clarity**: Parameter names make the code more readable
2. **Maintainability**: Easier to understand and modify
3. **Consistency**: Follows modern API design patterns
4. **Backward Compatibility**: Old syntax still works

## Technical Notes

- **Mathematical Operation**: Computes e^x for each element
- **Domain**: Works with all real numbers
- **Range**: Always produces positive values
- **Special Cases**: 
  - exp(0) = 1
  - exp(-∞) = 0
  - exp(∞) = ∞
- **Numerical Stability**: May overflow for large positive values
- **Memory Usage**: Creates a new tensor for the result

## Related Commands

- `torch::tensor_log` - Natural logarithm
- `torch::tensor_sqrt` - Square root
- `torch::tensor_pow` - Power function
- `torch::tensor_sigmoid` - Sigmoid activation (uses exp internally) 