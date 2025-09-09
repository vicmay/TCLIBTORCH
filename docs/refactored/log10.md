# torch::log10

Computes the base-10 logarithm of each element in the input tensor.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::log10 -input TENSOR
torch::log10 -tensor TENSOR    # Alternative parameter name
```

### Positional Parameters (Legacy)
```tcl
torch::log10 TENSOR
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `-input` | Tensor | Input tensor | Yes |
| `-tensor` | Tensor | Input tensor (alternative to `-input`) | Yes |

**Note**: Either `-input` or `-tensor` must be provided. They are equivalent parameter names.

## Returns

Returns a new tensor with the base-10 logarithm of each element in the input tensor.

## Mathematical Definition

The base-10 logarithm is computed as:
```
log₁₀(x) = ln(x) / ln(10)
```

Where:
- `log₁₀(10^n) = n` for any real number n
- `log₁₀(1) = 0`
- `log₁₀(10) = 1`
- `log₁₀(100) = 2`

## Examples

### Basic Usage (Named Parameters)

```tcl
# Create input tensor with powers of 10
set input [torch::tensor_create -data {1.0 10.0 100.0 1000.0} -dtype float32 -device cpu -requiresGrad false]

# Compute base-10 logarithm
set result [torch::log10 -input $input]
# Result: [0.0, 1.0, 2.0, 3.0]

# Using alternative parameter name
set result [torch::log10 -tensor $input]
# Same result: [0.0, 1.0, 2.0, 3.0]
```

### Legacy Syntax (Positional Parameters)

```tcl
# Create input tensor
set input [torch::tensor_create -data {10.0 100.0 1000.0} -dtype float32 -device cpu -requiresGrad false]

# Compute base-10 logarithm using legacy syntax
set result [torch::log10 $input]
# Result: [1.0, 2.0, 3.0]
```

### Working with Fractional Values

```tcl
# Create tensor with fractional powers of 10
set input [torch::tensor_create -data {0.1 0.01 0.001} -dtype float32 -device cpu -requiresGrad false]

# Compute base-10 logarithm
set result [torch::log10 -input $input]
# Result: [-1.0, -2.0, -3.0]
```

### Working with 2D Tensors

```tcl
# Create 2D tensor
set input [torch::tensor_create -data {{1.0 10.0} {100.0 1000.0}} -dtype float32 -device cpu -requiresGrad false]

# Compute base-10 logarithm
set result [torch::log10 -input $input]
# Result: [[0.0, 1.0], [2.0, 3.0]]
```

### High Precision Calculations

```tcl
# Create high precision tensor
set input [torch::tensor_create -data {1e6 1e7 1e8 1e9} -dtype float64 -device cpu -requiresGrad false]

# Compute base-10 logarithm with high precision
set result [torch::log10 -input $input]
# Result: [6.0, 7.0, 8.0, 9.0]
```

## Data Type Support

The `torch::log10` operation supports the following data types:

| Data Type | Supported | Notes |
|-----------|-----------|-------|
| `float32` | ✅ | Standard precision |
| `float64` | ✅ | Double precision |
| `double` | ✅ | Alias for float64 |

**Note**: Input values must be positive (> 0). Negative values or zero will result in NaN or -inf.

## Device Support

- **CPU**: Fully supported
- **CUDA**: Supported when PyTorch is built with CUDA support

## Error Handling

### Common Errors

1. **Missing Arguments**
   ```tcl
   # Error: Missing tensor argument
   torch::log10
   ```

2. **Invalid Tensor Reference**
   ```tcl
   # Error: Tensor doesn't exist
   torch::log10 nonexistent_tensor
   ```

3. **Missing Parameter Value**
   ```tcl
   # Error: -input requires a value
   torch::log10 -input
   ```

4. **Unknown Parameter**
   ```tcl
   # Error: Unknown parameter
   torch::log10 -unknown_param tensor1
   ```

## Mathematical Properties

### Special Values

| Input | log10(Input) | Notes |
|-------|--------------|-------|
| 1.0 | 0.0 | log₁₀(1) = 0 |
| 10.0 | 1.0 | log₁₀(10) = 1 |
| 100.0 | 2.0 | log₁₀(100) = 2 |
| 0.1 | -1.0 | log₁₀(0.1) = -1 |
| 0.0 | -∞ | Negative infinity |
| < 0 | NaN | Not a number |

### Relationship to Natural Logarithm

```tcl
# log10(x) = ln(x) / ln(10)
set input [torch::tensor_create -data {2.0 3.0 5.0} -dtype float64 -device cpu -requiresGrad false]

set result_log10 [torch::log10 -input $input]
set result_ln [torch::tensor_log $input]
set ln_10 [torch::tensor_create -data {2.302585092994046} -dtype float64 -device cpu -requiresGrad false]
set result_calculated [torch::tensor_div $result_ln $ln_10]

# result_log10 ≈ result_calculated
```

## Performance Considerations

1. **Precision**: Use `float64` for high-precision calculations
2. **Memory**: The operation creates a new tensor; original tensor is unchanged
3. **Vectorization**: Operation is applied element-wise and vectorized
4. **Domain**: Ensure input values are positive for valid results

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set result [torch::log10 $tensor]

# New named parameter syntax (recommended)
set result [torch::log10 -input $tensor]
```

### Parameter Equivalence

```tcl
# These are equivalent:
set result1 [torch::log10 -input $tensor]
set result2 [torch::log10 -tensor $tensor]
```

## Related Commands

- [`torch::tensor_log`](tensor_log.md) - Natural logarithm (base e)
- [`torch::log2`](log2.md) - Base-2 logarithm
- [`torch::exp10`](exp10.md) - Base-10 exponential (inverse operation)
- [`torch::tensor_exp`](tensor_exp.md) - Natural exponential

## See Also

- [Mathematical Operations](../mathematical_operations.md)
- [Tensor Creation](../tensor_creation.md)
- [Data Types](../data_types.md) 