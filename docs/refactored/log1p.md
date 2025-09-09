# torch::log1p

Computes the natural logarithm of (1 + x) for each element in the input tensor. This function provides improved numerical precision compared to `log(1 + x)` for small values of x.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::log1p -input TENSOR
torch::log1p -tensor TENSOR    # Alternative parameter name
```

### Positional Parameters (Legacy)
```tcl
torch::log1p TENSOR
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `-input` | Tensor | Input tensor | Yes |
| `-tensor` | Tensor | Input tensor (alternative to `-input`) | Yes |

**Note**: Either `-input` or `-tensor` must be provided. They are equivalent parameter names.

## Returns

Returns a new tensor with the natural logarithm of (1 + x) for each element in the input tensor.

## Mathematical Definition

The log1p function is computed as:
```
log1p(x) = ln(1 + x)
```

**Key Properties:**
- `log1p(0) = 0`
- For small x: `log1p(x) ≈ x - x²/2 + x³/3 - x⁴/4 + ...` (Taylor series)
- More numerically stable than `log(1 + x)` for small values of x
- Domain: x > -1 (x must be greater than -1)

## Numerical Advantage

The `log1p` function is specifically designed to provide accurate results when x is close to zero. For very small values of x, computing `log(1 + x)` directly can suffer from numerical precision loss due to floating-point arithmetic limitations. The `log1p` function uses specialized algorithms to maintain precision in these cases.

## Examples

### Basic Usage (Named Parameters)

```tcl
# Create input tensor
set input [torch::tensor_create -data {0.0 1.0 2.0} -dtype float32 -device cpu -requiresGrad false]

# Compute log1p
set result [torch::log1p -input $input]
# Result: [0.0, 0.693147, 1.098612]  # log(1), log(2), log(3)

# Using alternative parameter name
set result [torch::log1p -tensor $input]
# Same result: [0.0, 0.693147, 1.098612]
```

### Legacy Syntax (Positional Parameters)

```tcl
# Create input tensor
set input [torch::tensor_create -data {0.0 1.0 2.0} -dtype float32 -device cpu -requiresGrad false]

# Compute log1p using legacy syntax
set result [torch::log1p $input]
# Result: [0.0, 0.693147, 1.098612]
```

### Working with Small Values (Precision Advantage)

```tcl
# For very small values, log1p provides better precision than log(1 + x)
set small_input [torch::tensor_create -data {1e-10 1e-8 1e-6} -dtype float64 -device cpu -requiresGrad false]

# log1p maintains precision for small values
set result [torch::log1p -input $small_input]
# Result: [1e-10, 1e-8, 1e-6] (approximately, with high precision)

# Compare with manual log(1 + x) - less precise for small x
set ones [torch::tensor_create -data {1.0 1.0 1.0} -dtype float64 -device cpu -requiresGrad false]
set one_plus_x [torch::tensor_add $ones $small_input]
set manual_log [torch::tensor_log $one_plus_x]
# manual_log may have lower precision than log1p result
```

### Working with Negative Values

```tcl
# log1p works with negative values as long as x > -1
set input [torch::tensor_create -data {-0.5 -0.1 -0.01} -dtype float32 -device cpu -requiresGrad false]

set result [torch::log1p -input $input]
# Result: [-0.693147, -0.105361, -0.010050]  # log(0.5), log(0.9), log(0.99)
```

### Relationship with expm1

```tcl
# log1p and expm1 are inverse functions
set input [torch::tensor_create -data {0.1 0.5 1.0} -dtype float64 -device cpu -requiresGrad false]

# Forward: expm1 then log1p should return original value
set expm1_result [torch::tensor_expm1 $input]
set log1p_expm1_result [torch::log1p -input $expm1_result]
# log1p_expm1_result ≈ input

# Backward: log1p then expm1 should return original value (for x > -1)
set log1p_result [torch::log1p -input $input]
set expm1_log1p_result [torch::tensor_expm1 $log1p_result]
# expm1_log1p_result ≈ input
```

## Data Type Support

The `torch::log1p` operation supports the following data types:

| Data Type | Supported | Notes |
|-----------|-----------|-------|
| `float32` | ✅ | Standard precision |
| `float64` | ✅ | Double precision (recommended for high precision) |
| `double` | ✅ | Alias for float64 |

**Note**: Input values must satisfy x > -1. Values ≤ -1 will result in NaN or -inf.

## Device Support

- **CPU**: Fully supported
- **CUDA**: Supported when PyTorch is built with CUDA support

## Error Handling

### Common Errors

1. **Missing Arguments**
   ```tcl
   # Error: Missing tensor argument
   torch::log1p
   ```

2. **Invalid Tensor Reference**
   ```tcl
   # Error: Tensor doesn't exist
   torch::log1p nonexistent_tensor
   ```

3. **Missing Parameter Value**
   ```tcl
   # Error: -input requires a value
   torch::log1p -input
   ```

4. **Unknown Parameter**
   ```tcl
   # Error: Unknown parameter
   torch::log1p -unknown_param tensor1
   ```

## Mathematical Properties

### Special Values

| Input | log1p(Input) | Notes |
|-------|--------------|-------|
| 0.0 | 0.0 | log1p(0) = ln(1) = 0 |
| -1.0 | -∞ | Negative infinity |
| < -1 | NaN | Not a number (invalid domain) |

### Taylor Series Expansion

For small values of x, log1p(x) can be approximated using its Taylor series:
```
log1p(x) = x - x²/2 + x³/3 - x⁴/4 + x⁵/5 - ...
```

This series converges for |x| < 1 and is the basis for the high-precision implementation.

### Relationship to Other Functions

```tcl
# Relationship with natural logarithm
# log1p(x) = log(1 + x)

# Relationship with expm1
# log1p(expm1(x)) = x (for appropriate domains)
# expm1(log1p(x)) = x (for x > -1)

# Derivative
# d/dx log1p(x) = 1/(1 + x)
```

## Performance Considerations

1. **Precision**: Use `float64` for maximum precision, especially for small values
2. **Domain**: Ensure input values are > -1 for valid results
3. **Memory**: The operation creates a new tensor; original tensor is unchanged
4. **Vectorization**: Operation is applied element-wise and vectorized
5. **Numerical Stability**: Preferred over `log(1 + x)` for small x values

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set result [torch::log1p $tensor]

# New named parameter syntax (recommended)
set result [torch::log1p -input $tensor]
```

### Parameter Equivalence

```tcl
# These are equivalent:
set result1 [torch::log1p -input $tensor]
set result2 [torch::log1p -tensor $tensor]
```

### From Manual log(1 + x) to log1p

```tcl
# Less precise approach (avoid for small x)
set ones [torch::tensor_create -data {1.0 1.0 1.0} -dtype float64 -device cpu -requiresGrad false]
set one_plus_x [torch::tensor_add $ones $input]
set result_manual [torch::tensor_log $one_plus_x]

# Preferred approach (better precision)
set result_log1p [torch::log1p -input $input]
```

## Related Commands

- [`torch::tensor_log`](tensor_log.md) - Natural logarithm
- [`torch::tensor_expm1`](expm1.md) - exp(x) - 1 (inverse function)
- [`torch::log10`](log10.md) - Base-10 logarithm
- [`torch::log2`](log2.md) - Base-2 logarithm
- [`torch::tensor_exp`](tensor_exp.md) - Natural exponential

## See Also

- [Mathematical Operations](../mathematical_operations.md)
- [Numerical Precision](../numerical_precision.md)
- [Tensor Creation](../tensor_creation.md)
- [Data Types](../data_types.md) 