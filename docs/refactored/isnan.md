# torch::isnan

Returns a tensor indicating which elements are NaN (Not a Number).

## Syntax

### Modern Syntax (Recommended)
```tcl
torch::isnan -input TENSOR
torch::isNan -input TENSOR  ;# camelCase alias
```

### Legacy Syntax (Backward Compatible)
```tcl
torch::isnan TENSOR
torch::isNan TENSOR  ;# camelCase alias
```

## Parameters

### Named Parameters (Modern Syntax)
- **`-input TENSOR`** *(required)*: Input tensor to check for NaN values
- **`-tensor TENSOR`** *(alias)*: Alternative name for `-input` parameter

### Positional Parameters (Legacy Syntax)
1. **`TENSOR`** *(required)*: Input tensor to check for NaN values

## Return Value

Returns a new tensor of the same shape as the input, containing boolean values:
- **`1` (true)**: Element is NaN (Not a Number)
- **`0` (false)**: Element is a valid number (finite or infinite)

## Description

The `torch::isnan` function creates a boolean tensor that identifies NaN (Not a Number) values in the input tensor. This function is essential for:

- **Data validation and cleaning**
- **Debugging numerical computations**
- **Handling missing or undefined values**
- **Mathematical analysis and filtering**

NaN values typically arise from undefined mathematical operations like:
- `0.0 / 0.0`
- `infinity - infinity`
- `sqrt(-1.0)` (for real numbers)
- Operations involving existing NaN values

## Examples

### Basic Usage
```tcl
# Create a tensor with normal finite values
set tensor1 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]

# Check for NaN values using modern syntax
set result1 [torch::isnan -input $tensor1]
# Result: tensor with all 0s (false) - no NaN values

# Check using legacy syntax
set result2 [torch::isnan $tensor1]
# Result: identical to result1

# Check using camelCase alias
set result3 [torch::isNan $tensor1]
# Result: identical to result1 and result2
```

### Data Type Support
```tcl
# Works with different floating-point types
set float32_tensor [torch::tensorCreate -data {1.0 2.0} -dtype float32]
set float64_tensor [torch::tensorCreate -data {1.0 2.0} -dtype float64]

set nan_check_f32 [torch::isnan $float32_tensor]
set nan_check_f64 [torch::isnan $float64_tensor]

# Integer tensors (integers cannot be NaN)
set int_tensor [torch::tensorCreate -data {1 2 3} -dtype int32]
set nan_check_int [torch::isnan $int_tensor]
# Result: all 0s (false) - integers are never NaN
```

### Multi-dimensional Tensors
```tcl
# 2D tensor example
set matrix [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
set nan_matrix [torch::isnan $matrix]
# Result: 2x2 tensor with boolean values
```

### Error Handling Examples
```tcl
# Invalid usage examples (will raise errors)

# Missing arguments
catch {torch::isnan} error
# Error: Usage information displayed

# Invalid tensor name
catch {torch::isnan invalid_tensor} error
# Error: Invalid tensor name

# Unknown parameter  
set tensor [torch::tensorCreate -data {1.0 2.0} -dtype float32]
catch {torch::isnan -unknown_param $tensor} error
# Error: Unknown parameter: -unknown_param
```

## Mathematical Properties

1. **Non-NaN Values**: All finite numbers, positive/negative infinity return `false`
2. **NaN Values**: Only actual NaN values return `true`
3. **Type Preservation**: Output tensor has same shape as input
4. **Integer Safety**: Integer tensors always return `false` (integers cannot be NaN)

## Comparison with Related Functions

- **`torch::isfinite`**: Returns `true` for finite numbers (not NaN, not infinite)
- **`torch::isinf`**: Returns `true` for positive/negative infinity
- **`torch::isnan`**: Returns `true` only for NaN values

## Performance Notes

- **Efficient**: Operation is performed element-wise with optimized PyTorch kernels
- **Memory**: Creates new tensor with same shape as input
- **CUDA Support**: Automatically uses GPU acceleration when input is on GPU

## Migration Guide

### From Legacy to Modern Syntax

```tcl
# Old style (still supported)
set result [torch::isnan $my_tensor]

# New style (recommended)  
set result [torch::isnan -input $my_tensor]
```

### Benefits of Modern Syntax
- **Explicit parameter names** improve code readability
- **Parameter validation** provides better error messages
- **Extensibility** allows for future parameter additions
- **IDE support** enables better autocompletion

## Common Use Cases

### 1. Data Validation
```tcl
set data [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
set has_nan [torch::isnan $data]
# Check if any NaN values exist in dataset
```

### 2. Computational Debugging
```tcl
set computation_result [torch::div $numerator $denominator]
set nan_locations [torch::isnan $computation_result]
# Find where divisions produced NaN (e.g., 0/0)
```

### 3. Data Cleaning Pipeline
```tcl
set raw_data [load_data_from_file "dataset.txt"]
set nan_mask [torch::isnan $raw_data]
# Identify locations needing data imputation
```

## See Also

- [`torch::isfinite`](isfinite.md) - Check for finite values
- [`torch::isinf`](isinf.md) - Check for infinite values
- [`torch::isclose`](isclose.md) - Compare tensors with tolerance
- [Tensor Creation Guide](../tensor_creation.md)
- [Mathematical Operations Guide](../mathematical_operations.md) 