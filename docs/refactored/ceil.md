# torch::ceil

Computes the ceiling (smallest integer greater than or equal to each element) of each element in the input tensor.

## Syntax

### Current Syntax
```tcl
torch::ceil tensor
```

### Named Parameter Syntax  
```tcl
torch::ceil -input tensor
torch::ceil -tensor tensor
```

Both syntaxes are fully supported and equivalent.

## Parameters

### Named Parameters
- `-input` (required): Input tensor name
- `-tensor` (required): Alternative name for input tensor

### Positional Parameters
1. `tensor` (required): Input tensor name

## Description

The `torch::ceil` function computes the ceiling of each element in the input tensor. The ceiling function returns the smallest integer that is greater than or equal to the input value. This is also known as "rounding up" to the nearest integer.

## Mathematical Details

For any tensor element x:
- `ceil(x)` returns the smallest integer n such that `n ≥ x`
- Examples:
  - `ceil(2.3) = 3`
  - `ceil(-1.7) = -1`
  - `ceil(5.0) = 5`
  - `ceil(-3.0) = -3`
  - `ceil(0.1) = 1`
  - `ceil(-0.1) = 0`

## Examples

### Basic Usage

#### Positional Syntax
```tcl
# Create a tensor with floating point values
set input [torch::tensor_create {1.2 2.7 -0.5 -1.8 3.0}]
set result [torch::ceil $input]
# Result contains [2.0, 3.0, 0.0, -1.0, 3.0]
```

#### Named Parameter Syntax
```tcl
# Same operation using named parameters
set input [torch::tensor_create {1.2 2.7 -0.5 -1.8 3.0}]
set result [torch::ceil -input $input]

# Alternative parameter name
set result [torch::ceil -tensor $input]
```

### Mathematical Properties

```tcl
# Ceiling of integers remains unchanged
set integers [torch::tensor_create {1.0 2.0 -3.0 0.0}]
set result [torch::ceil $integers]
# Result: [1.0, 2.0, -3.0, 0.0] (unchanged)

# Ceiling always rounds up
set decimals [torch::tensor_create {0.1 0.9 -0.1 -0.9}]
set result [torch::ceil $decimals]
# Result: [1.0, 1.0, 0.0, 0.0]
```

### Working with Different Tensor Shapes

```tcl
# 2D tensor
set matrix [torch::tensor_create {1.1 2.7 3.3 4.9} float32 cpu false {2 2}]
set result [torch::ceil -input $matrix]  # Shape preserved: {2 2}

# 3D tensor  
set tensor3d [torch::zeros {2 2 2}]
set result [torch::ceil -input $tensor3d]  # All elements remain 0.0
```

### Edge Cases

```tcl
# Very small positive values
set small_pos [torch::tensor_create {0.001 0.999}]
set result [torch::ceil $small_pos]
# Result: [1.0, 1.0]

# Very small negative values  
set small_neg [torch::tensor_create {-0.001 -0.999}]
set result [torch::ceil $small_neg]
# Result: [0.0, 0.0]

# Large values
set large [torch::tensor_create {1000.1 -1000.1}]
set result [torch::ceil $large]
# Result: [1001.0, -1000.0]
```

## Input Requirements

- **Data Type**: Floating-point tensors (float32, float64)
- **Shape**: Any tensor shape is supported
- **Device**: CPU and CUDA tensors supported
- **Values**: Any real number (no domain restrictions)

## Output

Returns a new tensor with the same shape as the input, containing the ceiling values as floating-point numbers.

## Error Handling

The function will raise an error if:
- Input tensor name is invalid or doesn't exist
- Required parameters are missing
- Unknown parameters are provided
- Wrong number of positional arguments provided

## Performance Considerations

- The operation is element-wise and parallelizable
- GPU acceleration available for CUDA tensors
- Memory usage: Creates a new tensor, doesn't modify input
- Computational complexity: O(n) where n is the number of elements
- Very fast operation with minimal computational overhead

## Common Use Cases

1. **Data Quantization**: Converting continuous values to discrete levels
2. **Array Indexing**: Computing buffer sizes or array dimensions
3. **Mathematical Modeling**: Implementing step functions and discrete mappings
4. **Computer Graphics**: Pixel coordinate calculations and rasterization
5. **Financial Calculations**: Rounding up currency amounts or quantities

## Related Functions

- `torch::floor` - Floor function (rounds down)
- `torch::round` - Round to nearest integer
- `torch::trunc` - Truncate toward zero
- `torch::frac` - Fractional part (x - floor(x))

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax (still supported)
set result [torch::ceil $input_tensor]

# New named parameter syntax
set result [torch::ceil -input $input_tensor]

# Alternative parameter name
set result [torch::ceil -tensor $input_tensor]

# All produce identical results
```

### Benefits of Named Parameters

1. **Clarity**: Parameter purpose is explicit
2. **Flexibility**: Parameters can be provided in any order  
3. **Extensibility**: Easy to add optional parameters in the future
4. **Consistency**: Matches modern TCL conventions

## Comparison with Other Rounding Functions

```tcl
set values [torch::tensor_create {2.3 2.7 -2.3 -2.7}]

set ceil_result [torch::ceil $values]    # [3.0, 3.0, -2.0, -2.0]
set floor_result [torch::floor $values]  # [2.0, 2.0, -3.0, -3.0]  
set round_result [torch::round $values]  # [2.0, 3.0, -2.0, -3.0]
set trunc_result [torch::trunc $values]  # [2.0, 2.0, -2.0, -2.0]
```

## Technical Notes

- Implements PyTorch's `torch.ceil()` function
- Preserves tensor properties (device, requires_grad, etc.)
- Supports automatic differentiation when `requires_grad=true`
- Thread-safe operation
- Follows IEEE 754 floating-point standards

## Version History

- Added dual syntax support in refactoring phase
- Original positional syntax maintained for backward compatibility
- Named parameter syntax added for modern TCL conventions
- Added alternative `-tensor` parameter alias for flexibility 