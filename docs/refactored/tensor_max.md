# torch::tensor_max

Finds the maximum values in a tensor, either across the entire tensor or along specified dimensions.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_max tensor ?dim?
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_max -input tensor ?-dim dim?
```

### CamelCase Alias
```tcl
torch::tensorMax tensor ?dim?
torch::tensorMax -input tensor ?-dim dim?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor` | string | **required** | Name of the input tensor |
| `dim` | integer | **none** | Dimension along which to find maximum values (optional) |

## Return Value

Returns a tensor handle (string) containing the maximum values.

## Behavior

- **Without dimension**: Returns a scalar tensor containing the maximum value of the entire tensor
- **With dimension**: Returns a tensor containing the maximum values along the specified dimension

## Examples

### Basic Usage

```tcl
# Create a test tensor
set t [torch::tensor_create {1 5 3 2 4 6} {2 3}]

# Find maximum value of entire tensor
set max_val [torch::tensor_max $t]
set result [torch::tensor_item $max_val]
puts "Maximum value: $result"  # Output: Maximum value: 6

# Find maximum values along dimension 0 (columns)
set max_cols [torch::tensor_max $t 0]
set shape [torch::tensor_shape $max_cols]
puts "Shape of max along dim 0: $shape"  # Output: Shape of max along dim 0: 3

# Find maximum values along dimension 1 (rows)
set max_rows [torch::tensor_max $t 1]
set shape [torch::tensor_shape $max_rows]
puts "Shape of max along dim 1: $shape"  # Output: Shape of max along dim 1: 2
```

### Named Parameter Syntax

```tcl
# Using named parameters
set t [torch::tensor_create {1 8 3 2} {4}]

# Maximum of entire tensor
set result [torch::tensor_max -input $t]
set max_val [torch::tensor_item $result]
puts "Max: $max_val"  # Output: Max: 8

# Maximum along dimension
set t2d [torch::tensor_create {1 3 2 4} {2 2}]
set result [torch::tensor_max -input $t2d -dim 0]
```

### CamelCase Alias

```tcl
# Using camelCase alias
set t [torch::tensor_create {1 7 3} {3}]
set result [torch::tensorMax $t]
set max_val [torch::tensor_item $result]
puts "Max: $max_val"  # Output: Max: 7

# With named parameters
set result [torch::tensorMax -input $t]
```

### Advanced Examples

```tcl
# Working with 3D tensors
set t3d [torch::tensor_create {1 2 3 4 5 6 7 8} {2 2 2}]

# Max along each dimension
set max_dim0 [torch::tensor_max $t3d 0]  # Max along first dimension
set max_dim1 [torch::tensor_max $t3d 1]  # Max along second dimension
set max_dim2 [torch::tensor_max $t3d 2]  # Max along third dimension

# Global maximum
set global_max [torch::tensor_max $t3d]
set max_val [torch::tensor_item $global_max]
puts "Global maximum: $max_val"  # Output: Global maximum: 8
```

## Error Handling

### Missing Required Parameters
```tcl
# Error: Missing input tensor
torch::tensor_max
# Error: Input tensor is required
```

### Invalid Parameters
```tcl
# Error: Unknown parameter
set t [torch::tensor_create {1 2} {2}]
torch::tensor_max -foo $t
# Error: Unknown parameter: -foo

# Error: Missing value for parameter
torch::tensor_max -input
# Error: Missing value for parameter

# Error: Invalid tensor name
torch::tensor_max invalid_tensor
# Error: Invalid tensor name

# Error: Invalid dimension
set t [torch::tensor_create {1 2} {2}]
torch::tensor_max $t 10
# Error: Dimension out of range (expected to be in range of [-1, 0], but got 10)

# Error: Too many positional arguments
torch::tensor_max $t 0 extra
# Error: Invalid number of arguments
```

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
set result [torch::tensor_max $tensor 0]
```

**New (Named Parameters):**
```tcl
set result [torch::tensor_max -input $tensor -dim 0]
```

### Using CamelCase Alias

**Old (snake_case):**
```tcl
set result [torch::tensor_max $tensor]
```

**New (camelCase):**
```tcl
set result [torch::tensorMax $tensor]
```

## Mathematical Properties

The `tensor_max` function implements the mathematical maximum operation:

- **Commutative**: `max(a, b) = max(b, a)`
- **Associative**: `max(a, max(b, c)) = max(max(a, b), c)`
- **Idempotent**: `max(a, a) = a`

### Dimension Reduction

When specifying a dimension:
- The specified dimension is reduced (removed from the output shape)
- Other dimensions remain unchanged
- For a tensor of shape `(2, 3, 4)`:
  - `max(dim=0)` produces shape `(3, 4)`
  - `max(dim=1)` produces shape `(2, 4)`
  - `max(dim=2)` produces shape `(2, 3)`

## Performance Considerations

- **Memory**: Output tensor size depends on the operation:
  - Global max: scalar tensor (minimal memory)
  - Dimension max: reduced tensor size
- **Computation**: Linear time complexity O(n) where n is the number of elements
- **GPU**: Automatically uses GPU if input tensor is on GPU

## Related Commands

- `torch::tensor_min` - Find minimum values
- `torch::tensor_sum` - Sum tensor elements
- `torch::tensor_mean` - Calculate mean values
- `torch::tensor_argmax` - Find indices of maximum values

## Test Coverage

The refactored command includes comprehensive tests covering:

1. **Basic Functionality**: Both syntaxes work correctly
2. **Parameter Validation**: Error handling for invalid inputs
3. **Mathematical Correctness**: Results match expected maximum values
4. **Data Type Support**: Works with different tensor data types
5. **Edge Cases**: Single elements, all same values, negative values
6. **Syntax Consistency**: Both syntaxes produce same results
7. **Dimension Handling**: Correct behavior with different dimensions
8. **CamelCase Alias**: Alternative naming works correctly

## Backward Compatibility

✅ **100% Backward Compatible**: All existing code using positional parameters will continue to work without modification.

## Notes

- The function returns the maximum values, not the indices of maximum values
- For finding indices of maximum values, use `torch::tensor_argmax`
- When working with floating-point tensors, NaN values are handled according to IEEE 754 standards
- The function automatically handles tensors of any dimension (0D, 1D, 2D, 3D, etc.)
- Empty tensors will raise an error as there are no elements to find the maximum of 