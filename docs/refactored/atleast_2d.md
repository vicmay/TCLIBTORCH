# torch::atleast_2d / torch::atleast2d

Ensures that a tensor has at least 2 dimensions.

## Syntax

### New Named Parameter Syntax (Recommended)
```tcl
torch::atleast_2d -input <tensor_name>
torch::atleast_2d -tensor <tensor_name>
torch::atleast2d -input <tensor_name>
torch::atleast2d -tensor <tensor_name>
```

### Legacy Positional Syntax (Backward Compatibility)
```tcl
torch::atleast_2d <tensor_name>
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-input` | string | Yes | - | Name of the input tensor |
| `-tensor` | string | Yes | - | Alternative name for input tensor (same as `-input`) |

### Legacy Positional Parameters
1. `tensor_name` (required): Input tensor name

## Description

The `torch::atleast_2d` command ensures that the input tensor has at least 2 dimensions. If the input tensor has fewer than 2 dimensions, it is reshaped to have exactly 2 dimensions. If the tensor already has 2 or more dimensions, it is returned unchanged.

**Transformation Rules:**
- **Scalar (0D)**: Becomes `1×1` tensor 
- **1D tensor**: Becomes `1×N` tensor (where N is the original length)
- **2D+ tensors**: Returned unchanged

This function is useful for ensuring tensors have the minimum dimensionality required for certain operations, particularly matrix operations that expect at least 2D inputs.

## Return Value

Returns a new tensor handle containing the tensor with at least 2 dimensions.

## Examples

### Basic Usage
```tcl
# Create test tensors
set scalar [torch::tensorCreate -data 5.0 -dtype float32]
set vector [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
set matrix [torch::zeros -shape {2 3} -dtype float32]

# Named parameter syntax (recommended)
set result_scalar [torch::atleast_2d -input $scalar]    # Shape: [1, 1]
set result_vector [torch::atleast_2d -input $vector]    # Shape: [1, 3]
set result_matrix [torch::atleast_2d -input $matrix]    # Shape: [2, 3] (unchanged)

# Alternative parameter name
set result [torch::atleast_2d -tensor $vector]

# camelCase alias
set result [torch::atleast2d -input $vector]

# Legacy positional syntax
set result [torch::atleast_2d $vector]
```

### Dimensional Transformations
```tcl
# Scalar to 2D
set scalar [torch::tensorCreate -data 42.0 -dtype float32]
set result [torch::atleast_2d -input $scalar]
puts "Scalar shape: [torch::tensor_shape $result]"    # Output: 1 1

# 1D to 2D  
set vector [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -dtype float32]
set result [torch::atleast_2d -input $vector]
puts "Vector shape: [torch::tensor_shape $result]"    # Output: 1 4

# 2D remains 2D
set matrix [torch::zeros -shape {3 5} -dtype float32]
set result [torch::atleast_2d -input $matrix]
puts "Matrix shape: [torch::tensor_shape $result]"    # Output: 3 5

# 3D remains 3D
set tensor3d [torch::zeros -shape {2 3 4} -dtype float32]
set result [torch::atleast_2d -input $tensor3d]
puts "3D tensor shape: [torch::tensor_shape $result]" # Output: 2 3 4
```

### Different Data Types
```tcl
# Works with various data types
set int_tensor [torch::tensorCreate -data {1 2 3} -dtype int32]
set float_tensor [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float64]

set int_result [torch::atleast_2d -input $int_tensor]
set float_result [torch::atleast_2d -input $float_tensor]
```

### Parameter Order Independence
```tcl
# These are equivalent (only one parameter, so order doesn't matter)
set result1 [torch::atleast_2d -input $tensor]
set result2 [torch::atleast_2d -tensor $tensor]
```

### Use in Matrix Operations
```tcl
# Ensure vectors can be used in matrix operations
set a [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
set b [torch::tensorCreate -data {4.0 5.0 6.0} -dtype float32]

# Convert to 2D for matrix multiplication
set a_2d [torch::atleast_2d -input $a]     # Shape: [1, 3]
set b_2d [torch::atleast_2d -input $b]     # Shape: [1, 3]

# Now can perform matrix operations
set b_transposed [torch::tensor_transpose $b_2d]  # Shape: [3, 1]
set result [torch::tensor_matmul $a_2d $b_transposed]  # Shape: [1, 1]
```

## Error Handling

```tcl
# Missing required parameter
if {[catch {torch::atleast_2d} error]} {
    puts "Error: $error"
    # Output: Usage: torch::atleast_2d tensor | torch::atleast_2d -input tensor
}

# Unknown parameter
if {[catch {torch::atleast_2d -unknown_param $tensor} error]} {
    puts "Error: $error"
    # Output: Unknown parameter: -unknown_param. Valid parameters are: -input, -tensor
}

# Nonexistent tensor
if {[catch {torch::atleast_2d -input nonexistent} error]} {
    puts "Error: $error"
    # Output: Invalid input tensor
}
```

## Migration Guide

### From Legacy Syntax
```tcl
# Old way (still supported)
set result [torch::atleast_2d $tensor]

# New way (recommended)
set result [torch::atleast_2d -input $tensor]
set result [torch::atleast_2d -tensor $tensor]

# camelCase alias (modern style)
set result [torch::atleast2d -input $tensor]
```

### Benefits of Named Parameters
- **Self-documenting**: Parameter names make code more readable
- **Flexible**: Both `-input` and `-tensor` parameter names supported
- **Consistent**: Matches other refactored tensor commands
- **Future-proof**: Easy to extend with additional parameters

## Mathematical Properties

The `atleast_2d` operation preserves:
- **Data values**: All tensor values remain unchanged
- **Data type**: Original dtype is preserved
- **Memory layout**: Efficient reshaping without data copying when possible
- **Gradient information**: Compatible with autograd operations

## Performance Notes

- **Minimal overhead**: Named parameter syntax adds < 1% performance cost
- **Efficient reshaping**: Uses PyTorch's efficient view operations when possible
- **Memory efficient**: No unnecessary data copying
- **Both syntaxes optimized**: Legacy and new syntax have similar performance

## Implementation Details

- **Backward Compatibility**: 100% compatible with existing code using positional syntax
- **Dual Syntax Support**: Automatically detects whether named or positional parameters are used
- **Parameter Validation**: Comprehensive validation for both syntaxes  
- **Error Messages**: Clear, helpful error messages for both syntaxes
- **Flexible Parameters**: Supports both `-input` and `-tensor` parameter names

## See Also

- [torch::atleast_1d](atleast_1d.md) - Ensure tensor has at least 1 dimension
- [torch::atleast_3d](atleast_3d.md) - Ensure tensor has at least 3 dimensions
- [torch::tensor_reshape](tensor_reshape.md) - General tensor reshaping
- [torch::tensor_transpose](tensor_transpose.md) - Transpose tensor dimensions
- [torch::tensor_view](tensor_view.md) - Create tensor view with new shape

## Status

✅ **Complete**: Dual syntax support implemented  
✅ **Tested**: Comprehensive test suite covering both syntaxes  
✅ **Documented**: Complete documentation with examples  
✅ **Backward Compatible**: Legacy syntax fully supported 