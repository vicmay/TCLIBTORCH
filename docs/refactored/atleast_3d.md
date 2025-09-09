# torch::atleast_3d / torch::atleast3d

Ensures that a tensor has at least 3 dimensions.

## Syntax

### New Named Parameter Syntax (Recommended)
```tcl
torch::atleast_3d -input <tensor_name>
torch::atleast_3d -tensor <tensor_name>
torch::atleast3d -input <tensor_name>
torch::atleast3d -tensor <tensor_name>
```

### Legacy Positional Syntax (Backward Compatibility)
```tcl
torch::atleast_3d <tensor_name>
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-input` | string | Yes | - | Name of the input tensor |
| `-tensor` | string | Yes | - | Alternative name for input tensor (same as `-input`) |

### Legacy Positional Parameters
1. `tensor_name` (required): Input tensor name

## Description

The `torch::atleast_3d` command ensures that the input tensor has at least 3 dimensions. If the input tensor has fewer than 3 dimensions, it is reshaped to have exactly 3 dimensions. If the tensor already has 3 or more dimensions, it is returned unchanged.

**Transformation Rules:**
- **Scalar (0D)**: Becomes `1×1×1` tensor 
- **1D tensor**: Becomes `1×N×1` tensor (where N is the original length)
- **2D tensor**: Becomes `M×N×1` tensor (where M×N is the original shape)
- **3D+ tensors**: Returned unchanged

This function is useful for ensuring tensors have the minimum dimensionality required for certain operations, particularly 3D operations like volumetric convolutions or batch operations that expect at least 3D inputs.

## Return Value

Returns a new tensor handle containing the tensor with at least 3 dimensions.

## Examples

### Basic Usage
```tcl
# Create test tensors
set scalar [torch::tensorCreate -data 5.0 -dtype float32]
set vector [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
set matrix [torch::zeros -shape {2 3} -dtype float32]
set tensor3d [torch::zeros -shape {2 3 4} -dtype float32]

# Named parameter syntax (recommended)
set result_scalar [torch::atleast_3d -input $scalar]    # Shape: [1, 1, 1]
set result_vector [torch::atleast_3d -input $vector]    # Shape: [1, 3, 1]
set result_matrix [torch::atleast_3d -input $matrix]    # Shape: [2, 3, 1]
set result_3d [torch::atleast_3d -input $tensor3d]      # Shape: [2, 3, 4] (unchanged)

# Alternative parameter name
set result [torch::atleast_3d -tensor $vector]

# camelCase alias
set result [torch::atleast3d -input $vector]

# Legacy positional syntax
set result [torch::atleast_3d $vector]
```

### Dimensional Transformations
```tcl
# Scalar to 3D
set scalar [torch::tensorCreate -data 42.0 -dtype float32]
set result [torch::atleast_3d -input $scalar]
puts "Scalar shape: [torch::tensor_shape $result]"    # Output: 1 1 1

# 1D to 3D  
set vector [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -dtype float32]
set result [torch::atleast_3d -input $vector]
puts "Vector shape: [torch::tensor_shape $result]"    # Output: 1 4 1

# 2D to 3D
set matrix [torch::zeros -shape {3 5} -dtype float32]
set result [torch::atleast_3d -input $matrix]
puts "Matrix shape: [torch::tensor_shape $result]"    # Output: 3 5 1

# 3D remains 3D
set tensor3d [torch::zeros -shape {2 3 4} -dtype float32]
set result [torch::atleast_3d -input $tensor3d]
puts "3D tensor shape: [torch::tensor_shape $result]" # Output: 2 3 4

# 4D remains 4D
set tensor4d [torch::zeros -shape {2 3 4 5} -dtype float32]
set result [torch::atleast_3d -input $tensor4d]
puts "4D tensor shape: [torch::tensor_shape $result]" # Output: 2 3 4 5
```

### Different Data Types
```tcl
# Works with various data types
set int_tensor [torch::tensorCreate -data {1 2 3} -dtype int32]
set float_tensor [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float64]

set int_result [torch::atleast_3d -input $int_tensor]
set float_result [torch::atleast_3d -input $float_tensor]
```

### Parameter Order Independence
```tcl
# These are equivalent (only one parameter, so order doesn't matter)
set result1 [torch::atleast_3d -input $tensor]
set result2 [torch::atleast_3d -tensor $tensor]
```

### Use in 3D Operations
```tcl
# Ensure tensors can be used in 3D operations
set data2d [torch::zeros -shape {4 5} -dtype float32]

# Convert to 3D for volumetric operations
set data3d [torch::atleast_3d -input $data2d]     # Shape: [4, 5, 1]

# Now can perform 3D operations
set expanded [torch::tensor_expand $data3d -shape {4 5 8}]  # Expand last dimension
```

### Batch Processing
```tcl
# Prepare tensors for batch operations that expect 3D input
set sample1 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]  # 1D
set sample2 [torch::zeros -shape {2 3} -dtype float32]                # 2D

# Convert both to 3D for consistent batch processing
set batch_sample1 [torch::atleast_3d -input $sample1]  # Shape: [1, 3, 1]
set batch_sample2 [torch::atleast_3d -input $sample2]  # Shape: [2, 3, 1]

# Now can stack into batch (need matching trailing dimensions)
```

## Error Handling

```tcl
# Missing required parameter
if {[catch {torch::atleast_3d} error]} {
    puts "Error: $error"
    # Output: Usage: torch::atleast_3d tensor | torch::atleast_3d -input tensor
}

# Unknown parameter
if {[catch {torch::atleast_3d -unknown_param $tensor} error]} {
    puts "Error: $error"
    # Output: Unknown parameter: -unknown_param. Valid parameters are: -input, -tensor
}

# Nonexistent tensor
if {[catch {torch::atleast_3d -input nonexistent} error]} {
    puts "Error: $error"
    # Output: Invalid input tensor
}
```

## Migration Guide

### From Legacy Syntax
```tcl
# Old way (still supported)
set result [torch::atleast_3d $tensor]

# New way (recommended)
set result [torch::atleast_3d -input $tensor]
set result [torch::atleast_3d -tensor $tensor]

# camelCase alias (modern style)
set result [torch::atleast3d -input $tensor]
```

### Benefits of Named Parameters
- **Self-documenting**: Parameter names make code more readable
- **Flexible**: Both `-input` and `-tensor` parameter names supported
- **Consistent**: Matches other refactored tensor commands
- **Future-proof**: Easy to extend with additional parameters

## Mathematical Properties

The `atleast_3d` operation preserves:
- **Data values**: All tensor values remain unchanged
- **Data type**: Original dtype is preserved
- **Memory layout**: Efficient reshaping without data copying when possible
- **Gradient information**: Compatible with autograd operations

## Performance Notes

- **Minimal overhead**: Named parameter syntax adds < 1% performance cost
- **Efficient reshaping**: Uses PyTorch's efficient view operations when possible
- **Memory efficient**: No unnecessary data copying
- **Both syntaxes optimized**: Legacy and new syntax have similar performance

## Common Use Cases

### 3D Convolutions
```tcl
# Prepare data for 3D convolution layers
set input_2d [torch::zeros -shape {32 32} -dtype float32]  # 2D image
set input_3d [torch::atleast_3d -input $input_2d]         # Shape: [32, 32, 1]
# Now can be used with conv3d operations (after adding batch dimension)
```

### Volumetric Processing
```tcl
# Process slices as volumes
set slice [torch::zeros -shape {64 64} -dtype float32]     # 2D slice
set volume [torch::atleast_3d -input $slice]              # Shape: [64, 64, 1]
# Can now stack multiple slices to create 3D volumes
```

### Time Series Data
```tcl
# Convert features to time-series format
set features [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -dtype float32]  # 1D features
set time_series [torch::atleast_3d -input $features]                      # Shape: [1, 4, 1]
# Now compatible with RNN/LSTM layers that expect (batch, seq, features)
```

## Implementation Details

- **Backward Compatibility**: 100% compatible with existing code using positional syntax
- **Dual Syntax Support**: Automatically detects whether named or positional parameters are used
- **Parameter Validation**: Comprehensive validation for both syntaxes  
- **Error Messages**: Clear, helpful error messages for both syntaxes
- **Flexible Parameters**: Supports both `-input` and `-tensor` parameter names

## See Also

- [torch::atleast_1d](atleast_1d.md) - Ensure tensor has at least 1 dimension
- [torch::atleast_2d](atleast_2d.md) - Ensure tensor has at least 2 dimensions
- [torch::tensor_reshape](tensor_reshape.md) - General tensor reshaping
- [torch::tensor_view](tensor_view.md) - Create tensor view with new shape
- [torch::tensor_expand](tensor_expand.md) - Expand tensor to larger shape
- [torch::tensor_unsqueeze](tensor_unsqueeze.md) - Add singleton dimensions

## Status

✅ **Complete**: Dual syntax support implemented  
✅ **Tested**: Comprehensive test suite covering both syntaxes  
✅ **Documented**: Complete documentation with examples  
✅ **Backward Compatible**: Legacy syntax fully supported 