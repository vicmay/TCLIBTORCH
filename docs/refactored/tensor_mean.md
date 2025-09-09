# torch::tensor_mean

Computes the mean of tensor elements, optionally along specified dimensions.

## Description

The `torch::tensor_mean` command calculates the arithmetic mean of tensor elements. When no dimension is specified, it computes the mean of all elements in the tensor. When a dimension is specified, it computes the mean along that dimension, reducing the tensor's dimensionality.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_mean tensor ?dim?
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_mean -input tensor ?-dim dimension?
```

### CamelCase Alias
```tcl
torch::tensorMean -input tensor ?-dim dimension?
```

## Parameters

| Parameter | Type   | Required | Description                        |
|-----------|--------|----------|------------------------------------|
| input     | string | Yes      | Name of the input tensor           |
| dim       | int    | No       | Dimension along which to compute mean (default: all elements) |

## Return Value

Returns a string containing the handle name of the resulting mean tensor.

## Examples

### Basic Usage
```tcl
# Create a 1D tensor
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu]

# Using positional syntax - mean of all elements
set result1 [torch::tensor_mean $a]

# Using named parameter syntax
set result2 [torch::tensor_mean -input $a]

# Using camelCase alias
set result3 [torch::tensorMean -input $a]
```

### Computing Mean Along Dimensions
```tcl
# Create a 2D tensor
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
set a2d [torch::tensor_reshape $a {2 3}]

# Mean along dimension 0 (rows)
set mean_dim0 [torch::tensor_mean $a2d 0]

# Mean along dimension 1 (columns) using named syntax
set mean_dim1 [torch::tensor_mean -input $a2d -dim 1]

# Mean along dimension 1 using camelCase
set mean_dim1_camel [torch::tensorMean -input $a2d -dim 1]
```

### 3D Tensor Example
```tcl
# Create a 3D tensor
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
set a3d [torch::tensor_reshape $a {2 2 2}]

# Mean along dimension 0
set mean_3d_dim0 [torch::tensor_mean $a3d 0]

# Mean along dimension 1
set mean_3d_dim1 [torch::tensor_mean -input $a3d -dim 1]

# Mean along dimension 2
set mean_3d_dim2 [torch::tensorMean -input $a3d -dim 2]
```

## Error Handling

### Invalid Tensor Name
```tcl
catch {torch::tensor_mean invalid_tensor} result
# Returns: "Invalid tensor name"
```

### Missing Input Parameter
```tcl
catch {torch::tensor_mean -dim 0} result
# Returns: "Input tensor is required"
```

### Invalid Dimension Value
```tcl
set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
catch {torch::tensor_mean -input $a -dim invalid} result
# Returns: "Invalid dimension value: invalid"
```

### Too Many Arguments
```tcl
set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
catch {torch::tensor_mean $a 0 extra} result
# Returns: "Invalid number of arguments"
```

### Unknown Parameter
```tcl
set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
catch {torch::tensor_mean -input $a -unknown_param value} result
# Returns: "Unknown parameter: -unknown_param"
```

### Dimension Out of Bounds
```tcl
set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
catch {torch::tensor_mean $a 5} result
# Returns: PyTorch error about dimension out of bounds
```

## Mathematical Behavior

### 1D Tensor
- Computes the arithmetic mean of all elements
- Returns a scalar tensor

### 2D Tensor
- Without dimension: computes mean of all elements
- With dimension 0: computes mean along rows (reduces to 1D)
- With dimension 1: computes mean along columns (reduces to 1D)

### 3D Tensor
- Without dimension: computes mean of all elements
- With dimension: reduces the specified dimension by computing mean along it

## Data Type Support

The `tensor_mean` command supports the following data types:
- **float32**: Single precision floating point
- **float64**: Double precision floating point
- **complex64**: Single precision complex
- **complex128**: Double precision complex

**Note**: Integer tensors are not supported by PyTorch's mean operation.

## Edge Cases

### Single Element Tensor
```tcl
set a [torch::tensor_create -data {5.0} -dtype float32 -device cpu]
set result [torch::tensor_mean $a]
# Returns: tensor with value 5.0
```

### Zero Tensor
```tcl
set a [torch::tensor_create -data {0.0 0.0 0.0} -dtype float32 -device cpu]
set result [torch::tensor_mean $a]
# Returns: tensor with value 0.0
```

### Negative Values
```tcl
set a [torch::tensor_create -data {-1.0 -2.0 -3.0} -dtype float32 -device cpu]
set result [torch::tensor_mean $a]
# Returns: tensor with value -2.0
```

## Migration Guide

### From Old Syntax to New Syntax

**Old (Positional Only)**:
```tcl
# Old way - still supported
set result [torch::tensor_mean $tensor]
set result [torch::tensor_mean $tensor 0]
```

**New (Named Parameters)**:
```tcl
# New way - more explicit
set result [torch::tensor_mean -input $tensor]
set result [torch::tensor_mean -input $tensor -dim 0]
```

**CamelCase Alternative**:
```tcl
# Modern camelCase syntax
set result [torch::tensorMean -input $tensor]
set result [torch::tensorMean -input $tensor -dim 0]
```

### Benefits of New Syntax
- **Explicit parameter names**: No confusion about parameter order
- **Optional parameters**: Only specify what you need
- **Better error messages**: Clear indication of missing parameters
- **Future extensibility**: Easy to add new parameters

## Related Commands

- [torch::tensor_sum](tensor_sum.md) - Sum of tensor elements
- [torch::tensor_max](tensor_max.md) - Maximum value in tensor
- [torch::tensor_min](tensor_min.md) - Minimum value in tensor
- [torch::tensor_std](tensor_std.md) - Standard deviation of tensor elements
- [torch::tensor_var](tensor_var.md) - Variance of tensor elements

## Notes

- The mean operation requires floating point or complex data types
- Integer tensors will cause an error
- When no dimension is specified, the result is a scalar tensor
- When a dimension is specified, that dimension is reduced (removed from shape)
- The operation preserves the data type of the input tensor
- For complex tensors, the mean is computed separately for real and imaginary parts

---

**Migration Note**: 
- **Old:** `torch::tensor_mean $t ?dim?` (still supported)
- **New:** `torch::tensor_mean -input $t -dim dim` or `torch::tensorMean $t dim`
- Both syntaxes are fully supported and produce identical results. 