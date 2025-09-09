# torch::tensor_shape / torch::tensorShape

Get the shape (dimensions) of a tensor.

## Description

The `tensor_shape` command returns the shape of a tensor as a list of integers representing the size of each dimension. This is useful for understanding the structure of tensors and for debugging tensor operations.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_shape tensor_name
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_shape -tensor tensor_name
torch::tensor_shape -input tensor_name
```

### CamelCase Alias
```tcl
torch::tensorShape tensor_name
torch::tensorShape -tensor tensor_name
torch::tensorShape -input tensor_name
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor_name` | string | Yes | Name of the tensor to get shape for |
| `-tensor` | string | Yes* | Alternative parameter name for tensor |
| `-input` | string | Yes* | Alternative parameter name for tensor |

*Required when using named parameter syntax

## Return Value

Returns a TCL list containing the dimensions of the tensor. For example:
- Scalar tensor: `1`
- 1D tensor with 3 elements: `3`
- 2D tensor with 2 rows and 3 columns: `2 3`
- 3D tensor with shape (2, 3, 4): `2 3 4`
- Empty tensor: `0`

## Examples

### Basic Usage

```tcl
# Create a simple tensor
set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]

# Get shape using positional syntax
set shape [torch::tensor_shape $tensor]
puts "Shape: $shape"  # Output: Shape: 3

# Get shape using named syntax
set shape [torch::tensor_shape -tensor $tensor]
puts "Shape: $shape"  # Output: Shape: 3

# Get shape using camelCase alias
set shape [torch::tensorShape $tensor]
puts "Shape: $shape"  # Output: Shape: 3
```

### Multi-dimensional Tensors

```tcl
# Create a 2D tensor
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]

# Get shape
set shape [torch::tensor_shape $reshaped]
puts "Shape: $shape"  # Output: Shape: 2 2

# Create a 3D tensor
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32]
set reshaped [torch::tensor_reshape -input $tensor -shape {2 2 2}]

# Get shape
set shape [torch::tensor_shape $reshaped]
puts "Shape: $shape"  # Output: Shape: 2 2 2
```

### Different Data Types

```tcl
# Integer tensor
set int_tensor [torch::tensor_create -data {1 2 3 4} -dtype int32]
set shape [torch::tensor_shape $int_tensor]
puts "Integer tensor shape: $shape"  # Output: Integer tensor shape: 4

# Double precision tensor
set double_tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64]
set shape [torch::tensor_shape $double_tensor]
puts "Double tensor shape: $shape"  # Output: Double tensor shape: 3
```

### Edge Cases

```tcl
# Scalar tensor
set scalar [torch::tensor_create -data 5.0 -dtype float32]
set shape [torch::tensor_shape $scalar]
puts "Scalar shape: $shape"  # Output: Scalar shape: 1

# Empty tensor
set empty [torch::tensor_create -data {} -dtype float32]
set shape [torch::tensor_shape $empty]
puts "Empty tensor shape: $shape"  # Output: Empty tensor shape: 0

# Zero tensor
set zero [torch::tensor_create -data 0.0 -dtype float32]
set shape [torch::tensor_shape $zero]
puts "Zero tensor shape: $shape"  # Output: Zero tensor shape: 1
```

### After Tensor Operations

```tcl
# Create tensor and reshape
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32]
set reshaped [torch::tensor_reshape -input $tensor -shape {2 3}]

# Check shape after reshape
set shape [torch::tensor_shape $reshaped]
puts "After reshape: $shape"  # Output: After reshape: 2 3

# Permute dimensions
set permuted [torch::tensor_permute -input $reshaped -dims {1 0}]
set shape [torch::tensor_shape $permuted]
puts "After permute: $shape"  # Output: After permute: 3 2
```

## Error Handling

### Invalid Tensor Name
```tcl
catch {torch::tensor_shape invalid_tensor} result
puts $result  # Output: Invalid tensor name: invalid_tensor
```

### Missing Parameters
```tcl
# Missing tensor name
catch {torch::tensor_shape} result
puts $result  # Output: Required parameter missing: tensor

# Missing parameter value
catch {torch::tensor_shape -tensor} result
puts $result  # Output: Missing value for parameter
```

### Unknown Parameters
```tcl
set tensor [torch::tensor_create -data {1.0 2.0} -dtype float32]
catch {torch::tensor_shape -unknown $tensor} result
puts $result  # Output: Unknown parameter: -unknown
```

### Too Many Arguments
```tcl
set tensor [torch::tensor_create -data {1.0 2.0} -dtype float32]
catch {torch::tensor_shape $tensor extra_arg} result
puts $result  # Output: Usage: torch::tensor_shape tensor
```

## Migration Guide

### From Old Syntax to New Syntax

**Old (Positional Only):**
```tcl
set shape [torch::tensor_shape $tensor]
```

**New (Named Parameters):**
```tcl
set shape [torch::tensor_shape -tensor $tensor]
# or
set shape [torch::tensor_shape -input $tensor]
```

**New (CamelCase Alias):**
```tcl
set shape [torch::tensorShape $tensor]
set shape [torch::tensorShape -tensor $tensor]
```

### Backward Compatibility

The old positional syntax continues to work:
```tcl
# This still works
set shape [torch::tensor_shape $tensor]

# This also works
set shape [torch::tensor_shape -tensor $tensor]
```

## Notes

1. **Shape Representation**: The shape is returned as a TCL list where each element represents the size of that dimension.

2. **Scalar Tensors**: Scalar tensors (0-dimensional) return shape `1` rather than an empty list.

3. **Empty Tensors**: Empty tensors return shape `0`.

4. **Device Independence**: The shape is the same regardless of the device (CPU/CUDA) the tensor is on.

5. **Data Type Independence**: The shape is independent of the tensor's data type.

6. **Memory Efficiency**: This operation is very lightweight and doesn't copy any tensor data.

## Related Commands

- `torch::tensor_create` - Create tensors
- `torch::tensor_reshape` - Reshape tensors
- `torch::tensor_permute` - Permute tensor dimensions
- `torch::tensor_numel` - Get total number of elements
- `torch::tensor_dtype` - Get tensor data type
- `torch::tensor_device` - Get tensor device 