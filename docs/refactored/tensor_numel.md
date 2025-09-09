# torch::tensor_numel / torch::tensorNumel

Get the total number of elements in a tensor.

## Description

The `tensor_numel` command returns the total number of elements in a tensor, regardless of its shape or dimensions. This is useful for understanding the size of tensors and for memory management.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_numel tensor_name
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_numel -tensor tensor_name
torch::tensor_numel -input tensor_name
```

### CamelCase Alias
```tcl
torch::tensorNumel tensor_name
torch::tensorNumel -tensor tensor_name
torch::tensorNumel -input tensor_name
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor_name` | string | Yes | Name of the tensor to get element count for |
| `-tensor` | string | Yes* | Alternative parameter name for tensor |
| `-input` | string | Yes* | Alternative parameter name for tensor |

*Required when using named parameter syntax

## Return Value

Returns the total number of elements as a string. For example:
- Scalar tensor: `1`
- 1D tensor with 3 elements: `3`
- 2D tensor with 2×3=6 elements: `6`
- 3D tensor with 2×2×2=8 elements: `8`
- Empty tensor: `0`

## Examples

### Basic Usage

```tcl
# Create a simple tensor
set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]

# Get number of elements using positional syntax
set numel [torch::tensor_numel $tensor]
puts "Number of elements: $numel"  # Output: Number of elements: 3

# Get number of elements using named syntax
set numel [torch::tensor_numel -tensor $tensor]
puts "Number of elements: $numel"  # Output: Number of elements: 3

# Get number of elements using camelCase alias
set numel [torch::tensorNumel $tensor]
puts "Number of elements: $numel"  # Output: Number of elements: 3
```

### Multi-dimensional Tensors

```tcl
# Create a 2D tensor
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]

# Get number of elements
set numel [torch::tensor_numel $reshaped]
puts "Number of elements: $numel"  # Output: Number of elements: 4

# Create a 3D tensor
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32]
set reshaped [torch::tensor_reshape -input $tensor -shape {2 2 2}]

# Get number of elements
set numel [torch::tensor_numel $reshaped]
puts "Number of elements: $numel"  # Output: Number of elements: 8
```

### Different Data Types

```tcl
# Integer tensor
set int_tensor [torch::tensor_create -data {1 2 3 4} -dtype int32]
set numel [torch::tensor_numel $int_tensor]
puts "Integer tensor elements: $numel"  # Output: Integer tensor elements: 4

# Double precision tensor
set double_tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64]
set numel [torch::tensor_numel $double_tensor]
puts "Double tensor elements: $numel"  # Output: Double tensor elements: 3
```

### Edge Cases

```tcl
# Scalar tensor
set scalar [torch::tensor_create -data 5.0 -dtype float32]
set numel [torch::tensor_numel $scalar]
puts "Scalar elements: $numel"  # Output: Scalar elements: 1

# Empty tensor
set empty [torch::tensor_create -data {} -dtype float32]
set numel [torch::tensor_numel $empty]
puts "Empty tensor elements: $numel"  # Output: Empty tensor elements: 0

# Zero tensor
set zero [torch::tensor_create -data 0.0 -dtype float32]
set numel [torch::tensor_numel $zero]
puts "Zero tensor elements: $numel"  # Output: Zero tensor elements: 1
```

### After Tensor Operations

```tcl
# Create tensor and reshape
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32]
set reshaped [torch::tensor_reshape -input $tensor -shape {2 3}]

# Check number of elements after reshape
set numel [torch::tensor_numel $reshaped]
puts "After reshape: $numel"  # Output: After reshape: 6

# Permute dimensions (numel stays the same)
set permuted [torch::tensor_permute -input $reshaped -dims {1 0}]
set numel [torch::tensor_numel $permuted]
puts "After permute: $numel"  # Output: After permute: 6
```

### Different Devices

```tcl
# CPU tensor
set cpu_tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set numel [torch::tensor_numel $cpu_tensor]
puts "CPU tensor elements: $numel"  # Output: CPU tensor elements: 3

# CUDA tensor (if available)
if {[torch::cuda_is_available]} {
    set cuda_tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cuda]
    set numel [torch::tensor_numel $cuda_tensor]
    puts "CUDA tensor elements: $numel"  # Output: CUDA tensor elements: 3
}
```

### Relationship with tensor_shape

```tcl
# Create a 2D tensor
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32]
set reshaped [torch::tensor_reshape -input $tensor -shape {2 3}]

# Get shape and numel
set shape [torch::tensor_shape $reshaped]
set numel [torch::tensor_numel $reshaped]

puts "Shape: $shape"  # Output: Shape: 2 3
puts "Number of elements: $numel"  # Output: Number of elements: 6

# Verify relationship: numel = product of shape dimensions
set expected [expr {[lindex $shape 0] * [lindex $shape 1]}]
puts "Expected: $expected"  # Output: Expected: 6
puts "Match: [expr {$numel == $expected}]"  # Output: Match: 1
```

## Error Handling

### Invalid Tensor Name
```tcl
catch {torch::tensor_numel invalid_tensor} result
puts $result  # Output: Invalid tensor name: invalid_tensor
```

### Missing Parameters
```tcl
# Missing tensor name
catch {torch::tensor_numel} result
puts $result  # Output: Required parameter missing: tensor

# Missing parameter value
catch {torch::tensor_numel -tensor} result
puts $result  # Output: Missing value for parameter
```

### Unknown Parameters
```tcl
set tensor [torch::tensor_create -data {1.0 2.0} -dtype float32]
catch {torch::tensor_numel -unknown $tensor} result
puts $result  # Output: Unknown parameter: -unknown
```

### Too Many Arguments
```tcl
set tensor [torch::tensor_create -data {1.0 2.0} -dtype float32]
catch {torch::tensor_numel $tensor extra_arg} result
puts $result  # Output: Usage: torch::tensor_numel tensor
```

## Migration Guide

### From Old Syntax to New Syntax

**Old (Positional Only):**
```tcl
set numel [torch::tensor_numel $tensor]
```

**New (Named Parameters):**
```tcl
set numel [torch::tensor_numel -tensor $tensor]
# or
set numel [torch::tensor_numel -input $tensor]
```

**New (CamelCase Alias):**
```tcl
set numel [torch::tensorNumel $tensor]
set numel [torch::tensorNumel -tensor $tensor]
```

### Backward Compatibility

The old positional syntax continues to work:
```tcl
# This still works
set numel [torch::tensor_numel $tensor]

# This also works
set numel [torch::tensor_numel -tensor $tensor]
```

## Notes

1. **Element Count**: Returns the total number of elements, not the number of dimensions.

2. **Shape Independence**: The number of elements is independent of the tensor's shape. A tensor with shape `{2 3}` has the same number of elements as a tensor with shape `{6}`.

3. **Memory Efficiency**: This operation is very lightweight and doesn't copy any tensor data.

4. **Device Independence**: Works with tensors on any device (CPU/CUDA).

5. **Data Type Independence**: The number of elements is independent of the tensor's data type.

6. **Relationship with Shape**: For any tensor, `numel = product of all shape dimensions`.

7. **Common Use Cases**: 
   - Memory management and allocation
   - Understanding tensor size
   - Validation of tensor operations
   - Performance optimization

## Related Commands

- `torch::tensor_create` - Create tensors
- `torch::tensor_shape` - Get tensor shape
- `torch::tensor_reshape` - Reshape tensors
- `torch::tensor_permute` - Permute tensor dimensions
- `torch::tensor_item` - Extract scalar value from single-element tensor
- `torch::tensor_dtype` - Get tensor data type
- `torch::tensor_device` - Get tensor device 