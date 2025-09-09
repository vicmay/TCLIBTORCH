# torch::tensor_is_contiguous

Checks whether a tensor's memory layout is contiguous.

## Description

The `torch::tensor_is_contiguous` command checks if a tensor's memory layout is contiguous. A tensor is considered contiguous when its elements are stored in memory in the same order as they would be traversed in a row-major fashion.

This is useful for:
- Determining if a tensor needs to be made contiguous before certain operations
- Optimizing memory access patterns
- Understanding tensor memory layout after operations like transpose, permute, etc.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_is_contiguous tensor
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_is_contiguous -tensor tensor
torch::tensor_is_contiguous -input tensor
```

### CamelCase Alias
```tcl
torch::tensorIsContiguous tensor
torch::tensorIsContiguous -tensor tensor
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tensor` / `-tensor` / `-input` | string | Yes | - | Name of the input tensor to check |

## Return Value

Returns a string:
- `"1"` if the tensor is contiguous
- `"0"` if the tensor is not contiguous

## Examples

### Basic Usage

**Positional syntax:**
```tcl
# Create a contiguous tensor
set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]

# Check if it's contiguous
set result [torch::tensor_is_contiguous $tensor]
puts $result  ;# Output: 1
```

**Named parameter syntax:**
```tcl
# Same operation with named parameters
set result [torch::tensor_is_contiguous -tensor $tensor]
puts $result  ;# Output: 1
```

**CamelCase alias:**
```tcl
# Using camelCase alias
set result [torch::tensorIsContiguous $tensor]
puts $result  ;# Output: 1
```

### Checking Non-Contiguous Tensors

```tcl
# Create a 2D tensor
set tensor_2d [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]

# Check if original is contiguous
puts [torch::tensor_is_contiguous $tensor_2d]  ;# Output: 1

# Transpose the tensor (makes it non-contiguous)
set transposed [torch::tensor_permute $tensor_2d {1 0}]
puts [torch::tensor_is_contiguous $transposed]  ;# Output: 0
```

### Working with Different Tensor Shapes

```tcl
# 1D tensor
set tensor_1d [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
puts [torch::tensor_is_contiguous $tensor_1d]  ;# Output: 1

# 2D tensor
set tensor_2d [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
puts [torch::tensor_is_contiguous $tensor_2d]  ;# Output: 1

# 3D tensor (created by stacking)
set tensor_3d [torch::tensor_stack [list $tensor_2d $tensor_2d] 0]
puts [torch::tensor_is_contiguous $tensor_3d]  ;# Output: 1
```

### Operations That Affect Contiguity

```tcl
# Create a base tensor
set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
puts "Original: [torch::tensor_is_contiguous $tensor]"  ;# Output: 1

# Reshape (usually maintains contiguity)
set reshaped [torch::tensor_reshape $tensor {2 2}]
puts "After reshape: [torch::tensor_is_contiguous $reshaped]"  ;# Output: 1

# Permute (can make non-contiguous)
set permuted [torch::tensor_permute $reshaped {1 0}]
puts "After permute: [torch::tensor_is_contiguous $permuted]"  ;# Output: 0
```

## Migration Guide

### From Old Positional Syntax

**Old code:**
```tcl
set result [torch::tensor_is_contiguous my_tensor]
```

**New positional syntax (unchanged):**
```tcl
set result [torch::tensor_is_contiguous my_tensor]
```

**New named parameter syntax:**
```tcl
set result [torch::tensor_is_contiguous -tensor my_tensor]
set result [torch::tensor_is_contiguous -input my_tensor]
```

**New camelCase alias:**
```tcl
set result [torch::tensorIsContiguous my_tensor]
set result [torch::tensorIsContiguous -tensor my_tensor]
```

## Error Handling

The command provides clear error messages for various error conditions:

### Invalid Tensor Name
```tcl
catch {torch::tensor_is_contiguous nonexistent_tensor} result
puts $result  ;# Output: Invalid tensor name
```

### Missing Required Parameter
```tcl
catch {torch::tensor_is_contiguous} result
puts $result  ;# Output: Required tensor parameter missing
```

### Unknown Named Parameter
```tcl
catch {torch::tensor_is_contiguous -unknown value} result
puts $result  ;# Output: Unknown parameter: -unknown
```

### Missing Parameter Value
```tcl
catch {torch::tensor_is_contiguous -tensor} result
puts $result  ;# Output: Missing value for parameter
```

## Memory Layout and Contiguity

### What Makes a Tensor Contiguous

A tensor is contiguous when:
- Elements are stored in memory in row-major order
- No gaps exist between elements in memory
- The stride between adjacent elements is consistent

### Operations That Preserve Contiguity
- `torch::tensor_create` - New tensors are always contiguous
- `torch::tensor_reshape` - Usually preserves contiguity
- `torch::tensor_contiguous` - Explicitly makes tensor contiguous

### Operations That May Break Contiguity
- `torch::tensor_permute` - Reorders dimensions
- `torch::tensor_transpose` - Swaps dimensions
- `torch::tensor_slice` - May create non-contiguous views
- `torch::tensor_select` - May create non-contiguous views

### Performance Implications

**Contiguous tensors:**
- Faster memory access
- Better cache performance
- Optimized for most operations

**Non-contiguous tensors:**
- Slower memory access
- May require copying for certain operations
- Some operations may automatically make them contiguous

## Related Commands

- `torch::tensor_contiguous` - Make tensor contiguous
- `torch::tensor_reshape` - Reshape tensor (may affect contiguity)
- `torch::tensor_permute` - Permute dimensions (may break contiguity)
- `torch::tensor_transpose` - Transpose tensor (may break contiguity)
- `torch::tensor_shape` - Get tensor shape
- `torch::tensor_stride` - Get tensor strides

## Notes

1. **Backward Compatibility**: The original positional syntax is fully supported and unchanged.
2. **Performance**: Checking contiguity is a fast operation that doesn't modify the tensor.
3. **Memory**: The operation doesn't allocate new memory or modify the tensor.
4. **Use Cases**: Useful for optimizing operations that require contiguous tensors.

## Common Patterns

### Checking Before Operations
```tcl
# Check if tensor needs to be made contiguous
if {![torch::tensor_is_contiguous $tensor]} {
    set tensor [torch::tensor_contiguous $tensor]
}
```

### Debugging Memory Layout
```tcl
# Debug tensor memory layout
puts "Shape: [torch::tensor_shape $tensor]"
puts "Contiguous: [torch::tensor_is_contiguous $tensor]"
```

### Optimizing Performance
```tcl
# Ensure tensor is contiguous for optimal performance
if {![torch::tensor_is_contiguous $tensor]} {
    puts "Warning: Tensor is not contiguous, consider using tensor_contiguous"
}
```

## Version History

- **Refactored**: Added named parameter syntax and camelCase alias while maintaining 100% backward compatibility
- **Original**: Positional parameter syntax only 