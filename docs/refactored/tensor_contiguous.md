# torch::tensor_contiguous

Makes a tensor contiguous in memory. This operation ensures that the tensor's data is stored in a contiguous block of memory, which can improve performance for certain operations.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_contiguous tensor
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_contiguous -input tensor
torch::tensor_contiguous -tensor tensor
```

### CamelCase Alias
```tcl
torch::tensorContiguous tensor
torch::tensorContiguous -input tensor
torch::tensorContiguous -tensor tensor
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor` / `-input` / `-tensor` | string | Yes | Name of the input tensor to make contiguous |

## Return Value

Returns a string containing the handle of the new contiguous tensor.

## Description

The `torch::tensor_contiguous` command creates a new tensor that is contiguous in memory. This is useful when:

- You need to ensure optimal memory layout for performance-critical operations
- Working with tensors that have been reshaped, permuted, or sliced (which can make them non-contiguous)
- Preparing tensors for operations that require contiguous memory layout

A tensor is considered contiguous when its elements are stored in memory in the same order as they appear when iterating through the tensor. Operations like `reshape`, `permute`, or slicing can create non-contiguous tensors.

## Examples

### Basic Usage

```tcl
# Create a tensor
set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]

# Make it contiguous (positional syntax)
set result [torch::tensor_contiguous $tensor]

# Check if it's contiguous
set is_contiguous [torch::tensor_is_contiguous $result]
puts "Is contiguous: $is_contiguous"  ;# Output: Is contiguous: 1
```

### Named Parameter Syntax

```tcl
# Create a tensor
set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]

# Make it contiguous using named parameters
set result [torch::tensor_contiguous -input $tensor]

# Alternative parameter name
set result2 [torch::tensor_contiguous -tensor $tensor]
```

### Working with Non-Contiguous Tensors

```tcl
# Create a tensor
set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]

# Permute the tensor (this makes it non-contiguous)
set permuted [torch::tensor_permute $tensor {1 0}]

# Check if permuted tensor is contiguous
set is_contiguous_before [torch::tensor_is_contiguous $permuted]
puts "Before contiguous: $is_contiguous_before"  ;# Output: Before contiguous: 0

# Make it contiguous
set result [torch::tensor_contiguous $permuted]

# Check if result is contiguous
set is_contiguous_after [torch::tensor_is_contiguous $result]
puts "After contiguous: $is_contiguous_after"  ;# Output: After contiguous: 1
```

### Using CamelCase Alias

```tcl
# Create a tensor
set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]

# Make it contiguous using camelCase alias
set result [torch::tensorContiguous $tensor]

# With named parameters
set result2 [torch::tensorContiguous -input $tensor]
```

## Error Handling

### Missing Tensor
```tcl
# Error: Missing required parameter
catch {torch::tensor_contiguous} msg
puts $msg  ;# Output: Required parameter missing: -input
```

### Invalid Tensor Name
```tcl
# Error: Invalid tensor name
catch {torch::tensor_contiguous invalid_tensor} msg
puts $msg  ;# Output: Invalid tensor name
```

### Missing Named Parameter Value
```tcl
# Error: Missing value for parameter
catch {torch::tensor_contiguous -input} msg
puts $msg  ;# Output: Missing value for parameter
```

### Unknown Named Parameter
```tcl
# Error: Unknown parameter
set tensor [torch::tensor_create -data {1 2 3} -shape {3}]
catch {torch::tensor_contiguous -unknown $tensor} msg
puts $msg  ;# Output: Unknown parameter: -unknown
```

## Migration Guide

### From Positional to Named Parameters

**Old Code:**
```tcl
set result [torch::tensor_contiguous $tensor]
```

**New Code (Optional):**
```tcl
set result [torch::tensor_contiguous -input $tensor]
```

**Note:** The old positional syntax continues to work for backward compatibility.

### Using CamelCase Alias

**Old Code:**
```tcl
set result [torch::tensor_contiguous $tensor]
```

**New Code (Optional):**
```tcl
set result [torch::tensorContiguous $tensor]
```

## Performance Considerations

- **Contiguous tensors** generally perform better in operations that require sequential memory access
- **Non-contiguous tensors** may be slower for certain operations due to memory access patterns
- The `contiguous()` operation creates a new tensor with a copy of the data, so it has memory overhead
- If a tensor is already contiguous, calling `contiguous()` returns the same tensor (no copy)

## Related Commands

- `torch::tensor_is_contiguous` - Check if a tensor is contiguous
- `torch::tensor_reshape` - Reshape a tensor (may create non-contiguous tensor)
- `torch::tensor_permute` - Permute tensor dimensions (creates non-contiguous tensor)
- `torch::tensor_transpose` - Transpose tensor dimensions (creates non-contiguous tensor)

## Notes

- The operation preserves the tensor's data, shape, and data type
- If the input tensor is already contiguous, the operation returns a tensor with the same data
- The returned tensor handle is different from the input tensor handle
- This operation is commonly used as a preprocessing step before performance-critical operations 