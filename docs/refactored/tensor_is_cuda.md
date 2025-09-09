# torch::tensor_is_cuda

Check if a tensor is located on CUDA device.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_is_cuda tensor_handle
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_is_cuda -tensor tensor_handle
torch::tensor_is_cuda -input tensor_handle
```

### CamelCase Alias
```tcl
torch::tensorIsCuda tensor_handle
torch::tensorIsCuda -tensor tensor_handle
torch::tensorIsCuda -input tensor_handle
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor_handle` | string | Yes | The handle of the tensor to check |
| `-tensor` | string | Yes | Alternative parameter name for tensor_handle |
| `-input` | string | Yes | Alternative parameter name for tensor_handle |

## Return Value

Returns a boolean value:
- `1` (true) if the tensor is on CUDA device
- `0` (false) if the tensor is on CPU or other device

## Examples

### Basic Usage

```tcl
;# Create a CPU tensor
set cpu_tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]

;# Check if it's on CUDA (should return 0)
set is_cuda [torch::tensor_is_cuda $cpu_tensor]
puts "Is CUDA: $is_cuda"  ;# Output: Is CUDA: 0

;# Create a CUDA tensor (if CUDA is available)
if {[torch::cuda_is_available]} {
    set cuda_tensor [torch::tensor_create {1.0 2.0 3.0} float32 cuda true]
    set is_cuda [torch::tensor_is_cuda $cuda_tensor]
    puts "Is CUDA: $is_cuda"  ;# Output: Is CUDA: 1
}
```

### Named Parameter Syntax

```tcl
set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]

;# Using -tensor parameter
set result1 [torch::tensor_is_cuda -tensor $tensor]

;# Using -input parameter (alias)
set result2 [torch::tensor_is_cuda -input $tensor]

;# Both should return the same result
puts "Result 1: $result1, Result 2: $result2"
```

### CamelCase Alias

```tcl
set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]

;# Using camelCase alias
set result [torch::tensorIsCuda $tensor]
puts "Is CUDA: $result"

;# CamelCase with named parameters
set result [torch::tensorIsCuda -tensor $tensor]
puts "Is CUDA: $result"
```

### Error Handling

```tcl
;# Invalid tensor name
catch {torch::tensor_is_cuda invalid_tensor} result
puts "Error: $result"  ;# Output: Error: Invalid tensor name

;# Missing parameter
catch {torch::tensor_is_cuda} result
puts "Error: $result"  ;# Output: Error: Required tensor parameter missing

;# Unknown parameter
set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
catch {torch::tensor_is_cuda -unknown $tensor} result
puts "Error: $result"  ;# Output: Error: Unknown parameter: -unknown
```

## Migration Guide

### From Old Syntax to New Syntax

**Old (Positional Only):**
```tcl
set is_cuda [torch::tensor_is_cuda $tensor]
```

**New (Named Parameters):**
```tcl
set is_cuda [torch::tensor_is_cuda -tensor $tensor]
```

**New (CamelCase):**
```tcl
set is_cuda [torch::tensorIsCuda $tensor]
```

### Backward Compatibility

The old positional syntax continues to work:
```tcl
;# This still works
set is_cuda [torch::tensor_is_cuda $tensor]
```

## Notes

- This function is useful for checking device placement before operations
- Returns `0` for CPU tensors and `1` for CUDA tensors
- Works with tensors of any data type or shape
- The function is lightweight and doesn't modify the tensor
- Both snake_case and camelCase versions are functionally identical

## Related Commands

- `torch::tensor_create` - Create tensors on specific devices
- `torch::cuda_is_available` - Check if CUDA is available
- `torch::tensor_to` - Move tensors between devices
- `torch::tensor_device` - Get the device of a tensor 