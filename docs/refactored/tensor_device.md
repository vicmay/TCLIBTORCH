# torch::tensor_device

Returns the device of a tensor.

## Description

The `torch::tensor_device` command retrieves the device (CPU, CUDA, etc.) where the specified tensor is located. This is essential for understanding where tensor operations will be performed and for managing memory across different devices in GPU-accelerated computations.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_device tensor
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_device -input tensor
```

### CamelCase Alias
```tcl
torch::tensorDevice -input tensor
```

## Parameters

| Parameter | Type   | Required | Description                        |
|-----------|--------|----------|------------------------------------|
| input     | string | Yes      | Name of the input tensor           |

## Return Value

Returns a string containing the device of the tensor (e.g., "cpu", "cuda:0", "cuda:1").

## Examples

### Basic Usage
```tcl
# Create tensors on different devices
set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set b [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cuda]

# Using positional syntax
set device_a [torch::tensor_device $a]
set device_b [torch::tensor_device $b]

# Using named parameter syntax
set device_a_named [torch::tensor_device -input $a]
set device_b_named [torch::tensor_device -input $b]

# Using camelCase alias
set device_a_camel [torch::tensorDevice -input $a]
set device_b_camel [torch::tensorDevice -input $b]
```

### Device Management
```tcl
# Check device before operations
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu]
set device [torch::tensor_device $tensor]

if {[string match "*cuda*" $device]} {
    puts "Tensor is on GPU - fast computation available"
} else {
    puts "Tensor is on CPU - may need to move to GPU for acceleration"
}
```

### Multi-Device Operations
```tcl
# Create tensors on different devices
set cpu_tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]

if {[catch {torch::cuda_is_available} cuda_available] || $cuda_available} {
    set gpu_tensor [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cuda]
    
    # Check devices
    set cpu_device [torch::tensor_device $cpu_tensor]
    set gpu_device [torch::tensor_device $gpu_tensor]
    
    puts "CPU tensor device: $cpu_device"
    puts "GPU tensor device: $gpu_device"
    
    # Note: Operations between tensors on different devices require explicit movement
}
```

### Device Validation
```tcl
# Function to validate tensor device
proc validate_cuda_tensor {tensor_name} {
    set device [torch::tensor_device $tensor_name]
    if {[string match "*cuda*" $device]} {
        return 1
    } else {
        error "Tensor must be on CUDA device, got: $device"
    }
}

# Usage
if {[catch {torch::cuda_is_available} cuda_available] || $cuda_available} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cuda]
    validate_cuda_tensor $tensor
}
```

## Error Handling

### Invalid Tensor Name
```tcl
catch {torch::tensor_device invalid_tensor} result
# Returns: "Invalid tensor name"
```

### Missing Input Parameter
```tcl
catch {torch::tensor_device} result
# Returns: "Input tensor is required"
```

### Too Many Arguments
```tcl
set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
catch {torch::tensor_device $a extra} result
# Returns: "Invalid number of arguments"
```

### Unknown Parameter
```tcl
set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
catch {torch::tensor_device -input $a -unknown_param value} result
# Returns: "Unknown parameter: -unknown_param"
```

## Supported Devices

The `tensor_device` command returns the following device strings:

### CPU Device
- **"cpu"** - Central Processing Unit (default)

### CUDA Devices
- **"cuda:0"** - First CUDA GPU
- **"cuda:1"** - Second CUDA GPU
- **"cuda:2"** - Third CUDA GPU
- etc.

### Other Devices (if supported)
- **"mps"** - Apple Metal Performance Shaders (macOS)
- **"xpu"** - Intel XPU
- **"cuda"** - Default CUDA device (equivalent to "cuda:0")

### Notes
- The exact string returned may vary slightly depending on the PyTorch version
- CUDA devices are only available if CUDA is installed and the system has compatible GPUs
- Device strings are case-sensitive and should be used exactly as returned

## Edge Cases

### Empty Tensor
```tcl
set a [torch::tensor_create -data {} -dtype float32 -device cpu]
set device [torch::tensor_device $a]
# Returns: "cpu" (or equivalent)
```

### Single Element Tensor
```tcl
set a [torch::tensor_create -data {5.0} -dtype float32 -device cpu]
set device [torch::tensor_device $a]
# Returns: "cpu" (or equivalent)
```

### Large Tensor
```tcl
set data [list]
for {set i 0} {$i < 10000} {incr i} {
    lappend data [expr {$i * 1.0}]
}
set a [torch::tensor_create -data $data -dtype float32 -device cpu]
set device [torch::tensor_device $a]
# Returns: "cpu" (or equivalent)
```

## Migration Guide

### From Old Syntax to New Syntax

**Old (Positional Only)**:
```tcl
# Old way - still supported
set device [torch::tensor_device $tensor]
```

**New (Named Parameters)**:
```tcl
# New way - more explicit
set device [torch::tensor_device -input $tensor]
```

**CamelCase Alternative**:
```tcl
# Modern camelCase syntax
set device [torch::tensorDevice -input $tensor]
```

### Benefits of New Syntax
- **Explicit parameter names**: No confusion about parameter order
- **Better error messages**: Clear indication of missing parameters
- **Future extensibility**: Easy to add new parameters
- **Consistency**: Matches other refactored commands

## Related Commands

- [torch::tensor_dtype](tensor_dtype.md) - Get tensor data type
- [torch::tensor_requires_grad](tensor_requires_grad.md) - Check if tensor requires gradients
- [torch::tensor_to](tensor_to.md) - Move tensor to different device/dtype
- [torch::cuda_is_available](cuda_is_available.md) - Check CUDA availability
- [torch::cuda_device_count](cuda_device_count.md) - Get number of CUDA devices

## Performance Considerations

### Device Selection
- **CPU**: Good for small tensors and operations that don't benefit from GPU acceleration
- **CUDA**: Excellent for large tensors and computationally intensive operations
- **Memory**: GPU memory is typically more limited than system RAM

### Best Practices
```tcl
# Check CUDA availability before using GPU
if {[catch {torch::cuda_is_available} cuda_available] && $cuda_available} {
    set device "cuda"
} else {
    set device "cpu"
}

# Create tensor on appropriate device
set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device $device]

# Verify device placement
set actual_device [torch::tensor_device $tensor]
puts "Tensor created on: $actual_device"
```

### Memory Management
- Tensors on GPU consume GPU memory
- Use `torch::tensor_to` to move tensors between devices
- Monitor GPU memory usage with `torch::cuda_memory_info`

## Notes

- The device is a fundamental property of the tensor and affects where operations are performed
- Use `torch::tensor_to` to move a tensor to a different device
- GPU operations are typically much faster for large tensors and complex computations
- CPU operations are more reliable and have larger memory capacity
- Device information is essential for debugging and optimizing performance
- Always check CUDA availability before attempting to use GPU devices

---

**Migration Note**: 
- **Old:** `torch::tensor_device $t` (still supported)
- **New:** `torch::tensor_device -input $t` or `torch::tensorDevice -input $t`
- Both syntaxes are fully supported and produce identical results. 