# torch::cuda_device_count

Returns the number of available CUDA devices in the system.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::cuda_device_count
```

### Positional Syntax (Legacy)
```tcl
torch::cuda_device_count
```

### camelCase Alias
```tcl
torch::cudaDeviceCount
```

## Parameters

This command takes no parameters.

## Return Value

Returns an integer representing the number of available CUDA devices:
- **0**: No CUDA devices available or CUDA not installed/configured
- **Positive integer**: Number of CUDA devices available (typically 1-8 for most systems)

## Description

The `torch::cuda_device_count` command queries the system for the number of available CUDA-capable GPU devices. This is useful for:

- **Hardware Detection**: Determining if CUDA acceleration is available
- **Multi-GPU Setup**: Planning parallel computation across multiple GPUs
- **Conditional Logic**: Enabling GPU-specific features only when devices are available
- **Resource Management**: Allocating tensors across available devices

The command returns the count immediately without initialization overhead, making it suitable for frequent calls.

## CUDA Environment

The behavior depends on your CUDA environment:

| CUDA Status | Return Value | Description |
|-------------|--------------|-------------|
| Not installed | 0 | CUDA runtime not found |
| Installed, no GPUs | 0 | CUDA available but no compatible devices |
| Installed with GPUs | 1+ | Number of available CUDA devices |
| Driver issues | 0 | CUDA devices present but not accessible |

## Examples

### Basic Usage
```tcl
# Check if any CUDA devices are available
set gpu_count [torch::cuda_device_count]
if {$gpu_count > 0} {
    puts "Found $gpu_count CUDA device(s)"
} else {
    puts "No CUDA devices available - using CPU"
}
```

### Multi-GPU Setup Detection
```tcl
# Plan computation based on available GPUs
set num_gpus [torch::cuda_device_count]

if {$num_gpus >= 4} {
    puts "Multi-GPU setup detected - enabling distributed training"
} elseif {$num_gpus >= 2} {
    puts "Dual GPU setup - enabling data parallel training"
} elseif {$num_gpus == 1} {
    puts "Single GPU available - using GPU acceleration"
} else {
    puts "CPU-only mode"
}
```

### Integration with CUDA Availability Check
```tcl
# Comprehensive CUDA status check
set cuda_available [torch::cuda_is_available]
set device_count [torch::cuda_device_count]

puts "CUDA Runtime Available: [expr {$cuda_available ? "Yes" : "No"}]"
puts "CUDA Devices Found: $device_count"

if {$cuda_available && $device_count > 0} {
    puts "CUDA acceleration ready"
} else {
    puts "Using CPU mode"
}
```

### camelCase Alias Usage
```tcl
# Using the modern camelCase syntax
set gpus [torch::cudaDeviceCount]
puts "System has $gpus GPU(s)"
```

### Resource Planning Loop
```tcl
# Distribute work across available GPUs
set device_count [torch::cuda_device_count]

if {$device_count > 0} {
    for {set i 0} {$i < $device_count} {incr i} {
        puts "GPU $i available for computation"
        # Could initialize tensors on specific devices here
    }
}
```

## Performance Characteristics

- **Execution Time**: < 1ms per call (very fast)
- **Scalability**: Tested up to 1000 rapid calls (< 2 seconds total)
- **Overhead**: Minimal - suitable for frequent polling
- **Consistency**: Always returns the same value for a given system state

## Error Handling

This command is designed to be robust and should not produce errors under normal circumstances:

- **No CUDA**: Returns 0 (not an error)
- **Driver Issues**: Returns 0 (graceful degradation)
- **Invalid Arguments**: Returns error for extra parameters

```tcl
# This will produce an error
catch {torch::cuda_device_count extra_arg} error
puts "Error: $error"
# Output: Error: Wrong number of arguments...
```

## Integration with Other Commands

### Related CUDA Commands
```tcl
# Complete CUDA environment check
set available [torch::cuda_is_available]
set count [torch::cuda_device_count]
set memory [torch::cuda_memory_info]  # If implemented

puts "CUDA Available: $available"
puts "Device Count: $count"
```

### Conditional Tensor Creation
```tcl
# Create tensors on GPU if available
set gpu_count [torch::cuda_device_count]

if {$gpu_count > 0} {
    set tensor [torch::randn -size {1000 1000} -device cuda:0]
} else {
    set tensor [torch::randn -size {1000 1000} -device cpu]
}
```

## Migration Guide

### From Positional to Named Syntax

Since this command takes no parameters, both syntaxes are identical:

```tcl
# Both of these are equivalent
set count1 [torch::cuda_device_count]
set count2 [torch::cudaDeviceCount]
```

### From Legacy Code

If you have existing code using this command, no changes are required for backward compatibility:

```tcl
# Existing code continues to work
set gpu_count [torch::cuda_device_count]

# New code can use camelCase for consistency
set gpu_count [torch::cudaDeviceCount]
```

## Technical Notes

### Implementation Details
- Uses `torch::cuda::device_count()` from LibTorch
- No CUDA context initialization required
- Thread-safe implementation
- Cached result for performance (updates on device changes)

### System Requirements
- LibTorch with CUDA support
- CUDA runtime installed (for non-zero results)
- Compatible NVIDIA drivers

### Troubleshooting
If the command returns 0 but you expect CUDA devices:

1. **Check CUDA Installation**:
   ```bash
   nvidia-smi  # Should show GPU(s)
   nvcc --version  # Should show CUDA compiler
   ```

2. **Verify LibTorch CUDA Support**:
   ```tcl
   torch::cuda_is_available  # Should return 1
   ```

3. **Check Driver Compatibility**: Ensure NVIDIA drivers match CUDA version

## See Also

- [`torch::cuda_is_available`](cuda_is_available.md) - Check if CUDA runtime is available
- [`torch::cuda_device_info`](cuda_device_info.md) - Get detailed device information
- [`torch::cuda_memory_info`](cuda_memory_info.md) - Check GPU memory status
- [Tensor Device Placement](../tensor_device_placement.md) - Using GPUs for computation

## Version History

- **1.0.0**: Initial implementation with positional syntax
- **2.0.0**: Added dual syntax support and camelCase alias
- **2.0.1**: Enhanced error handling and performance optimization 