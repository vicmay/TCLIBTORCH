# torch::cuda_is_available

Checks if CUDA acceleration is available in the current environment.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::cuda_is_available
```

### Positional Syntax (Legacy)
```tcl
torch::cuda_is_available
```

### camelCase Alias
```tcl
torch::cudaIsAvailable
```

## Parameters

This command takes no parameters.

## Return Value

Returns a boolean integer indicating CUDA availability:
- **1**: CUDA is available and ready for use
- **0**: CUDA is not available (not installed, no devices, or driver issues)

## Description

The `torch::cuda_is_available` command determines whether CUDA acceleration is available in the current environment. This is the primary way to check if GPU computation can be used with LibTorch.

This command is essential for:

- **Runtime Detection**: Determining if CUDA acceleration can be used
- **Conditional Logic**: Enabling GPU-specific features only when available
- **Device Selection**: Choosing between CPU and GPU backends
- **Error Prevention**: Avoiding CUDA-related errors in CPU-only environments
- **Performance Optimization**: Using the fastest available backend

The command performs a comprehensive check that includes CUDA runtime availability, driver compatibility, and device presence.

## CUDA Environment Detection

The behavior varies based on your system configuration:

| System Configuration | Return Value | Description |
|----------------------|--------------|-------------|
| CUDA installed + GPU present | 1 | Full CUDA support available |
| CUDA installed + No GPU | 0 | CUDA runtime but no devices |
| No CUDA installation | 0 | CUDA not available |
| Driver incompatibility | 0 | CUDA devices exist but unusable |
| LibTorch CPU-only build | 0 | No CUDA support compiled in |

## Examples

### Basic CUDA Detection
```tcl
# Check if CUDA is available
set has_cuda [torch::cuda_is_available]
if {$has_cuda} {
    puts "CUDA acceleration is available!"
} else {
    puts "Using CPU mode"
}
```

### Device Backend Selection
```tcl
# Choose appropriate device backend
proc get_device {} {
    if {[torch::cuda_is_available]} {
        return "cuda"
    } else {
        return "cpu"
    }
}

set device [get_device]
puts "Using device: $device"
```

### Conditional Tensor Creation
```tcl
# Create tensors on the best available device
set cuda_available [torch::cuda_is_available]

if {$cuda_available} {
    # Use GPU for large computations
    set large_tensor [torch::randn -size {10000 10000} -device cuda]
    puts "Created large tensor on GPU"
} else {
    # Use CPU with smaller sizes
    set large_tensor [torch::randn -size {1000 1000} -device cpu]
    puts "Created tensor on CPU"
}
```

### Performance-Aware Model Training
```tcl
# Configure training based on hardware availability
proc setup_training_config {} {
    set config [dict create]
    
    if {[torch::cuda_is_available]} {
        dict set config device "cuda"
        dict set config batch_size 128
        dict set config workers 4
        puts "GPU training configuration"
    } else {
        dict set config device "cpu" 
        dict set config batch_size 32
        dict set config workers 2
        puts "CPU training configuration"
    }
    
    return $config
}

set training_config [setup_training_config]
```

### Integration with Device Count
```tcl
# Comprehensive hardware detection
set cuda_available [torch::cuda_is_available]
set gpu_count [torch::cuda_device_count]

puts "CUDA Status Report:"
puts "  Runtime Available: [expr {$cuda_available ? "Yes" : "No"}]"
puts "  GPU Devices: $gpu_count"

if {$cuda_available && $gpu_count > 1} {
    puts "  Multi-GPU setup detected"
} elseif {$cuda_available} {
    puts "  Single GPU available"
} else {
    puts "  CPU-only environment"
}
```

### camelCase Alias Usage
```tcl
# Using the modern camelCase syntax
if {[torch::cudaIsAvailable]} {
    puts "GPU acceleration enabled"
    set device_type "cuda"
} else {
    puts "CPU fallback mode"
    set device_type "cpu"
}
```

### Library Initialization Check
```tcl
# Check CUDA before performing heavy operations
proc safe_gpu_operation {} {
    if {![torch::cuda_is_available]} {
        error "This operation requires CUDA support"
    }
    
    # Proceed with GPU operations
    set gpu_tensor [torch::ones -size {1000 1000} -device cuda]
    return $gpu_tensor
}

# Safe usage
if {[torch::cuda_is_available]} {
    set result [safe_gpu_operation]
    puts "GPU operation completed successfully"
} else {
    puts "Skipping GPU-only operation"
}
```

## Performance Characteristics

- **Execution Time**: < 1ms per call (very fast)
- **Scalability**: Tested up to 1000 rapid calls (< 2 seconds total)
- **Caching**: Result is cached internally for performance
- **Overhead**: Minimal - suitable for frequent checks

## Error Handling

This command is designed to never throw exceptions - it always returns a boolean result:

- **Hardware Issues**: Returns 0 instead of error
- **Driver Problems**: Returns 0 with graceful degradation
- **Missing CUDA**: Returns 0 (not an error condition)
- **Invalid Arguments**: Returns error only for syntax violations

```tcl
# This will produce an error (invalid syntax)
catch {torch::cuda_is_available extra_parameter} error
puts "Error: $error"
# Output: Error: Wrong number of arguments...

# This will NOT produce an error (graceful handling)
set result [torch::cuda_is_available]
# Always succeeds, returns 0 or 1
```

## Integration with Other Commands

### CUDA Command Family
```tcl
# Complete CUDA environment assessment
proc cuda_status_report {} {
    set available [torch::cuda_is_available]
    set count [torch::cuda_device_count]
    
    puts "CUDA Environment Report:"
    puts "  Available: [expr {$available ? "Yes" : "No"}]"
    puts "  Device Count: $count"
    
    if {$available && $count > 0} {
        for {set i 0} {$i < $count} {incr i} {
            # Could call torch::cuda_device_info here when available
            puts "  Device $i: Available"
        }
    }
}

cuda_status_report
```

### Tensor Operations Integration
```tcl
# Smart device placement for tensors
proc create_tensor_smart {args} {
    if {[torch::cuda_is_available]} {
        # Add CUDA device to arguments if not specified
        if {[lsearch $args "-device"] == -1} {
            lappend args -device cuda
        }
    }
    
    # Create tensor with appropriate device
    return [torch::randn {*}$args]
}

set tensor [create_tensor_smart -size {100 100}]
```

### Model Deployment Logic
```tcl
# Deploy model to best available device
proc deploy_model {model_path} {
    set device [expr {[torch::cuda_is_available] ? "cuda" : "cpu"}]
    
    puts "Loading model on $device"
    # Load and configure model for the device
    # set model [torch::load $model_path]
    # torch::model_to $model $device
    
    return $device
}
```

## Migration Guide

### From Positional to Named Syntax

Since this command takes no parameters, both syntaxes are identical:

```tcl
# Legacy positional syntax
set available1 [torch::cuda_is_available]

# Modern named syntax (same as positional for this command)
set available2 [torch::cuda_is_available]

# Modern camelCase alias
set available3 [torch::cudaIsAvailable]

# All three are equivalent and return the same value
```

### From Legacy CUDA Detection

If you previously used other methods to detect CUDA:

```tcl
# OLD: Manual environment checking (unreliable)
# set cuda_available [expr {[info exists env(CUDA_VISIBLE_DEVICES)]}]

# NEW: Proper LibTorch-based detection (recommended)
set cuda_available [torch::cuda_is_available]
```

### Error Handling Migration

```tcl
# OLD: Manual try-catch for CUDA operations
# try {
#     # Some CUDA operation
# } catch {
#     puts "Falling back to CPU"
# }

# NEW: Proactive checking (better performance)
if {[torch::cuda_is_available]} {
    # GPU operations
} else {
    # CPU operations
}
```

## Technical Notes

### Implementation Details
- Uses `torch::cuda::is_available()` from LibTorch C++ API
- Performs comprehensive runtime and device checks
- Thread-safe implementation
- No CUDA context creation overhead
- Cached result for optimal performance

### System Requirements
- LibTorch with CUDA support (for non-zero results)
- CUDA runtime installed (for CUDA detection)
- Compatible NVIDIA drivers (for device access)
- No requirements for CPU-only operation

### Relationship to Device Count
The relationship with `torch::cuda_device_count` is:
- `cuda_is_available() == 1` implies `cuda_device_count() > 0`
- `cuda_is_available() == 0` implies `cuda_device_count() == 0`

### Troubleshooting

If the command returns 0 but you expect CUDA to be available:

1. **Check CUDA Installation**:
   ```bash
   nvidia-smi  # Should show GPU status
   nvcc --version  # Should show CUDA compiler
   ```

2. **Verify LibTorch CUDA Support**:
   ```tcl
   # This command should work without errors
   torch::cuda_is_available
   ```

3. **Check Environment Variables**:
   ```bash
   echo $CUDA_VISIBLE_DEVICES  # Should show available devices
   echo $LD_LIBRARY_PATH  # Should include CUDA libraries
   ```

4. **Driver Compatibility**: Ensure NVIDIA drivers support your CUDA version

## See Also

- [`torch::cuda_device_count`](cuda_device_count.md) - Get number of CUDA devices
- [`torch::cuda_device_info`](cuda_device_info.md) - Get detailed device information
- [`torch::cuda_memory_info`](cuda_memory_info.md) - Check GPU memory status
- [Device Placement Guide](../device_placement.md) - Using different devices for computation

## Version History

- **1.0.0**: Initial implementation with simple CUDA checking
- **2.0.0**: Added dual syntax support and camelCase alias
- **2.0.1**: Enhanced robustness and error handling
- **2.1.0**: Performance optimization with result caching 