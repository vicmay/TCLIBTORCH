# torch::autocast_is_enabled

**Check if automatic mixed precision (AMP) is enabled for PyTorch operations**

## Overview

The `torch::autocast_is_enabled` command checks whether automatic mixed precision (AMP) is currently enabled for PyTorch operations on the specified device. This command returns a boolean value indicating the autocast state, which is useful for conditional logic and verification in AMP workflows.

## Syntax

### Original Syntax (Positional Parameters)
```tcl
torch::autocast_is_enabled [device_type]
```

### New Syntax (Named Parameters)  
```tcl
torch::autocast_is_enabled [-device_type <device>] [-device <device>]
```

### camelCase Alias
```tcl
torch::autocastIsEnabled [device_type]
torch::autocastIsEnabled [-device_type <device>] [-device <device>]
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| device_type | string | No | "cuda" | Device to check autocast status for ("cuda" or "cpu") |

### Named Parameter Aliases

| Long Form | Short Form | Description |
|-----------|------------|-------------|
| `-device_type` | `-device` | Device to check autocast status for |

## Returns

Returns a boolean value:
- `1` (true): Autocast is enabled for the specified device
- `0` (false): Autocast is disabled for the specified device

## Device Types

- **cuda**: Check autocast status for CUDA operations (GPU)
- **cpu**: Check autocast status for CPU operations

## Examples

### Basic Usage

```tcl
# Check autocast status for CUDA (default)
set cuda_enabled [torch::autocast_is_enabled]
puts "CUDA autocast enabled: $cuda_enabled"

# Check autocast status for CUDA explicitly
set cuda_status [torch::autocast_is_enabled cuda]

# Check autocast status for CPU
set cpu_status [torch::autocast_is_enabled cpu]
```

### Named Parameter Syntax

```tcl
# Check CUDA autocast status using named parameters
set cuda_enabled [torch::autocast_is_enabled -device_type cuda]

# Check CPU autocast status using short parameter name
set cpu_enabled [torch::autocast_is_enabled -device cpu]
```

### camelCase Alias

```tcl
# Using camelCase with positional parameters
set status [torch::autocastIsEnabled cuda]

# Using camelCase with named parameters
set status [torch::autocastIsEnabled -device_type cpu]
```

### Complete Workflow Examples

#### Basic Enable/Check/Disable Workflow
```tcl
# Check initial state
set initial_state [torch::autocast_is_enabled cuda]
puts "Initial CUDA autocast state: $initial_state"

# Enable autocast
torch::autocast_enable cuda float16

# Verify it's enabled
set enabled_state [torch::autocast_is_enabled cuda]
puts "After enabling: $enabled_state"

# Disable autocast
torch::autocast_disable cuda

# Verify it's disabled
set disabled_state [torch::autocast_is_enabled cuda]
puts "After disabling: $disabled_state"
```

#### Conditional Training Logic
```tcl
# Check if autocast is available and use it conditionally
proc train_model {model data} {
    set autocast_available [torch::autocast_is_enabled cuda]
    
    if {$autocast_available} {
        puts "Training with autocast enabled"
        # Training with mixed precision
        set output [$model $data]
    } else {
        puts "Training without autocast"
        # Regular training
        set output [$model $data]
    }
    
    return $output
}

# Enable autocast for training
torch::autocast_enable cuda float16
train_model $my_model $training_data
```

#### Multi-Device Autocast Management
```tcl
# Check autocast status for both devices
set cuda_status [torch::autocast_is_enabled cuda]
set cpu_status [torch::autocast_is_enabled cpu]

puts "Device autocast status:"
puts "  CUDA: $cuda_status"
puts "  CPU: $cpu_status"

# Enable autocast for both devices if needed
if {!$cuda_status} {
    torch::autocast_enable cuda float16
    puts "Enabled CUDA autocast"
}

if {!$cpu_status} {
    torch::autocast_enable cpu bfloat16
    puts "Enabled CPU autocast"
}

# Verify both are now enabled
set final_cuda [torch::autocast_is_enabled cuda]
set final_cpu [torch::autocast_is_enabled cpu]
puts "Final status - CUDA: $final_cuda, CPU: $final_cpu"
```

#### Integration with Performance Monitoring
```tcl
proc benchmark_with_autocast {model data iterations} {
    # Test without autocast
    torch::autocast_disable cuda
    set start_time [clock clicks -milliseconds]
    for {set i 0} {$i < $iterations} {incr i} {
        set output [$model $data]
    }
    set end_time [clock clicks -milliseconds]
    set time_without_autocast [expr {$end_time - $start_time}]
    
    # Test with autocast
    torch::autocast_enable cuda float16
    set autocast_enabled [torch::autocast_is_enabled cuda]
    if {$autocast_enabled} {
        set start_time [clock clicks -milliseconds]
        for {set i 0} {$i < $iterations} {incr i} {
            set output [$model $data]
        }
        set end_time [clock clicks -milliseconds]
        set time_with_autocast [expr {$end_time - $start_time}]
        
        puts "Performance comparison:"
        puts "  Without autocast: ${time_without_autocast}ms"
        puts "  With autocast: ${time_with_autocast}ms"
        puts "  Speedup: [expr {double($time_without_autocast) / $time_with_autocast}]x"
    }
}
```

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
set cuda_enabled [torch::autocast_is_enabled cuda]
set cpu_enabled [torch::autocast_is_enabled cpu]
```

**New (Named):**
```tcl
set cuda_enabled [torch::autocast_is_enabled -device_type cuda]
set cpu_enabled [torch::autocast_is_enabled -device_type cpu]
```

### Advantages of Named Parameters

1. **Self-documenting**: Parameter names make code more readable
2. **Explicit intent**: Clear what device is being checked
3. **Consistency**: Matches other autocast commands' named parameter style
4. **Future-proof**: Easier to extend with additional parameters

## Error Handling

### Invalid Device Type
```tcl
# This will fail
set status [torch::autocast_is_enabled invalid_device]
# Error: Invalid device type. Use cuda or cpu
```

### Unknown Parameter
```tcl
# This will fail  
set status [torch::autocast_is_enabled -invalid_param cuda]
# Error: Unknown parameter: -invalid_param. Valid parameters are: -device_type, -device
```

### Missing Parameter Value
```tcl
# This will fail
set status [torch::autocast_is_enabled -device_type]
# Error: Missing value for parameter
```

### Too Many Positional Arguments
```tcl
# This will fail
set status [torch::autocast_is_enabled cuda extra_arg]
# Error: Usage: torch::autocast_is_enabled [device_type]
```

## Use Cases

### 1. Conditional Logic
Check autocast status before performing operations that depend on mixed precision.

### 2. State Verification
Verify that autocast enable/disable operations worked correctly.

### 3. Performance Testing
Compare performance with and without autocast enabled.

### 4. Multi-Device Management
Manage autocast state independently for CUDA and CPU devices.

### 5. Configuration Validation
Ensure proper autocast configuration in training pipelines.

## Performance Notes

- **Very Fast Operation**: Checking autocast status is extremely fast (sub-millisecond)
- **No Overhead**: Does not affect model performance or memory usage
- **Thread-Safe**: Safe to call from multiple threads
- **State Query Only**: Does not modify any autocast settings

## Related Commands

- [torch::autocast_enable](autocast_enable.md) - Enable automatic mixed precision
- [torch::autocast_disable](autocast_disable.md) - Disable automatic mixed precision  
- [torch::autocast_set_dtype](autocast_set_dtype.md) - Set autocast data type

## Technical Details

This command uses PyTorch's native autocast query functionality:
- For CUDA: `at::autocast::is_autocast_enabled(at::kCUDA)`
- For CPU: `at::autocast::is_autocast_enabled(at::kCPU)`
- Returns the current autocast enabled state as a boolean

## Return Value Details

The command returns a TCL boolean object:
- `1` when autocast is enabled
- `0` when autocast is disabled

This can be used directly in conditional expressions:
```tcl
if {[torch::autocast_is_enabled cuda]} {
    puts "Autocast is enabled"
} else {
    puts "Autocast is disabled"
}
```

## Backward Compatibility

✅ **Full backward compatibility maintained**
- All existing positional syntax continues to work
- No breaking changes to existing code
- New named parameter syntax is purely additive

## Integration Examples

### With PyTorch Lightning-style Training
```tcl
proc setup_training {use_amp} {
    if {$use_amp} {
        torch::autocast_enable cuda float16
        set amp_enabled [torch::autocast_is_enabled cuda]
        puts "AMP enabled: $amp_enabled"
    }
}

proc training_step {model batch} {
    set autocast_enabled [torch::autocast_is_enabled cuda]
    if {$autocast_enabled} {
        # Mixed precision training step
        puts "Using mixed precision"
    } else {
        # Full precision training step  
        puts "Using full precision"
    }
}
```

### With Configuration Management
```tcl
proc apply_autocast_config {config} {
    dict with config {
        if {[info exists use_cuda_autocast] && $use_cuda_autocast} {
            torch::autocast_enable cuda $cuda_dtype
        }
        if {[info exists use_cpu_autocast] && $use_cpu_autocast} {
            torch::autocast_enable cpu $cpu_dtype
        }
    }
    
    # Verify configuration applied correctly
    puts "Autocast configuration applied:"
    puts "  CUDA: [torch::autocast_is_enabled cuda]"
    puts "  CPU: [torch::autocast_is_enabled cpu]"
}
```

## See Also

- [PyTorch Automatic Mixed Precision Documentation](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA Automatic Mixed Precision Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) 