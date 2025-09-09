# torch::layer_device

Get the current device of a neural network layer.

## Syntax

### Current Syntax (Backward Compatible)
```tcl
torch::layer_device layer_name
```

### New Syntax (Named Parameters)
```tcl
torch::layer_device -layer layer_name
torch::layer_device -input layer_name  ;# alternative parameter name
```

### camelCase Alias
```tcl
torch::layerDevice layer_name
torch::layerDevice -layer layer_name
```

## Parameters

### Named Parameters
- **`-layer`** (string, required): Name/handle of the layer to query
- **`-input`** (string, required): Alternative parameter name for the layer (same as `-layer`)

### Positional Parameters  
- **`layer_name`** (string, required): Name/handle of the layer to query

## Description

The `torch::layer_device` command returns the current device of a neural network layer/module. This is useful for:

- Debugging device placement issues
- Verifying device assignments before computations
- Monitoring device usage in mixed-device training
- Ensuring optimal device utilization

The command supports both the original positional syntax for backward compatibility and the new named parameter syntax for improved readability and flexibility.

## Return Value

Returns a string representing the current device:
- **`"cpu"`** - Layer is on CPU
- **`"cuda:N"`** - Layer is on CUDA device N (where N is the device index)

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a layer
set layer [torch::linear 128 64]

# Check current device
set device [torch::layer_device $layer]
puts "Layer device: $device"  ;# Output: "Layer device: cpu"
```

### Named Parameter Syntax
```tcl
# Create a layer  
set layer [torch::linear 256 128]

# Check device using named parameters
set device [torch::layer_device -layer $layer]
puts "Current device: $device"

# Alternative parameter name
set device [torch::layer_device -input $layer]
puts "Device: $device"
```

### camelCase Alias
```tcl
# Create a layer
set layer [torch::linear 512 256]

# Check device using camelCase alias
set device [torch::layerDevice $layer]
puts "Device: $device"

# Named parameter with camelCase
set device [torch::layerDevice -layer $layer]
puts "Device: $device"
```

### Device Monitoring Workflow
```tcl
# Create multiple layers
set encoder [torch::linear 784 512]
set decoder [torch::linear 512 784]

# Check initial device placement
puts "Encoder device: [torch::layer_device $encoder]"
puts "Decoder device: [torch::layer_device $decoder]"

# Move encoder to CUDA (if available)
if {[torch::cuda_is_available]} {
    torch::layer_cuda $encoder
    puts "Encoder moved to: [torch::layer_device $encoder]"
}

# Verify final placement
puts "Final encoder device: [torch::layer_device $encoder]"
puts "Final decoder device: [torch::layer_device $decoder]"
```

### Debug Device Placement
```tcl
# Function to debug layer device placement
proc debug_layer_devices {layer_list} {
    puts "=== Layer Device Debug ==="
    foreach layer $layer_list {
        set device [torch::layer_device $layer]
        puts "Layer $layer: $device"
    }
    puts "========================="
}

# Usage
set layers [list [torch::linear 100 50] [torch::linear 50 10]]
debug_layer_devices $layers
```

### Mixed Device Training Setup
```tcl
# Create model components
set embedding [torch::embedding 1000 128]
set encoder [torch::linear 128 256]
set decoder [torch::linear 256 128]
set output [torch::linear 128 10]

# Strategic device placement
torch::layer_cpu $embedding      ;# Keep embeddings on CPU
torch::layer_cuda $encoder       ;# Move compute-heavy layers to GPU
torch::layer_cuda $decoder
torch::layer_cpu $output         ;# Output layer on CPU

# Verify placement
puts "Device placement:"
puts "Embedding: [torch::layer_device $embedding]"
puts "Encoder: [torch::layer_device $encoder]"
puts "Decoder: [torch::layer_device $decoder]"
puts "Output: [torch::layer_device $output]"
```

### Performance Optimization
```tcl
# Monitor device usage for optimization
proc optimize_device_placement {layers} {
    set cpu_count 0
    set cuda_count 0
    
    foreach layer $layers {
        set device [torch::layer_device $layer]
        if {$device eq "cpu"} {
            incr cpu_count
        } elseif {[string match "cuda*" $device]} {
            incr cuda_count
        }
    }
    
    puts "Device distribution:"
    puts "CPU layers: $cpu_count"
    puts "CUDA layers: $cuda_count"
    
    # Recommendation logic
    if {$cpu_count > $cuda_count && [torch::cuda_is_available]} {
        puts "Recommendation: Consider moving more layers to CUDA"
    }
}

# Usage
set model_layers [list [torch::linear 512 256] [torch::linear 256 128] [torch::linear 128 64]]
optimize_device_placement $model_layers
```

### Conditional Device Operations
```tcl
# Conditional processing based on device
proc process_layer_conditionally {layer} {
    set device [torch::layer_device $layer]
    
    if {$device eq "cpu"} {
        puts "Processing $layer on CPU - using optimized CPU routines"
        # CPU-specific optimizations
    } elseif {[string match "cuda*" $device]} {
        puts "Processing $layer on CUDA device $device - using GPU acceleration"
        # GPU-specific optimizations
    } else {
        puts "Unknown device: $device"
    }
}

# Usage
set layer [torch::linear 1024 512]
process_layer_conditionally $layer
```

### Device Transition Verification
```tcl
# Verify device transitions work correctly
proc verify_device_transition {layer} {
    # Check initial device
    set initial_device [torch::layer_device $layer]
    puts "Initial device: $initial_device"
    
    # Move to CPU and verify
    torch::layer_cpu $layer
    set cpu_device [torch::layer_device $layer]
    puts "After CPU move: $cpu_device"
    
    # Move to CUDA if available and verify
    if {[torch::cuda_is_available]} {
        torch::layer_cuda $layer
        set cuda_device [torch::layer_device $layer]
        puts "After CUDA move: $cuda_device"
        
        # Move back to CPU
        torch::layer_cpu $layer
        set final_device [torch::layer_device $layer]
        puts "Final device: $final_device"
    }
}

# Usage
set layer [torch::linear 256 128]
verify_device_transition $layer
```

## Error Handling

The command performs comprehensive error checking:

```tcl
# Invalid layer name
catch {torch::layer_device "nonexistent_layer"} error
puts $error  ;# "Invalid layer name"

# Missing parameter value
catch {torch::layer_device -layer} error
puts $error  ;# "Missing value for parameter"

# Unknown parameter
catch {torch::layer_device -unknown_param value} error
puts $error  ;# "Unknown parameter: -unknown_param"

# Empty layer name
catch {torch::layer_device -layer ""} error
puts $error  ;# "Required parameter missing: layer"
```

## Device Types

### CPU Device
- **Format**: `"cpu"`
- **Usage**: Default device for all layers
- **Best for**: Small models, debugging, CPU-only systems

### CUDA Device
- **Format**: `"cuda:N"` where N is the device index
- **Usage**: GPU acceleration for compute-intensive operations
- **Best for**: Large models, training, high-performance inference

### Device Detection Examples
```tcl
# Check device type
set device [torch::layer_device $layer]

if {$device eq "cpu"} {
    puts "Layer is on CPU"
} elseif {[string match "cuda:*" $device]} {
    # Extract CUDA device number
    set cuda_id [string range $device 5 end]
    puts "Layer is on CUDA device $cuda_id"
} else {
    puts "Unknown device type: $device"
}
```

## Integration with Device Management

The `torch::layer_device` command works seamlessly with other device management commands:

```tcl
# Complete device management workflow
set layer [torch::linear 1024 512]

# 1. Check initial device
set initial [torch::layer_device $layer]
puts "Initial: $initial"

# 2. Move to specific device
torch::layer_cpu $layer
set after_cpu [torch::layer_device $layer]
puts "After CPU: $after_cpu"

# 3. Move to CUDA if available
if {[torch::cuda_is_available]} {
    torch::layer_cuda $layer
    set after_cuda [torch::layer_device $layer]
    puts "After CUDA: $after_cuda"
}

# 4. Move to specific device
torch::layer_to $layer "cpu"
set after_to [torch::layer_device $layer]
puts "After torch::layer_to: $after_to"
```

## Device Management Commands

Related commands for comprehensive device management:

- **`torch::layer_cpu`** - Move layer to CPU device
- **`torch::layer_cuda`** - Move layer to CUDA device
- **`torch::layer_to`** - Move layer to specific device
- **`torch::cuda_is_available`** - Check CUDA availability
- **`torch::cuda_device_count`** - Get number of CUDA devices

## Performance Considerations

### Device Query Overhead
- Device queries are fast operations
- Minimal performance impact
- Safe to use in performance-critical code

### Best Practices
```tcl
# Cache device info for repeated use
set device [torch::layer_device $layer]
if {$device eq "cpu"} {
    # Use cached info for multiple operations
    cpu_specific_operation1 $layer
    cpu_specific_operation2 $layer
}
```

### Memory Management
- Device queries don't affect memory usage
- No memory transfer involved
- Safe for memory-constrained environments

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set device [torch::layer_device $layer]

# New named parameter syntax  
set device [torch::layer_device -layer $layer]
```

### From snake_case to camelCase

```tcl
# Old snake_case command
set device [torch::layer_device $layer]

# New camelCase command
set device [torch::layerDevice $layer]
```

## Troubleshooting

### Common Issues

1. **Invalid Layer Name**
   ```tcl
   # Problem: Layer doesn't exist
   catch {torch::layer_device "bad_layer"} error
   # Solution: Verify layer exists before querying
   ```

2. **Empty Results**
   ```tcl
   # Problem: Layer has no parameters
   # Solution: Some layers may not have parameters, default is CPU
   ```

3. **Unexpected Device**
   ```tcl
   # Problem: Layer on wrong device
   # Solution: Use torch::layer_cpu or torch::layer_cuda to move
   ```

## Compatibility

- **Backward Compatible**: All existing positional syntax continues to work
- **Forward Compatible**: New named parameter syntax is preferred for new code
- **Alias Support**: camelCase aliases provide modern API style
- **Error Handling**: Comprehensive validation with clear error messages
- **Cross-platform**: Works on all supported PyTorch platforms

## Version History

- **v1.0**: Original positional syntax implementation
- **v2.0**: Added dual syntax support with named parameters and camelCase aliases

## See Also

- [torch::layer_cpu](layer_cpu.md) - Move layer to CPU device
- [torch::layer_cuda](layer_cuda.md) - Move layer to CUDA device
- [torch::layer_to](layer_to.md) - Move layer to specific device
- [Device Management Guide](../guides/device_management.md)
- [Performance Optimization](../guides/performance.md)
- [CUDA Setup Guide](../guides/cuda_setup.md) 