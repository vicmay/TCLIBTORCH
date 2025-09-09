# torch::autocast_set_dtype

**Set the data type for automatic mixed precision (AMP) operations**

## Overview

The `torch::autocast_set_dtype` command sets the data type used for automatic mixed precision (AMP) operations on the specified device. This command allows you to change the precision level used by autocast without having to disable and re-enable it, providing fine-grained control over mixed precision behavior.

## Syntax

### Original Syntax (Positional Parameters)
```tcl
torch::autocast_set_dtype dtype [device_type]
```

### New Syntax (Named Parameters)  
```tcl
torch::autocast_set_dtype -dtype <dtype> [-device_type <device>] [-data_type <dtype>] [-device <device>]
```

### camelCase Alias
```tcl
torch::autocastSetDtype dtype [device_type]
torch::autocastSetDtype -dtype <dtype> [-device_type <device>] [-data_type <dtype>] [-device <device>]
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| dtype | string | Yes | - | Data type for mixed precision ("float16", "bfloat16", or "float32") |
| device_type | string | No | "cuda" | Device to set autocast dtype for ("cuda" or "cpu") |

### Named Parameter Aliases

| Long Form | Short Form | Description |
|-----------|------------|-------------|
| `-dtype` | `-data_type` | Data type for mixed precision |
| `-device_type` | `-device` | Device to set autocast dtype for |

## Returns

Returns the string `"autocast dtype set"` on success.

## Data Types

- **float16**: Half precision floating point (16-bit) - fastest, lowest memory, good for most cases
- **bfloat16**: Brain floating point (16-bit) - more stable than float16, better numerical properties  
- **float32**: Single precision floating point (32-bit) - highest precision, disables mixed precision benefits

## Device Types

- **cuda**: Set autocast dtype for CUDA operations (GPU)
- **cpu**: Set autocast dtype for CPU operations

## Examples

### Basic Usage

```tcl
# Set dtype to float16 for CUDA (default device)
torch::autocast_set_dtype float16

# Set dtype to bfloat16 for CUDA explicitly
torch::autocast_set_dtype bfloat16 cuda

# Set dtype to float32 for CPU
torch::autocast_set_dtype float32 cpu
```

### Named Parameter Syntax

```tcl
# Set CUDA autocast dtype using long parameter names
torch::autocast_set_dtype -dtype float16 -device_type cuda

# Set CPU autocast dtype using short parameter names
torch::autocast_set_dtype -data_type bfloat16 -device cpu

# Parameter order independence
torch::autocast_set_dtype -device_type cuda -dtype float32
```

### camelCase Alias

```tcl
# Using camelCase with positional parameters
torch::autocastSetDtype float16 cuda

# Using camelCase with named parameters
torch::autocastSetDtype -dtype bfloat16 -device_type cpu
```

### Complete Workflow Examples

#### Dynamic Precision Control During Training
```tcl
# Start with standard mixed precision
torch::autocast_enable cuda float16

# Train for a few epochs
for {set epoch 0} {$epoch < 5} {incr epoch} {
    puts "Training epoch $epoch with float16"
    # Training code here...
}

# Switch to more stable precision for fine-tuning
torch::autocast_set_dtype bfloat16 cuda
puts "Switched to bfloat16 for stable fine-tuning"

# Continue training
for {set epoch 5} {$epoch < 10} {incr epoch} {
    puts "Fine-tuning epoch $epoch with bfloat16"
    # Fine-tuning code here...
}
```

#### Multi-Stage Training with Different Precisions
```tcl
proc train_with_precision_schedule {model data epochs} {
    # Stage 1: Aggressive mixed precision for initial training
    torch::autocast_enable cuda float16
    torch::autocast_set_dtype float16 cuda
    
    puts "Stage 1: Training with float16 (aggressive mixed precision)"
    for {set epoch 0} {$epoch < [expr {$epochs / 2}]} {incr epoch} {
        # Fast initial training
        set loss [train_epoch $model $data]
        puts "Epoch $epoch: Loss = $loss (float16)"
    }
    
    # Stage 2: More stable precision for convergence
    torch::autocast_set_dtype bfloat16 cuda
    
    puts "Stage 2: Training with bfloat16 (stable mixed precision)"
    for {set epoch [expr {$epochs / 2}]} {$epoch < $epochs} {incr epoch} {
        # Stable convergence
        set loss [train_epoch $model $data]
        puts "Epoch $epoch: Loss = $loss (bfloat16)"
    }
}

# Use the precision schedule
train_with_precision_schedule $my_model $training_data 20
```

#### Performance vs Accuracy Benchmarking
```tcl
proc benchmark_precision_types {model data} {
    set precisions [list "float16" "bfloat16" "float32"]
    set results [dict create]
    
    foreach precision $precisions {
        puts "Benchmarking with $precision..."
        
        # Set the precision
        torch::autocast_set_dtype $precision cuda
        
        # Measure performance
        set start_time [clock clicks -milliseconds]
        set output [$model $data]
        set end_time [clock clicks -milliseconds]
        set time_ms [expr {$end_time - $start_time}]
        
        # Store results
        dict set results $precision time_ms $time_ms
        dict set results $precision output $output
        
        puts "  $precision: ${time_ms}ms"
    }
    
    return $results
}

# Enable autocast and benchmark
torch::autocast_enable cuda float16
set benchmark_results [benchmark_precision_types $my_model $test_data]

# Analyze results
dict for {precision data} $benchmark_results {
    set time [dict get $data time_ms]
    puts "$precision precision: ${time}ms"
}
```

#### Adaptive Precision Based on Loss Stability
```tcl
proc adaptive_precision_training {model data} {
    set losses [list]
    set current_precision "float16"
    
    # Start with aggressive mixed precision
    torch::autocast_enable cuda
    torch::autocast_set_dtype $current_precision cuda
    
    for {set epoch 0} {$epoch < 100} {incr epoch} {
        set loss [train_epoch $model $data]
        lappend losses $loss
        
        # Check loss stability every 5 epochs
        if {$epoch > 5 && $epoch % 5 == 0} {
            set recent_losses [lrange $losses end-4 end]
            set loss_variance [calculate_variance $recent_losses]
            
            if {$loss_variance > 0.1 && $current_precision eq "float16"} {
                puts "Loss unstable, switching to bfloat16"
                torch::autocast_set_dtype bfloat16 cuda
                set current_precision "bfloat16"
            } elseif {$loss_variance > 0.05 && $current_precision eq "bfloat16"} {
                puts "Loss still unstable, switching to float32"
                torch::autocast_set_dtype float32 cuda
                set current_precision "float32"
            }
        }
        
        puts "Epoch $epoch: Loss = $loss (precision: $current_precision)"
    }
}
```

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
torch::autocast_set_dtype float16
torch::autocast_set_dtype bfloat16 cuda
torch::autocast_set_dtype float32 cpu
```

**New (Named):**
```tcl
torch::autocast_set_dtype -dtype float16
torch::autocast_set_dtype -dtype bfloat16 -device_type cuda  
torch::autocast_set_dtype -dtype float32 -device_type cpu
```

### Advantages of Named Parameters

1. **Self-documenting**: Parameter names make the code more readable
2. **Order independent**: Parameters can be specified in any order
3. **Explicit intent**: Clear what dtype and device are being set
4. **Future-proof**: Easier to extend with additional parameters
5. **Consistency**: Matches other autocast commands' parameter style

## Error Handling

### Missing Required Dtype Parameter
```tcl
# This will fail
torch::autocast_set_dtype
# Error: Missing required dtype parameter

# This will also fail (named syntax without dtype)
torch::autocast_set_dtype -device_type cuda
# Error: Missing required dtype parameter. Valid dtypes: float16, bfloat16, float32
```

### Invalid Data Type
```tcl
# This will fail
torch::autocast_set_dtype invalid_dtype
# Error: Invalid parameters. Device type: cuda or cpu. Dtype: float16, bfloat16, or float32
```

### Invalid Device Type
```tcl
# This will fail  
torch::autocast_set_dtype float16 invalid_device
# Error: Invalid parameters. Device type: cuda or cpu. Dtype: float16, bfloat16, or float32
```

### Unknown Parameter
```tcl
# This will fail
torch::autocast_set_dtype -invalid_param float16
# Error: Unknown parameter: -invalid_param. Valid parameters are: -dtype, -data_type, -device_type, -device
```

### Missing Parameter Value
```tcl
# This will fail
torch::autocast_set_dtype -dtype
# Error: Missing value for parameter
```

### Too Many Positional Arguments
```tcl
# This will fail
torch::autocast_set_dtype float16 cuda extra_arg
# Error: Usage: torch::autocast_set_dtype dtype [device_type]
```

## Integration with Other Autocast Commands

### Complete Autocast Workflow
```tcl
# 1. Enable autocast with initial dtype
torch::autocast_enable cuda float16
set enabled [torch::autocast_is_enabled cuda]
puts "Autocast enabled: $enabled"

# 2. Change dtype dynamically
torch::autocast_set_dtype bfloat16 cuda
puts "Changed to bfloat16"

# 3. Verify autocast is still enabled
set still_enabled [torch::autocast_is_enabled cuda]
puts "Still enabled after dtype change: $still_enabled"

# 4. Change dtype again
torch::autocast_set_dtype -dtype float32 -device cuda
puts "Changed to float32"

# 5. Disable when done
torch::autocast_disable cuda
```

### Multi-Device Management
```tcl
# Enable autocast for both devices with different dtypes
torch::autocast_enable cuda float16
torch::autocast_enable cpu bfloat16

# Change CUDA dtype without affecting CPU
torch::autocast_set_dtype -dtype bfloat16 -device cuda

# Change CPU dtype without affecting CUDA
torch::autocast_set_dtype -dtype float32 -device cpu

# Verify independent management
set cuda_enabled [torch::autocast_is_enabled cuda]
set cpu_enabled [torch::autocast_is_enabled cpu]
puts "CUDA enabled: $cuda_enabled, CPU enabled: $cpu_enabled"
```

## Use Cases

### 1. Training Schedule Optimization
Start with aggressive mixed precision, then switch to more stable precision for convergence.

### 2. Model Architecture Adaptation
Different model layers may benefit from different precision levels.

### 3. Dataset-Specific Tuning
Adjust precision based on data characteristics and numerical stability requirements.

### 4. Performance Benchmarking
Compare training speed and accuracy across different precision levels.

### 5. Dynamic Stability Control
Automatically adjust precision based on loss stability or gradient behavior.

## Performance Notes

- **Very Fast Operation**: Setting autocast dtype is extremely fast (sub-millisecond)
- **No Overhead**: Does not affect ongoing training performance
- **Immediate Effect**: Changes take effect for subsequent autocast operations
- **No State Loss**: Preserves autocast enabled/disabled state

## Technical Details

This command uses PyTorch's native autocast dtype setting functionality:
- For CUDA: `at::autocast::set_autocast_dtype(at::kCUDA, dtype)`
- For CPU: `at::autocast::set_autocast_dtype(at::kCPU, dtype)`
- Changes the data type used for future autocast operations
- Does not affect the enabled/disabled state of autocast

## Data Type Characteristics

| Data Type | Bits | Exponent | Mantissa | Range | Precision | Use Case |
|-----------|------|----------|----------|-------|-----------|----------|
| float16 | 16 | 5 | 10 | ±65,504 | ~3 decimal digits | Maximum speed |
| bfloat16 | 16 | 8 | 7 | ±3.4×10³⁸ | ~2 decimal digits | Stable training |
| float32 | 32 | 8 | 23 | ±3.4×10³⁸ | ~7 decimal digits | Full precision |

## Related Commands

- [torch::autocast_enable](autocast_enable.md) - Enable automatic mixed precision
- [torch::autocast_disable](autocast_disable.md) - Disable automatic mixed precision
- [torch::autocast_is_enabled](autocast_is_enabled.md) - Check if autocast is enabled

## Backward Compatibility

✅ **Full backward compatibility maintained**
- All existing positional syntax continues to work
- No breaking changes to existing code
- New named parameter syntax is purely additive

## Best Practices

### 1. Start Conservative
Begin with `float16` for maximum performance, then adjust based on stability.

### 2. Monitor Loss Stability
Watch for oscillating or exploding losses that might indicate precision issues.

### 3. Use bfloat16 for Stability
When `float16` causes instability, `bfloat16` often provides a good balance.

### 4. Device-Specific Tuning
Different devices may benefit from different precision settings.

### 5. Profile Performance
Measure actual performance impact rather than assuming faster dtype = faster training.

## Troubleshooting

### Loss Exploding or NaN
- Switch from `float16` to `bfloat16` or `float32`
- Use gradient clipping with mixed precision
- Check model architecture for numerical instabilities

### Slow Convergence
- Ensure autocast is properly enabled
- Verify dtype is appropriate for your model size
- Consider loss scaling for `float16`

### Memory Issues
- `float16` and `bfloat16` should reduce memory usage
- `float32` will use more memory than mixed precision
- Monitor GPU memory usage with different dtypes

## See Also

- [PyTorch Automatic Mixed Precision Documentation](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA Automatic Mixed Precision Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- [Mixed Precision Training Paper](https://arxiv.org/abs/1710.03740) 