# torch::autocast_disable / torch::autocastDisable

Disables automatic mixed precision (AMP) autocast for a specified device type.

## Syntax

### New Named Parameter Syntax (Recommended)
```tcl
torch::autocast_disable -device_type <device_type>
torch::autocast_disable -device <device_type>
torch::autocastDisable -device_type <device_type>
torch::autocastDisable -device <device_type>
```

### Legacy Positional Syntax (Backward Compatibility)
```tcl
torch::autocast_disable [device_type]
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-device_type` | string | No | "cuda" | Device type for autocast ("cuda" or "cpu") |
| `-device` | string | No | "cuda" | Alternative name for device_type (same as `-device_type`) |

### Legacy Positional Parameters
1. `device_type` (optional): Device type for autocast ("cuda" or "cpu"), defaults to "cuda"

## Description

The `torch::autocast_disable` command disables automatic mixed precision (AMP) autocast for the specified device type. When autocast is disabled, operations will use their default precision (typically float32) instead of automatically selecting lower precision data types for performance optimization.

**Supported Device Types:**
- **"cuda"**: Disables autocast for CUDA GPU operations
- **"cpu"**: Disables autocast for CPU operations

This command is typically used when you want to ensure full precision for critical computations or when debugging numerical stability issues that might be related to automatic precision reduction.

## Return Value

Returns the string "autocast disabled" upon successful completion.

## Examples

### Basic Usage
```tcl
# Named parameter syntax (recommended)
torch::autocast_disable -device_type cuda       # Disable CUDA autocast
torch::autocast_disable -device cpu             # Disable CPU autocast
torch::autocast_disable -device_type cuda       # Alternative parameter name

# camelCase alias
torch::autocastDisable -device cuda
torch::autocastDisable -device_type cpu

# Legacy positional syntax
torch::autocast_disable                         # Default: disable CUDA autocast
torch::autocast_disable cuda                    # Disable CUDA autocast
torch::autocast_disable cpu                     # Disable CPU autocast
```

### Working with Mixed Precision Training
```tcl
# Training setup with autocast control
proc train_epoch {model data} {
    # Enable autocast for forward pass (for performance)
    torch::autocast_enable cuda
    
    # Forward pass with autocast
    set outputs [model_forward $model $data]
    set loss [compute_loss $outputs $targets]
    
    # Disable autocast for backward pass (for numerical stability)
    torch::autocast_disable -device_type cuda
    
    # Backward pass in full precision
    torch::tensor_backward $loss
    
    return $loss
}
```

### Checking Autocast Status
```tcl
# Check current autocast status
set cuda_enabled [torch::autocast_is_enabled cuda]
set cpu_enabled [torch::autocast_is_enabled cpu]

puts "CUDA autocast: $cuda_enabled"
puts "CPU autocast: $cpu_enabled"

# Disable autocast for both devices
torch::autocast_disable -device cuda
torch::autocast_disable -device cpu

# Verify autocast is disabled
set cuda_after [torch::autocast_is_enabled cuda]
set cpu_after [torch::autocast_is_enabled cpu]

puts "CUDA autocast after disable: $cuda_after"   # Should be 0 (false)
puts "CPU autocast after disable: $cpu_after"     # Should be 0 (false)
```

### Different Device Types
```tcl
# Disable autocast for specific devices
torch::autocast_disable -device_type cuda       # GPU only
torch::autocast_disable -device_type cpu        # CPU only

# Or disable for both
torch::autocast_disable -device cuda
torch::autocast_disable -device cpu
```

### Parameter Order Independence
```tcl
# These are equivalent (only one parameter)
torch::autocast_disable -device_type cuda
torch::autocast_disable -device cuda
```

### Integration with Model Training
```tcl
# Precision-sensitive operations
proc critical_computation {tensor} {
    # Ensure full precision for critical math
    torch::autocast_disable -device_type cuda
    
    # Perform sensitive computation
    set result [torch::tensor_svd $tensor]
    
    # Can re-enable autocast later if needed
    # torch::autocast_enable cuda
    
    return $result
}
```

### Debugging Numerical Issues
```tcl
# Compare results with and without autocast
proc debug_precision {input} {
    # First run with autocast enabled
    torch::autocast_enable cuda
    set result_autocast [some_operation $input]
    
    # Then run with autocast disabled for comparison
    torch::autocast_disable -device cuda
    set result_full_precision [some_operation $input]
    
    # Compare for numerical differences
    set diff [torch::tensor_sub $result_autocast $result_full_precision]
    set max_diff [torch::tensor_max [torch::tensor_abs $diff]]
    
    puts "Max difference between autocast and full precision: $max_diff"
    
    return [list $result_autocast $result_full_precision]
}
```

## Error Handling

```tcl
# Invalid device type
if {[catch {torch::autocast_disable -device_type invalid} error]} {
    puts "Error: $error"
    # Output: Invalid device type. Use cuda or cpu
}

# Unknown parameter
if {[catch {torch::autocast_disable -unknown_param cuda} error]} {
    puts "Error: $error"
    # Output: Unknown parameter: -unknown_param. Valid parameters are: -device_type, -device
}

# Missing value for parameter
if {[catch {torch::autocast_disable -device_type} error]} {
    puts "Error: $error"
    # Output: Missing value for parameter
}
```

## Migration Guide

### From Legacy Syntax
```tcl
# Old way (still supported)
torch::autocast_disable
torch::autocast_disable cuda
torch::autocast_disable cpu

# New way (recommended)
torch::autocast_disable -device_type cuda
torch::autocast_disable -device cpu

# camelCase alias (modern style)
torch::autocastDisable -device cuda
torch::autocastDisable -device_type cpu
```

### Benefits of Named Parameters
- **Self-documenting**: Parameter names make device type explicit
- **Flexible**: Both `-device_type` and `-device` parameter names supported
- **Consistent**: Matches other AMP-related commands
- **Future-proof**: Easy to extend with additional parameters

## Automatic Mixed Precision (AMP) Overview

Autocast automatically selects the appropriate data type for each operation to optimize performance while maintaining numerical accuracy:

- **float16/bfloat16**: Used for operations that can tolerate lower precision
- **float32**: Used for operations requiring higher precision
- **Automatic promotion**: Operations that mix precisions are handled appropriately

Disabling autocast forces all operations to use their default precision (typically float32).

## Performance Considerations

- **Training Speed**: Disabling autocast may reduce training speed but improve numerical stability
- **Memory Usage**: Full precision operations use more GPU memory
- **Numerical Accuracy**: Disabling autocast can resolve precision-related issues
- **Selective Disable**: Consider disabling only for specific operations rather than globally

## Common Use Cases

### Critical Computations
```tcl
# Matrix decompositions often need full precision
torch::autocast_disable -device cuda
set [U S V] [torch::tensor_svd $matrix]
```

### Loss Computation
```tcl
# Some loss functions are sensitive to precision
torch::autocast_disable -device_type cuda
set loss [torch::cross_entropy_loss $logits $targets]
```

### Numerical Debugging
```tcl
# Debug NaN or inf issues
torch::autocast_disable -device cuda
# Run problematic code with full precision
```

### Gradient Computations
```tcl
# Ensure gradient accuracy
torch::autocast_disable -device_type cuda
torch::tensor_backward $loss
```

## Implementation Details

- **Backward Compatibility**: 100% compatible with existing code using positional syntax
- **Dual Syntax Support**: Automatically detects whether named or positional parameters are used
- **Parameter Validation**: Comprehensive validation for both syntaxes
- **Error Messages**: Clear, helpful error messages for both syntaxes
- **Device Support**: Supports both CUDA and CPU devices

## See Also

- [torch::autocast_enable](autocast_enable.md) - Enable automatic mixed precision
- [torch::autocast_is_enabled](autocast_is_enabled.md) - Check autocast status
- [torch::autocast_set_dtype](autocast_set_dtype.md) - Set autocast data type
- [torch::grad_scaler_new](grad_scaler_new.md) - Create gradient scaler for AMP training

## Status

✅ **Complete**: Dual syntax support implemented  
✅ **Tested**: Comprehensive test suite covering both syntaxes  
✅ **Documented**: Complete documentation with examples  
✅ **Backward Compatible**: Legacy syntax fully supported 