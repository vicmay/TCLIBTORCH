# torch::autocast_enable

**Enable automatic mixed precision (AMP) for PyTorch operations**

## Overview

The `torch::autocast_enable` command enables automatic mixed precision (AMP) for PyTorch operations on the specified device. AMP automatically casts operations to lower precision data types (like float16 or bfloat16) when safe to do so, which can significantly improve performance and reduce memory usage while maintaining numerical accuracy.

## Syntax

### Original Syntax (Positional Parameters)
```tcl
torch::autocast_enable [device_type] [dtype]
```

### New Syntax (Named Parameters)  
```tcl
torch::autocast_enable [-device_type <device>] [-device <device>] [-dtype <dtype>] [-data_type <dtype>]
```

### camelCase Alias
```tcl
torch::autocastEnable [device_type] [dtype]
torch::autocastEnable [-device_type <device>] [-device <device>] [-dtype <dtype>] [-data_type <dtype>]
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| device_type | string | No | "cuda" | Device to enable autocast for ("cuda" or "cpu") |
| dtype | string | No | "float16" | Data type for mixed precision ("float16", "bfloat16", or "float32") |

### Named Parameter Aliases

| Long Form | Short Form | Description |
|-----------|------------|-------------|
| `-device_type` | `-device` | Device to enable autocast for |
| `-dtype` | `-data_type` | Data type for mixed precision |

## Returns

Returns the string `"autocast enabled"` on success.

## Device Types

- **cuda**: Enable autocast for CUDA operations (GPU)
- **cpu**: Enable autocast for CPU operations

## Data Types

- **float16**: Half precision floating point (16-bit) - fastest, lowest memory, good for most cases
- **bfloat16**: Brain floating point (16-bit) - more stable than float16, good numerical properties
- **float32**: Single precision floating point (32-bit) - highest precision, disables mixed precision benefits

## Examples

### Basic Usage

```tcl
# Enable with defaults (CUDA, float16)
torch::autocast_enable

# Enable for CUDA with float16
torch::autocast_enable cuda

# Enable for CPU with bfloat16
torch::autocast_enable cpu bfloat16
```

### Named Parameter Syntax

```tcl
# Enable for CUDA with float16
torch::autocast_enable -device_type cuda -dtype float16

# Enable for CPU with bfloat16
torch::autocast_enable -device cpu -dtype bfloat16

# Using short parameter names
torch::autocast_enable -device cuda -data_type float16
```

### camelCase Alias

```tcl
# Using camelCase with positional parameters
torch::autocastEnable cuda float16

# Using camelCase with named parameters
torch::autocastEnable -device_type cpu -dtype bfloat16
```

### Complete Workflow Example

```tcl
# Enable autocast for training
torch::autocast_enable cuda float16

# Create model and data
set model [torch::sequential [list \
    [torch::linear -in_features 784 -out_features 128] \
    [torch::relu] \
    [torch::linear -in_features 128 -out_features 10]]]

set input [torch::randn -size [list 32 784] -dtype float32]
set target [torch::randint -high 10 -size [list 32]]

# Forward pass with autocast (automatic precision conversion)
set output [$model $input]
set loss [torch::cross_entropy_loss $output $target]

# Check if autocast is enabled
set autocast_status [torch::autocast_is_enabled cuda]
puts "Autocast enabled: $autocast_status"

# Disable when done (optional)
torch::autocast_disable cuda
```

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
torch::autocast_enable cuda float16
torch::autocast_enable cpu bfloat16
```

**New (Named):**
```tcl
torch::autocast_enable -device_type cuda -dtype float16
torch::autocast_enable -device_type cpu -dtype bfloat16
```

### Advantages of Named Parameters

1. **Self-documenting**: Parameter names make code more readable
2. **Order independent**: Parameters can be specified in any order
3. **Optional parameters**: Easy to specify only needed parameters
4. **Less error-prone**: Parameter names prevent argument misplacement

## Error Handling

### Invalid Device Type
```tcl
# This will fail
torch::autocast_enable invalid_device
# Error: Invalid parameters. Device type: cuda or cpu. Dtype: float16, bfloat16, or float32
```

### Invalid Data Type
```tcl
# This will fail
torch::autocast_enable -device cuda -dtype invalid_dtype
# Error: Invalid parameters. Device type: cuda or cpu. Dtype: float16, bfloat16, or float32
```

### Unknown Parameter
```tcl
# This will fail
torch::autocast_enable -invalid_param cuda
# Error: Unknown parameter: -invalid_param. Valid parameters are: -device_type, -device, -dtype, -data_type
```

### Missing Parameter Value
```tcl
# This will fail
torch::autocast_enable -device_type
# Error: Missing value for parameter
```

### Too Many Positional Arguments
```tcl
# This will fail
torch::autocast_enable cuda float16 extra_arg
# Error: Usage: torch::autocast_enable [device_type] [dtype]
```

## Performance Notes

- **GPU Performance**: float16 typically provides 1.5-2x speedup on modern GPUs
- **Memory Usage**: float16 reduces memory usage by approximately 50%
- **Numerical Stability**: bfloat16 provides better numerical stability than float16
- **CPU Performance**: Mixed precision on CPU may have limited benefits depending on hardware

## Related Commands

- [torch::autocast_disable](autocast_disable.md) - Disable automatic mixed precision
- [torch::autocast_is_enabled](autocast_is_enabled.md) - Check if autocast is enabled
- [torch::autocast_set_dtype](autocast_set_dtype.md) - Set autocast data type

## Technical Details

This command uses PyTorch's native autocast functionality:
- For CUDA: `at::autocast::set_autocast_enabled(at::kCUDA, true)`
- For CPU: `at::autocast::set_autocast_enabled(at::kCPU, true)`
- Sets the specified data type for autocast operations

## Backward Compatibility

✅ **Full backward compatibility maintained**
- All existing positional syntax continues to work
- No breaking changes to existing code
- New named parameter syntax is purely additive

## See Also

- [PyTorch Automatic Mixed Precision Documentation](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA Automatic Mixed Precision Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) 