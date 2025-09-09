# torch::grad_scaler_get_scale

Get the current scale value from a gradient scaler used for automatic mixed precision training.

## Syntax

```tcl
# Positional syntax (backward compatibility)
torch::grad_scaler_get_scale scaler

# Named parameter syntax
torch::grad_scaler_get_scale -scaler scaler
torch::grad_scaler_get_scale -gradscaler scaler

# CamelCase alias
torch::gradScalerGetScale scaler
torch::gradScalerGetScale -scaler scaler
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `scaler` | string | Gradient scaler handle | Required |

### Parameter Aliases

- `-scaler` and `-gradscaler` are equivalent

## Description

The `torch::grad_scaler_get_scale` command retrieves the current scale value from a gradient scaler. The gradient scaler is used in automatic mixed precision (AMP) training to scale gradients to prevent underflow in half-precision computations.

The scale value is dynamically adjusted during training:
- It increases when no infinite gradients are detected (successful training steps)
- It decreases when infinite gradients are detected (overflow situations)

This command is essential for monitoring the scale behavior during training and debugging gradient scaling issues.

## Return Value

Returns the current scale value as a floating-point number.

## Examples

### Basic Usage

```tcl
# Create a gradient scaler with default initial scale (65536.0)
set scaler [torch::grad_scaler_new]

# Get scale using positional syntax
set scale1 [torch::grad_scaler_get_scale $scaler]
puts "Current scale: $scale1"  ;# Current scale: 65536.0

# Get scale using named parameter syntax
set scale2 [torch::grad_scaler_get_scale -scaler $scaler]
puts "Current scale: $scale2"  ;# Current scale: 65536.0
```

### Using CamelCase Alias

```tcl
# CamelCase alias
set scale3 [torch::gradScalerGetScale $scaler]
set scale4 [torch::gradScalerGetScale -scaler $scaler]

puts "Scale values: $scale3, $scale4"  ;# Scale values: 65536.0, 65536.0
```

### Custom Initial Scale

```tcl
# Create scaler with custom initial scale
set custom_scaler [torch::grad_scaler_new 1024.0]

# Get the custom scale value
set custom_scale [torch::grad_scaler_get_scale $custom_scaler]
puts "Custom scale: $custom_scale"  ;# Custom scale: 1024.0
```

### Parameter Aliases

```tcl
# Both parameters are equivalent
set scale_a [torch::grad_scaler_get_scale -scaler $scaler]
set scale_b [torch::grad_scaler_get_scale -gradscaler $scaler]

puts "Equivalent results: [expr {$scale_a == $scale_b}]"  ;# Equivalent results: 1
```

### Monitoring Scale Changes

```tcl
# Create scaler and monitor scale changes
set scaler [torch::grad_scaler_new 2048.0]

# Initial scale
set initial_scale [torch::grad_scaler_get_scale $scaler]
puts "Initial scale: $initial_scale"

# Update scaler (may change scale based on gradient status)
torch::grad_scaler_update $scaler

# Check scale after update
set updated_scale [torch::grad_scaler_get_scale $scaler]
puts "Updated scale: $updated_scale"
```

### Multiple Scalers

```tcl
# Create multiple scalers with different initial scales
set scaler1 [torch::grad_scaler_new 1000.0]
set scaler2 [torch::grad_scaler_new 2000.0]
set scaler3 [torch::grad_scaler_new 4000.0]

# Get scales from all scalers
set scale1 [torch::grad_scaler_get_scale $scaler1]
set scale2 [torch::grad_scaler_get_scale $scaler2]
set scale3 [torch::grad_scaler_get_scale $scaler3]

puts "Scales: $scale1, $scale2, $scale3"  ;# Scales: 1000.0, 2000.0, 4000.0
```

## Error Handling

The command performs comprehensive error checking:

```tcl
# Missing scaler parameter
torch::grad_scaler_get_scale  ;# Error: Usage: torch::grad_scaler_get_scale scaler

# Invalid scaler handle
torch::grad_scaler_get_scale "invalid_scaler"  ;# Error: Gradient scaler not found

# Unknown parameter
torch::grad_scaler_get_scale -scaler $scaler -unknown "value"  ;# Error: Unknown parameter: -unknown

# Missing parameter value
torch::grad_scaler_get_scale -scaler  ;# Error: Named parameters must come in pairs
```

## Backward Compatibility

The command maintains 100% backward compatibility:

```tcl
# These all work and produce identical results
set scaler [torch::grad_scaler_new 8192.0]

set result1 [torch::grad_scaler_get_scale $scaler]
set result2 [torch::grad_scaler_get_scale -scaler $scaler]
set result3 [torch::gradScalerGetScale $scaler]

puts "All equal: [expr {$result1 == $result2 && $result2 == $result3}]"  ;# All equal: 1
```

## Implementation Notes

- The scale value is stored internally as a `torch::Tensor` but returned as a double-precision floating-point number
- Scale values can range from very small (e.g., 1e-10) to very large (e.g., 1e+10) numbers
- The default initial scale is 65536.0, which is commonly used in mixed precision training
- Scale values are automatically adjusted during training based on gradient overflow detection
- Multiple gradient scalers can exist simultaneously with independent scale values

## Use Cases

### Training Monitoring

```tcl
# Monitor scale during training
proc log_scale {scaler epoch} {
    set scale [torch::grad_scaler_get_scale $scaler]
    puts "Epoch $epoch: Scale = $scale"
}

set scaler [torch::grad_scaler_new]
for {set epoch 1} {$epoch <= 10} {incr epoch} {
    # ... training code ...
    log_scale $scaler $epoch
    torch::grad_scaler_update $scaler
}
```

### Scale Validation

```tcl
# Validate scale is within expected range
proc validate_scale {scaler min_scale max_scale} {
    set current_scale [torch::grad_scaler_get_scale $scaler]
    
    if {$current_scale < $min_scale} {
        puts "Warning: Scale too low ($current_scale < $min_scale)"
        return false
    } elseif {$current_scale > $max_scale} {
        puts "Warning: Scale too high ($current_scale > $max_scale)"
        return false
    }
    
    return true
}
```

### Debugging Scale Issues

```tcl
# Debug scale behavior
proc debug_scaler {scaler} {
    set scale [torch::grad_scaler_get_scale $scaler]
    
    if {$scale < 1.0} {
        puts "Scale very low: $scale - possible gradient overflow issues"
    } elseif {$scale > 100000.0} {
        puts "Scale very high: $scale - gradients may be too small"
    } else {
        puts "Scale normal: $scale"
    }
}
```

## Return Value Precision

The command returns scale values with full floating-point precision:

```tcl
# Fractional scales are preserved
set scaler [torch::grad_scaler_new 128.25]
set scale [torch::grad_scaler_get_scale $scaler]
puts "Precise scale: $scale"  ;# Precise scale: 128.25

# Very small scales
set tiny_scaler [torch::grad_scaler_new 1e-8]
set tiny_scale [torch::grad_scaler_get_scale $tiny_scaler]
puts "Tiny scale: $tiny_scale"  ;# Tiny scale: 1e-08

# Very large scales
set huge_scaler [torch::grad_scaler_new 1e8]
set huge_scale [torch::grad_scaler_get_scale $huge_scaler]
puts "Huge scale: $huge_scale"  ;# Huge scale: 1e+08
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set scale [torch::grad_scaler_get_scale $scaler]

# New named parameter syntax
set scale [torch::grad_scaler_get_scale -scaler $scaler]

# Both syntaxes supported - no breaking changes
```

### Parameter Aliases

```tcl
# Multiple ways to specify the scaler parameter
torch::grad_scaler_get_scale -scaler $scaler
torch::grad_scaler_get_scale -gradscaler $scaler
```

## See Also

- [torch::grad_scaler_new](grad_scaler_new.md) - Create gradient scaler
- [torch::grad_scaler_scale](grad_scaler_scale.md) - Scale tensors/gradients
- [torch::grad_scaler_step](grad_scaler_step.md) - Perform optimizer step with scaling
- [torch::grad_scaler_update](grad_scaler_update.md) - Update scaler based on gradient status
- [torch::autocast_enable](autocast_enable.md) - Enable automatic mixed precision

## Technical Details

### Scale Value Storage

The scale value is internally stored as a PyTorch tensor:
```cpp
torch::Tensor scale = torch::tensor(init_scale, torch::kFloat32);
```

### Scale Value Access

The scale is accessed using the tensor's item method:
```cpp
double get_scale() const { 
    return scale.item<double>(); 
}
```

### Precision Considerations

- Scale values are stored as single-precision (float32) tensors internally
- Returned values are converted to double-precision for TCL compatibility
- Very small or very large values may experience floating-point precision limitations
- Scale values of exactly 0.0 are not recommended as they would eliminate all gradients 