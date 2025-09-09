# torch::grad_scaler_new

Creates a gradient scaler for automatic mixed precision (AMP) training.

## Syntax

### Modern Syntax (Named Parameters)
```tcl
torch::grad_scaler_new ?-initScale init_scale? ?-growthFactor growth_factor? ?-backoffFactor backoff_factor? ?-growthInterval growth_interval?
torch::gradScalerNew ?-initScale init_scale? ?-growthFactor growth_factor? ?-backoffFactor backoff_factor? ?-growthInterval growth_interval?
```

### Legacy Syntax (Positional Parameters)
```tcl
torch::grad_scaler_new ?init_scale? ?growth_factor? ?backoff_factor? ?growth_interval?
torch::gradScalerNew ?init_scale? ?growth_factor? ?backoff_factor? ?growth_interval?
```

## Parameters

### Named Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-initScale` | double | 65536.0 | Initial scale value for gradient scaling |
| `-growthFactor` | double | 2.0 | Factor to multiply scale by when no overflow detected |
| `-backoffFactor` | double | 0.5 | Factor to multiply scale by when overflow detected |
| `-growthInterval` | int | 2000 | Number of iterations before attempting to increase scale |

### Parameter Aliases

Both camelCase and snake_case parameter names are supported:
- `-initScale` / `-init_scale`
- `-growthFactor` / `-growth_factor`
- `-backoffFactor` / `-backoff_factor`
- `-growthInterval` / `-growth_interval`

## Return Value

Returns a gradient scaler handle (string) that can be used with other gradient scaling functions.

## Description

The gradient scaler is essential for automatic mixed precision training. It:

1. **Scales gradients** to prevent underflow when using reduced precision (float16)
2. **Automatically adjusts scale** based on overflow detection
3. **Provides stable training** by maintaining appropriate gradient magnitudes
4. **Optimizes memory usage** by enabling mixed precision computation

### How It Works

1. **Initialize**: Creates a scaler with specified initial scale value
2. **Scale gradients**: Multiplies gradients by scale factor before backpropagation
3. **Check for overflow**: Detects infinite or NaN gradients
4. **Adjust scale**: Increases scale when stable, decreases when overflow detected
5. **Update optimizer**: Only steps optimizer when gradients are finite

## Examples

### Basic Usage

```tcl
# Create scaler with default parameters
set scaler [torch::grad_scaler_new]

# Create scaler with custom initial scale
set scaler [torch::grad_scaler_new -initScale 1024.0]

# Create scaler with multiple parameters
set scaler [torch::grad_scaler_new -initScale 2048.0 -growthFactor 2.5 -backoffFactor 0.25]
```

### Complete AMP Training Setup

```tcl
# Create model and optimizer
set model [torch::sequential]
set optimizer [torch::optimizer_adam ...]

# Create gradient scaler
set scaler [torch::grad_scaler_new -initScale 1024.0 -growthFactor 2.0 -backoffFactor 0.5 -growthInterval 1000]

# Training loop
for {set epoch 0} {$epoch < 10} {incr epoch} {
    # Forward pass with autocast
    torch::autocast_enable
    set output [torch::sequential_forward $model $input]
    set loss [torch::mse_loss $output $target]
    torch::autocast_disable
    
    # Backward pass with scaled gradients
    torch::tensor_backward $loss
    set scaled_loss [torch::grad_scaler_scale $scaler $loss]
    torch::tensor_backward $scaled_loss
    
    # Update parameters
    torch::grad_scaler_step $scaler $optimizer
    torch::grad_scaler_update $scaler
    torch::optimizer_zero_grad $optimizer
}
```

### Parameter Tuning

```tcl
# Conservative scaling (slower convergence, more stable)
set conservative_scaler [torch::grad_scaler_new -initScale 1024.0 -growthFactor 1.5 -backoffFactor 0.8 -growthInterval 3000]

# Aggressive scaling (faster convergence, less stable)
set aggressive_scaler [torch::grad_scaler_new -initScale 8192.0 -growthFactor 4.0 -backoffFactor 0.25 -growthInterval 500]

# Minimal scaling (for debugging)
set minimal_scaler [torch::grad_scaler_new -initScale 1.0 -growthFactor 1.0 -backoffFactor 1.0 -growthInterval 1]
```

### Using Different Syntax Styles

```tcl
# Modern named parameter syntax
set scaler1 [torch::grad_scaler_new -initScale 2048.0 -growthFactor 3.0]

# Legacy positional syntax (backward compatibility)
set scaler2 [torch::grad_scaler_new 2048.0 3.0]

# CamelCase alias
set scaler3 [torch::gradScalerNew -initScale 2048.0 -growthFactor 3.0]

# Snake case parameter names
set scaler4 [torch::grad_scaler_new -init_scale 2048.0 -growth_factor 3.0]
```

## Error Handling

The function validates all parameters and provides clear error messages:

```tcl
# Invalid parameters
catch {torch::grad_scaler_new -initScale -1.0} error
puts $error  ;# "Invalid parameters: all values must be positive"

# Unknown parameter
catch {torch::grad_scaler_new -invalidParam 1.0} error  
puts $error  ;# "Unknown parameter: -invalidParam"

# Incomplete parameter pairs
catch {torch::grad_scaler_new -initScale} error
puts $error  ;# "Named parameters must come in pairs"
```

## Integration with Other Functions

The gradient scaler works with these related functions:

- `torch::grad_scaler_scale` - Scale gradients before backpropagation
- `torch::grad_scaler_step` - Step optimizer with gradient scaling
- `torch::grad_scaler_update` - Update scale based on overflow detection
- `torch::grad_scaler_get_scale` - Get current scale value
- `torch::autocast_enable` - Enable mixed precision computation
- `torch::autocast_disable` - Disable mixed precision computation

## Performance Considerations

### Scale Value Selection

- **Too small**: May cause gradient underflow (vanishing gradients)
- **Too large**: May cause gradient overflow (exploding gradients)
- **Default (65536)**: Good starting point for most applications

### Growth Parameters

- **Higher growth_factor**: Faster scale increase, but more risk of overflow
- **Lower growth_factor**: Slower scale increase, more conservative
- **Higher growth_interval**: More iterations before scale increase
- **Lower growth_interval**: Faster scale adaptation

### Backoff Parameters

- **Higher backoff_factor**: Less aggressive scale reduction on overflow
- **Lower backoff_factor**: More aggressive scale reduction on overflow

## Common Use Cases

### 1. Standard Training
```tcl
set scaler [torch::grad_scaler_new]  ;# Use defaults
```

### 2. Stable Training (Conservative)
```tcl
set scaler [torch::grad_scaler_new -initScale 1024.0 -growthFactor 1.2 -backoffFactor 0.8 -growthInterval 5000]
```

### 3. Fast Training (Aggressive)
```tcl
set scaler [torch::grad_scaler_new -initScale 32768.0 -growthFactor 4.0 -backoffFactor 0.125 -growthInterval 200]
```

### 4. Debugging (Minimal Scaling)
```tcl
set scaler [torch::grad_scaler_new -initScale 1.0 -growthFactor 1.0 -backoffFactor 1.0 -growthInterval 1]
```

## Version History

- **v1.0**: Initial implementation with positional parameters
- **v2.0**: Added dual syntax support with named parameters and camelCase aliases

## See Also

- [torch::grad_scaler_scale](grad_scaler_scale.md) - Scale gradients
- [torch::grad_scaler_step](grad_scaler_step.md) - Step optimizer with scaling
- [torch::grad_scaler_update](grad_scaler_update.md) - Update gradient scaler
- [torch::grad_scaler_get_scale](grad_scaler_get_scale.md) - Get current scale
- [torch::autocast_enable](autocast_enable.md) - Enable mixed precision
- [Automatic Mixed Precision Training Guide](../guides/amp_training.md) 