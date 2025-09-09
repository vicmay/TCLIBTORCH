# torch::randn

Generate a tensor filled with random numbers from a standard normal distribution (mean=0, std=1).

## Syntax

### Positional Parameters (Legacy)
```tcl
torch::randn shape ?device? ?dtype?
```

### Named Parameters (Recommended)
```tcl
torch::randn -shape shape ?-device device? ?-dtype dtype?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `shape` | list | Yes | - | Shape of the tensor as a list of integers |
| `device` | string | No | "cpu" | Device to place the tensor on (cpu, cuda, cuda:0, etc.) |
| `dtype` | string | No | "float32" | Data type of the tensor (float32, float64, etc.) |

## Returns

**string** - Handle to the created tensor

## Description

The `torch::randn` command creates a new tensor filled with random values drawn from a standard normal (Gaussian) distribution with mean 0 and standard deviation 1. Each element in the tensor is independently sampled from this distribution.

This is useful for:
- Initializing neural network weights (especially with proper scaling)
- Statistical sampling and simulations
- Adding noise to tensors
- Monte Carlo methods requiring normal distributions

## Statistical Properties

- **Mean**: 0 (expected value)
- **Standard Deviation**: 1
- **Variance**: 1
- **Range**: Theoretically (-∞, +∞), practically most values fall within [-3, 3]

## Examples

### Basic Usage

```tcl
# Create a 3x3 normal random tensor (positional syntax)
set tensor [torch::randn {3 3}]

# Create a 3x3 normal random tensor (named parameters)
set tensor [torch::randn -shape {3 3}]
```

### Specifying Device and Data Type

```tcl
# Create tensor on CPU with float32 (positional)
set tensor [torch::randn {2 4} cpu float32]

# Create tensor with float64 precision (named parameters)
set tensor [torch::randn -shape {2 4} -dtype float64]

# Create tensor on GPU if available (named parameters)
set tensor [torch::randn -shape {5 5} -device cuda:0 -dtype float32]
```

### Different Tensor Shapes

```tcl
# Scalar tensor (0-dimensional)
set scalar [torch::randn {}]

# 1D tensor (vector)
set vector [torch::randn {10}]

# 2D tensor (matrix)
set matrix [torch::randn {3 4}]

# 3D tensor
set tensor_3d [torch::randn {2 3 4}]

# 4D tensor (batch of 3D tensors)
set tensor_4d [torch::randn {8 3 32 32}]
```

### Parameter Order Flexibility (Named Syntax)

```tcl
# These are equivalent
set t1 [torch::randn -shape {2 2} -device cpu -dtype float32]
set t2 [torch::randn -dtype float32 -shape {2 2} -device cpu]
set t3 [torch::randn -device cpu -dtype float32 -shape {2 2}]
```

## Practical Examples

### Neural Network Weight Initialization

```tcl
# Xavier/Glorot normal initialization for linear layer
# weights = randn * sqrt(2.0 / (fan_in + fan_out))
set fan_in 784
set fan_out 128
set weights [torch::randn {$fan_out $fan_in}]
set scale [expr {sqrt(2.0 / ($fan_in + $fan_out))}]
set scaled_weights [torch::tensor_mul $weights $scale]

# He normal initialization for ReLU networks
# weights = randn * sqrt(2.0 / fan_in)
set conv_weights [torch::randn {32 3 3 3}]  ;# 32 filters, 3 channels, 3x3 kernel
set he_scale [expr {sqrt(2.0 / (3 * 3 * 3))}]
set he_weights [torch::tensor_mul $conv_weights $he_scale]
```

### Adding Gaussian Noise

```tcl
# Add noise to existing tensor
set clean_data [torch::ones {100 50}]
set noise [torch::randn {100 50}]
set noise_scale 0.1
set scaled_noise [torch::tensor_mul $noise $noise_scale]
set noisy_data [torch::tensor_add $clean_data $scaled_noise]
```

### Statistical Sampling

```tcl
# Generate samples from normal distribution with custom mean and std
set mean 5.0
set std 2.0
set samples [torch::randn {1000}]
set scaled_samples [torch::tensor_mul $samples $std]
set final_samples [torch::tensor_add $scaled_samples $mean]

# Multi-dimensional normal sampling
set points [torch::randn {1000 3}]  ;# 1000 random 3D points from standard normal
```

### Monte Carlo Simulation

```tcl
# Simulate Brownian motion
set n_steps 1000
set dt 0.01
set random_steps [torch::randn {$n_steps}]
set step_size [expr {sqrt($dt)}]
set scaled_steps [torch::tensor_mul $random_steps $step_size]

# Generate random portfolio returns (financial modeling)
set n_assets 10
set n_scenarios 10000
set returns [torch::randn {$n_scenarios $n_assets}]
```

## Distribution Comparison

### torch::randn vs torch::rand

```tcl
# Normal distribution (Gaussian)
set normal_data [torch::randn {1000}]

# Uniform distribution
set uniform_data [torch::rand {1000}]

# Normal: values mostly in [-3, 3], can be negative
# Uniform: values always in [0, 1)
```

| Command | Distribution | Mean | Std Dev | Range | Use Case |
|---------|-------------|------|---------|-------|----------|
| `torch::randn` | Normal (Gaussian) | 0 | 1 | (-∞, +∞) | Weight initialization, statistical sampling |
| `torch::rand` | Uniform | 0.5 | ~0.289 | [0, 1) | General random numbers, probabilities |

## Statistical Validation

```tcl
# Verify statistical properties
set large_sample [torch::randn {100000}]

# Check mean (should be close to 0)
set mean_tensor [torch::tensor_mean $large_sample]
set mean_val [torch::tensor_item $mean_tensor]
puts "Mean: $mean_val"  ;# Should be close to 0.0

# Check standard deviation (should be close to 1)
set std_tensor [torch::tensor_std $large_sample]
set std_val [torch::tensor_item $std_tensor]
puts "Std Dev: $std_val"  ;# Should be close to 1.0
```

## Error Handling

### Common Errors

```tcl
# Missing required shape parameter
catch {torch::randn} err
puts $err  ;# "Required parameter missing: shape"

# Invalid device (current implementation may not error)
catch {torch::randn {2 2} invalid_device} err

# Invalid data type (current implementation may not error)
catch {torch::randn {2 2} cpu invalid_dtype} err

# Missing value for named parameter
catch {torch::randn -shape} err
puts $err  ;# "Missing value for parameter"

# Unknown parameter
catch {torch::randn -shape {2 2} -unknown_param value} err
puts $err  ;# "Unknown parameter: -unknown_param"
```

## Tensor Properties

After creation, you can inspect the tensor properties:

```tcl
set tensor [torch::randn {3 4}]

# Check shape
puts [torch::tensor_shape $tensor]     ;# "3 4"

# Check data type
puts [torch::tensor_dtype $tensor]     ;# "Float"

# Check device
puts [torch::tensor_device $tensor]    ;# Shows device info

# Check number of elements
puts [torch::tensor_numel $tensor]     ;# "12"
```

## Performance Notes

- Normal random generation is slightly slower than uniform random generation
- GPU generation (CUDA) can be significantly faster for large tensors
- Consider data type precision (float32 vs float64) based on requirements
- For repeated sampling, consider generating larger batches

## Weight Initialization Patterns

### Common Initialization Schemes

```tcl
# Standard normal (mean=0, std=1)
set weights [torch::randn {out_features in_features}]

# Xavier/Glorot Normal
set xavier_scale [expr {sqrt(2.0 / ($in_features + $out_features))}]
set xavier_weights [torch::tensor_mul [torch::randn {$out_features $in_features}] $xavier_scale]

# He/Kaiming Normal (for ReLU networks)
set he_scale [expr {sqrt(2.0 / $in_features)}]
set he_weights [torch::tensor_mul [torch::randn {$out_features $in_features}] $he_scale]

# LeCun Normal
set lecun_scale [expr {sqrt(1.0 / $in_features)}]
set lecun_weights [torch::tensor_mul [torch::randn {$out_features $in_features}] $lecun_scale]
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old style (still supported)
set tensor [torch::randn {3 3} cuda:0 float64]

# New style (recommended)
set tensor [torch::randn -shape {3 3} -device cuda:0 -dtype float64]
```

### Benefits of Named Parameters
- **Clarity**: Parameter names make code self-documenting
- **Flexibility**: Parameters can be specified in any order
- **Maintainability**: Easier to modify and understand code
- **Error Prevention**: Less likely to mix up parameter positions

## Mathematical Background

The standard normal distribution has probability density function:

```
f(x) = (1/√(2π)) * e^(-x²/2)
```

Where:
- Mean (μ) = 0
- Variance (σ²) = 1
- Standard deviation (σ) = 1

## See Also

- [`torch::rand`](rand.md) - Uniform distribution random tensors
- [`torch::randint`](randint.md) - Random integer tensors
- [`torch::randn_like`](randn_like.md) - Normal random tensor with same shape as input
- [`torch::zeros`](zeros.md) - Zero-filled tensors
- [`torch::ones`](ones.md) - One-filled tensors 