# torch::rand

Generate a tensor filled with random numbers from a uniform distribution on the interval [0, 1).

## Syntax

### Positional Parameters (Legacy)
```tcl
torch::rand shape ?device? ?dtype?
```

### Named Parameters (Recommended)
```tcl
torch::rand -shape shape ?-device device? ?-dtype dtype?
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

The `torch::rand` command creates a new tensor filled with random values drawn from a uniform distribution on the interval [0, 1). Each element in the tensor is independently sampled from this distribution.

This is useful for:
- Initializing neural network weights
- Creating random test data
- Monte Carlo simulations
- Random sampling operations

## Examples

### Basic Usage

```tcl
# Create a 3x3 random tensor (positional syntax)
set tensor [torch::rand {3 3}]

# Create a 3x3 random tensor (named parameters)
set tensor [torch::rand -shape {3 3}]
```

### Specifying Device and Data Type

```tcl
# Create tensor on CPU with float32 (positional)
set tensor [torch::rand {2 4} cpu float32]

# Create tensor with float64 precision (named parameters)
set tensor [torch::rand -shape {2 4} -dtype float64]

# Create tensor on GPU if available (named parameters)
set tensor [torch::rand -shape {5 5} -device cuda:0 -dtype float32]
```

### Different Tensor Shapes

```tcl
# Scalar tensor (0-dimensional)
set scalar [torch::rand {}]

# 1D tensor (vector)
set vector [torch::rand {10}]

# 2D tensor (matrix)
set matrix [torch::rand {3 4}]

# 3D tensor
set tensor_3d [torch::rand {2 3 4}]

# 4D tensor (batch of 3D tensors)
set tensor_4d [torch::rand {8 3 32 32}]
```

### Parameter Order Flexibility (Named Syntax)

```tcl
# These are equivalent
set t1 [torch::rand -shape {2 2} -device cpu -dtype float32]
set t2 [torch::rand -dtype float32 -shape {2 2} -device cpu]
set t3 [torch::rand -device cpu -dtype float32 -shape {2 2}]
```

## Practical Examples

### Neural Network Weight Initialization

```tcl
# Initialize weights for a linear layer (input_size=784, output_size=128)
set weights [torch::rand {128 784}]

# Initialize convolutional layer weights (out_channels=32, in_channels=3, kernel=3x3)
set conv_weights [torch::rand {32 3 3 3}]
```

### Random Data Generation

```tcl
# Create random input data for testing
set batch_size 16
set input_dim 256
set test_data [torch::rand {$batch_size $input_dim}]

# Create random target labels (uniform distribution)
set targets [torch::rand {$batch_size 10}]
```

### Sampling and Monte Carlo

```tcl
# Generate random samples for Monte Carlo simulation
set n_samples 10000
set samples [torch::rand {$n_samples}]

# Multi-dimensional random sampling
set points [torch::rand {1000 3}]  ;# 1000 random 3D points
```

## Value Range

The `torch::rand` command generates values in the range **[0, 1)**, which means:
- Minimum possible value: 0.0 (inclusive)
- Maximum possible value: approaching 1.0 (exclusive)
- Values are uniformly distributed across this range

```tcl
# Verify the range
set tensor [torch::rand {1000}]
set min_val [torch::tensor_item [torch::tensor_min $tensor]]
set max_val [torch::tensor_item [torch::tensor_max $tensor]]
puts "Min: $min_val, Max: $max_val"  ;# Min: ~0.0, Max: ~0.999...
```

## Error Handling

### Common Errors

```tcl
# Missing required shape parameter
catch {torch::rand} err
puts $err  ;# "Required parameter missing: shape"

# Invalid device
catch {torch::rand {2 2} invalid_device} err
puts $err  ;# Error related to invalid device

# Invalid data type
catch {torch::rand {2 2} cpu invalid_dtype} err
puts $err  ;# Error related to invalid dtype

# Missing value for named parameter
catch {torch::rand -shape} err
puts $err  ;# "Missing value for parameter"

# Unknown parameter
catch {torch::rand -shape {2 2} -unknown_param value} err
puts $err  ;# "Unknown parameter: -unknown_param"
```

## Tensor Properties

After creation, you can inspect the tensor properties:

```tcl
set tensor [torch::rand {3 4}]

# Check shape
puts [torch::tensor_shape $tensor]     ;# "3 4"

# Check data type
puts [torch::tensor_dtype $tensor]     ;# "float32"

# Check device
puts [torch::tensor_device $tensor]    ;# "cpu"

# Check number of elements
puts [torch::tensor_numel $tensor]     ;# "12"
```

## Comparison with Related Commands

| Command | Distribution | Range | Use Case |
|---------|-------------|-------|----------|
| `torch::rand` | Uniform | [0, 1) | General random initialization |
| `torch::randn` | Normal (Gaussian) | (-∞, +∞) | Statistical sampling, some NN initializations |
| `torch::randint` | Uniform integers | [low, high) | Integer sampling, indexing |

## Performance Notes

- Random number generation is relatively fast but scales with tensor size
- GPU random generation (CUDA) can be faster for large tensors
- Consider using appropriate data types (float32 vs float64) based on precision needs

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old style (still supported)
set tensor [torch::rand {3 3} cuda:0 float64]

# New style (recommended)
set tensor [torch::rand -shape {3 3} -device cuda:0 -dtype float64]
```

### Benefits of Named Parameters
- **Clarity**: Parameter names make code self-documenting
- **Flexibility**: Parameters can be specified in any order
- **Maintainability**: Easier to modify and understand code
- **Error Prevention**: Less likely to mix up parameter positions

## See Also

- [`torch::randn`](randn.md) - Normal distribution random tensors
- [`torch::randint`](randint.md) - Random integer tensors
- [`torch::rand_like`](rand_like.md) - Random tensor with same shape as input
- [`torch::zeros`](zeros.md) - Zero-filled tensors
- [`torch::ones`](ones.md) - One-filled tensors 