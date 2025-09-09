# torch::rand_like / torch::randLike

Create a tensor filled with random values from uniform distribution [0, 1) using the same shape as an input tensor.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::rand_like tensor ?dtype? ?device?
```

### Named Parameter Syntax
```tcl
torch::rand_like -input tensor ?-dtype dtype? ?-device device? ?-requiresGrad bool?
torch::randLike -input tensor ?-dtype dtype? ?-device device? ?-requiresGrad bool?
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input` / `tensor` | string | Handle of the input tensor to match shape | Required |
| `dtype` | string | Data type for the result tensor | Same as input |
| `device` | string | Device for the result tensor | Same as input |
| `requiresGrad` | boolean | Whether result tensor requires gradients | false |

### Supported Data Types
- `float32`, `float64` (recommended for random values)
- `int32`, `int64` (values will be 0 due to truncation)

### Supported Devices
- `cpu`
- `cuda` (if available)

## Return Value

Returns a string handle representing the new tensor filled with random values from uniform distribution [0, 1).

## Examples

### Basic Usage
```tcl
# Create reference tensor
set input [torch::zeros {3 4}]

# Create random tensor with same shape
set result [torch::rand_like $input]
# Result: 3x4 tensor filled with random values in [0, 1)
```

### Positional Syntax Examples
```tcl
# Basic usage
set result [torch::rand_like $input]

# With dtype
set result [torch::rand_like $input float64]

# With dtype and device
set result [torch::rand_like $input float32 cpu]
```

### Named Parameter Syntax Examples
```tcl
# Basic usage
set result [torch::rand_like -input $input]

# With specific dtype
set result [torch::rand_like -input $input -dtype float64]

# With all parameters
set result [torch::rand_like -input $input -dtype float32 -device cpu -requiresGrad true]

# Parameter order doesn't matter
set result [torch::rand_like -dtype float32 -input $input -device cpu]
```

### CamelCase Alias Examples
```tcl
# Using camelCase alias
set result [torch::randLike -input $input]

# With parameters
set result [torch::randLike -input $input -dtype float64 -requiresGrad true]
```

## Integration Examples

### Random Initialization
```tcl
set weights [torch::zeros {256 512}]
set random_weights [torch::rand_like -input $weights -dtype float32]
# Initialize weights with random values
```

### Data Augmentation
```tcl
set input_data [torch::zeros {32 3 224 224}]
set noise [torch::rand_like $input_data]
set noisy_data [torch::tensor_add $input_data $noise]
# Add random noise to data
```

### Mask Generation
```tcl
set input [torch::zeros {100 50}]
set random_mask [torch::rand_like $input]
# Can be used to create dropout masks, etc.
```

### Statistical Sampling
```tcl
# Generate random samples for Monte Carlo methods
set sample_shape [torch::zeros {1000 10}]
set samples [torch::rand_like -input $sample_shape -dtype float64]
```

## Mathematical Properties

### Distribution
- **Type**: Uniform distribution
- **Range**: [0, 1) - includes 0, excludes 1
- **Mean**: ~0.5
- **Variance**: ~1/12 ≈ 0.0833

### Statistical Validation
```tcl
set large_tensor [torch::zeros {1000 1000}]
set random_values [torch::rand_like $large_tensor]
set mean_val [torch::tensor_mean $random_values]
set mean_scalar [torch::tensor_item $mean_val]
# mean_scalar should be approximately 0.5
```

## Error Handling

### Invalid Tensor
```tcl
catch {torch::rand_like invalid_tensor} error
# Error: "Invalid tensor name"
```

### Missing Required Parameters
```tcl
catch {torch::rand_like -dtype float32} error
# Error: "Input tensor is required"
```

### Unknown Parameters
```tcl
catch {torch::rand_like -input $tensor -invalid param} error
# Error: "Unknown parameter: -invalid"
```

### Invalid Data Types
```tcl
catch {torch::rand_like -input $tensor -dtype invalid_type} error
# Error message about invalid dtype
```

## Migration Guide

### From Positional to Named Parameters

**Old positional syntax:**
```tcl
set result [torch::rand_like $input float64 cpu]
```

**New named parameter syntax:**
```tcl
set result [torch::rand_like -input $input -dtype float64 -device cpu]
```

**Or using camelCase alias:**
```tcl
set result [torch::randLike -input $input -dtype float64 -device cpu]
```

### Benefits of Named Parameters

1. **Clarity**: Parameter names make code self-documenting
2. **Flexibility**: Parameters can be specified in any order
3. **Optional parameters**: Easier to specify only needed parameters
4. **Future-proof**: New parameters can be added without breaking existing code

## Performance Notes

- The command creates a new tensor with the same shape as the input
- Memory usage is proportional to the input tensor size
- Random number generation is performed on the specified device
- CUDA random generation may be faster for large tensors
- Each call produces different random values (unless seed is controlled)

## Common Use Cases

### Weight Initialization
```tcl
# Xavier/Glorot uniform initialization (scaled)
set layer_weights [torch::zeros {784 512}]
set random_init [torch::rand_like $layer_weights]
# Scale by appropriate factor for initialization
```

### Dropout Implementation
```tcl
# Create dropout mask
set activations [torch::randn {32 512}]
set dropout_mask [torch::rand_like $activations]
# Use mask to randomly zero elements
```

### Data Augmentation
```tcl
# Add random noise to inputs
set clean_data [torch::randn {64 3 32 32}]
set noise [torch::rand_like $clean_data]
# Combine for augmented data
```

### Monte Carlo Simulations
```tcl
# Generate random samples for simulation
set simulation_space [torch::zeros {10000 100}]
set random_samples [torch::rand_like $simulation_space]
```

## Reproducibility

### Random Seed Control
To ensure reproducible results, control the random seed before calling:
```tcl
# Note: Actual seed control depends on PyTorch's random state
# This is handled at the PyTorch level, not the TCL level
```

## See Also

- [`torch::randn_like`](randn_like.md) - Create tensor with normal distribution
- [`torch::randint_like`](randint_like.md) - Create tensor with random integers
- [`torch::zeros_like`](zeros_like.md) - Create tensor filled with zeros
- [`torch::ones_like`](ones_like.md) - Create tensor filled with ones
- [`torch::rand`](rand.md) - Create random tensor with specified shape
- [`torch::tensor_shape`](tensor_shape.md) - Get tensor shape
- [`torch::tensor_dtype`](tensor_dtype.md) - Get tensor data type 