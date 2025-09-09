# torch::randn_like / torch::randnLike

Create a tensor filled with random values from standard normal distribution (mean=0, std=1) using the same shape as an input tensor.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::randn_like tensor ?dtype? ?device?
```

### Named Parameter Syntax
```tcl
torch::randn_like -input tensor ?-dtype dtype? ?-device device? ?-requiresGrad bool?
torch::randnLike -input tensor ?-dtype dtype? ?-device device? ?-requiresGrad bool?
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input` / `tensor` | string | Handle of the input tensor to match shape | Required |
| `dtype` | string | Data type for the result tensor | Same as input |
| `device` | string | Device for the result tensor | Same as input |
| `requiresGrad` | boolean | Whether result tensor requires gradients | false |

### Supported Data Types
- `float32`, `float64` (recommended for normal random values)
- `int32`, `int64` (values will be truncated to integers)

### Supported Devices
- `cpu`
- `cuda` (if available)

## Return Value

Returns a string handle representing the new tensor filled with random values from standard normal distribution N(0,1).

## Examples

### Basic Usage
```tcl
# Create reference tensor
set input [torch::zeros {3 4}]

# Create normal random tensor with same shape
set result [torch::randn_like $input]
# Result: 3x4 tensor filled with normally distributed values
```

### Positional Syntax Examples
```tcl
# Basic usage
set result [torch::randn_like $input]

# With dtype
set result [torch::randn_like $input float64]

# With dtype and device
set result [torch::randn_like $input float32 cpu]
```

### Named Parameter Syntax Examples
```tcl
# Basic usage
set result [torch::randn_like -input $input]

# With specific dtype
set result [torch::randn_like -input $input -dtype float64]

# With all parameters
set result [torch::randn_like -input $input -dtype float32 -device cpu -requiresGrad true]

# Parameter order doesn't matter
set result [torch::randn_like -dtype float32 -input $input -device cpu]
```

### CamelCase Alias Examples
```tcl
# Using camelCase alias
set result [torch::randnLike -input $input]

# With parameters
set result [torch::randnLike -input $input -dtype float64 -requiresGrad true]
```

## Integration Examples

### Neural Network Weight Initialization
```tcl
# Xavier/Glorot normal initialization
set layer_weights [torch::zeros {256 512}]
set initialized_weights [torch::randn_like -input $layer_weights -dtype float32]
# Scale appropriately: multiply by sqrt(2/(fan_in + fan_out))
```

### Noise Addition for Data Augmentation
```tcl
set input_data [torch::zeros {32 3 224 224}]
set noise [torch::randn_like $input_data]
set scale 0.1
set scaled_noise [torch::tensor_mul $noise $scale]
set noisy_data [torch::tensor_add $input_data $scaled_noise]
```

### Sampling for Variational Models
```tcl
# Generate samples for VAE reparameterization trick
set latent_shape [torch::zeros {64 128}]
set epsilon [torch::randn_like $latent_shape]
# Use with mean and std tensors for reparameterization
```

### Gaussian Process Sampling
```tcl
# Sample from Gaussian process
set sample_points [torch::zeros {1000 10}]
set gp_samples [torch::randn_like -input $sample_points -dtype float64]
```

## Mathematical Properties

### Distribution
- **Type**: Standard Normal Distribution (Gaussian)
- **Mean**: 0.0
- **Standard Deviation**: 1.0
- **Variance**: 1.0
- **Range**: (-∞, +∞) theoretically, typically within [-4, +4] for most values

### Probability Density Function
f(x) = (1/√(2π)) * e^(-x²/2)

### Statistical Validation
```tcl
# Verify distribution properties with large sample
set large_tensor [torch::zeros {1000 1000}]
set samples [torch::randn_like $large_tensor]

set mean_val [torch::tensor_mean $samples]
set mean_scalar [torch::tensor_item $mean_val]
# mean_scalar should be approximately 0.0

set std_val [torch::tensor_std $samples]  
set std_scalar [torch::tensor_item $std_val]
# std_scalar should be approximately 1.0
```

## Error Handling

### Invalid Tensor
```tcl
catch {torch::randn_like invalid_tensor} error
# Error: "Invalid tensor name"
```

### Missing Required Parameters
```tcl
catch {torch::randn_like -dtype float32} error
# Error: "Input tensor is required"
```

### Unknown Parameters
```tcl
catch {torch::randn_like -input $tensor -invalid param} error
# Error: "Unknown parameter: -invalid"
```

### Invalid Data Types
```tcl
catch {torch::randn_like -input $tensor -dtype invalid_type} error
# Error message about invalid dtype
```

## Migration Guide

### From Positional to Named Parameters

**Old positional syntax:**
```tcl
set result [torch::randn_like $input float64 cpu]
```

**New named parameter syntax:**
```tcl
set result [torch::randn_like -input $input -dtype float64 -device cpu]
```

**Or using camelCase alias:**
```tcl
set result [torch::randnLike -input $input -dtype float64 -device cpu]
```

### Benefits of Named Parameters

1. **Clarity**: Parameter names make code self-documenting
2. **Flexibility**: Parameters can be specified in any order
3. **Optional parameters**: Easier to specify only needed parameters
4. **Future-proof**: New parameters can be added without breaking existing code

## Performance Notes

- The command creates a new tensor with the same shape as the input
- Memory usage is proportional to the input tensor size
- Normal random generation is performed on the specified device
- CUDA random generation may be faster for large tensors
- Each call produces different random values (unless seed is controlled)

## Common Use Cases

### Weight Initialization Schemes

#### Xavier/Glorot Normal
```tcl
set layer_weights [torch::zeros {784 512}]
set xavier_weights [torch::randn_like $layer_weights]
# Scale by sqrt(2/(fan_in + fan_out)) = sqrt(2/(784+512))
```

#### He Normal (for ReLU networks)
```tcl
set conv_weights [torch::zeros {64 3 3 3}]
set he_weights [torch::randn_like $conv_weights]
# Scale by sqrt(2/fan_in)
```

### Variational Autoencoders
```tcl
# Reparameterization trick
set mu [torch::zeros {32 128}]      # Mean tensor
set logvar [torch::zeros {32 128}]  # Log variance tensor
set eps [torch::randn_like $mu]     # Random noise

# Sample: z = mu + std * eps
# where std = exp(0.5 * logvar)
```

### Monte Carlo Methods
```tcl
# Generate samples for Monte Carlo integration
set sample_space [torch::zeros {10000 50}]
set mc_samples [torch::randn_like $sample_space]
```

### Gaussian Mixture Models
```tcl
# Sample from mixture components
set component_shape [torch::zeros {1000 10}]
set component_samples [torch::randn_like $component_shape]
# Transform with component-specific mean and covariance
```

## Statistical Analysis Tools

### Central Limit Theorem Verification
```tcl
# Generate multiple samples and check convergence to normal
set samples [torch::randn_like $tensor]
set histogram_bins [torch::arange -start -3.0 -end 3.0 -step 0.1]
# Use with histogram functions to verify distribution shape
```

### Hypothesis Testing
```tcl
# Generate null hypothesis samples
set null_samples [torch::randn_like $data_tensor]
# Compare with actual data for statistical tests
```

## Advanced Examples

### Batch Normalization Initialization
```tcl
# Initialize batch norm parameters
set batch_size 32
set channels 64
set bn_shape [torch::zeros [list $batch_size $channels]]
set bn_weight [torch::randn_like $bn_shape]
set bn_bias [torch::zeros_like $bn_shape]
```

### Dropout Noise
```tcl
# Alternative dropout using normal noise (for research)
set activations [torch::randn {32 512}]
set dropout_noise [torch::randn_like $activations]
# Apply threshold and scaling
```

### Data Augmentation Pipeline
```tcl
# Add controlled gaussian noise to inputs
set clean_images [torch::randn {64 3 32 32}]
set noise_strength 0.05
set augmentation_noise [torch::randn_like $clean_images]
set noise_scaled [torch::tensor_mul $augmentation_noise $noise_strength]
set augmented_images [torch::tensor_add $clean_images $noise_scaled]
```

## Reproducibility

### Random Seed Control
To ensure reproducible results, control the random seed at the PyTorch level:
```tcl
# Note: Seed control happens at PyTorch C++ level
# This ensures reproducible sequences across calls
```

### Deterministic Behavior
```tcl
# For deterministic testing, save and reuse tensor values
set reference_normal [torch::randn_like $shape_tensor]
# Use reference_normal for consistent test results
```

## Comparison with Other Random Functions

| Function | Distribution | Range | Use Case |
|----------|-------------|-------|----------|
| `torch::randn_like` | Normal (0,1) | (-∞,+∞) | Weight initialization, sampling |
| `torch::rand_like` | Uniform [0,1) | [0,1) | Dropout masks, uniform sampling |
| `torch::randint_like` | Discrete uniform | [low,high) | Integer sampling, indices |

## See Also

- [`torch::rand_like`](rand_like.md) - Create tensor with uniform distribution
- [`torch::randint_like`](randint_like.md) - Create tensor with random integers
- [`torch::zeros_like`](zeros_like.md) - Create tensor filled with zeros
- [`torch::ones_like`](ones_like.md) - Create tensor filled with ones
- [`torch::randn`](randn.md) - Create normal random tensor with specified shape
- [`torch::tensor_mean`](tensor_mean.md) - Calculate tensor mean
- [`torch::tensor_std`](tensor_std.md) - Calculate tensor standard deviation 