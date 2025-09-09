# torch::normal / torch::Normal

Generates a tensor of random numbers from a normal (Gaussian) distribution.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::normal -mean value -std value ?-size list? ?-dtype type? ?-device device?
torch::Normal -mean value -std value ?-size list? ?-dtype type? ?-device device?
```

### Positional Parameters (Legacy)
```tcl
torch::normal mean std ?size? ?dtype? ?device?
torch::Normal mean std ?size? ?dtype? ?device?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-mean` | float | Required | Mean (μ) of the normal distribution |
| `-std` | float | Required | Standard deviation (σ) of the normal distribution |
| `-size` | list | {1} | Shape of the output tensor |
| `-dtype` | string | float32 | Data type of the output tensor |
| `-device` | string | cpu | Device to place the output tensor on |

## Description

The `torch::normal` command generates random numbers from a normal distribution with the specified mean (μ) and standard deviation (σ). The probability density function of the normal distribution is:

```
f(x) = (1 / (σ * sqrt(2π))) * exp(-(x - μ)² / (2σ²))
```

The command supports generating both single values and tensors of any shape. The generated values follow the specified normal distribution, where approximately:
- 68% of values fall within 1 standard deviation of the mean (μ ± σ)
- 95% of values fall within 2 standard deviations of the mean (μ ± 2σ)
- 99.7% of values fall within 3 standard deviations of the mean (μ ± 3σ)

## Return Value

Returns a handle to a new tensor containing random values drawn from the specified normal distribution.

## Examples

### Basic Usage
```tcl
# Generate a single value from N(0, 1) (standard normal distribution)
set result [torch::normal 0.0 1.0]

# Generate a 2x3 tensor from N(5, 2)
set result [torch::normal -mean 5.0 -std 2.0 -size {2 3}]
```

### Specifying Data Type
```tcl
# Generate double-precision values
set result [torch::normal 0.0 1.0 {2 3} float64]

# Using named parameters
set result [torch::normal -mean 0.0 -std 1.0 -size {2 3} -dtype float64]
```

### Using camelCase Alias
```tcl
# The command is also available with camelCase alias
set result [torch::Normal -mean 0.0 -std 1.0 -size {2 3}]
```

### Statistical Properties
```tcl
# Generate many samples to verify distribution properties
set samples [torch::normal 5.0 2.0 {10000}]

# Calculate mean (should be close to 5.0)
set mean [torch::tensor_mean $samples]

# Calculate standard deviation (should be close to 2.0)
set std [torch::tensor_std $samples]
```

## Error Handling

The command will raise an error if:
- Required parameters (mean, std) are missing
- Invalid size list is provided
- Invalid dtype is specified
- Invalid device is specified

## See Also

- `torch::randn` - Generate standard normal distribution (μ=0, σ=1)
- `torch::uniform` - Generate uniform distribution
- `torch::bernoulli` - Generate Bernoulli distribution
- `torch::multinomial` - Generate multinomial distribution 