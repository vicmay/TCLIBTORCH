# torch::poisson

**Poisson Distribution Sampling**

Generates a tensor of samples drawn from a Poisson distribution with the given rate (lambda) parameter.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::poisson -size list -lambda double ?-dtype string? ?-device string?
torch::poissonDist -size list -lambda double ?-dtype string? ?-device string?
```

### Positional Syntax (Legacy)
```tcl
torch::poisson size lambda ?dtype? ?device?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **size** | list | - | Shape of the output tensor |
| **lambda** | double | - | Rate parameter (λ) of the Poisson distribution |
| **dtype** | string | "float32" | Data type of the output tensor ("float32" or "float64") |
| **device** | string | "cpu" | Device to place the output tensor on ("cpu" or "cuda") |

## Returns

Returns a tensor handle containing samples drawn from a Poisson distribution with the specified rate parameter.

## Mathematical Details

The Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant mean rate and independently of the time since the last event.

For a Poisson distribution with rate λ:
- Mean (expected value) = λ
- Variance = λ
- Support: Non-negative integers (0, 1, 2, ...)

The probability mass function is:

P(X = k) = (λ^k * e^(-λ)) / k!

where:
- k is the number of occurrences (k = 0, 1, 2, ...)
- λ is the rate parameter
- e is Euler's number

## Examples

### Named Parameter Syntax

#### Basic Usage
```tcl
# Generate 2x3 tensor of Poisson samples with λ=2.0
set samples [torch::poisson -size {2 3} -lambda 2.0]
```

#### With Custom Data Type
```tcl
# Generate samples with double precision
set samples [torch::poisson -size {3 2} -lambda 5.0 -dtype float64]
```

#### With Device Specification
```tcl
# Generate samples on CPU
set samples [torch::poisson -size {4 4} -lambda 1.5 -dtype float32 -device cpu]
```

### Positional Syntax (Legacy)

```tcl
# Basic usage
set samples [torch::poisson {2 3} 2.0]

# With custom dtype
set samples [torch::poisson {3 2} 5.0 float64]

# With device specification
set samples [torch::poisson {4 4} 1.5 float32 cpu]
```

## Error Handling

The command will throw an error in the following cases:

- Invalid size dimensions (non-positive values)
- Negative lambda value
- Invalid dtype (must be "float32" or "float64")
- Invalid device (must be "cpu" or "cuda")
- Missing required parameters (size and lambda)

## See Also

- `torch::normal` - Samples from a normal distribution
- `torch::exponential` - Samples from an exponential distribution
- `torch::gamma` - Samples from a gamma distribution 