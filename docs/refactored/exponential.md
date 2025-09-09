# torch::exponential

**Generate random samples from an exponential distribution**

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::exponential size rate ?dtype? ?device?
```

### Named Parameter Syntax (New)
```tcl
torch::exponential -size {shape} -rate value ?-dtype type? ?-device dev?
```

### Both syntaxes are supported and produce identical distribution properties.

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `size` | list | Shape of the output tensor as a list of integers | Yes |
| `rate` | double | Rate parameter (λ) of the exponential distribution (must be > 0) | Yes |
| `dtype` | string | Data type of the output tensor (default: float32) | No |
| `device` | string | Device to place the tensor on (default: cpu) | No |

## Returns

Returns a new tensor with the specified shape containing random samples drawn from an exponential distribution with the given rate parameter.

## Description

The `torch::exponential` function generates random samples from an exponential distribution using the inverse transform method. The exponential distribution is commonly used to model the time between events in a Poisson process, such as the time between arrivals at a service center.

### Mathematical Properties

- **Probability Density Function**: `f(x; λ) = λ * exp(-λx)` for x ≥ 0
- **Mean**: `μ = 1/λ` (where λ is the rate parameter)
- **Variance**: `σ² = 1/λ²`
- **Standard Deviation**: `σ = 1/λ`
- **Memoryless Property**: `P(X > s + t | X > s) = P(X > t)`

### Implementation Method

The function uses the inverse transform method:
1. Generate uniform random numbers U ~ Uniform(0,1)
2. Apply transformation: X = -ln(U) / λ

## Examples

### Basic Usage
```tcl
# Create samples from exponential distribution
set samples [torch::exponential {1000} 2.0]

# Using named parameters
set samples2 [torch::exponential -size {1000} -rate 2.0]

# Both produce tensors with 1000 samples from Exp(λ=2.0)
puts "Mean should be ≈ 0.5: [torch::tensor_item [torch::tensor_mean $samples]]"
```

### Multi-dimensional Tensors
```tcl
# Create a 2D matrix of exponential samples
set matrix [torch::exponential {100 50} 1.0]
puts "Shape: [torch::tensor_shape $matrix]"  ;# {100 50}

# 3D tensor with named parameters
set tensor3d [torch::exponential -size {10 20 30} -rate 0.5]
puts "Total elements: [expr 10*20*30]"  ;# 6000 samples
```

### Different Rate Parameters
```tcl
# Low rate = high mean (longer intervals)
set long_intervals [torch::exponential {5} 0.1]  ;# Mean = 10.0

# High rate = low mean (shorter intervals)
set short_intervals [torch::exponential -size {5} -rate 10.0]  ;# Mean = 0.1

# Compare means
set mean1 [torch::tensor_item [torch::tensor_mean $long_intervals]]
set mean2 [torch::tensor_item [torch::tensor_mean $short_intervals]]
puts "Low rate mean: $mean1, High rate mean: $mean2"
```

### Data Type Specification
```tcl
# Float32 (default)
set f32_samples [torch::exponential {100} 1.0]

# Float64 for higher precision
set f64_samples [torch::exponential -size {100} -rate 1.0 -dtype float64]

# Specify both dtype and device
set samples [torch::exponential {50} 2.0 float64 cpu]
```

### Statistical Validation
```tcl
# Generate large sample to verify statistical properties
set large_sample [torch::exponential {100000} 2.0]
set sample_mean [torch::tensor_item [torch::tensor_mean $large_sample]]
set theoretical_mean [expr 1.0 / 2.0]

puts "Sample mean: $sample_mean"
puts "Theoretical mean: $theoretical_mean"
puts "Difference: [expr abs($sample_mean - $theoretical_mean)]"
```

## Real-World Applications

### 1. Service Time Modeling
```tcl
# Model service times at a bank (mean service time = 3 minutes)
set rate [expr 1.0 / 3.0]  ;# Rate parameter for mean = 3
set service_times [torch::exponential -size {1000} -rate $rate]

# Calculate statistics
set mean_time [torch::tensor_item [torch::tensor_mean $service_times]]
puts "Average service time: $mean_time minutes"
```

### 2. Reliability Engineering
```tcl
# Model component failure times (MTBF = 1000 hours)
set failure_rate [expr 1.0 / 1000.0]
set failure_times [torch::exponential {500} $failure_rate]

# Find probability of failure before 500 hours
set threshold [torch::tensor_create {500.0}]
set early_failures [torch::tensor_lt $failure_times $threshold]
set failure_count [torch::tensor_item [torch::tensor_sum $early_failures]]
set failure_prob [expr $failure_count / 500.0]

puts "Probability of failure before 500 hours: $failure_prob"
```

### 3. Network Traffic Modeling
```tcl
# Model inter-arrival times of network packets (mean = 10ms)
set arrival_rate [expr 1.0 / 10.0]  ;# packets per ms
set inter_arrival_times [torch::exponential -size {10000} -rate $arrival_rate -dtype float64]

# Calculate network statistics
set min_interval [torch::tensor_item [torch::tensor_min $inter_arrival_times]]
set max_interval [torch::tensor_item [torch::tensor_max $inter_arrival_times]]

puts "Minimum inter-arrival: $min_interval ms"
puts "Maximum inter-arrival: $max_interval ms"
```

### 4. Queueing Theory
```tcl
# Model waiting times in M/M/1 queue
set arrival_rate 0.8  ;# customers per minute
set service_rate 1.0  ;# customers served per minute

# Generate arrival intervals
set arrivals [torch::exponential {1000} $arrival_rate]

# Generate service times
set services [torch::exponential -size {1000} -rate $service_rate]

puts "System utilization: [expr $arrival_rate / $service_rate]"
```

## Device and Precision Support

```tcl
# Different precisions
set single_precision [torch::exponential {100} 1.0 float32]
set double_precision [torch::exponential -size {100} -rate 1.0 -dtype float64]

# Different devices (when available)
set cpu_samples [torch::exponential -size {100} -rate 1.0 -device cpu]
# set gpu_samples [torch::exponential -size {100} -rate 1.0 -device cuda:0]
```

## Error Handling

```tcl
# Invalid rate parameter
catch {torch::exponential {10} -1.0} result
puts $result  ;# Error: rate must be positive

# Missing required parameters
catch {torch::exponential -size {10}} result
puts $result  ;# Usage information

# Invalid size specification
catch {torch::exponential {} 1.0} result
puts $result  ;# Error: empty size

# Invalid data type
catch {torch::exponential -size {10} -rate 1.0 -dtype invalid} result
puts $result  ;# Error: Invalid dtype
```

## Statistical Properties Verification

```tcl
# Verify exponential distribution properties
proc verify_exponential {rate sample_size tolerance} {
    set samples [torch::exponential $sample_size $rate]
    
    # Calculate sample statistics
    set sample_mean [torch::tensor_item [torch::tensor_mean $samples]]
    set theoretical_mean [expr 1.0 / $rate]
    
    # Check if sample mean is close to theoretical mean
    set mean_error [expr abs($sample_mean - $theoretical_mean)]
    set mean_ok [expr $mean_error < $tolerance]
    
    # Check if all values are positive (required for exponential)
    set min_val [torch::tensor_item [torch::tensor_min $samples]]
    set positive_ok [expr $min_val > 0.0]
    
    return [list $mean_ok $positive_ok $sample_mean $theoretical_mean]
}

# Test with different rates
set result1 [verify_exponential 1.0 10000 0.1]
set result2 [verify_exponential 0.5 10000 0.2]

puts "Rate 1.0 - Mean OK: [lindex $result1 0], Positive OK: [lindex $result1 1]"
puts "Rate 0.5 - Mean OK: [lindex $result2 0], Positive OK: [lindex $result2 1]"
```

## Mathematical Comparison

| Distribution | PDF | Mean | Use Case |
|--------------|-----|------|----------|
| Exponential(λ) | λe^(-λx) | 1/λ | Inter-arrival times, lifetimes |
| Normal(μ,σ) | (1/σ√2π)e^(-(x-μ)²/2σ²) | μ | General continuous data |
| Uniform(a,b) | 1/(b-a) | (a+b)/2 | When all values equally likely |
| Poisson(λ) | λ^k e^(-λ)/k! | λ | Count of events in fixed time |

## Performance Notes

- **Memory efficient**: Uses PyTorch's optimized random number generation
- **Vectorized**: Generates all samples in one operation
- **Thread-safe**: Safe for concurrent use with proper RNG management
- **GPU support**: Can generate samples on CUDA devices (when available)

## Migration Guide

### From Old Syntax to New Syntax

```tcl
# Old positional syntax (still supported)
set result [torch::exponential {100 200} 2.0 float32 cpu]

# New named parameter syntax (recommended)
set result [torch::exponential -size {100 200} -rate 2.0 -dtype float32 -device cpu]

# Mixed usage - both work identically
```

### Best Practices

1. **Use appropriate sample sizes**: Larger samples give better statistical properties
2. **Choose rate based on domain**: Rate = 1/mean_desired
3. **Consider precision**: Use float64 for high-precision applications
4. **Validate parameters**: Ensure rate > 0 before calling
5. **Check for edge cases**: Handle very small/large rate values appropriately

## See Also

- [torch::normal](normal.md) - Normal distribution sampling
- [torch::uniform](uniform.md) - Uniform distribution sampling
- [torch::gamma](gamma.md) - Gamma distribution sampling
- [torch::poisson](poisson.md) - Poisson distribution sampling
- [torch::bernoulli](bernoulli.md) - Bernoulli distribution sampling

## Technical Details

- **Implementation**: Uses inverse transform method with uniform random numbers
- **Random seed**: Respects global PyTorch random seed settings
- **Precision**: IEEE 754 floating-point precision
- **Domain**: x ∈ [0, ∞) for exponential distribution
- **Thread safety**: Generates independent samples per call 