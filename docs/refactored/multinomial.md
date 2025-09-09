# torch::multinomial

**Multinomial Sampling from Probability Distributions**

Samples from multinomial probability distributions. This function takes probability weights for a multinomial distribution and returns indices of sampled categories.

## Syntax

### New Syntax (Named Parameters)
```tcl
torch::multinomial -input tensor -numSamples int ?-replacement bool?
torch::multinomial -input tensor -num_samples int ?-replacement bool?  # Alternative parameter name
```

### Legacy Syntax (Positional Parameters) 
```tcl
torch::multinomial tensor num_samples ?replacement?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-input` | tensor | Yes | - | Input tensor containing probability weights |
| `-numSamples` / `-num_samples` | int | Yes | - | Number of samples to draw |
| `-replacement` | bool | No | `true` | Whether to sample with replacement |

### Positional Parameter Order (Legacy)
1. `tensor` - Input tensor containing probability weights
2. `num_samples` - Number of samples to draw  
3. `replacement` - Whether to sample with replacement (optional, default: true)

## Description

Draws samples from a multinomial probability distribution located in the rightmost dimension of the input tensor. The input tensor represents the probability weights for each category. 

**Key behaviors:**
- **1D input**: Returns a 1D tensor of sampled indices
- **2D input**: Performs independent sampling for each row (batch processing)
- **With replacement**: Can sample the same category multiple times
- **Without replacement**: Each category can only be sampled once per distribution

The input tensor does not need to be normalized (doesn't need to sum to 1), as the function automatically normalizes the weights internally.

## Returns

Returns a tensor containing the indices of sampled categories:
- For 1D input of size `[n]`: returns tensor of size `[num_samples]`
- For 2D input of size `[batch_size, n]`: returns tensor of size `[batch_size, num_samples]`

## Examples

### Basic Multinomial Sampling
```tcl
# Create probability weights (doesn't need to sum to 1)
set weights [torch::tensor_create -data {0.1 0.2 0.3 0.4} -dtype float32 -device cpu]

# New syntax - sample 3 times with replacement
set result [torch::multinomial -input $weights -numSamples 3 -replacement true]

# Legacy syntax - same operation
set result [torch::multinomial $weights 3 true]

# Alternative parameter name
set result [torch::multinomial -input $weights -num_samples 3 -replacement true]
```

### Sampling Without Replacement
```tcl
# Sample 3 different categories (no repeats)
set weights [torch::tensor_create -data {0.25 0.25 0.25 0.25} -dtype float32 -device cpu]
set result [torch::multinomial -input $weights -numSamples 3 -replacement false]

# Legacy syntax
set result [torch::multinomial $weights 3 false]
```

### Batch Processing
```tcl
# Create batch of probability distributions (2 distributions of 4 categories each)
set batch_weights [torch::tensor_create -data {0.1 0.2 0.3 0.4 0.4 0.3 0.2 0.1} -dtype float32 -device cpu]
set batch_2d [torch::tensor_reshape $batch_weights {2 4}]

# Sample 2 times from each distribution
set result [torch::multinomial -input $batch_2d -numSamples 2 -replacement true]
# Result shape: [2, 2] - 2 samples from each of 2 distributions
```

### Different Probability Distributions
```tcl
# Uniform distribution
set uniform [torch::tensor_create -data {0.25 0.25 0.25 0.25} -dtype float32 -device cpu]
set uniform_samples [torch::multinomial $uniform 10 true]

# Skewed distribution (category 0 is much more likely)
set skewed [torch::tensor_create -data {0.9 0.05 0.03 0.02} -dtype float32 -device cpu]
set skewed_samples [torch::multinomial $skewed 10 true]

# Binary distribution
set binary [torch::tensor_create -data {0.7 0.3} -dtype float32 -device cpu]
set binary_samples [torch::multinomial $binary 20 true]
```

### Large-Scale Sampling
```tcl
# Sample many times from a small distribution
set weights [torch::tensor_create -data {0.2 0.3 0.5} -dtype float32 -device cpu]
set many_samples [torch::multinomial -input $weights -numSamples 1000 -replacement true]
```

## Mathematical Notes

The multinomial distribution is a generalization of the binomial distribution to multiple categories. For a probability vector `p = [p1, p2, ..., pk]`, the multinomial sampling:

1. **Normalizes weights**: Internally converts weights to probabilities by dividing by sum
2. **Categorical sampling**: Each sample selects one category according to the probability distribution
3. **Independent samples**: When `replacement=true`, each sample is independent

## Error Handling

The function validates:
- Input tensor must exist and be valid
- Number of samples must be positive
- For sampling without replacement: `num_samples ≤ number_of_categories`

```tcl
# Error: Invalid tensor
catch {torch::multinomial -input "invalid" -numSamples 3} error
puts $error  # Will show invalid tensor error

# Error: Zero or negative samples
catch {torch::multinomial -input $weights -numSamples 0} error
puts $error  # Will show invalid arguments error

# Error: Too many samples without replacement
set weights [torch::tensor_create -data {0.5 0.5} -dtype float32 -device cpu]
catch {torch::multinomial $weights 5 false} error  # Only 2 categories available
puts $error  # Will show sampling error
```

## Migration Guide

### From Positional to Named Parameters

**Before:**
```tcl
set result [torch::multinomial $weights 5 true]
set result [torch::multinomial $weights 3 false]
```

**After:**
```tcl
set result [torch::multinomial -input $weights -numSamples 5 -replacement true]
set result [torch::multinomial -input $weights -numSamples 3 -replacement false]
```

### Benefits of New Syntax
- **Clarity**: Parameter names make code self-documenting
- **Flexibility**: Parameters can be specified in any order
- **Safety**: Reduced risk of confusing num_samples and replacement parameters
- **Consistency**: Follows TCL best practices for named parameters

## Use Cases

### Reinforcement Learning
```tcl
# Action selection from policy probabilities
set action_probs [torch::tensor_create -data {0.6 0.3 0.1} -dtype float32 -device cpu]
set action [torch::multinomial $action_probs 1 true]
```

### Natural Language Processing
```tcl
# Word sampling from vocabulary distribution
set vocab_probs [torch::tensor_create -data {0.05 0.15 0.25 0.35 0.20} -dtype float32 -device cpu]
set sampled_words [torch::multinomial $vocab_probs 10 true]
```

### Data Augmentation
```tcl
# Random class selection for synthetic data generation
set class_weights [torch::tensor_create -data {0.4 0.3 0.2 0.1} -dtype float32 -device cpu]
set synthetic_labels [torch::multinomial $class_weights 100 true]
```

## Performance Notes

- **Efficient sampling**: Uses optimized PyTorch multinomial implementation
- **Batch processing**: Multiple distributions processed in parallel
- **Memory efficient**: Minimal memory overhead for large sample counts
- **GPU compatible**: Works with CUDA tensors for accelerated sampling

## See Also

- [torch::tensor_create](tensor_create.md) - Creating input probability tensors
- [torch::rand](rand.md) - Uniform random sampling
- [torch::randn](randn.md) - Normal distribution sampling

## Implementation Status

- ✅ **Dual Syntax**: Supports both positional and named parameters
- ✅ **No camelCase Change**: Command name stays the same (single word)
- ✅ **Tests**: Comprehensive test suite in `tests/refactored/multinomial_test.tcl`
- ✅ **Documentation**: Complete API documentation with examples 