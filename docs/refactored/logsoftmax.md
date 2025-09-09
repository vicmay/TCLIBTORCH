# torch::logsoftmax

Applies the log softmax function element-wise to the input tensor along a specified dimension.

## Syntax

### Current (Positional) - Backward Compatible
```tcl
torch::logsoftmax tensor ?dim?
```

### New (Named Parameters) - Recommended
```tcl
torch::logsoftmax -input tensor ?-dim dimension?
torch::logsoftmax -tensor tensor ?-dimension dimension?
```

### camelCase Alias
```tcl
torch::logSoftmax -input tensor ?-dim dimension?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-input` or `-tensor` | string | required | Name of input tensor |
| `-dim` or `-dimension` | integer | -1 | Dimension to apply log softmax along |

**Note**: Default dimension `-1` means the last dimension of the tensor.

## Returns

Returns a new tensor handle containing the log softmax result.

## Description

The log softmax function is defined as:

```
log_softmax(x_i) = log(softmax(x_i)) = log(exp(x_i) / Σ_j exp(x_j))
                 = x_i - log(Σ_j exp(x_j))
```

This is mathematically equivalent to applying softmax followed by logarithm, but is numerically more stable for large input values.

### Mathematical Properties

1. **Numerically Stable**: Unlike `log(softmax(x))`, this implementation avoids overflow/underflow
2. **Sum Property**: `exp(log_softmax(x))` sums to 1.0 along the specified dimension
3. **Maximum Value**: The maximum value of log_softmax output is 0.0
4. **Monotonic**: Preserves the relative ordering of input values

## Examples

### Basic Usage

```tcl
# Create a simple 1D tensor
set input [torch::tensor_create {1.0 2.0 3.0} float32]

# Apply log softmax (positional syntax)
set result1 [torch::logsoftmax $input]

# Apply log softmax (named syntax)
set result2 [torch::logsoftmax -input $input]

# Using camelCase alias
set result3 [torch::logSoftmax -input $input]

# All three results are identical
```

### Multi-dimensional Tensors

```tcl
# Create a 2D tensor
set input [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
set input [torch::tensor_reshape $input {2 3}]

# Apply log softmax along dimension 0 (rows)
set result_dim0 [torch::logsoftmax -input $input -dim 0]

# Apply log softmax along dimension 1 (columns) 
set result_dim1 [torch::logsoftmax -input $input -dim 1]

# Using default dimension (-1, last dimension)
set result_default [torch::logsoftmax -input $input]
# result_default is same as result_dim1 for 2D tensor
```

### Classification Probabilities

```tcl
# Typical usage in neural networks for classification
set logits [torch::tensor_create {2.1 1.0 0.1} float32]
set log_probs [torch::logsoftmax $logits]

# Convert back to probabilities if needed
set probs [torch::tensor_exp $log_probs]

# Verify probabilities sum to 1.0
set sum_probs [torch::tensor_sum $probs]
set total [torch::tensor_item $sum_probs]
# total ≈ 1.0
```

### Batch Processing

```tcl
# Process batch of samples
set batch_logits [torch::tensor_create {
    1.0 2.0 3.0
    0.5 1.5 2.5  
    2.0 1.0 0.5
} float32]
set batch_logits [torch::tensor_reshape $batch_logits {3 3}]

# Apply log softmax to each sample (along dimension 1)
set batch_log_probs [torch::logsoftmax -input $batch_logits -dim 1]
```

## Common Use Cases

### 1. Neural Network Output
```tcl
# Final layer of classification network
set final_logits [torch::linear $hidden $output_weights]
set log_probabilities [torch::logsoftmax -input $final_logits -dim 1]
```

### 2. Loss Function Preparation
```tcl
# Prepare for negative log likelihood loss
set predictions [torch::logsoftmax -input $model_output -dim 1]
set loss [torch::nll_loss $predictions $targets]
```

### 3. Attention Mechanisms
```tcl
# Compute attention weights in transformer
set attention_scores [torch::matmul $query $key_transposed]
set attention_weights [torch::logsoftmax -input $attention_scores -dim -1]
```

## Error Handling

```tcl
# Missing input parameter
catch {torch::logsoftmax} error
# Error: Required parameter missing: -input or -tensor

# Invalid tensor name
catch {torch::logsoftmax invalid_tensor} error  
# Error: Invalid tensor name

# Invalid dimension type
catch {torch::logsoftmax $input "invalid"} error
# Error: Invalid dimension parameter

# Integer tensors not supported
set int_tensor [torch::tensor_create {1 2 3} int32]
catch {torch::logsoftmax $int_tensor} error
# Error: log_softmax not implemented for integer types
```

## Data Type Support

| Data Type | Supported | Notes |
|-----------|-----------|-------|
| `float32` | ✅ | Recommended for most use cases |
| `float64` | ✅ | Higher precision |
| `int32` | ❌ | Not supported by PyTorch |
| `int64` | ❌ | Not supported by PyTorch |
| `bool` | ❌ | Not supported |

## Performance Considerations

1. **Numerical Stability**: LogSoftmax is more stable than `log(softmax(x))`
2. **Memory Efficiency**: Single operation, no intermediate softmax tensor
3. **Gradient Flow**: Better gradient properties for training
4. **Large Values**: Handles large input values without overflow

## Comparison with Alternatives

### vs. log(softmax(x))
```tcl
# Less stable approach
set softmax_result [torch::softmax $input]
set log_result_unstable [torch::tensor_log $softmax_result]

# More stable approach (recommended)
set log_result_stable [torch::logsoftmax $input]
```

### vs. softmax
```tcl
# For probability distributions
set probabilities [torch::softmax $input]

# For log probability distributions (better for loss computation)
set log_probabilities [torch::logsoftmax $input]
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set result [torch::logsoftmax $input]
set result [torch::logsoftmax $input 1]

# New named parameter syntax  
set result [torch::logsoftmax -input $input]
set result [torch::logsoftmax -input $input -dim 1]
```

### Parameter Name Flexibility

```tcl
# These are equivalent
torch::logsoftmax -input $tensor -dim 0
torch::logsoftmax -tensor $tensor -dimension 0
```

### camelCase Usage

```tcl
# snake_case (traditional)
torch::logsoftmax -input $tensor

# camelCase (modern)
torch::logSoftmax -input $tensor
```

## Mathematical Verification

### Property Tests
```tcl
# Test that exp(log_softmax(x)) sums to 1
set input [torch::tensor_create {1.0 2.0 3.0} float32]
set log_probs [torch::logsoftmax $input]
set probs [torch::tensor_exp $log_probs]
set sum_val [torch::tensor_item [torch::tensor_sum $probs]]
# sum_val ≈ 1.0

# Test numerical stability with large values
set large_input [torch::tensor_create {100.0 101.0 102.0} float32]
set stable_result [torch::logsoftmax $large_input]
# No overflow or NaN values
```

### Manual Computation Verification
```tcl
# Verify against manual computation
set x [torch::tensor_create {1.0 2.0 3.0} float32]
set log_softmax_result [torch::logsoftmax $x]

# Manual: log_softmax(x) = x - log(sum(exp(x)))
set exp_x [torch::tensor_exp $x]
set sum_exp_x [torch::tensor_sum $exp_x]
set log_sum_exp_x [torch::tensor_log $sum_exp_x]
set manual_result [torch::tensor_sub $x $log_sum_exp_x]

# Results should be identical
```

## See Also

- [`torch::softmax`](softmax.md) - Standard softmax function
- [`torch::log`](log.md) - Natural logarithm
- [`torch::exp`](exp.md) - Exponential function
- [`torch::nll_loss`](nll_loss.md) - Negative log likelihood loss
- [`torch::cross_entropy_loss`](cross_entropy_loss.md) - Cross entropy loss

## Implementation Details

- **Backend**: Uses PyTorch's `torch::log_softmax` function
- **Numerical Method**: Logarithm of softmax computed in single stable operation
- **Memory**: Single output tensor, no intermediate allocations
- **Thread Safety**: Safe for concurrent use with different tensors

---

*This documentation covers LibTorch TCL Extension v2.0+ with dual syntax support.* 