# torch::adaptive_maxpool1d

1D adaptive max pooling operation for neural networks.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::adaptive_maxpool1d input output_size
```

### Named Parameters (New)
```tcl
torch::adaptive_maxpool1d -input input -output_size output_size
torch::adaptive_maxpool1d -tensor input -outputSize output_size
```

### camelCase Alias
```tcl
torch::adaptiveMaxpool1d input output_size
torch::adaptiveMaxpool1d -input input -outputSize output_size
```

## Description

Applies a 1D adaptive max pooling operation over an input tensor. Unlike regular max pooling with fixed kernel sizes, adaptive max pooling allows you to specify the desired output size, and the operation automatically determines the appropriate kernel size and stride to achieve that output.

This operation is commonly used in neural networks where you need to reduce the spatial dimensions of feature maps to a fixed size, regardless of the input size.

## Parameters

| Parameter | Aliases | Type | Required | Description |
|-----------|---------|------|----------|-------------|
| `input` | `-input`, `-tensor` | string | Yes | Handle to input tensor |
| `output_size` | `-output_size`, `-outputSize` | integer | Yes | Target output size for the pooled dimension |

### Input Requirements
- **input**: Must be a valid tensor handle
- **output_size**: Must be a positive integer

### Input Shape
- Input tensor should have shape `(N, C, L)` where:
  - `N` = batch size
  - `C` = number of channels
  - `L` = length dimension

### Output Shape
- Output tensor will have shape `(N, C, output_size)`

## Return Value

Returns a string handle to the resulting tensor containing the pooled values.

## Mathematical Background

Adaptive max pooling divides the input into approximately equal regions and takes the maximum value from each region. For a given output size `S` and input size `L`, the operation:

1. Divides the input length `L` into `S` intervals
2. For each interval, computes the maximum value
3. Returns a tensor of size `S` with these maximum values

The pooling is "adaptive" because it automatically adjusts the kernel size and stride based on the desired output size.

## Examples

### Basic Usage - Positional Syntax
```tcl
# Create a 3D tensor with shape (2, 3, 8)
set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0 48.0}
set tensor [torch::tensor_create $data float32 cpu false]
set input [torch::tensor_reshape $tensor {2 3 8}]

# Apply adaptive max pooling to reduce length dimension to 4
set result [torch::adaptive_maxpool1d $input 4]
set shape [torch::tensor_shape $result]  ;# Returns: 2 3 4
```

### Named Parameter Syntax
```tcl
# Same operation using named parameters
set result [torch::adaptive_maxpool1d -input $input -output_size 4]

# Alternative parameter names
set result [torch::adaptive_maxpool1d -tensor $input -outputSize 4]

# Parameters can be in any order
set result [torch::adaptive_maxpool1d -output_size 4 -input $input]
```

### camelCase Alias
```tcl
# Using the modern camelCase syntax
set result [torch::adaptiveMaxpool1d $input 4]
set result [torch::adaptiveMaxpool1d -input $input -outputSize 4]
```

### Different Output Sizes
```tcl
# Reduce to single value (global max pooling)
set global_max [torch::adaptive_maxpool1d $input 1]

# Reduce to half the original size
set half_size [torch::adaptive_maxpool1d -input $input -output_size 4]

# Reduce to smaller size
set small [torch::adaptive_maxpool1d -tensor $input -outputSize 2]
```

### Neural Network Context
```tcl
# Typical usage in a CNN
set conv_output [torch::conv1d $input $weight $bias 1 0 1]  ;# Shape: (batch, 64, 128)
set pooled [torch::adaptive_maxpool1d $conv_output 32]      ;# Shape: (batch, 64, 32)
set flattened [torch::tensor_flatten $pooled 1]             ;# Shape: (batch, 2048)
```

## Error Handling

### Missing Parameters
```tcl
# Error: Missing output_size
set result [catch {torch::adaptive_maxpool1d $input} msg]
# msg: "Wrong number of arguments: expected 'input output_size'"

# Error: Missing parameters in named syntax
set result [catch {torch::adaptive_maxpool1d -input $input} msg]
# msg: "Required parameters: input tensor and positive output_size"
```

### Invalid Parameters
```tcl
# Error: Invalid tensor handle
set result [catch {torch::adaptive_maxpool1d "invalid_tensor" 4} msg]
# msg: "Invalid input tensor name"

# Error: Invalid output_size
set result [catch {torch::adaptive_maxpool1d $input "invalid"} msg]
# msg: "Invalid output_size: must be an integer"

# Error: Zero output_size
set result [catch {torch::adaptive_maxpool1d -input $input -output_size 0} msg]
# msg: "Required parameters: input tensor and positive output_size"

# Error: Unknown parameter
set result [catch {torch::adaptive_maxpool1d -input $input -unknown_param 4} msg]
# msg: "Unknown parameter: -unknown_param"
```

## Use Cases

### 1. Feature Map Standardization
When building neural networks that need to handle variable-size inputs, adaptive pooling ensures consistent output dimensions:

```tcl
# Input sequences of different lengths all become same size
set seq1 [torch::adaptive_maxpool1d $variable_length_input 64]
set seq2 [torch::adaptive_maxpool1d $another_input 64]
# Both have same output length: 64
```

### 2. Global Feature Extraction
Extract a single representative value from each channel:

```tcl
# Global max pooling - one value per channel
set global_features [torch::adaptive_maxpool1d $feature_maps 1]
```

### 3. Multi-Scale Feature Processing
Create features at different scales:

```tcl
set scale1 [torch::adaptive_maxpool1d $input 32]  ;# Fine scale
set scale2 [torch::adaptive_maxpool1d $input 16]  ;# Medium scale
set scale3 [torch::adaptive_maxpool1d $input 8]   ;# Coarse scale
```

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
set result [torch::adaptive_maxpool1d $input 8]
```

**New (Named):**
```tcl
set result [torch::adaptive_maxpool1d -input $input -output_size 8]
```

**Modern (camelCase):**
```tcl
set result [torch::adaptiveMaxpool1d -input $input -outputSize 8]
```

### Benefits of Named Parameters
- **Clarity**: Parameter purpose is explicit
- **Flexibility**: Parameters can be specified in any order
- **Maintainability**: Code is more self-documenting
- **Error Prevention**: Less likely to mix up parameter positions

## Performance Notes

- Adaptive max pooling is computationally efficient
- The operation is differentiable and works with gradient computation
- Memory usage scales with output size, not input size
- Both syntax forms have identical performance

## Related Commands

- `torch::maxpool1d` - Regular 1D max pooling with fixed kernel size
- `torch::adaptive_avgpool1d` - 1D adaptive average pooling
- `torch::adaptive_maxpool2d` - 2D adaptive max pooling
- `torch::adaptive_maxpool3d` - 3D adaptive max pooling
- `torch::avgpool1d` - Regular 1D average pooling

## Implementation Details

- Uses PyTorch's `torch::adaptive_max_pool1d` function internally
- Returns only the pooled values (indices are discarded)
- Supports all PyTorch tensor data types
- Maintains gradient tracking for autograd

## Version Information

- **Introduced**: LibTorch TCL Extension v1.0
- **Refactored**: Added dual syntax support with named parameters and camelCase alias
- **Backward Compatibility**: All existing positional syntax code continues to work unchanged 