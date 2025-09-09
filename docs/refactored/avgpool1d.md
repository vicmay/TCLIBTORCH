# torch::avgpool1d

Applies 1D average pooling over a 1D input tensor.

## Syntax

### Snake_case (Original)
```tcl
torch::avgpool1d input kernel_size ?stride? ?padding? ?count_include_pad?
```

### Named Parameters  
```tcl
torch::avgpool1d -input tensor -kernel_size size ?-stride size? ?-padding size? ?-count_include_pad bool?
```

### CamelCase Alias
```tcl
torch::avgPool1d ...
```

## Parameters

### Required Parameters
- **input** (tensor): Input tensor handle
  - Aliases: `-input`, `-tensor`
  - Shape: `(N, C, L)` where N=batch, C=channels, L=length

- **kernel_size** (int): Size of the pooling kernel
  - Aliases: `-kernel_size`, `-kernelSize`
  - Must be a positive integer

### Optional Parameters
- **stride** (int): Stride of the pooling operation
  - Aliases: `-stride`
  - Default: Same as kernel_size
  - Must be a positive integer

- **padding** (int): Padding applied to the input
  - Aliases: `-padding`
  - Default: `0` (no padding)
  - Must be non-negative

- **count_include_pad** (bool): Whether to include padding in the average calculation
  - Aliases: `-count_include_pad`, `-countIncludePad`
  - Default: `1` (true - include padding)
  - Values: `0` (false) or `1` (true)

## Return Value

Returns a tensor handle containing the average pooled output.

## Examples

### Basic Usage
```tcl
# Create a 1D input tensor
set input [torch::randn {2 3 8}]

# Simple average pooling with kernel size 2
set result [torch::avgpool1d $input 2]
```

### Positional Syntax
```tcl
# With all parameters
set result [torch::avgpool1d $input 2 2 1 1]

# With stride
set result [torch::avgpool1d $input 3 2]

# With padding
set result [torch::avgpool1d $input 2 2 1]
```

### Named Parameter Syntax
```tcl
# Basic named parameters
set result [torch::avgpool1d -input $input -kernel_size 2]

# With stride
set result [torch::avgpool1d -input $input -kernel_size 2 -stride 2]

# With all parameters
set result [torch::avgpool1d -input $input -kernel_size 3 -stride 2 -padding 1 -count_include_pad 0]
```

### CamelCase Alias
```tcl
# Using camelCase alias
set result [torch::avgPool1d $input 2]
set result [torch::avgPool1d -input $input -kernelSize 2]
```

## Parameter Aliases

| Standard | Aliases |
|----------|---------|
| input | `-input`, `-tensor` |
| kernel_size | `-kernel_size`, `-kernelSize` |
| stride | `-stride` |
| padding | `-padding` |
| count_include_pad | `-count_include_pad`, `-countIncludePad` |

## Output Shape Calculation

```
output_length = floor((input_length + 2 * padding - kernel_size) / stride + 1)
```

## Error Handling

The command will throw an error if:
- Input tensor handle is invalid
- Required parameters are missing
- Unknown parameters are provided
- Parameter values are invalid

### Common Error Messages
- `"Invalid input tensor name"` - Invalid tensor handle
- `"Named parameters require pairs: -param value"` - Missing required parameters or incomplete parameter pairs
- `"Unknown parameter: param_name"` - Unrecognized parameter

## Use Cases

### 1. Basic 1D Average Pooling
```tcl
# Reduce sequence length by half
set input [torch::randn {1 64 100}]
set pooled [torch::avgpool1d $input 2]
# Output shape: {1 64 50}
```

### 2. Overlapping Pooling
```tcl
# Overlapping pooling with stride smaller than kernel
set input [torch::randn {1 32 20}]
set pooled [torch::avgpool1d $input 3 2]
# Creates overlapping pooling windows
```

### 3. With Padding
```tcl
# Preserve sequence length with padding
set input [torch::randn {1 32 10}]
set pooled [torch::avgpool1d $input 3 1 1]
# Padding helps maintain sequence length
```

### 4. Count Include Pad Control
```tcl
# Include padding in average calculation
set result1 [torch::avgpool1d $input 3 3 1 1]

# Exclude padding from average calculation
set result2 [torch::avgpool1d $input 3 3 1 0]
```

## Integration with Other Operations

### In RNN/CNN Architectures
```tcl
# Common pattern in sequence processing
set x [torch::conv1d $input $conv_weights -stride 1 -padding 1]
set x [torch::relu $x]
set x [torch::avgpool1d $x 2]  # Downsample sequence
```

### For Time Series Processing
```tcl
# Process time series data
set timeseries [torch::randn {1 1 1000}]  # 1000 time steps
set pooled [torch::avgpool1d $timeseries 10]  # Reduce to 100 steps
```

### For Signal Processing
```tcl
# Process 1D signals
set signal [torch::randn {1 1 256}]  # 256 samples
set smoothed [torch::avgpool1d $signal 4]  # Smooth and downsample
```

## Performance Considerations

1. **Memory Usage**: Average pooling reduces memory requirements by downsampling
2. **Computation**: Efficient implementation for long sequences
3. **Kernel Size**: Larger kernels reduce output size more aggressively
4. **Stride**: Smaller strides create more overlap but larger outputs

## Backward Compatibility

- All existing positional syntax calls continue to work unchanged
- New named parameter syntax is additive, not replacement
- CamelCase aliases provide modern API consistency

## See Also

- [torch::maxpool1d](maxpool1d.md) - 1D max pooling
- [torch::avgpool2d](avgpool2d.md) - 2D average pooling
- [torch::conv1d](conv1d.md) - 1D convolution

## Migration Guide

### From Positional to Named Parameters
```tcl
# Old positional syntax
torch::avgpool1d $input 2 2 1 1

# New named parameter syntax
torch::avgpool1d -input $input -kernel_size 2 -stride 2 -padding 1 -count_include_pad 1
```

### Using CamelCase Aliases
```tcl
# Convert to modern camelCase
torch::avgPool1d -input $input -kernelSize 2
```

## Mathematical Details

### Average Calculation
For each pooling window, the output is:
```
output[i] = mean(input[i*stride:i*stride+kernel_size])
```

### Padding Behavior
- When `count_include_pad=1`: Padding values (zeros) are included in the average
- When `count_include_pad=0`: Only actual input values are averaged, padding is excluded

### Example with Padding
```tcl
# Input: [1, 2, 3, 4] with padding=1, kernel_size=3
# Padded input: [0, 1, 2, 3, 4, 0]

# count_include_pad=1: averages include zeros
# count_include_pad=0: averages exclude zeros (more accurate for edge cases)
```

## Best Practices

1. **Choose appropriate kernel size**: Balance between smoothing and information loss
2. **Consider stride carefully**: Smaller strides preserve more information
3. **Use padding judiciously**: Padding can help maintain sequence length but may introduce bias
4. **Count include pad**: Set to false when padding should not affect averages
5. **Sequence processing**: Useful for reducing temporal resolution in time series
6. **Signal smoothing**: Effective for noise reduction in 1D signals

## Common Patterns

### Temporal Downsampling
```tcl
# Reduce temporal resolution in video/audio processing
set features [torch::conv1d $input $weights]
set downsampled [torch::avgpool1d $features 2]
```

### Signal Smoothing
```tcl
# Smooth noisy 1D signals
set noisy_signal [torch::randn {1 1 1000}]
set smooth_signal [torch::avgpool1d $noisy_signal 5 1 2]
```

### Multi-scale Processing
```tcl
# Create multiple scales of the same sequence
set scale1 [torch::avgpool1d $input 2]    # Half resolution
set scale2 [torch::avgpool1d $scale1 2]   # Quarter resolution
set scale3 [torch::avgpool1d $scale2 2]   # Eighth resolution
```