# torch::tensor_conv1d

**1D Convolution Operation**

Applies a 1D convolution over an input tensor.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_conv1d input weight [bias] [stride] [padding] [dilation] [groups]
```

### Named Parameter Syntax (Recommended)
```tcl
torch::tensor_conv1d -input input_tensor -weight weight_tensor [-bias bias_tensor] [-stride stride_value] [-padding padding_value] [-dilation dilation_value] [-groups groups_value]
```

### camelCase Alias
```tcl
torch::tensorConv1d [same parameters as above]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | Tensor | **Required** | Input tensor of shape `(batch_size, in_channels, sequence_length)` or `(in_channels, sequence_length)` |
| `weight` | Tensor | **Required** | Filter weights of shape `(out_channels, in_channels/groups, kernel_size)` |
| `bias` | Tensor | `none` | Bias tensor of shape `(out_channels)`. Use "none" or empty string to omit |
| `stride` | Integer | `1` | Stride of the convolution |
| `padding` | Integer | `0` | Zero-padding added to both sides of the input |
| `dilation` | Integer | `1` | Spacing between kernel elements |
| `groups` | Integer | `1` | Number of blocked connections from input channels to output channels |

## Returns

A tensor containing the convolution result.

## Examples

### Basic Convolution
```tcl
# Create input tensor (batch=2, in_channels=3, sequence_length=10)
set input [torch::ones -shape {2 3 10} -dtype float32]

# Create weight tensor (out_channels=4, in_channels=3, kernel_size=3)  
set weight [torch::ones -shape {4 3 3} -dtype float32]

# Apply 1D convolution
set result [torch::tensor_conv1d -input $input -weight $weight]
```

### Convolution with Bias
```tcl
set input [torch::ones -shape {2 3 10} -dtype float32]
set weight [torch::ones -shape {4 3 3} -dtype float32]
set bias [torch::zeros -shape {4} -dtype float32]

# Named parameter syntax
set result [torch::tensor_conv1d -input $input -weight $weight -bias $bias]

# Positional syntax (backward compatible)
set result [torch::tensor_conv1d $input $weight $bias]
```

### Advanced Parameters
```tcl
set input [torch::ones -shape {1 6 20} -dtype float32]
set weight [torch::ones -shape {8 3 5} -dtype float32]

# Convolution with stride, padding, and groups
set result [torch::tensor_conv1d -input $input -weight $weight -stride 2 -padding 2 -groups 2]
```

### Grouped Convolution
```tcl
# Input with 4 channels, using 2 groups
set input [torch::ones -shape {1 4 10} -dtype float32]

# Weight: out_channels=4, in_channels_per_group=2, kernel_size=3
set weight [torch::ones -shape {4 2 3} -dtype float32]

set result [torch::tensor_conv1d -input $input -weight $weight -groups 2]
```

### Dilated Convolution
```tcl
set input [torch::ones -shape {1 3 15} -dtype float32]
set weight [torch::ones -shape {6 3 3} -dtype float32]

# Dilation increases the effective kernel size
set result [torch::tensor_conv1d -input $input -weight $weight -dilation 2]
```

### Using camelCase Alias
```tcl
set input [torch::ones -shape {1 3 10} -dtype float32]
set weight [torch::ones -shape {5 3 3} -dtype float32]

# camelCase alias works with both syntaxes
set result1 [torch::tensorConv1d $input $weight]
set result2 [torch::tensorConv1d -input $input -weight $weight -stride 2]
```

## Output Shape Calculation

For an input of shape `(N, C_in, L_in)` and kernel size `K`:

```
L_out = floor((L_in + 2*padding - dilation*(K-1) - 1) / stride + 1)
```

Output shape: `(N, C_out, L_out)`

Where:
- `N` = batch size
- `C_in` = input channels  
- `C_out` = output channels (from weight tensor)
- `L_in` = input sequence length
- `L_out` = output sequence length

## Error Handling

The command validates all parameters and provides clear error messages:

```tcl
# Missing required parameters
torch::tensor_conv1d -weight $weight  ;# Error: input parameter required

# Invalid tensor names
torch::tensor_conv1d -input invalid_tensor -weight $weight  ;# Error: invalid tensor

# Invalid parameter values
torch::tensor_conv1d -input $input -weight $weight -stride invalid  ;# Error: invalid stride

# Unknown parameters
torch::tensor_conv1d -input $input -weight $weight -unknown_param 1  ;# Error: unknown parameter
```

## Notes

- Input tensors can be 2D (unbatched) or 3D (batched)
- All parameters except `input` and `weight` are optional
- The bias parameter can be omitted by using "none" or an empty string
- When using groups > 1, ensure `in_channels` is divisible by `groups`
- Both syntaxes produce identical results and can be used interchangeably
- This command maintains full backward compatibility with existing code

## Migration from Old Syntax

```tcl
# Old positional syntax
set result [torch::tensor_conv1d $input $weight $bias 2 1 1 1]

# New named parameter syntax (equivalent)
set result [torch::tensor_conv1d -input $input -weight $weight -bias $bias -stride 2 -padding 1 -dilation 1 -groups 1]

# camelCase alias
set result [torch::tensorConv1d -input $input -weight $weight -bias $bias -stride 2 -padding 1 -dilation 1 -groups 1]
``` 