# torch::tensor_conv_transpose1d

**1D Transposed Convolution Operation**

Applies a 1D transposed convolution (also known as deconvolution) over an input tensor. Transposed convolution is commonly used for upsampling in neural networks.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_conv_transpose1d input weight [bias] [stride] [padding] [output_padding] [groups] [dilation]
```

### Named Parameter Syntax (Recommended)
```tcl
torch::tensor_conv_transpose1d -input input_tensor -weight weight_tensor [-bias bias_tensor] [-stride stride_value] [-padding padding_value] [-output_padding output_padding_value] [-groups groups_value] [-dilation dilation_value]
```

### camelCase Alias
```tcl
torch::tensorConvTranspose1d [same parameters as above]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | Tensor | **Required** | Input tensor of shape `(batch_size, in_channels, sequence_length)` or `(in_channels, sequence_length)` |
| `weight` | Tensor | **Required** | Filter weights of shape `(in_channels, out_channels/groups, kernel_size)` |
| `bias` | Tensor | `none` | Bias tensor of shape `(out_channels)`. Use "none" or empty string to omit |
| `stride` | Integer | `1` | Stride of the transposed convolution |
| `padding` | Integer | `0` | Zero-padding added to both sides of the input |
| `output_padding` | Integer | `0` | Additional size added to one side of the output shape |
| `groups` | Integer | `1` | Number of blocked connections from input channels to output channels |
| `dilation` | Integer | `1` | Spacing between kernel elements |

## Returns

A tensor containing the transposed convolution result.

## Examples

### Basic Transposed Convolution
```tcl
# Create input tensor (batch=2, in_channels=3, sequence_length=8)
set input [torch::ones -shape {2 3 8} -dtype float32]

# Create weight tensor (in_channels=3, out_channels=4, kernel_size=3)  
set weight [torch::ones -shape {3 4 3} -dtype float32]

# Apply 1D transposed convolution
set result [torch::tensor_conv_transpose1d -input $input -weight $weight]
puts "Output shape: [torch::tensor_shape $result]"  ;# Should be {2 4 10}
```

### Transposed Convolution with Bias
```tcl
set input [torch::ones -shape {2 3 8} -dtype float32]
set weight [torch::ones -shape {3 4 3} -dtype float32]
set bias [torch::zeros -shape {4} -dtype float32]

# Named parameter syntax
set result [torch::tensor_conv_transpose1d -input $input -weight $weight -bias $bias]

# Positional syntax (backward compatible)
set result [torch::tensor_conv_transpose1d $input $weight $bias]
```

### Upsampling with Stride
```tcl
set input [torch::ones -shape {1 3 5} -dtype float32]
set weight [torch::ones -shape {3 6 3} -dtype float32]

# Upsample by factor of 2
set result [torch::tensor_conv_transpose1d -input $input -weight $weight -stride 2]
puts "Input length: 5, Output length: [lindex [torch::tensor_shape $result] 2]"
```

### Fine-grained Output Control
```tcl
set input [torch::ones -shape {1 2 4} -dtype float32]
set weight [torch::ones -shape {2 3 3} -dtype float32]

# Control exact output size with output_padding
set result [torch::tensor_conv_transpose1d -input $input -weight $weight -stride 2 -output_padding 1]
puts "Output shape: [torch::tensor_shape $result]"
```

### Grouped Transposed Convolution
```tcl
# Input with 4 channels, using 2 groups
set input [torch::ones -shape {1 4 6} -dtype float32]

# Weight: in_channels=4, out_channels_per_group=3, kernel_size=3
set weight [torch::ones -shape {4 3 3} -dtype float32]

set result [torch::tensor_conv_transpose1d -input $input -weight $weight -groups 2]
puts "Output channels: [lindex [torch::tensor_shape $result] 1]"  ;# Should be 6 (2 groups × 3 out_channels)
```

### Dilated Transposed Convolution
```tcl
set input [torch::ones -shape {1 3 6} -dtype float32]
set weight [torch::ones -shape {3 5 3} -dtype float32]

# Dilation increases the effective kernel size
set result [torch::tensor_conv_transpose1d -input $input -weight $weight -dilation 2]
```

### Complex Parameter Combination
```tcl
set input [torch::ones -shape {2 3 8} -dtype float32]
set weight [torch::ones -shape {3 4 3} -dtype float32]
set bias [torch::ones -shape {4} -dtype float32]

# Multiple parameters for precise control
set result [torch::tensor_conv_transpose1d -input $input -weight $weight -bias $bias -stride 2 -padding 1 -output_padding 1 -dilation 1 -groups 1]
```

### Using camelCase Alias
```tcl
set input [torch::ones -shape {1 3 5} -dtype float32]
set weight [torch::ones -shape {3 5 3} -dtype float32]

# camelCase alias works with both syntaxes
set result1 [torch::tensorConvTranspose1d $input $weight]
set result2 [torch::tensorConvTranspose1d -input $input -weight $weight -stride 2]
```

## Output Shape Calculation

For an input of shape `(N, C_in, L_in)` and kernel size `K`:

```
L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
```

Output shape: `(N, C_out, L_out)`

Where:
- `N` = batch size
- `C_in` = input channels  
- `C_out` = output channels (from weight tensor: `in_channels * out_channels_per_group`)
- `L_in` = input sequence length
- `L_out` = output sequence length

## Weight Tensor Shape

**Important**: The weight tensor shape for transposed convolution is different from regular convolution:

- **Regular conv1d**: `(out_channels, in_channels/groups, kernel_size)`
- **Transposed conv1d**: `(in_channels, out_channels/groups, kernel_size)`

## Common Use Cases

### Upsampling
```tcl
# Upsample a signal by factor of 2
set input [torch::ones -shape {1 16 50} -dtype float32]
set weight [torch::randn -shape {16 32 4} -dtype float32]  # kernel_size=4 for smooth upsampling
set upsampled [torch::tensor_conv_transpose1d -input $input -weight $weight -stride 2 -padding 1]
```

### Decoder Networks
```tcl
# Typical decoder layer in autoencoders
set encoded [torch::ones -shape {8 64 25} -dtype float32]  # Batch=8, features=64, length=25
set decoder_weight [torch::randn -shape {64 32 3} -dtype float32]
set decoder_bias [torch::zeros -shape {32} -dtype float32]

set decoded [torch::tensor_conv_transpose1d -input $encoded -weight $decoder_weight -bias $decoder_bias -stride 2 -padding 1]
```

## Error Handling

The command validates all parameters and provides clear error messages:

```tcl
# Missing required parameters
torch::tensor_conv_transpose1d -weight $weight  ;# Error: input parameter required

# Invalid tensor names
torch::tensor_conv_transpose1d -input invalid_tensor -weight $weight  ;# Error: invalid tensor

# Invalid parameter values
torch::tensor_conv_transpose1d -input $input -weight $weight -stride invalid  ;# Error: invalid stride

# Unknown parameters
torch::tensor_conv_transpose1d -input $input -weight $weight -unknown_param 1  ;# Error: unknown parameter
```

## Notes

- Input tensors can be 2D (unbatched) or 3D (batched)
- All parameters except `input` and `weight` are optional
- The bias parameter can be omitted by using "none" or an empty string
- When using groups > 1, ensure `in_channels` is divisible by `groups`
- Output channels = `in_channels * out_channels_per_group`
- Both syntaxes produce identical results and can be used interchangeably
- This command maintains full backward compatibility with existing code

## Relationship to Regular Convolution

Transposed convolution is the mathematical transpose of regular convolution:

```tcl
# If regular conv1d transforms input(H_in) → output(H_out)
# Then transposed conv1d transforms input(H_out) → output(H_in) (approximately)

# Example:
set original [torch::ones -shape {1 3 10} -dtype float32]
set conv_weight [torch::ones -shape {5 3 3} -dtype float32]          # For regular conv
set tconv_weight [torch::ones -shape {3 5 3} -dtype float32]         # For transposed conv (note the swap)

# Regular convolution: 10 → 8
set conv_result [torch::tensor_conv1d -input $original -weight $conv_weight]

# Transposed convolution: 8 → 10
set tconv_result [torch::tensor_conv_transpose1d -input $conv_result -weight $tconv_weight]
```

## Migration from Old Syntax

```tcl
# Old positional syntax (limited parameters)
set result [torch::tensor_conv_transpose1d $input $weight $bias 2]

# New named parameter syntax (all parameters available)
set result [torch::tensor_conv_transpose1d -input $input -weight $weight -bias $bias -stride 2 -padding 0 -output_padding 0 -groups 1 -dilation 1]

# camelCase alias
set result [torch::tensorConvTranspose1d -input $input -weight $weight -bias $bias -stride 2 -padding 0 -output_padding 0 -groups 1 -dilation 1]
``` 