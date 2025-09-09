# torch::conv_transpose1d

Applies a 1D transposed convolution operator over an input signal composed of several input planes. Also known as fractionally-strided convolution or deconvolution.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::conv_transpose1d input weight ?bias? ?stride? ?padding? ?output_padding? ?groups? ?dilation?
```

### Named Parameter Syntax  
```tcl
torch::conv_transpose1d -input input_tensor -weight weight_tensor ?-bias bias_tensor? ?-stride stride_value? ?-padding padding_value? ?-output_padding output_padding_value? ?-groups groups_value? ?-dilation dilation_value?
```

### CamelCase Alias
```tcl
torch::convTranspose1d ...
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | tensor | required | Input tensor of shape `(N, C_in, L_in)` |
| `weight` | tensor | required | Convolution kernel of shape `(C_in, C_out/groups, kS)` |
| `bias` | tensor | none | Optional bias tensor of shape `(C_out)` |
| `stride` | integer | 1 | Stride of the convolution |
| `padding` | integer | 0 | Padding added to both sides of the input |
| `output_padding` | integer | 0 | Additional size added to the output shape |
| `groups` | integer | 1 | Number of blocked connections from input to output channels |
| `dilation` | integer | 1 | Spacing between kernel elements |

## Returns

Returns a tensor handle containing the result of the 1D transposed convolution operation.

**Output Shape**: `(N, C_out, L_out)` where:
```
L_out = (L_in - 1) × stride - 2 × padding + kernel_size + output_padding
```

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create input tensor (batch=1, channels=16, length=25)
set input [torch::randn -shape {1 16 25}]

# Create weight tensor (in_channels=16, out_channels=32, kernel_size=4)
set weight [torch::randn -shape {16 32 4}]

# Apply 1D transposed convolution
set result [torch::conv_transpose1d $input $weight]
puts "Output shape: [torch::tensor_shape $result]"
```

### With Bias and Stride (Positional Syntax)
```tcl
# Create tensors
set input [torch::randn -shape {1 8 50}]
set weight [torch::randn -shape {8 16 3}]
set bias [torch::randn -shape {16}]

# Apply with stride=2 for upsampling
set result [torch::conv_transpose1d $input $weight $bias 2]
puts "Upsampled shape: [torch::tensor_shape $result]"
```

### Named Parameter Syntax
```tcl
# Create tensors
set input [torch::randn -shape {2 32 128}]
set weight [torch::randn -shape {32 64 5}]
set bias [torch::randn -shape {64}]

# Apply with named parameters
set result [torch::conv_transpose1d -input $input -weight $weight -bias $bias \
                                   -stride 2 -padding 1 -output_padding 1]
puts "Result shape: [torch::tensor_shape $result]"
```

### CamelCase Alias
```tcl
# Using camelCase alias with named parameters
set input [torch::randn -shape {1 16 100}]
set weight [torch::randn -shape {16 8 3}]

set result [torch::convTranspose1d -input $input -weight $weight -stride 1 -padding 1]
puts "Shape: [torch::tensor_shape $result]"
```

### Advanced Usage with All Parameters
```tcl
# Create complex setup
set input [torch::randn -shape {4 64 256}]
set weight [torch::randn -shape {32 128 7}]  # Note: 64/2 = 32 for groups=2
set bias [torch::randn -shape {128}]

# Apply with all parameters using named syntax
set result [torch::conv_transpose1d -input $input -weight $weight -bias $bias \
                                   -stride 3 -padding 2 -output_padding 1 \
                                   -groups 2 -dilation 2]
puts "Complex result shape: [torch::tensor_shape $result]"
```

## Use Cases

### 1. Upsampling for Audio Processing
```tcl
# Upsample audio signal from 16kHz to 32kHz
set audio_16k [torch::randn -shape {1 1 16000}]  # 1 second at 16kHz
set upsample_filter [torch::randn -shape {1 1 4}]

# Use stride=2 to double the sampling rate
set audio_32k [torch::conv_transpose1d $audio_16k $upsample_filter none 2]
puts "Upsampled from [lindex [torch::tensor_shape $audio_16k] 2] to [lindex [torch::tensor_shape $audio_32k] 2] samples"
```

### 2. Generative Models (Decoder)
```tcl
# Decoder layer in a generative model
set latent [torch::randn -shape {8 512 16}]      # Batch of latent codes
set decoder_weight [torch::randn -shape {512 256 4}]
set decoder_bias [torch::randn -shape {256}]

# Expand latent space
set decoded [torch::conv_transpose1d $latent $decoder_weight $decoder_bias 2 1]
puts "Decoded shape: [torch::tensor_shape $decoded]"
```

### 3. Signal Reconstruction
```tcl
# Reconstruct signal from compressed representation
set compressed [torch::randn -shape {1 128 64}]
set reconstruction_filter [torch::randn -shape {128 1 8}]

# Reconstruct with overlapping windows
set reconstructed [torch::conv_transpose1d $compressed $reconstruction_filter \
                                          none 4 3 0 1 1]
puts "Reconstructed signal shape: [torch::tensor_shape $reconstructed]"
```

### 4. Feature Map Upsampling
```tcl
# Upsample feature maps in a neural network
set features [torch::randn -shape {16 256 32}]   # Batch of feature maps
set upsample_weights [torch::randn -shape {256 128 3}]

# Upsample with learnable filters
set upsampled [torch::convTranspose1d -input $features -weight $upsample_weights \
                                      -stride 2 -padding 1 -output_padding 1]
puts "Upsampled features: [torch::tensor_shape $upsampled]"
```

## Parameter Details

### Stride
- Controls the upsampling factor
- `stride=1`: No upsampling
- `stride=2`: Doubles the output length
- Higher values create more aggressive upsampling

### Padding
- Reduces the output size by removing border effects
- Applied symmetrically to both sides of the input
- `padding=0`: No padding removal
- `padding=1`: Removes 1 element from each side of output

### Output Padding
- Adds extra size to one side of the output
- Useful for controlling exact output dimensions
- `output_padding=0`: No extra padding
- `output_padding=1`: Adds 1 to the output length

### Groups
- Controls grouped convolutions
- `groups=1`: Standard convolution
- `groups=C_in`: Depthwise convolution
- Must satisfy: `C_in % groups == 0` and `C_out % groups == 0`

### Dilation
- Controls spacing between kernel elements
- `dilation=1`: Standard convolution
- `dilation=2`: Skip every other input element
- Effective kernel size = `(kernel_size - 1) × dilation + 1`

## Error Handling

The function performs comprehensive error checking:

```tcl
# Missing required parameters
catch {torch::conv_transpose1d -weight $weight} error
puts $error  # "Required parameters missing: input and weight must be specified"

# Invalid tensor handles
catch {torch::conv_transpose1d invalid_input $weight} error  
puts $error  # "Invalid input tensor name"

# Invalid parameter values
catch {torch::conv_transpose1d -input $input -weight $weight -stride invalid} error
puts $error  # "Invalid stride value"

# Unknown parameters
catch {torch::conv_transpose1d -input $input -weight $weight -unknown_param 1} error
puts $error  # "Unknown parameter: -unknown_param"
```

## Mathematical Background

1D transposed convolution is the gradient of 1D convolution with respect to its input. It's commonly used for:

- **Upsampling**: Increasing the resolution of 1D signals
- **Deconvolution**: Reversing the effect of convolution (approximately)
- **Generative modeling**: Expanding compressed representations
- **Signal reconstruction**: Rebuilding original signals from processed versions

The operation can be viewed as:
1. Insert zeros between input elements (controlled by stride)
2. Apply standard convolution
3. Add output padding if specified

## Performance Considerations

- **Memory usage**: Transposed convolution can be memory-intensive for large inputs
- **Computational cost**: O(N × C_in × C_out × kernel_size) operations
- **Groups parameter**: Reduces computation by factor of `groups`
- **Dilation**: Increases effective receptive field without additional parameters

## Migration Guide

### From Positional to Named Syntax

```tcl
# Old positional syntax
set result [torch::conv_transpose1d $input $weight $bias 2 1 0 1 1]

# New named syntax  
set result [torch::conv_transpose1d -input $input -weight $weight -bias $bias \
                                   -stride 2 -padding 1 -output_padding 0 \
                                   -groups 1 -dilation 1]

# Or using camelCase alias
set result [torch::convTranspose1d -input $input -weight $weight -bias $bias \
                                   -stride 2 -padding 1]
```

### Benefits of Named Syntax
- **Clarity**: Parameter names make code self-documenting
- **Flexibility**: Parameters can be specified in any order
- **Maintainability**: Easier to modify and understand
- **Error prevention**: Less likely to pass parameters in wrong order

## See Also

- [conv1d](conv1d.md) - 1D convolution (forward operation)
- [conv2d](conv2d.md) - 2D convolution
- [conv_transpose2d](conv_transpose2d.md) - 2D transposed convolution
- [conv_transpose3d](conv_transpose3d.md) - 3D transposed convolution

## Version History

- **v1.0**: Initial implementation with positional syntax
- **v2.0**: Added named parameter syntax and camelCase alias
- **v2.1**: Enhanced error handling and documentation 