# torch::fftshift

Shifts the zero-frequency component to the center of the spectrum. This is typically used for visualizing the frequency spectrum and preparing data for FFT operations.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::fftshift tensor ?dim?
```

### Named Parameter Syntax  
```tcl
torch::fftshift -input tensor [-dim dimension]
torch::fftshift -tensor tensor [-dimension dimension]
```

### CamelCase Alias
```tcl
torch::fftShift tensor ?dim?
torch::fftShift -input tensor [-dim dimension]
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor` / `-input` / `-tensor` | tensor | Yes | Input tensor to shift |
| `dim` / `-dim` / `-dimension` | integer | No | Dimension along which to shift. If not specified, shifts all dimensions |

## Returns

Returns a tensor with the same shape and dtype as the input, but with the zero-frequency component shifted to the center.

## Description

The FFT shift operation rearranges the elements of a tensor by shifting the zero-frequency component to the center. This is particularly useful for:

- **Frequency Domain Visualization**: Moving the DC component to the center for better visualization
- **Image Processing**: Centering the frequency components in 2D FFT operations  
- **Signal Processing**: Preparing data for frequency domain analysis

The shift is performed by rolling the tensor by half its size along the specified dimension(s). For even-sized dimensions, the shift is `size // 2`. For odd-sized dimensions, the behavior is similar but naturally handles the center position.

### Mathematical Operation

For a 1D tensor of size N, fftshift moves the elements as follows:
- Elements at indices [0, 1, ..., N//2-1] move to [N//2, N//2+1, ..., N-1]
- Elements at indices [N//2, N//2+1, ..., N-1] move to [0, 1, ..., N//2-1]

## Examples

### Basic Usage

```tcl
# Create a 1D tensor
set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4}]

# Shift using positional syntax
set result1 [torch::fftshift $input]

# Shift using named parameters
set result2 [torch::fftshift -input $input]

# Using camelCase alias
set result3 [torch::fftShift $input]
```

### Dimension-Specific Shifting

```tcl
# Create a 2D tensor
set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 4}]

# Shift only dimension 0 (rows)
set result1 [torch::fftshift $input 0]

# Shift only dimension 1 (columns) using named syntax
set result2 [torch::fftshift -input $input -dim 1]

# Shift all dimensions
set result3 [torch::fftshift $input]
```

### Complex Signal Processing

```tcl
# Simulate frequency domain data
set freq_data [torch::tensor_create -data {0.1 0.2 0.9 0.8 0.7 0.3 0.2 0.1} -shape {8}]

# Shift zero frequency to center for visualization
set centered [torch::fftshift $freq_data]

# Process in frequency domain (example: apply filter)
# ... processing code ...

# Shift back using ifftshift if needed
set restored [torch::ifftshift $centered]
```

### 2D Image Processing

```tcl
# Create a 2D frequency spectrum
set spectrum [torch::tensor_create -data {1.0 0.1 0.05 0.1 0.1 0.8 0.1 0.05 0.05 0.1 0.2 0.05 0.1 0.05 0.05 1.0} -shape {4 4}]

# Center the DC component
set centered_spectrum [torch::fftshift $spectrum]

# Shift only horizontally
set h_shifted [torch::fftshift $spectrum 1]

# Shift only vertically  
set v_shifted [torch::fftshift $spectrum 0]
```

## Use Cases

### 1. Frequency Domain Visualization

```tcl
# After taking FFT of a signal
set signal [torch::tensor_create -data {1.0 0.5 -0.2 -0.8 -0.2 0.5} -shape {6}]
set fft_result [torch::fft $signal]
set centered_fft [torch::fftshift $fft_result]
# Now the spectrum is centered for better visualization
```

### 2. Image Frequency Analysis

```tcl
# For 2D image in frequency domain
set image [torch::tensor_create -data {...} -shape {256 256}]
set fft2d [torch::fft2d $image]
set centered [torch::fftshift $fft2d]
# Low frequencies are now at the center of the image
```

### 3. Filter Design

```tcl
# Design a frequency domain filter
set filter [torch::zeros -shape {64}]
# Set desired frequency response
# ...
set shifted_filter [torch::fftshift $filter]
# Apply to signal in frequency domain
```

## Error Handling

The function provides comprehensive error checking:

- **Missing Input**: Returns error if no tensor is provided
- **Invalid Tensor**: Returns error if tensor handle is invalid
- **Invalid Dimension**: Returns error if dimension is out of range
- **Invalid Parameters**: Returns error for unknown parameter names
- **Missing Values**: Returns error if parameter values are missing

```tcl
# Error examples
catch {torch::fftshift} result
# Returns: "Usage: torch::fftshift tensor ?dim? | torch::fftshift -input tensor [-dim dimension]"

catch {torch::fftshift invalid_tensor} result  
# Returns: "Error in fftshift: Invalid tensor handle"

catch {torch::fftshift $tensor 10} result
# Returns: "Error in fftshift: Dimension out of range"
```

## Performance Notes

- **Memory Efficient**: The operation uses PyTorch's roll function internally, which is memory efficient
- **Dimension Handling**: Specifying a dimension is more efficient than shifting all dimensions
- **Data Types**: Works with all numeric data types (float32, float64, int32, etc.)
- **GPU Support**: Fully compatible with CUDA tensors when available

## Relationship to Other Functions

- **torch::ifftshift**: Inverse operation that shifts the center back to the origin
- **torch::roll**: Underlying operation used to implement the shift
- **torch::fft**: Often used together for frequency domain processing

## Mathematical Properties

1. **Reversible**: `fftshift(fftshift(x))` returns the original tensor for even sizes
2. **Dimension Preserving**: Output has the same shape as input
3. **Type Preserving**: Output has the same data type as input
4. **Frequency Centering**: Moves the zero-frequency component to the center

## Migration Guide

### From Legacy Positional Syntax

```tcl
# Old positional syntax
set result [torch::fftshift $tensor]
set result [torch::fftshift $tensor $dim]

# New named parameter syntax (recommended)
set result [torch::fftshift -input $tensor]
set result [torch::fftshift -input $tensor -dim $dim]

# CamelCase alternative
set result [torch::fftShift -input $tensor]
set result [torch::fftShift -input $tensor -dim $dim]
```

### Benefits of Named Parameters

1. **Clarity**: Parameter names make the code self-documenting
2. **Flexibility**: Parameters can be specified in any order
3. **Extensibility**: New parameters can be added without breaking existing code
4. **Consistency**: Matches the pattern used across the modern PyTorch TCL API

## Best Practices

1. **Use Named Parameters**: For new code, prefer the named parameter syntax
2. **Specify Dimensions**: When working with multi-dimensional data, explicitly specify the dimension for clarity
3. **Combine with FFT**: Often used in conjunction with FFT operations for proper frequency domain visualization
4. **Check Dimensions**: Ensure the dimension parameter is valid for your tensor's shape

## See Also

- [torch::ifftshift](ifftshift.md) - Inverse FFT shift operation
- [torch::roll](roll.md) - General tensor rolling operation  
- [torch::fft](fft.md) - Fast Fourier Transform
- [torch::fft2d](fft2d.md) - 2D Fast Fourier Transform 