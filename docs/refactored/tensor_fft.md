# torch::tensor_fft

Computes the Fast Fourier Transform (FFT) of a tensor.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::tensor_fft tensor ?dim?
```

### Named Parameters (New)
```tcl
torch::tensor_fft -tensor tensor -dim dim
```

### CamelCase Alias
```tcl
torch::tensorFft tensor ?dim?
torch::tensorFft -tensor tensor -dim dim
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tensor` | string | Yes | - | Name of the input tensor |
| `dim` | int | No | -1 | Dimension along which to compute FFT. -1 means last dimension |

## Description

The `torch::tensor_fft` command computes the Fast Fourier Transform of a tensor. The FFT is a mathematical algorithm that transforms a signal from the time domain to the frequency domain.

- **Input**: A tensor of any shape
- **Output**: A complex tensor with the same shape as the input
- **Default behavior**: If no dimension is specified, FFT is computed along the last dimension

## Examples

### Basic Usage

```tcl
# Create a simple signal
set signal [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]

# Compute FFT using positional syntax
set fft_result [torch::tensor_fft $signal]

# Compute FFT using named parameters
set fft_result [torch::tensor_fft -tensor $signal]

# Compute FFT using camelCase alias
set fft_result [torch::tensorFft $signal]
```

### Specifying Dimension

```tcl
# Create a 2D tensor
set tensor_2d [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]

# FFT along dimension 0 (rows)
set fft_dim0 [torch::tensor_fft $tensor_2d 0]

# FFT along dimension 1 (columns) using named parameters
set fft_dim1 [torch::tensor_fft -tensor $tensor_2d -dim 1]

# FFT along dimension 1 using camelCase alias
set fft_dim1 [torch::tensorFft -tensor $tensor_2d -dim 1]
```

### Working with Complex Results

```tcl
# Create a signal
set signal [torch::tensor_create {1.0 0.0 1.0 0.0} float32 cpu true]

# Compute FFT
set fft_result [torch::tensor_fft $signal]

# Get the shape of the result
set shape [torch::tensor_shape $fft_result]
puts "FFT result shape: $shape"
```

## Error Handling

The command provides clear error messages for various error conditions:

- **Invalid tensor name**: When the specified tensor doesn't exist
- **Invalid dim parameter**: When the dimension parameter is not a valid integer
- **Missing tensor parameter**: When no tensor is provided
- **Unknown parameter**: When using named parameters with invalid parameter names

## Migration Guide

### From Old Syntax to New Syntax

**Old (Positional Only)**:
```tcl
torch::tensor_fft tensor_name 0
```

**New (Named Parameters)**:
```tcl
torch::tensor_fft -tensor tensor_name -dim 0
```

**New (CamelCase)**:
```tcl
torch::tensorFft tensor_name 0
torch::tensorFft -tensor tensor_name -dim 0
```

### Backward Compatibility

The old positional syntax is fully supported and will continue to work:

```tcl
# This still works exactly as before
torch::tensor_fft my_tensor
torch::tensor_fft my_tensor 1
```

## Mathematical Background

The FFT transforms a signal from the time domain to the frequency domain. For a signal of length N, the FFT computes:

```
X[k] = Σ x[n] * e^(-j*2π*k*n/N)
```

where:
- `X[k]` is the k-th frequency component
- `x[n]` is the n-th time sample
- `N` is the length of the signal

## Related Commands

- `torch::tensor_ifft` - Inverse Fast Fourier Transform
- `torch::tensor_fft2d` - 2D Fast Fourier Transform
- `torch::tensor_ifft2d` - 2D Inverse Fast Fourier Transform
- `torch::tensor_rfft` - Real-valued Fast Fourier Transform
- `torch::tensor_irfft` - Inverse Real-valued Fast Fourier Transform

## Notes

- The FFT result is a complex tensor, even if the input is real
- The output has the same shape as the input
- For real input signals, the FFT result has conjugate symmetry
- The FFT is most efficient when the signal length is a power of 2 