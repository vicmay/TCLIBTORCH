# torch::tensor_ifft

Computes the Inverse Fast Fourier Transform (IFFT) of a tensor.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::tensor_ifft tensor ?dim?
```

### Named Parameters (New)
```tcl
torch::tensor_ifft -tensor tensor -dim dim
```

### CamelCase Alias
```tcl
torch::tensorIfft tensor ?dim?
torch::tensorIfft -tensor tensor -dim dim
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tensor` | string | Yes | - | Name of the input tensor |
| `dim` | int | No | -1 | Dimension along which to compute IFFT. -1 means use default (last dimension) |

## Description

The `torch::tensor_ifft` command computes the Inverse Fast Fourier Transform of a tensor. The IFFT is the mathematical inverse of the FFT, transforming a signal from the frequency domain back to the time domain.

- **Input**: A tensor of any shape (typically complex, but can be real)
- **Output**: A complex tensor with the same shape as the input
- **Default behavior**: If no dimension is specified, IFFT is computed along the last dimension

## Examples

### Basic Usage

```tcl
# Create a simple signal
set signal [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]

# Compute IFFT using positional syntax
set ifft_result [torch::tensor_ifft $signal]

# Compute IFFT using named parameters
set ifft_result [torch::tensor_ifft -tensor $signal]

# Compute IFFT using camelCase alias
set ifft_result [torch::tensorIfft $signal]
```

### Specifying Dimension

```tcl
# Create a 2D tensor
set tensor_2d [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]

# IFFT along dimension 0 (rows)
set ifft_dim0 [torch::tensor_ifft $tensor_2d 0]

# IFFT along dimension 1 (columns) using named parameters
set ifft_dim1 [torch::tensor_ifft -tensor $tensor_2d -dim 1]

# IFFT along dimension 1 using camelCase alias
set ifft_dim1 [torch::tensorIfft -tensor $tensor_2d -dim 1]
```

### FFT-IFFT Round Trip

```tcl
# Create a signal
set signal [torch::tensor_create {1.0 0.0 1.0 0.0} float32 cpu true]

# Apply FFT then IFFT - should recover original signal
set fft_result [torch::tensor_fft $signal]
set ifft_result [torch::tensor_ifft $fft_result]

# Get the shape of the result
set shape [torch::tensor_shape $ifft_result]
puts "IFFT result shape: $shape"
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
torch::tensor_ifft tensor_name 0
```

**New (Named Parameters)**:
```tcl
torch::tensor_ifft -tensor tensor_name -dim 0
```

**New (CamelCase)**:
```tcl
torch::tensorIfft tensor_name 0
torch::tensorIfft -tensor tensor_name -dim 0
```

### Backward Compatibility

The old positional syntax is fully supported and will continue to work:

```tcl
# This still works exactly as before
torch::tensor_ifft my_tensor
torch::tensor_ifft my_tensor 1
```

## Mathematical Background

The IFFT is the inverse operation of the FFT. For a signal of length N, the IFFT computes:

```
x[n] = (1/N) * Σ X[k] * e^(j*2π*k*n/N)
```

where:
- `x[n]` is the n-th time sample
- `X[k]` is the k-th frequency component
- `N` is the length of the signal

The IFFT transforms a signal from the frequency domain back to the time domain.

## Related Commands

- `torch::tensor_fft` - Fast Fourier Transform
- `torch::tensor_ifft2d` - 2D Inverse Fast Fourier Transform
- `torch::tensor_rfft` - Real-valued Fast Fourier Transform
- `torch::tensor_irfft` - Inverse Real-valued Fast Fourier Transform

## Notes

- The IFFT result is typically a complex tensor, even if the input is real
- The output has the same shape as the input
- For real input signals, the IFFT result may have conjugate symmetry
- The IFFT is most efficient when the signal length is a power of 2
- IFFT is the mathematical inverse of FFT: `IFFT(FFT(x)) ≈ x`

## Applications

- Signal reconstruction from frequency domain
- Audio and image processing
- Spectral analysis and filtering
- Digital signal processing (DSP)
- Scientific computing and analysis 