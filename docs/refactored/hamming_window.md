# torch::hamming_window

Creates a Hamming window tensor for signal processing applications.

## Syntax

### Positional Syntax (Original)
```tcl
torch::hamming_window window_length
```

### Named Parameter Syntax (New)
```tcl
torch::hamming_window -length window_length [-dtype dtype] [-device device] [-periodic boolean]
torch::hamming_window -window_length window_length [-dtype dtype] [-device device] [-periodic boolean]
```

### camelCase Alias
```tcl
torch::hammingWindow window_length
torch::hammingWindow -length window_length [-dtype dtype] [-device device] [-periodic boolean]
```

## Description

The `torch::hamming_window` command creates a Hamming window tensor with the specified length. A Hamming window is a windowing function commonly used in signal processing to reduce spectral leakage when performing FFT operations on finite-length signals.

The Hamming window function is defined as:
```
w[n] = 0.54 - 0.46 * cos(2π * n / (N-1))
```
where N is the window length and n ranges from 0 to N-1.

## Parameters

### Positional Parameters
- `window_length` - Length of the window (must be positive integer)

### Named Parameters
- `-length` or `-window_length` - Length of the window (must be positive integer)
- `-dtype` - Data type of the output tensor (optional, default: "float32")
  - Supported types: "float32", "float", "float64", "double", "int32", "int", "int64", "long"
- `-device` - Device to place the tensor on (optional, default: "cpu")
  - Supported devices: "cpu", "cuda", "cuda:0", etc.
- `-periodic` - Whether to return a periodic window (optional, default: true)
  - true: Window suitable for use with FFT functions
  - false: Symmetric window suitable for filter design

## Returns

A tensor handle containing the Hamming window values. The tensor has shape `[window_length]` and contains floating-point values between 0 and 1.

## Examples

### Basic Usage

```tcl
# Create a basic Hamming window with 10 points
set window [torch::hamming_window 10]

# Named parameter syntax
set window [torch::hamming_window -length 10]

# camelCase alias
set window [torch::hammingWindow 10]
```

### With Data Type Options

```tcl
# Create window with double precision
set window [torch::hamming_window -length 256 -dtype float64]

# Create window with different data types
set window_float [torch::hamming_window -length 128 -dtype float32]
set window_double [torch::hamming_window -length 128 -dtype double]
```

### With Device Options

```tcl
# Create window on CPU (default)
set cpu_window [torch::hamming_window -length 512 -device cpu]

# Create window on GPU (if available)
set gpu_window [torch::hamming_window -length 512 -device cuda]
```

### Periodic vs Non-Periodic Windows

```tcl
# Periodic window (default) - suitable for FFT
set periodic_window [torch::hamming_window -length 64 -periodic true]

# Non-periodic (symmetric) window - suitable for filter design
set symmetric_window [torch::hamming_window -length 64 -periodic false]
```

### Complete Example with All Parameters

```tcl
# Create a comprehensive Hamming window
set window [torch::hamming_window -length 1024 -dtype float64 -device cpu -periodic true]

# Verify the window properties
set shape [torch::tensor_shape $window]
puts "Window shape: $shape"  ;# Output: Window shape: 1024
```

## Signal Processing Applications

### FFT Windowing
```tcl
# Apply Hamming window to a signal before FFT
set signal [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 4.0 3.0 2.0} float32]
set window [torch::hamming_window 8]
set windowed_signal [torch::tensor_mul $signal $window]
```

### Spectral Analysis
```tcl
# Create window for spectrogram analysis
set window_size 512
set hamming_win [torch::hamming_window $window_size]
# Use with STFT for spectral analysis
```

## Backward Compatibility

The original positional syntax is fully supported and will continue to work:

```tcl
# This will always work
set window [torch::hamming_window 256]
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old syntax
torch::hamming_window 512

# New equivalent syntax
torch::hamming_window -length 512

# With additional options
torch::hamming_window -length 512 -dtype float64 -periodic true
```

### From snake_case to camelCase

```tcl
# Original command
torch::hamming_window 256

# camelCase alias
torch::hammingWindow 256
```

## Error Handling

The command provides clear error messages for common mistakes:

```tcl
# Missing arguments
torch::hamming_window
# Error: Usage: torch::hamming_window window_length | torch::hamming_window -length window_length [...]

# Invalid window length
torch::hamming_window 0
# Error: Window length must be positive

# Unknown parameter
torch::hamming_window -unknown 256
# Error: Unknown parameter: -unknown. Valid parameters are: -length, -window_length, -dtype, -device, -periodic

# Invalid data type
torch::hamming_window -length 256 -dtype invalid_type
# Error: Unsupported dtype: invalid_type
```

## Mathematical Properties

### Window Characteristics
- **Main lobe width**: 4π/N (normalized frequency)
- **Peak side lobe level**: -43 dB
- **Side lobe roll-off**: -6 dB/octave
- **Scalloping loss**: 1.36 dB
- **Coherent gain**: 0.54
- **Equivalent noise bandwidth**: 1.36 bins

### Comparison with Other Windows
- **Hamming**: Good frequency resolution, moderate side lobe suppression
- **Hanning**: Better side lobe suppression, slightly wider main lobe
- **Blackman**: Excellent side lobe suppression, wider main lobe

## Performance Notes

- The window is computed using PyTorch's optimized functions
- Both syntaxes have identical performance characteristics
- Named parameter parsing adds minimal overhead
- Window values are computed exactly using the mathematical formula

## Related Commands

- `torch::hann_window` - Hann (Hanning) window function
- `torch::blackman_window` - Blackman window function
- `torch::bartlett_window` - Bartlett (triangular) window function
- `torch::kaiser_window` - Kaiser window function with adjustable β parameter

## See Also

- [Signal Processing Operations](../signal_processing.md)
- [Window Functions](../window_functions.md)
- [FFT Operations](../fft_operations.md)
- [Tensor Creation](../tensor_creation.md) 