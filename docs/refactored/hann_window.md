# torch::hann_window

Creates a Hann (Hanning) window tensor for signal processing applications.

## Syntax

### Positional Syntax (Original)
```tcl
torch::hann_window window_length
```

### Named Parameter Syntax (New)
```tcl
torch::hann_window -length window_length [-dtype dtype] [-device device] [-periodic boolean]
torch::hann_window -window_length window_length [-dtype dtype] [-device device] [-periodic boolean]
```

### camelCase Alias
```tcl
torch::hannWindow window_length
torch::hannWindow -length window_length [-dtype dtype] [-device device] [-periodic boolean]
```

## Description

The `torch::hann_window` command creates a Hann window tensor with the specified length. A Hann window (also known as a Hanning window) is a windowing function commonly used in signal processing to reduce spectral leakage when performing FFT operations on finite-length signals.

The Hann window function is defined as:
```
w[n] = 0.5 * (1 - cos(2π * n / (N-1)))
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

A tensor handle containing the Hann window values. The tensor has shape `[window_length]` and contains floating-point values between 0 and 1.

## Examples

### Basic Usage

```tcl
# Create a basic Hann window with 10 points
set window [torch::hann_window 10]

# Named parameter syntax
set window [torch::hann_window -length 10]

# camelCase alias
set window [torch::hannWindow 10]
```

### With Data Type Options

```tcl
# Create window with double precision
set window [torch::hann_window -length 256 -dtype float64]

# Create window with different data types
set window_float [torch::hann_window -length 128 -dtype float32]
set window_double [torch::hann_window -length 128 -dtype double]
```

### With Device Options

```tcl
# Create window on CPU (default)
set cpu_window [torch::hann_window -length 512 -device cpu]

# Create window on GPU (if available)
set gpu_window [torch::hann_window -length 512 -device cuda]
```

### Periodic vs Non-Periodic Windows

```tcl
# Periodic window (default) - suitable for FFT
set periodic_window [torch::hann_window -length 64 -periodic true]

# Non-periodic (symmetric) window - suitable for filter design
set symmetric_window [torch::hann_window -length 64 -periodic false]
```

### Complete Example with All Parameters

```tcl
# Create a comprehensive Hann window
set window [torch::hann_window -length 1024 -dtype float64 -device cpu -periodic true]

# Verify the window properties
set shape [torch::tensor_shape $window]
puts "Window shape: $shape"  ;# Output: Window shape: 1024
```

## Signal Processing Applications

### FFT Windowing
```tcl
# Apply Hann window to a signal before FFT
set signal [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 4.0 3.0 2.0} float32]
set window [torch::hann_window 8]
set windowed_signal [torch::tensor_mul $signal $window]
```

### Spectral Analysis
```tcl
# Create window for spectrogram analysis
set window_size 512
set hann_win [torch::hann_window $window_size]
# Use with STFT for spectral analysis
```

### Overlap-Add Processing
```tcl
# Hann window is perfect for overlap-add processing
set frame_size 1024
set hop_size 512  ;# 50% overlap
set window [torch::hann_window $frame_size]
# Perfect reconstruction guaranteed with 50% overlap
```

## Backward Compatibility

The original positional syntax is fully supported and will continue to work:

```tcl
# This will always work
set window [torch::hann_window 256]
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old syntax
torch::hann_window 512

# New equivalent syntax
torch::hann_window -length 512

# With additional options
torch::hann_window -length 512 -dtype float64 -periodic true
```

### From snake_case to camelCase

```tcl
# Original command
torch::hann_window 256

# camelCase alias
torch::hannWindow 256
```

## Error Handling

The command provides clear error messages for common mistakes:

```tcl
# Missing arguments
torch::hann_window
# Error: Usage: torch::hann_window window_length | torch::hann_window -length window_length [...]

# Invalid window length
torch::hann_window 0
# Error: Window length must be positive

# Unknown parameter
torch::hann_window -unknown 256
# Error: Unknown parameter: -unknown. Valid parameters are: -length, -window_length, -dtype, -device, -periodic

# Invalid data type
torch::hann_window -length 256 -dtype invalid_type
# Error: Unsupported dtype: invalid_type
```

## Mathematical Properties

### Window Characteristics
- **Main lobe width**: 4π/N (normalized frequency)
- **Peak side lobe level**: -31.5 dB
- **Side lobe roll-off**: -18 dB/octave
- **Scalloping loss**: 1.42 dB
- **Coherent gain**: 0.5
- **Equivalent noise bandwidth**: 1.5 bins

### Overlap-Add Properties
- **Perfect reconstruction**: Achieved with 50% overlap
- **Complementary property**: w[n] + w[n+N/2] = 1 for 50% overlap
- **Linear phase**: Zero phase when centered

## Comparison with Other Windows

### Hann vs Hamming
- **Hann**: Better side lobe suppression (-31.5 dB vs -43 dB)
- **Hamming**: Better frequency resolution, lower scalloping loss
- **Hann**: Smoother transition to zero at edges
- **Hamming**: Non-zero values at window edges

### Hann vs Blackman
- **Hann**: Narrower main lobe, better frequency resolution
- **Blackman**: Excellent side lobe suppression (-58 dB)
- **Hann**: Lower computational complexity
- **Blackman**: Better for high dynamic range applications

## Performance Notes

- The window is computed using PyTorch's optimized functions
- Both syntaxes have identical performance characteristics
- Named parameter parsing adds minimal overhead
- Window values are computed exactly using the mathematical formula
- Memory usage is linear with window length

## Common Use Cases

### 1. Short-Time Fourier Transform (STFT)
```tcl
set frame_size 1024
set hop_size 512
set window [torch::hann_window $frame_size]
# Use with torch::stft for time-frequency analysis
```

### 2. Overlap-Add Signal Processing
```tcl
set window [torch::hann_window 2048]
# Process signal in overlapping frames
# Perfect reconstruction with 50% overlap
```

### 3. Filter Design
```tcl
set window [torch::hann_window -length 101 -periodic false]
# Use for FIR filter design windowing
```

### 4. Spectral Analysis
```tcl
set analysis_window [torch::hann_window 4096]
# Apply to signal before FFT analysis
```

## Related Commands

- `torch::hamming_window` - Hamming window function (different from Hann)
- `torch::blackman_window` - Blackman window function  
- `torch::bartlett_window` - Bartlett (triangular) window function
- `torch::kaiser_window` - Kaiser window function with adjustable β parameter

## Technical Notes

### Mathematical Derivation
The Hann window is derived from the cosine function:
```
w[n] = 0.5 * (1 - cos(2π * n / (N-1)))
```

This can also be written as:
```
w[n] = sin²(π * n / (N-1))
```

### Frequency Response
The Hann window has a frequency response with:
- Main lobe width: 4π/N
- First side lobe at -31.5 dB
- Roll-off rate: -18 dB/octave

### Implementation Details
- Uses PyTorch's built-in `torch::hann_window` function
- Supports all standard PyTorch data types
- Device placement handled automatically
- Periodic vs symmetric modes available

## See Also

- [Signal Processing Operations](../signal_processing.md)
- [Window Functions](../window_functions.md)
- [FFT Operations](../fft_operations.md)
- [Tensor Creation](../tensor_creation.md)
- [Spectral Analysis](../spectral_analysis.md) 