# torch::kaiser_window

Creates a Kaiser window tensor for signal processing applications.

## Syntax

### Positional Syntax (Original)
```tcl
torch::kaiser_window window_length ?beta?
```

### Named Parameter Syntax (New)
```tcl
torch::kaiser_window -windowLength window_length [-beta beta_value] [-dtype dtype] [-device device] [-periodic boolean]
```

### camelCase Alias
```tcl
torch::kaiserWindow window_length ?beta?
torch::kaiserWindow -windowLength window_length [-beta beta_value] [-dtype dtype] [-device device] [-periodic boolean]
```

## Description

The `torch::kaiser_window` command creates a Kaiser window tensor with the specified length and beta parameter. A Kaiser window is a near-optimal windowing function used in signal processing that provides excellent control over the trade-off between main lobe width and side lobe level through the beta parameter.

The Kaiser window function is defined as:
```
w[n] = I₀(β * √(1 - ((n - α) / α)²)) / I₀(β)
```
where:
- N is the window length
- α = (N-1)/2
- β is the beta parameter (shape factor)
- I₀ is the zeroth-order modified Bessel function of the first kind
- n ranges from 0 to N-1

## Parameters

### Positional Parameters
- `window_length` - Length of the window (must be positive integer)
- `beta` - Beta parameter controlling window shape (optional, default: 12.0)

### Named Parameters
- `-windowLength` - Length of the window (must be positive integer)
- `-beta` - Beta parameter controlling window shape (optional, default: 12.0)
  - Higher values create narrower main lobe with lower side lobes
  - Lower values create wider main lobe with higher side lobes
  - Typical range: 0.0 to 20.0
- `-dtype` - Data type of the output tensor (optional, default: "float32")
  - Supported types: "float32", "float", "float64", "double", "int32", "int", "int64", "long"
- `-device` - Device to place the tensor on (optional, default: "cpu")
  - Supported devices: "cpu", "cuda", "cuda:0", etc.
- `-periodic` - Whether to return a periodic window (optional, default: true)
  - true: Window suitable for use with FFT functions
  - false: Symmetric window suitable for filter design

## Returns

A tensor handle containing the Kaiser window values. The tensor has shape `[window_length]` and contains floating-point values between 0 and 1.

## Examples

### Basic Usage

```tcl
# Create a basic Kaiser window with 10 points (default beta=12.0)
set window [torch::kaiser_window 10]

# Create Kaiser window with specific beta parameter
set window [torch::kaiser_window 10 8.0]

# Named parameter syntax
set window [torch::kaiser_window -windowLength 10]
set window [torch::kaiser_window -windowLength 10 -beta 8.0]

# camelCase alias
set window [torch::kaiserWindow 10]
set window [torch::kaiserWindow -windowLength 10 -beta 8.0]
```

### Beta Parameter Examples

```tcl
# Low beta (wider main lobe, higher side lobes)
set window_low [torch::kaiser_window -windowLength 64 -beta 2.0]

# Medium beta (balanced trade-off)
set window_med [torch::kaiser_window -windowLength 64 -beta 8.0]

# High beta (narrow main lobe, lower side lobes)
set window_high [torch::kaiser_window -windowLength 64 -beta 16.0]

# Very high beta (very narrow main lobe, very low side lobes)
set window_vhigh [torch::kaiser_window -windowLength 64 -beta 20.0]
```

### With Data Type Options

```tcl
# Create window with double precision
set window [torch::kaiser_window -windowLength 256 -beta 10.0 -dtype float64]

# Create window with different data types
set window_float [torch::kaiser_window -windowLength 128 -beta 12.0 -dtype float32]
set window_double [torch::kaiser_window -windowLength 128 -beta 12.0 -dtype double]
```

### With Device Options

```tcl
# Create window on CPU (default)
set cpu_window [torch::kaiser_window -windowLength 512 -beta 14.0 -device cpu]

# Create window on GPU (if available)
set gpu_window [torch::kaiser_window -windowLength 512 -beta 14.0 -device cuda]
```

### Periodic vs Non-Periodic Windows

```tcl
# Periodic window (default) - suitable for FFT
set periodic_window [torch::kaiser_window -windowLength 64 -beta 10.0 -periodic true]

# Non-periodic (symmetric) window - suitable for filter design
set symmetric_window [torch::kaiser_window -windowLength 64 -beta 10.0 -periodic false]
```

### Complete Example with All Parameters

```tcl
# Create a comprehensive Kaiser window
set window [torch::kaiser_window -windowLength 1024 -beta 8.6 -dtype float64 -device cpu -periodic true]

# Verify the window properties
set shape [torch::tensor_shape $window]
puts "Window shape: $shape"  ;# Output: Window shape: 1024
```

## Signal Processing Applications

### FFT Windowing
```tcl
# Apply Kaiser window to a signal before FFT
set signal [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 4.0 3.0 2.0} float32]
set window [torch::kaiser_window 8 6.0]
set windowed_signal [torch::tensor_mul $signal $window]
```

### Filter Design
```tcl
# Kaiser window for FIR filter design
set filter_length 64
set beta 8.0  ;# For approximately 50 dB stopband attenuation
set window [torch::kaiser_window $filter_length $beta]
# Use with sinc function for optimal FIR filter design
```

### Spectral Analysis
```tcl
# Create window for high-resolution spectral analysis
set window_size 512
set beta 12.0  ;# Good balance for spectral analysis
set kaiser_win [torch::kaiser_window $window_size $beta]
# Use with STFT for spectral analysis
```

### Beamforming
```tcl
# Kaiser window for array beamforming
set array_size 32
set beta 6.0  ;# For sidelobe control in beamforming
set window [torch::kaiser_window $array_size $beta]
# Apply to array weights for sidelobe suppression
```

## Beta Parameter Guidelines

### Common Beta Values and Their Applications

| Beta Value | Main Lobe Width | Side Lobe Level | Application |
|------------|-----------------|-----------------|-------------|
| 0.0        | 2π/N           | -13.3 dB       | Rectangular window equivalent |
| 2.0        | 2.4π/N         | -25 dB          | Low spectral leakage |
| 4.0        | 2.8π/N         | -35 dB          | General purpose |
| 6.0        | 3.2π/N         | -45 dB          | Filter design |
| 8.0        | 3.6π/N         | -55 dB          | High dynamic range |
| 10.0       | 4.0π/N         | -65 dB          | Very high dynamic range |
| 12.0       | 4.4π/N         | -75 dB          | Default (excellent balance) |
| 16.0       | 5.2π/N         | -95 dB          | Extremely low side lobes |

### Selecting Beta for Your Application

```tcl
# For general FFT analysis
set window [torch::kaiser_window 1024 8.0]

# For filter design with 60 dB stopband
set window [torch::kaiser_window 128 8.5]

# For high-resolution spectral analysis
set window [torch::kaiser_window 2048 12.0]

# For beamforming with low sidelobes
set window [torch::kaiser_window 64 10.0]
```

## Backward Compatibility

The original positional syntax is fully supported and will continue to work:

```tcl
# This will always work
set window [torch::kaiser_window 256]
set window [torch::kaiser_window 256 10.0]
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old syntax
torch::kaiser_window 512
torch::kaiser_window 512 8.0

# New equivalent syntax
torch::kaiser_window -windowLength 512
torch::kaiser_window -windowLength 512 -beta 8.0

# With additional options
torch::kaiser_window -windowLength 512 -beta 8.0 -dtype float64 -periodic true
```

### From snake_case to camelCase

```tcl
# Original command
torch::kaiser_window 256 10.0

# camelCase alias
torch::kaiserWindow 256 10.0
```

## Error Handling

The command provides clear error messages for common mistakes:

```tcl
# Missing arguments
torch::kaiser_window
# Error: Usage: kaiser_window window_length ?beta?

# Invalid window length
torch::kaiser_window 0
# Error: Window length must be positive

# Unknown parameter
torch::kaiser_window -unknown 256
# Error: Unknown parameter: -unknown

# Invalid data type
torch::kaiser_window -windowLength 256 -dtype invalid_type
# Error: Unsupported dtype: invalid_type
```

## Mathematical Properties

### Window Characteristics
The Kaiser window properties depend on the beta parameter:

- **Main lobe width**: (2β + 2)π/N (normalized frequency)
- **Peak side lobe level**: Approximately -20log₁₀(β) dB for β > 2
- **Transition width**: Controllable via beta parameter
- **Coherent gain**: Varies with beta (approximately 0.4 to 0.5)

### Optimal Beta Selection
For FIR filter design, the beta parameter can be chosen based on desired stopband attenuation:

```
β = 0.1102 * (A - 8.7)     for A > 50 dB
β = 0.5842 * (A - 21)^0.4 + 0.07886 * (A - 21)  for 21 < A ≤ 50 dB
β = 0.0                     for A ≤ 21 dB
```

where A is the desired stopband attenuation in dB.

## Comparison with Other Windows

### Kaiser vs Hann Window
```tcl
# Hann window (fixed characteristics)
set hann_win [torch::hann_window 64]

# Kaiser window (adjustable characteristics)
set kaiser_win [torch::kaiser_window 64 8.0]  ;# Similar to Hann
```

### Kaiser vs Hamming Window
```tcl
# Kaiser window equivalent to Hamming (approximately)
set kaiser_hamming_like [torch::kaiser_window 64 6.0]
```

## Advanced Usage

### Designing Custom Windows
```tcl
# Create multiple Kaiser windows with different beta values
set betas {2.0 4.0 6.0 8.0 10.0 12.0}
set windows {}
foreach beta $betas {
    set window [torch::kaiser_window 256 $beta]
    lappend windows $window
}
```

### Time-Frequency Analysis
```tcl
# Kaiser window for time-frequency analysis
set window_length 1024
set overlap 0.75
set beta 8.0
set window [torch::kaiser_window $window_length $beta]
# Use with STFT for time-frequency analysis
```

## Performance Notes

- Kaiser windows are computationally more expensive than simple windows (Hann, Hamming) due to Bessel function calculations
- The computation time increases slightly with higher beta values
- For real-time applications, consider pre-computing and storing Kaiser windows
- GPU acceleration is available for large window sizes

## See Also

- [torch::hann_window](hann_window.md) - Hann window function
- [torch::hamming_window](hamming_window.md) - Hamming window function
- [torch::bartlett_window](bartlett_window.md) - Bartlett window function
- [torch::blackman_window](blackman_window.md) - Blackman window function 