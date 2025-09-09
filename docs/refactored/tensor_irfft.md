# torch::tensor_irfft

Computes the inverse real FFT (Inverse Real Fast Fourier Transform) of a tensor.

## Description

The `torch::tensor_irfft` command performs an inverse real FFT on a tensor, transforming complex frequency domain data back to real time domain data. This is the inverse operation of `torch::tensor_rfft`.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_irfft tensor ?n? ?dim?
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_irfft -tensor tensor ?-n n? ?-dim dim?
torch::tensor_irfft -input tensor ?-n n? ?-dim dim?
```

### CamelCase Alias
```tcl
torch::tensorIrfft tensor ?n? ?dim?
torch::tensorIrfft -tensor tensor ?-n n? ?-dim dim?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tensor` / `-tensor` / `-input` | string | Yes | - | Name of the input tensor |
| `n` / `-n` | integer | No | `nullopt` | Size of the output signal. If not specified, uses the default size |
| `dim` / `-dim` | integer | No | `-1` | Dimension along which to compute the inverse FFT. Defaults to the last dimension |

## Return Value

Returns a string containing the handle of the resulting tensor.

## Examples

### Basic Usage

**Positional syntax:**
```tcl
# Create input tensor
set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]

# Basic inverse real FFT
set result [torch::tensor_irfft $input]
puts [torch::tensor_shape $result]  ;# Output: 6
```

**Named parameter syntax:**
```tcl
# Same operation with named parameters
set result [torch::tensor_irfft -tensor $input]
puts [torch::tensor_shape $result]  ;# Output: 6
```

**CamelCase alias:**
```tcl
# Using camelCase alias
set result [torch::tensorIrfft $input]
puts [torch::tensor_shape $result]  ;# Output: 6
```

### Specifying Output Size

**Positional syntax:**
```tcl
# Specify output size
set result [torch::tensor_irfft $input 8]
puts [torch::tensor_shape $result]  ;# Output: 8
```

**Named parameter syntax:**
```tcl
# Same with named parameters
set result [torch::tensor_irfft -tensor $input -n 8]
puts [torch::tensor_shape $result]  ;# Output: 8
```

### Specifying Dimension

**Positional syntax:**
```tcl
# Specify dimension for 2D tensor
set input_2d [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
set result [torch::tensor_irfft $input_2d 4 0]
puts [torch::tensor_shape $result]  ;# Output: 4 2
```

**Named parameter syntax:**
```tcl
# Same with named parameters
set result [torch::tensor_irfft -tensor $input_2d -n 4 -dim 0]
puts [torch::tensor_shape $result]  ;# Output: 4 2
```

### Complete Example with All Parameters

```tcl
# Create a complex input tensor
set input [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu true]

# Perform inverse real FFT with all parameters
set result [torch::tensor_irfft -tensor $input -n 10 -dim 0]
puts [torch::tensor_shape $result]  ;# Output: 10

# Get the actual values
puts [torch::tensor_to_list $result]
```

## Migration Guide

### From Old Positional Syntax

**Old code:**
```tcl
set result [torch::tensor_irfft input_tensor 8 0]
```

**New positional syntax (unchanged):**
```tcl
set result [torch::tensor_irfft input_tensor 8 0]
```

**New named parameter syntax:**
```tcl
set result [torch::tensor_irfft -tensor input_tensor -n 8 -dim 0]
```

**New camelCase alias:**
```tcl
set result [torch::tensorIrfft input_tensor 8 0]
set result [torch::tensorIrfft -tensor input_tensor -n 8 -dim 0]
```

## Error Handling

The command provides clear error messages for various error conditions:

### Invalid Tensor Name
```tcl
catch {torch::tensor_irfft nonexistent_tensor} result
puts $result  ;# Output: Invalid tensor name
```

### Missing Required Parameter
```tcl
catch {torch::tensor_irfft} result
puts $result  ;# Output: Required tensor parameter missing
```

### Invalid Parameter Types
```tcl
catch {torch::tensor_irfft $input_tensor invalid_n} result
puts $result  ;# Output: Invalid n parameter

catch {torch::tensor_irfft $input_tensor 8 invalid_dim} result
puts $result  ;# Output: Invalid dim parameter
```

### Unknown Named Parameter
```tcl
catch {torch::tensor_irfft -tensor $input_tensor -unknown value} result
puts $result  ;# Output: Unknown parameter: -unknown
```

### Missing Parameter Value
```tcl
catch {torch::tensor_irfft -tensor $input_tensor -n} result
puts $result  ;# Output: Missing value for parameter
```

### Edge Cases

**Single element tensor (will fail):**
```tcl
set single [torch::tensor_create {1.0} float32 cpu true]
catch {torch::tensor_irfft $single} result
puts $result  ;# Output: Invalid number of data points (0) specified
```

**Zero tensor:**
```tcl
set zero [torch::tensor_create {0.0 0.0 0.0 0.0} float32 cpu true]
set result [torch::tensor_irfft $zero]
puts [torch::tensor_shape $result]  ;# Output: 6
```

## Mathematical Background

The inverse real FFT (`torch::fft::irfft`) transforms complex frequency domain data back to real time domain data. It's the inverse operation of the real FFT (`torch::fft::rfft`).

Key properties:
- Input: Complex tensor (typically from `torch::tensor_rfft`)
- Output: Real tensor
- The `n` parameter controls the size of the output signal
- The `dim` parameter specifies which dimension to transform

## Related Commands

- `torch::tensor_fft` - Forward FFT
- `torch::tensor_ifft` - Inverse FFT
- `torch::tensor_rfft` - Real FFT
- `torch::tensor_fft2d` - 2D FFT
- `torch::tensor_ifft2d` - Inverse 2D FFT

## Notes

1. **Backward Compatibility**: The original positional syntax is fully supported and unchanged.
2. **Performance**: The inverse real FFT is optimized for real-valued output signals.
3. **Memory**: The output tensor is automatically managed and will be cleaned up when no longer referenced.
4. **Precision**: Results may have small numerical differences due to floating-point arithmetic.

## Version History

- **Refactored**: Added named parameter syntax and camelCase alias while maintaining 100% backward compatibility
- **Original**: Positional parameter syntax only 