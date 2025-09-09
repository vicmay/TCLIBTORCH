# torch::tensor_stft / torch::tensorStft

Compute the Short-Time Fourier Transform (STFT) of a tensor.

## Description

The `torch::tensor_stft` command computes the Short-Time Fourier Transform of tensor elements. It supports both positional and named parameter syntax, with full backward compatibility.

**Alias**: `torch::tensorStft` (camelCase)

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_stft tensor n_fft ?hop_length? ?win_length? ?window?
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_stft -input tensor -n_fft n_fft ?-hop_length hop_length? ?-win_length win_length? ?-window window?
torch::tensor_stft -tensor tensor -nfft n_fft ?-hopLength hop_length? ?-winLength win_length? ?-window window?
```

### CamelCase Alias
```tcl
torch::tensorStft tensor n_fft ?hop_length? ?win_length? ?window?
torch::tensorStft -input tensor -n_fft n_fft ?-hop_length hop_length? ?-win_length win_length? ?-window window?
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor` / `-input` / `-tensor` | string | Yes | Tensor handle to compute STFT |
| `n_fft` / `-n_fft` / `-nfft` | integer | Yes | Number of FFT bins |
| `hop_length` / `-hop_length` / `-hopLength` | integer | No | Number of samples between windows (default: n_fft//4) |
| `win_length` / `-win_length` / `-winLength` | integer | No | Window length (default: n_fft) |
| `window` / `-window` | string | No | Window tensor handle (default: Hann window) |

## Return Value

Returns a tensor handle containing the complex-valued STFT result.

## Examples

### Basic Usage

**Positional syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
set result [torch::tensor_stft $tensor 4]
puts "STFT result: $result"
```

**Named parameter syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
set result [torch::tensor_stft -input $tensor -n_fft 4]
puts "STFT result: $result"
```

**CamelCase alias:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
set result [torch::tensorStft -input $tensor -n_fft 4]
puts "STFT result: $result"
```

### With Hop Length

**Positional syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
set result [torch::tensor_stft $tensor 4 2]  ;# hop_length = 2
puts "STFT with hop: $result"
```

**Named parameter syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
set result [torch::tensor_stft -tensor $tensor -nfft 4 -hopLength 2]
puts "STFT with hop: $result"
```

### With Window Length

**Positional syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
set result [torch::tensor_stft $tensor 4 2 4]  ;# win_length = 4
puts "STFT with window length: $result"
```

**Named parameter syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
set result [torch::tensor_stft -input $tensor -n_fft 4 -hop_length 2 -win_length 4]
puts "STFT with window length: $result"
```

### With Custom Window

**Positional syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
set window [torch::tensor_create {0.5 1.0 0.5 0.5}]
set result [torch::tensor_stft $tensor 4 2 4 $window]
puts "STFT with custom window: $result"
```

**Named parameter syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
set window [torch::tensor_create {0.5 1.0 0.5 0.5}]
set result [torch::tensor_stft -input $tensor -n_fft 4 -hop_length 2 -win_length 4 -window $window]
puts "STFT with custom window: $result"
```

### Different Data Types

```tcl
set tensor1 [torch::tensor_create {1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5} float32]
set tensor2 [torch::tensor_create {1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5} float64]
set result1 [torch::tensor_stft $tensor1 4]
set result2 [torch::tensor_stft $tensor2 4]
puts "Float32 STFT: $result1"
puts "Float64 STFT: $result2"
```

### Sine Wave Analysis

```tcl
set data {}
for {set i 0} {$i < 100} {incr i} {
    lappend data [expr {sin($i * 0.1)}]
}
set tensor [torch::tensor_create $data]
set result [torch::tensor_stft $tensor 16 8]
puts "Sine wave STFT: $result"
```

## Error Handling

The command provides clear error messages for various error conditions:

```tcl
# Invalid tensor name
catch {torch::tensor_stft invalid_tensor 4} result
puts $result  ;# Output: Invalid tensor name

# Missing tensor parameter
catch {torch::tensor_stft} result
puts $result  ;# Output: Required input and n_fft parameters missing

# Missing n_fft parameter
set tensor [torch::tensor_create {1 2 3}]
catch {torch::tensor_stft $tensor} result
puts $result  ;# Output: Invalid number of arguments

# Invalid n_fft value
catch {torch::tensor_stft $tensor invalid} result
puts $result  ;# Output: Invalid n_fft value

# Invalid hop_length value
catch {torch::tensor_stft $tensor 4 invalid} result
puts $result  ;# Output: Invalid hop_length value

# Invalid win_length value
catch {torch::tensor_stft $tensor 4 2 invalid} result
puts $result  ;# Output: Invalid win_length value

# Unknown named parameter
catch {torch::tensor_stft -invalid $tensor} result
puts $result  ;# Output: Unknown parameter: -invalid

# Missing parameter value
catch {torch::tensor_stft -tensor $tensor -n_fft} result
puts $result  ;# Output: Missing value for parameter

# Integer tensor (not supported)
set tensor [torch::tensor_create {1 2 3 4} int32]
catch {torch::tensor_stft $tensor 4} result
puts $result  ;# Output: expected a tensor of floating point or complex values
```

## Migration Guide

### From Positional to Named Parameters

**Old code:**
```tcl
set result [torch::tensor_stft $tensor 4 2 4 $window]
```

**New code (equivalent):**
```tcl
set result [torch::tensor_stft -input $tensor -n_fft 4 -hop_length 2 -win_length 4 -window $window]
```

### Using CamelCase Alias

**Old code:**
```tcl
set result [torch::tensor_stft $tensor 4]
```

**New code (equivalent):**
```tcl
set result [torch::tensorStft $tensor 4]
```

## Notes

- The input tensor must be floating point (float32, float64) or complex
- The `n_fft` parameter determines the number of frequency bins in the output
- The `hop_length` parameter controls the overlap between consecutive windows
- The `win_length` parameter must be <= `n_fft`
- If no window is provided, a Hann window is automatically created
- Custom windows must have size equal to `win_length` (or `n_fft` if `win_length` not specified)
- The result is a complex-valued tensor with shape `(n_fft//2 + 1, n_frames)`
- The STFT is normalized and one-sided by default
- The command supports tensors of various floating point data types
- For real input, the output contains only positive frequency components

## See Also

- `torch::tensor_istft` - Inverse Short-Time Fourier Transform
- `torch::tensor_fft` - Fast Fourier Transform
- `torch::tensor_rfft` - Real-valued FFT
- `torch::tensor_irfft` - Inverse real-valued FFT
- `torch::tensor_fft2d` - 2D FFT
- `torch::tensor_ifft2d` - 2D inverse FFT 