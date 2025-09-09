# torch::tensor_istft

Inverse Short-Time Fourier Transform (ISTFT) of a complex-valued input tensor.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_istft tensor n_fft ?hop_length? ?win_length? ?window?
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_istft -input tensor_handle -n_fft n_fft ?-hop_length hop_length? ?-win_length win_length? ?-window window? ?-center bool? ?-normalized bool? ?-onesided bool? ?-length length?
torch::tensor_istft -tensor tensor_handle -n_fft n_fft ...
```

### CamelCase Alias
```tcl
torch::tensorIstft tensor n_fft ?hop_length? ?win_length? ?window?
torch::tensorIstft -input tensor_handle -n_fft n_fft ...
```

## Parameters

| Parameter         | Type    | Required | Description                                                      |
|------------------|---------|----------|------------------------------------------------------------------|
| `tensor_handle`  | string  | Yes      | The handle of the complex-valued input tensor (from STFT)        |
| `n_fft`          | int     | Yes      | Size of FFT window                                               |
| `hop_length`     | int     | No       | Number of audio frames between STFT columns                      |
| `win_length`     | int     | No       | Window size                                                      |
| `window`         | tensor  | No       | Window tensor to use                                             |
| `center`         | bool    | No       | Whether the signal was padded on both sides (default: true)      |
| `normalized`     | bool    | No       | Whether the STFT was normalized (default: true)                  |
| `onesided`       | bool    | No       | Whether the STFT was onesided (default: true)                    |
| `length`         | int     | No       | Output length                                                    |

## Return Value

Returns a tensor handle containing the reconstructed time-domain signal.

## Examples

### Basic Usage

```tcl
set signal [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu true]
set stft_result [torch::tensor_stft $signal 4 2]
set istft_result [torch::tensor_istft $stft_result 4 2]
puts "Reconstructed: [torch::tensor_to_list $istft_result]"
```

### With Window

```tcl
set signal [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu true]
set window [torch::tensor_create {1.0 1.0 1.0 1.0} float32 cpu true]
set stft_result [torch::tensor_stft $signal 4 2 4 $window]
set istft_result [torch::tensor_istft $stft_result 4 2 4 $window]
puts "Reconstructed: [torch::tensor_to_list $istft_result]"
```

### Named Parameter Syntax

```tcl
set signal [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu true]
set stft_result [torch::tensor_stft $signal 4 2]
set istft_result [torch::tensor_istft -input $stft_result -n_fft 4 -hop_length 2]
puts "Reconstructed: [torch::tensor_to_list $istft_result]"
```

### CamelCase Alias

```tcl
set signal [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu true]
set stft_result [torch::tensor_stft $signal 4 2]
set istft_result [torch::tensorIstft $stft_result 4 2]
puts "Reconstructed: [torch::tensor_to_list $istft_result]"
```

### Error Handling

```tcl
catch {torch::tensor_istft invalid_tensor 4} result
puts "Error: $result"  ;# Output: Error: Invalid tensor name

catch {torch::tensor_istft} result
puts "Error: $result"  ;# Output: Error: Required input and n_fft parameters missing

set signal [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
catch {torch::tensor_istft -input $signal} result
puts "Error: $result"  ;# Output: Error: istft requires a complex-valued input tensor
```

## Migration Guide

### From Old Syntax to New Syntax

**Old (Positional Only):**
```tcl
set istft_result [torch::tensor_istft $tensor 4 2]
set istft_result [torch::tensor_istft $tensor 4 2 4 $window]
```

**New (Named Parameters):**
```tcl
set istft_result [torch::tensor_istft -input $tensor -n_fft 4 -hop_length 2]
set istft_result [torch::tensor_istft -input $tensor -n_fft 4 -hop_length 2 -win_length 4 -window $window]
```

**New (CamelCase):**
```tcl
set istft_result [torch::tensorIstft $tensor 4 2]
set istft_result [torch::tensorIstft -input $tensor -n_fft 4 -hop_length 2]
```

### Backward Compatibility

The old positional syntax continues to work:
```tcl
set istft_result [torch::tensor_istft $tensor 4 2]
```

## Notes

- Both snake_case and camelCase versions are functionally identical
- Input tensor must be complex-valued (output from STFT)
- All parameters are optional except input and n_fft
- The output is a real-valued time-domain signal
- For best results, use the same parameters for STFT and ISTFT

## Related Commands

- `torch::tensor_stft` - Short-Time Fourier Transform
- `torch::tensor_fft` - FFT
- `torch::tensor_ifft` - IFFT
- `torch::tensor_create` - Create tensors 