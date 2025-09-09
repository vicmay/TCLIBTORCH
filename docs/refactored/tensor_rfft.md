# torch::tensor_rfft / torch::tensorRfft

Compute the one-dimensional real-valued Fast Fourier Transform (RFFT) of a tensor.

## Description

The `torch::tensor_rfft` command computes the one-dimensional real-valued Fast Fourier Transform of tensor elements. It supports both positional and named parameter syntax, with full backward compatibility.

**Alias**: `torch::tensorRfft` (camelCase)

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_rfft tensor ?n? ?dim?
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_rfft -input tensor ?-n n? ?-dim dim?
torch::tensor_rfft -tensor tensor ?-n n? ?-dim dim?
```

### CamelCase Alias
```tcl
torch::tensorRfft tensor ?n? ?dim?
torch::tensorRfft -input tensor ?-n n? ?-dim dim?
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor` / `-input` / `-tensor` | string | Yes | Tensor handle to compute RFFT |
| `n` / `-n` | integer | No | Number of points to use for FFT (default: size of input) |
| `dim` / `-dim` | integer | No | Dimension along which to compute FFT (default: -1, last dimension) |

## Return Value

Returns a tensor handle containing the complex-valued RFFT result.

## Examples

### Basic Usage

**Positional syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5}]
set result [torch::tensor_rfft $tensor]
puts "RFFT result: $result"
```

**Named parameter syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5}]
set result [torch::tensor_rfft -input $tensor]
puts "RFFT result: $result"
```

**CamelCase alias:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5}]
set result [torch::tensorRfft -input $tensor]
puts "RFFT result: $result"
```

### With N Parameter (Zero Padding)

**Positional syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5}]
set result [torch::tensor_rfft $tensor 8]  ;# Zero pad to 8 points
puts "RFFT with padding: $result"
```

**Named parameter syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5}]
set result [torch::tensor_rfft -tensor $tensor -n 8]
puts "RFFT with padding: $result"
```

### With Dimension Specification

**Positional syntax:**
```tcl
set tensor [torch::tensor_create {{1 2 3} {4 5 6}}]
set result [torch::tensor_rfft $tensor 4 1]  ;# Along second dimension
puts "RFFT along dim 1: $result"
```

**Named parameter syntax:**
```tcl
set tensor [torch::tensor_create {{1 2 3} {4 5 6}}]
set result [torch::tensor_rfft -input $tensor -n 4 -dim 1]
puts "RFFT along dim 1: $result"
```

### Multi-dimensional Tensors

```tcl
set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9}}]
set result1 [torch::tensor_rfft $tensor 3 0]  ;# Along first dimension
set result2 [torch::tensor_rfft $tensor 3 1]  ;# Along second dimension
puts "RFFT along dim 0: $result1"
puts "RFFT along dim 1: $result2"
```

### Different Data Types

```tcl
set tensor1 [torch::tensor_create {1.5 2.5 3.5 4.5} float32]
set tensor2 [torch::tensor_create {1.5 2.5 3.5 4.5} float64]
set result1 [torch::tensor_rfft $tensor1]
set result2 [torch::tensor_rfft $tensor2]
puts "Float32 RFFT: $result1"
puts "Float64 RFFT: $result2"
```

## Error Handling

The command provides clear error messages for various error conditions:

```tcl
# Invalid tensor name
catch {torch::tensor_rfft invalid_tensor} result
puts $result  ;# Output: Invalid tensor name

# Missing tensor parameter
catch {torch::tensor_rfft} result
puts $result  ;# Output: Required input parameter missing

# Invalid n parameter
set tensor [torch::tensor_create {1 2 3}]
catch {torch::tensor_rfft $tensor invalid} result
puts $result  ;# Output: Invalid n parameter

# Invalid dim parameter
catch {torch::tensor_rfft $tensor 4 invalid} result
puts $result  ;# Output: Invalid dim parameter

# Unknown named parameter
catch {torch::tensor_rfft -invalid $tensor} result
puts $result  ;# Output: Unknown parameter: -invalid

# Missing parameter value
catch {torch::tensor_rfft -input $tensor -n} result
puts $result  ;# Output: Missing value for parameter
```

## Migration Guide

### From Positional to Named Parameters

**Old code:**
```tcl
set result [torch::tensor_rfft $tensor 8 1]
```

**New code (equivalent):**
```tcl
set result [torch::tensor_rfft -input $tensor -n 8 -dim 1]
```

### Using CamelCase Alias

**Old code:**
```tcl
set result [torch::tensor_rfft $tensor]
```

**New code (equivalent):**
```tcl
set result [torch::tensorRfft $tensor]
```

## Notes

- The `n` parameter specifies the number of points for the FFT. If not specified, uses the size of the input tensor along the specified dimension.
- The `dim` parameter defaults to -1 (last dimension) if not specified.
- The result is a complex-valued tensor with shape `(n//2 + 1,)` along the transformed dimension.
- For real input, the RFFT output contains only the positive frequency components.
- The command supports tensors of various data types (float32, float64, int32, etc.).
- Zero padding can be achieved by specifying an `n` value larger than the input size.
- The RFFT is more efficient than the full FFT for real-valued input data.

## See Also

- `torch::tensor_irfft` - Inverse real-valued FFT
- `torch::tensor_fft` - Complex-valued FFT
- `torch::tensor_ifft` - Inverse complex-valued FFT
- `torch::tensor_fft2d` - 2D FFT
- `torch::tensor_ifft2d` - 2D inverse FFT 