# torch::zero_pad1d

Applies 1D zero padding to a tensor. This operation pads the input tensor with zeros on both sides along its last dimension.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::zero_pad1d tensor padding
```

### Named Parameter Syntax
```tcl
torch::zero_pad1d -input tensor -padding {left right}
torch::zero_pad1d -tensor tensor -pad {left right}
```

### CamelCase Alias
```tcl
torch::zeroPad1d tensor padding
torch::zeroPad1d -input tensor -padding {left right}
torch::zeroPad1d -tensor tensor -pad {left right}
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor` / `-input` / `-tensor` | tensor handle | required | Input tensor to pad |
| `padding` / `-padding` / `-pad` | list of 2 ints | required | Amount of padding on each side: `{left right}` |

## Description

Zero padding adds zeros to the beginning and end of the last dimension of the input tensor. This is commonly used in convolutional neural networks to control the spatial size of the output.

## Examples

### Basic Usage
```tcl
set t [torch::arange -start 1 -end 5 -dtype float32]
set padded [torch::zero_pad1d $t {2 3}]
puts [torch::tensorToList $padded]
;# Output: 0.0 0.0 1.0 2.0 3.0 4.0 0.0 0.0 0.0
```

### Using Named Parameters
```tcl
set t [torch::arange -start 1 -end 5 -dtype float32]
set padded [torch::zero_pad1d -input $t -padding {1 2}]
puts [torch::tensorToList $padded]
;# Output: 0.0 1.0 2.0 3.0 4.0 0.0 0.0
```

### Using CamelCase Alias
```tcl
set t [torch::arange -start 1 -end 5 -dtype float32]
set padded [torch::zeroPad1d $t {1 1}]
puts [torch::tensorToList $padded]
;# Output: 0.0 1.0 2.0 3.0 4.0 0.0
```

## Error Handling

- If the padding list is not of length 2, an error is raised:
  ```tcl
  set t [torch::arange -start 1 -end 5 -dtype float32]
  catch {torch::zero_pad1d $t {1 2 3}} result
  puts $result
  ;# Output: Padding must be a list of 2 values for 1D
  ```
- If required parameters are missing, a usage error is raised.
- If an unknown parameter is provided, an error is raised:
  ```tcl
  catch {torch::zero_pad1d -foo $t -padding {1 1}} result
  puts $result
  ;# Output: Unknown parameter: -foo. Valid parameters are: -input, -tensor, -padding, -pad
  ```

## Edge Cases

- Zero padding with `{0 0}` returns the original tensor unchanged.
- Negative padding is not supported and will result in an error from the underlying library.
- Zero-length tensors are accepted but may not be meaningful for padding.

## Migration from Positional to Named Syntax

### Before (Positional Syntax)
```tcl
set padded [torch::zero_pad1d $t {2 1}]
```

### After (Named Parameter Syntax)
```tcl
set padded [torch::zero_pad1d -input $t -padding {2 1}]
```

### Using CamelCase Alias
```tcl
set padded [torch::zeroPad1d -input $t -padding {2 1}]
```

## Return Value

Returns a tensor handle to the padded tensor. The shape of the last dimension is increased by the sum of the left and right padding.

## Notes
- This command is an alias for constant padding with value 0.
- Both snake_case (`torch::zero_pad1d`) and camelCase (`torch::zeroPad1d`) aliases are available.
- 100% backward compatibility is maintained with the original positional syntax. 