# torch::zero_pad2d / torch::zeroPad2d

Zero-pad a 2D tensor (matrix) on all sides. This is an alias for constant padding with value 0.

## Syntax

### Positional (backward compatible)
```tcl
torch::zero_pad2d tensor padding
```
- `tensor`: Input 2D tensor (matrix)
- `padding`: List of 4 integers: `{pad_left pad_right pad_top pad_bottom}`

### Named Parameters (modern)
```tcl
torch::zero_pad2d -input tensor -padding {pad_left pad_right pad_top pad_bottom}
torch::zero_pad2d -tensor tensor -pad {pad_left pad_right pad_top pad_bottom}
```

### CamelCase Alias
```tcl
torch::zeroPad2d ...
```

## Parameters
| Name      | Alias   | Type   | Required | Description                                 |
|-----------|---------|--------|----------|---------------------------------------------|
| -input    | -tensor | tensor | yes      | Input 2D tensor (matrix)                    |
| -padding  | -pad    | list   | yes      | List of 4 ints: `{left right top bottom}`   |

## Return Value
A new tensor with zero padding applied.

## Examples

### Positional
```tcl
set t [torch::tensor_create {{1 2} {3 4}} float32 cpu false]
set padded [torch::zero_pad2d $t {1 1 1 1}]
```

### Named
```tcl
set t [torch::tensor_create {{1 2} {3 4}} float32 cpu false]
set padded [torch::zero_pad2d -input $t -padding {2 0 1 3}]
```

### CamelCase
```tcl
set t [torch::tensor_create {{1 2} {3 4}} float32 cpu false]
set padded [torch::zeroPad2d $t {1 1 1 1}]
```

## Error Handling
- If arguments are missing or invalid, a clear error message is returned.
- Padding must be a list of 4 integers.
- Input must be a 2D tensor.

## Migration Guide
- Old code using positional syntax will continue to work.
- New code should prefer named parameters for clarity and future compatibility.

## See Also
- [`torch::constant_pad2d`](./constant_pad2d.md)
- [`torch::reflection_pad2d`](./reflection_pad2d.md)
- [`torch::replication_pad2d`](./replication_pad2d.md) 