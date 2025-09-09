# torch::full_like

Creates a tensor filled with a specified constant value that has the **same shape** as a reference tensor.

## Syntax

### Named Parameters (Recommended)
```tcl
# snake_case
torch::full_like -input tensor -value number ?-dtype dtype? ?-device device? ?-requiresGrad bool?

# camelCase alias
torch::fullLike -input tensor -value number ?-dtype dtype? ?-device device? ?-requiresGrad bool?
```

### Positional Parameters (Legacy)
```tcl
torch::full_like input value ?dtype? ?device?
```

## Parameters
| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `-input`, `input` | string | Handle of the reference tensor whose shape will be copied. | **Yes** |
| `-value`, `value` | double | The constant value to fill the new tensor with. | **Yes** |
| `-dtype` | string | Desired data type (e.g. `float32`, `int64`). Defaults to dtype of input tensor. | No |
| `-device` | string | Target device (`cpu`, `cuda`, etc.). Defaults to device of input tensor. | No |
| `-requiresGrad` | bool | Whether the resulting tensor tracks gradients. Defaults to `false`. | No |

## Description
`full_like` is equivalent to PyTorch's `torch.full_like`.  It quickly creates a tensor of identical size and (optionally) data type/device as the given input tensor, filled with a constant value.  This is often more convenient and faster than manually querying the shape and calling `torch::full`.

## Examples

### Named Parameter Syntax
```tcl
set a [torch::randn {2 3} float32 cpu false]
# Create tensor filled with 7.0 on same device/dtype
set b [torch::full_like -input $a -value 7.0]

# Override dtype and ask for gradients
set c [torch::full_like -input $a -value 0 -dtype int64 -requiresGrad true]

# Using camelCase alias
set d [torch::fullLike -input $a -value 1.5]
```

### Positional Syntax
```tcl
set a [torch::zeros {4 4} float32 cpu false]
set b [torch::full_like $a 3.14 float32 cpu]
```

## Return Value
Returns a **tensor handle** that can be used with other LibTorch TCL commands.

## Error Handling
* **Missing input/value** – both the reference tensor and constant value are required.
* **Invalid tensor handle** – errors if the given input tensor does not exist.
* **Unknown parameter** – any unrecognised named parameter triggers an error.

## Compatibility
* ✅ **Backward compatible** – original positional syntax still works.
* ✅ **camelCase alias** – `torch::fullLike` is registered.

## See Also
* [`torch::empty_like`](empty_like.md) – uninitialised tensor with same shape.
* [`torch::zeros_like`](zeros_like.md) – same-shape tensor filled with zeros.
* [`torch::ones_like`](ones_like.md) – same-shape tensor filled with ones.

## Implementation Notes

- **Backward Compatibility**: All existing positional syntax continues to work unchanged
- **Parameter Validation**: All parameters are validated with clear error messages
- **Data Type Handling**: Supports all PyTorch data types (float32, float64, int32, int64, etc.)
- **Device Support**: Supports CPU and CUDA devices when available
- **Memory Efficiency**: Creates new tensor with optimized memory allocation

## Common Use Cases

1. **Initialization**: Creating tensors for neural network weights or biases
2. **Masking**: Creating mask tensors with specific values
3. **Testing**: Generating test data with known values
4. **Preprocessing**: Creating constant tensors for mathematical operations

## Related Commands

- `torch::zeros_like` - Creates tensor filled with zeros
- `torch::ones_like` - Creates tensor filled with ones
- `torch::empty_like` - Creates uninitialized tensor with same shape
- `torch::rand_like` - Creates tensor with random values
- `torch::randn_like` - Creates tensor with normal distribution values

## Migration Guide

```tcl
# Old style (still works)
set result [torch::full_like $input 5.0 float32]

# New style (recommended)
set result [torch::full_like -input $input -value 5.0 -dtype float32]

# Using camelCase alias
set result [torch::fullLike -input $input -value 5.0 -dtype float32]
``` 