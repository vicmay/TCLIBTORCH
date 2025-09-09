# torch::full

Creates a tensor filled with a specified value.

## Syntax

### Named Parameters (New Syntax)
```tcl
torch::full -shape list -value number ?-dtype type? ?-device device? ?-requiresGrad bool?
```

### Positional Parameters (Legacy Syntax)
```tcl
torch::full shape value ?dtype? ?device? ?requires_grad?
```

## Parameters

### Named Parameters
- **`-shape`** (required): List specifying the dimensions of the tensor
- **`-value`** (required): The value to fill the tensor with
- **`-dtype`** (optional): Data type of the tensor (default: "float32")
- **`-device`** (optional): Device to create tensor on (default: "cpu")
- **`-requiresGrad`** (optional): Whether the tensor requires gradients (default: false)

### Positional Parameters
- **`shape`** (required): List specifying the dimensions of the tensor
- **`value`** (required): The value to fill the tensor with
- **`dtype`** (optional): Data type of the tensor (default: "float32")
- **`device`** (optional): Device to create tensor on (default: "cpu")
- **`requires_grad`** (optional): Whether the tensor requires gradients (default: false)

## Examples

### Basic Usage

```tcl
# Create 2x2 tensor filled with 5.0
set tensor [torch::full {2 2} 5.0]
# Result: tensor([[5., 5.],
#                 [5., 5.]])

# Same with named parameters
set tensor [torch::full -shape {2 2} -value 5.0]
# Result: tensor([[5., 5.],
#                 [5., 5.]])
```

### With Different Data Types

```tcl
# Create integer tensor filled with 3
set tensor [torch::full {3 3} 3 int32]
# Result: tensor([[3, 3, 3],
#                 [3, 3, 3],
#                 [3, 3, 3]], dtype=torch.int32)

# Same with named parameters
set tensor [torch::full -shape {3 3} -value 3 -dtype int32]
# Result: tensor([[3, 3, 3],
#                 [3, 3, 3],
#                 [3, 3, 3]], dtype=torch.int32)
```

### With Device Specification

```tcl
# Create tensor on CPU
set tensor [torch::full {2 3} 1.5 float32 cpu]
# Result: tensor([[1.5000, 1.5000, 1.5000],
#                 [1.5000, 1.5000, 1.5000]])

# Same with named parameters
set tensor [torch::full -shape {2 3} -value 1.5 -dtype float32 -device cpu]
# Result: tensor([[1.5000, 1.5000, 1.5000],
#                 [1.5000, 1.5000, 1.5000]])
```

### With All Parameters

```tcl
# Create tensor with all parameters specified
set tensor [torch::full {2 2} 5.0 float32 cpu false]
# Result: tensor([[5., 5.],
#                 [5., 5.]])

# Same with named parameters
set tensor [torch::full -shape {2 2} -value 5.0 -dtype float32 -device cpu -requiresGrad false]
# Result: tensor([[5., 5.],
#                 [5., 5.]])
```

### With Gradient Tracking

```tcl
# Create tensor that requires gradients
set tensor [torch::full -shape {3 3} -value 2.0 -requiresGrad true]
# Result: tensor([[2., 2., 2.],
#                 [2., 2., 2.],
#                 [2., 2., 2.]], requires_grad=True)
```

### Different Shapes

```tcl
# 1D tensor
set tensor [torch::full -shape {5} -value 10]
# Result: tensor([10., 10., 10., 10., 10.])

# 3D tensor
set tensor [torch::full -shape {2 3 4} -value 0.5]
# Result: tensor([[[0.5000, 0.5000, 0.5000, 0.5000],
#                  [0.5000, 0.5000, 0.5000, 0.5000],
#                  [0.5000, 0.5000, 0.5000, 0.5000]],
#                 [[0.5000, 0.5000, 0.5000, 0.5000],
#                  [0.5000, 0.5000, 0.5000, 0.5000],
#                  [0.5000, 0.5000, 0.5000, 0.5000]]])
```

## Supported Data Types

- `float32` (default)
- `float64`
- `int32`
- `int64`
- `uint8`
- `int8`
- `int16`
- `uint16`
- `bool`

## Supported Devices

- `cpu` (default)
- `cuda` (if CUDA is available)

## Error Handling

### Missing Required Parameters
```tcl
# This will fail - missing -value parameter
torch::full -shape {2 2} -dtype float32
# Error: Missing required parameter: -value
```

### Invalid Parameters
```tcl
# This will fail - unknown parameter
torch::full -invalid 5
# Error: Unknown parameter: -invalid
```

### Invalid Data Types
```tcl
# This will fail - invalid dtype
torch::full -shape {2 2} -value 1 -dtype bad
# Error: Invalid dtype: bad
```

### Missing Values
```tcl
# This will fail - missing value for parameter
torch::full -shape
# Error: Missing value for parameter
```

## Performance Notes

- Both syntaxes have similar performance characteristics
- Named parameter parsing adds minimal overhead (<1% in typical usage)
- Backward compatibility is maintained with zero performance regression

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old syntax
torch::full {2 2} 5.0 float32 cpu false

# New syntax (equivalent)
torch::full -shape {2 2} -value 5.0 -dtype float32 -device cpu -requiresGrad false
```

### Benefits of Named Parameters

1. **Clarity**: Parameter meaning is explicit
2. **Flexibility**: Parameters can be specified in any order
3. **Maintainability**: Code is more self-documenting
4. **Future-proof**: Easier to add new parameters

## Backward Compatibility

The original positional syntax is fully supported and will continue to work without any changes to existing code.

## Use Cases

- **Initialization**: Create tensors with specific initial values
- **Constants**: Create tensors filled with constant values
- **Masks**: Create boolean masks filled with True/False
- **Padding**: Create padding tensors with specific values
- **Testing**: Create test tensors with known values

## Related Commands

- `torch::zeros` - Create tensor filled with zeros
- `torch::ones` - Create tensor filled with ones
- `torch::empty` - Create uninitialized tensor
- `torch::eye` - Create identity matrix
- `torch::full_like` - Create tensor with same shape as input, filled with value 