# torch::ones_like / torch::onesLike

Creates a tensor of ones with the same shape and properties as an input tensor.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::ones_like -input tensor_name [-dtype dtype] [-device device] [-requiresGrad boolean]
torch::onesLike -input tensor_name [-dtype dtype] [-device device] [-requiresGrad boolean]
```

### Positional Syntax (Legacy)
```tcl
torch::ones_like tensor_name ?dtype? ?device?
torch::onesLike tensor_name ?dtype? ?device?
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| input | string | Yes | Name of the input tensor to match shape and properties |
| dtype | string | No | Data type for the new tensor (if different from input) |
| device | string | No | Device for the new tensor (if different from input) |
| requiresGrad | boolean | No | Whether the tensor should require gradients (default: false) |

## Returns

Returns a string handle to the newly created tensor filled with ones.

## Examples

### Named Parameter Syntax

```tcl
# Basic usage - create ones tensor with same shape as input
set input [torch::randn -shape {3 4}]
set ones_tensor [torch::ones_like -input $input]
puts [torch::shape $ones_tensor]  ;# {3 4}

# Create with different dtype
set input [torch::randn -shape {2 3} -dtype float32]
set ones_int [torch::ones_like -input $input -dtype int32]
puts [torch::dtype $ones_int]  ;# int32

# Create with gradient tracking
set input [torch::randn -shape {2 2}]
set ones_grad [torch::ones_like -input $input -requiresGrad true]
puts [torch::requires_grad $ones_grad]  ;# true

# Create on specific device (if CUDA available)
set input [torch::randn -shape {3 3}]
set ones_cpu [torch::ones_like -input $input -device cpu]
puts [torch::device $ones_cpu]  ;# cpu

# Using camelCase alias
set input [torch::randn -shape {2 5}]
set result [torch::onesLike -input $input -dtype float64]
```

### Positional Syntax

```tcl
# Basic usage
set input [torch::randn -shape {3 4}]
set ones_tensor [torch::ones_like $input]

# With dtype
set ones_int [torch::ones_like $input int32]

# With dtype and device
set ones_gpu [torch::ones_like $input float32 cuda:0]
```

## Data Type Support

Supports all PyTorch data types:
- **Floating point**: `float16`, `float32`, `float64`
- **Integer**: `int8`, `int16`, `int32`, `int64`
- **Unsigned**: `uint8`
- **Boolean**: `bool`
- **Complex**: `complex64`, `complex128`

## Device Support

- `cpu` - CPU tensor
- `cuda` - Default CUDA device
- `cuda:0`, `cuda:1`, etc. - Specific CUDA device

## Mathematical Properties

Given an input tensor `X` with shape `(d1, d2, ..., dn)`, creates tensor `Y` where:
- `Y.shape = X.shape`
- `Y[i1, i2, ..., in] = 1` for all valid indices
- `Y.dtype = specified_dtype or X.dtype`
- `Y.device = specified_device or X.device`

## Error Handling

The command will raise an error if:
- Input tensor handle is invalid or doesn't exist
- Invalid data type specified
- Invalid device specified
- Missing required parameters in named syntax

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set result [torch::ones_like $input]
set result [torch::ones_like $input float32]
set result [torch::ones_like $input float32 cpu]

# New named parameter syntax
set result [torch::ones_like -input $input]
set result [torch::ones_like -input $input -dtype float32]
set result [torch::ones_like -input $input -dtype float32 -device cpu]

# Additional capability in named syntax
set result [torch::ones_like -input $input -requiresGrad true]
```

### Advantages of Named Parameters

1. **Clarity**: Parameter purpose is explicit
2. **Flexibility**: Parameters can be specified in any order
3. **Optional parameters**: Can skip intermediate optional parameters
4. **Extensibility**: New parameters can be added without breaking existing code
5. **Gradient support**: Can specify `requiresGrad` parameter

## Performance Notes

- Operation is memory allocation bound
- Performance scales with output tensor size
- Device specification affects memory allocation location

## See Also

- [torch::zeros_like](zeros_like.md) - Create zeros tensor with same shape
- [torch::empty_like](empty_like.md) - Create uninitialized tensor with same shape
- [torch::full_like](full_like.md) - Create tensor filled with specific value
- [torch::rand_like](rand_like.md) - Create random tensor with same shape
- [torch::ones](ones.md) - Create ones tensor with specified shape

## Implementation Details

The command uses PyTorch's `torch::ones_like()` function with proper tensor options for dtype, device, and gradient requirements. 