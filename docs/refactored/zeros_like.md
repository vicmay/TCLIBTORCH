# torch::zeros_like / torch::zerosLike

Create a tensor filled with zeros using the same shape as an input tensor.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::zeros_like tensor ?dtype? ?device?
```

### Named Parameter Syntax
```tcl
torch::zeros_like -input tensor ?-dtype dtype? ?-device device? ?-requiresGrad bool?
torch::zerosLike -input tensor ?-dtype dtype? ?-device device? ?-requiresGrad bool?
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input` / `tensor` | string | Handle of the input tensor to match shape | Required |
| `dtype` | string | Data type for the result tensor | Same as input |
| `device` | string | Device for the result tensor | Same as input |
| `requiresGrad` | boolean | Whether result tensor requires gradients | false |

### Supported Data Types
- `int32`, `int64`
- `float32`, `float64`
- `bool`

### Supported Devices
- `cpu`
- `cuda` (if available)

## Return Value

Returns a string handle representing the new tensor filled with zeros.

## Examples

### Basic Usage
```tcl
# Create reference tensor
set input [torch::ones {3 4}]

# Create zeros tensor with same shape
set result [torch::zeros_like $input]
# Result: 3x4 tensor filled with 0.0
```

### Positional Syntax Examples
```tcl
# Basic usage
set result [torch::zeros_like $input]

# With dtype
set result [torch::zeros_like $input float64]

# With dtype and device
set result [torch::zeros_like $input int32 cpu]
```

### Named Parameter Syntax Examples
```tcl
# Basic usage
set result [torch::zeros_like -input $input]

# With specific dtype
set result [torch::zeros_like -input $input -dtype float64]

# With all parameters
set result [torch::zeros_like -input $input -dtype int32 -device cpu -requiresGrad true]

# Parameter order doesn't matter
set result [torch::zeros_like -dtype float32 -input $input -device cpu]
```

### CamelCase Alias Examples
```tcl
# Using camelCase alias
set result [torch::zerosLike -input $input]

# With parameters
set result [torch::zerosLike -input $input -dtype float64 -requiresGrad true]
```

## Integration Examples

### Mathematical Operations
```tcl
set input [torch::ones {2 3}]
set zeros [torch::zeros_like -input $input]
set sum [torch::tensor_add $input $zeros]
# Result: 2x3 tensor filled with 1.0 (adding zeros doesn't change values)
```

### Shape Preservation
```tcl
# Works with any input shape
set input_1d [torch::ones {5}]
set zeros_1d [torch::zeros_like $input_1d]

set input_3d [torch::ones {2 3 4}]
set zeros_3d [torch::zeros_like $input_3d]
```

### Gradient Computation
```tcl
set input [torch::ones {2 2}]
set zeros [torch::zeros_like -input $input -requiresGrad true]
# Result tensor will require gradients
```

### Baseline Creation
```tcl
# Common use case: create baseline for comparison
set predictions [torch::ones {10}]
set baseline [torch::zeros_like $predictions]
set difference [torch::tensor_sub $predictions $baseline]
# Calculate difference from zero baseline
```

## Error Handling

### Invalid Tensor
```tcl
catch {torch::zeros_like invalid_tensor} error
# Error: "Invalid tensor name"
```

### Missing Required Parameters
```tcl
catch {torch::zeros_like -dtype float32} error
# Error: "Input tensor is required"
```

### Unknown Parameters
```tcl
catch {torch::zeros_like -input $tensor -invalid param} error
# Error: "Unknown parameter: -invalid"
```

### Invalid Data Types
```tcl
catch {torch::zeros_like -input $tensor -dtype invalid_type} error
# Error message about invalid dtype
```

## Migration Guide

### From Positional to Named Parameters

**Old positional syntax:**
```tcl
set result [torch::zeros_like $input float64 cpu]
```

**New named parameter syntax:**
```tcl
set result [torch::zeros_like -input $input -dtype float64 -device cpu]
```

**Or using camelCase alias:**
```tcl
set result [torch::zerosLike -input $input -dtype float64 -device cpu]
```

### Benefits of Named Parameters

1. **Clarity**: Parameter names make code self-documenting
2. **Flexibility**: Parameters can be specified in any order
3. **Optional parameters**: Easier to specify only needed parameters
4. **Future-proof**: New parameters can be added without breaking existing code

## Performance Notes

- The command creates a new tensor with the same shape as the input
- Memory usage is proportional to the input tensor size
- All elements are initialized to 0.0 (or equivalent for the specified dtype)
- Device operations (e.g., CPU to CUDA) may involve memory transfers
- Zero-filled tensors are commonly used as baselines and initializers

## Common Use Cases

### Baseline Creation
```tcl
# Create zero baseline for loss calculation
set predictions [torch::randn {100 10}]
set baseline [torch::zeros_like $predictions]
```

### Mask Initialization
```tcl
# Create zero mask (to be filled selectively)
set input [torch::randn {32 128}]
set mask [torch::zeros_like $input]
```

### Gradient Accumulation
```tcl
# Initialize gradient accumulator
set weights [torch::randn {256 512}]
set grad_accum [torch::zeros_like $weights]
```

## See Also

- [`torch::ones_like`](ones_like.md) - Create tensor filled with ones
- [`torch::empty_like`](empty_like.md) - Create uninitialized tensor
- [`torch::full_like`](full_like.md) - Create tensor filled with specific value
- [`torch::zeros`](zeros.md) - Create zeros tensor with specified shape
- [`torch::tensor_shape`](tensor_shape.md) - Get tensor shape
- [`torch::tensor_dtype`](tensor_dtype.md) - Get tensor data type 