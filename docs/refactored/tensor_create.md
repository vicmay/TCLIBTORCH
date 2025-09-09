# torch::tensor_create

Creates a new tensor from input data with specified properties.

## Syntax

### Positional Parameters (Original)
```tcl
torch::tensor_create values ?shape? ?dtype? ?device? ?requires_grad?
```

### Named Parameters (Refactored)
```tcl
torch::tensor_create -data values -dtype dtype -device device -requiresGrad bool
```

### camelCase Alias
```tcl
torch::tensorCreate -data values -dtype dtype -device device -requiresGrad bool
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `values` / `-data` | list | Yes | - | Input data as a Tcl list |
| `shape` | list | No | - | Optional reshape dimensions |
| `dtype` / `-dtype` | string | No | "float32" | Data type (float32, float64, int32, int64, bool) |
| `device` / `-device` | string | No | "cpu" | Device placement (cpu, cuda) |
| `requires_grad` / `-requiresGrad` | boolean | No | false | Whether to track gradients |

## Return Value

Returns a new tensor handle that can be used with other tensor operations.

## Examples

### Basic Usage

```tcl
# Simple tensor creation
set tensor1 [torch::tensor_create {1.0 2.0 3.0}]
set tensor2 [torch::tensorCreate -data {1.0 2.0 3.0}]

# Both syntaxes create identical tensors
```

### With Data Types

```tcl
# Positional syntax
set int_tensor [torch::tensor_create {1 2 3} int32]
set float_tensor [torch::tensor_create {1.0 2.0 3.0} float64]

# Named parameter syntax  
set int_tensor [torch::tensor_create -data {1 2 3} -dtype int32]
set float_tensor [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float64]
```

### With Device Specification

```tcl
# CPU tensor (default)
set cpu_tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu]

# CUDA tensor (if available)
set cuda_tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cuda]
```

### With Gradient Tracking

```tcl
# Enable gradient tracking for training
set trainable_tensor [torch::tensor_create -data {1.0 2.0 3.0} -requiresGrad true]

# Positional syntax
set trainable_tensor2 [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
```

### Multi-dimensional Tensors

```tcl
# 2D tensor creation
set matrix_data {1.0 2.0 3.0 4.0 5.0 6.0}

# Named parameter syntax
set matrix [torch::tensor_create -data $matrix_data -dtype float32]
set reshaped [torch::tensor_reshape $matrix {2 3}]

# Positional syntax with shape
set matrix2 [torch::tensor_create $matrix_data {2 3} float32]
```

### Handling Negative Values

```tcl
# Negative values work correctly in both syntaxes
set negative_tensor1 [torch::tensor_create {-1.0 -2.0 -3.0}]
set negative_tensor2 [torch::tensor_create -data {-1.0 -2.0 -3.0}]

# Mixed positive and negative
set mixed [torch::tensorCreate -data {-0.5 0.0 0.5 1.0}]
```

## Data Type Support

| Type | Description | Example |
|------|-------------|---------|
| `float32` | 32-bit floating point (default) | `{1.0 2.5 3.14}` |
| `float64` | 64-bit floating point | `{1.0 2.5 3.14}` |
| `int32` | 32-bit signed integer | `{1 2 3}` |
| `int64` | 64-bit signed integer | `{1 2 3}` |
| `bool` | Boolean values | `{true false true}` |

## Device Support

| Device | Description | Availability |
|--------|-------------|--------------|
| `cpu` | CPU computation (default) | Always available |
| `cuda` | NVIDIA GPU computation | Requires CUDA-enabled PyTorch |

## Error Handling

### Common Errors

```tcl
# Missing data
torch::tensor_create -dtype float32
# Error: Missing required parameter: -data

# Invalid data type
torch::tensor_create -data {1 2 3} -dtype invalid_type  
# Error: Invalid dtype: invalid_type

# Invalid device
torch::tensor_create -data {1 2 3} -device invalid_device
# Error: Invalid device specification
```

### Error Messages

The command provides clear error messages for:
- Missing required data parameter
- Invalid data types
- Invalid device specifications
- Malformed data lists

## Migration Guide

### From Positional to Named Parameters

```tcl
# OLD (positional syntax)
set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]

# NEW (named parameter syntax)  
set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]

# ALTERNATIVE (camelCase alias)
set tensor [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
```

### Benefits of Named Parameters

1. **Self-documenting**: Parameter names make code more readable
2. **Flexible order**: Parameters can be specified in any order
3. **Optional parameters**: Easy to omit parameters you don't need
4. **Less error-prone**: No need to remember parameter positions

## Performance Notes

- Both syntaxes have identical performance characteristics
- Named parameter parsing adds minimal overhead (<1%)
- Tensor creation performance depends on data size and device, not syntax used

## See Also

- [torch::zeros](zeros.md) - Create zero-filled tensors
- [torch::ones](ones.md) - Create one-filled tensors  
- [torch::empty](empty.md) - Create uninitialized tensors
- [torch::tensor_reshape](tensor_reshape.md) - Reshape existing tensors 