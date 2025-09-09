# torch::tensor_randn

Creates a tensor filled with random numbers from a normal (Gaussian) distribution with mean 0 and standard deviation 1.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_randn shape ?device? ?dtype?
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_randn -shape shape ?-device device? ?-dtype dtype?
```

### CamelCase Alias
```tcl
torch::tensorRandn shape ?device? ?dtype?
torch::tensorRandn -shape shape ?-device device? ?-dtype dtype?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | list | **required** | Shape of the tensor as a Tcl list |
| `device` | string | `"cpu"` | Device to create tensor on (`"cpu"`, `"cuda"`, etc.) |
| `dtype` | string | `"float32"` | Data type (`"float32"`, `"float64"`, etc.) |

## Return Value

Returns a tensor handle (string) that can be used with other tensor operations.

## Examples

### Basic Usage

```tcl
# Create a 1D tensor with 5 elements
set t1 [torch::tensor_randn {5}]

# Create a 2D tensor with shape (3, 4)
set t2 [torch::tensor_randn {3 4}]

# Create a 3D tensor with shape (2, 2, 2)
set t3 [torch::tensor_randn {2 2 2}]
```

### With Device and Data Type

```tcl
# Positional syntax
set t1 [torch::tensor_randn {2 3} cpu float64]

# Named parameter syntax
set t2 [torch::tensor_randn -shape {2 3} -device cpu -dtype float64]

# CamelCase alias
set t3 [torch::tensorRandn {2 3} cpu float64]
```

### Scalar Tensor

```tcl
# Create a scalar tensor (0D)
set scalar [torch::tensor_randn {}]
```

### Advanced Examples

```tcl
# Create tensors for neural network weights
set weights1 [torch::tensor_randn {100 50}]
set weights2 [torch::tensor_randn {50 10}]

# Create bias vectors
set bias1 [torch::tensor_randn {50}]
set bias2 [torch::tensor_randn {10}]

# Create tensors with specific precision
set high_precision [torch::tensor_randn {1000 1000} -dtype float64]
```

## Error Handling

### Missing Required Parameters
```tcl
# Error: Missing shape parameter
torch::tensor_randn
# Error: Required parameter missing: shape
```

### Invalid Parameters
```tcl
# Error: Unknown parameter
torch::tensor_randn -foo {2 2}
# Error: Unknown parameter: -foo

# Error: Missing value for parameter
torch::tensor_randn -shape
# Error: Missing value for parameter

# Error: Too many positional arguments
torch::tensor_randn {2 2} cpu float32 extra
# Error: Invalid number of arguments
```

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
set t [torch::tensor_randn {2 3} cpu float64]
```

**New (Named Parameters):**
```tcl
set t [torch::tensor_randn -shape {2 3} -device cpu -dtype float64]
```

### Using CamelCase Alias

**Old (snake_case):**
```tcl
set t [torch::tensor_randn {2 3}]
```

**New (camelCase):**
```tcl
set t [torch::tensorRandn {2 3}]
```

## Mathematical Properties

The `tensor_randn` function creates tensors with values drawn from a standard normal distribution:

- **Mean**: 0
- **Standard Deviation**: 1
- **Distribution**: Normal (Gaussian)

This makes it ideal for:
- Initializing neural network weights
- Creating random noise for data augmentation
- Generating test data with known statistical properties

## Performance Considerations

- **Memory**: Tensor size is determined by the product of shape dimensions
- **Device**: CPU tensors are created by default; use CUDA for GPU acceleration
- **Precision**: `float32` is faster but less precise than `float64`

## Related Commands

- `torch::tensor_rand` - Uniform distribution tensors
- `torch::tensor_zeros` - Zero tensors
- `torch::tensor_ones` - One tensors
- `torch::tensor_empty` - Uninitialized tensors

## Test Coverage

The refactored command includes comprehensive tests covering:

1. **Basic Functionality**: Both syntaxes work correctly
2. **Parameter Validation**: Error handling for invalid inputs
3. **Mathematical Correctness**: Values follow normal distribution
4. **Data Type Support**: Different dtypes work as expected
5. **Edge Cases**: Empty shapes, large tensors
6. **Syntax Consistency**: Both syntaxes produce same results
7. **Device Support**: CPU device handling
8. **CamelCase Alias**: Alternative naming works correctly

## Backward Compatibility

✅ **100% Backward Compatible**: All existing code using positional parameters will continue to work without modification.

## Notes

- The normal distribution is the most commonly used distribution for initializing neural network weights
- Values are generated using PyTorch's `torch.randn()` function
- Empty shape `{}` creates a scalar tensor (0D)
- The distribution is truly random; each call produces different values 