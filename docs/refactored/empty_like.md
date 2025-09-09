# torch::empty_like

Creates an uninitialized tensor with the same shape as the input tensor.

## Syntax

### Current Syntax
```tcl
torch::empty_like tensor ?dtype? ?device?
```

### Named Parameter Syntax  
```tcl
torch::empty_like -input tensor ?-dtype dtype? ?-device device? ?-requiresGrad bool?
```

### CamelCase Alias
```tcl
torch::emptyLike tensor ?dtype? ?device?
torch::emptyLike -input tensor ?-dtype dtype? ?-device device? ?-requiresGrad bool?
```

All syntaxes are fully supported and equivalent.

## Parameters

### Named Parameters
- `-input` (required): Input tensor to match shape from
- `-dtype` (optional): Data type for the new tensor (e.g., "float32", "float64", "int32")
- `-device` (optional): Device placement (e.g., "cpu", "cuda", "cuda:0")
- `-requiresGrad` (optional): Boolean flag for gradient computation (true/false)

### Positional Parameters
1. `tensor` (required): Input tensor to match shape from
2. `dtype` (optional): Data type for the new tensor
3. `device` (optional): Device placement

## Description

The `torch::empty_like` function creates a new tensor with the same shape as the input tensor, but with uninitialized values. This is useful for creating output tensors that will be filled with computed values, as it's more memory-efficient than initializing with specific values that will be overwritten.

The new tensor inherits properties from the input tensor unless explicitly overridden by parameters.

## Examples

### Basic Usage

#### Positional Syntax
```tcl
# Create reference tensor
set input [torch::ones {3 4}]

# Create empty tensor with same shape
set result [torch::empty_like $input]
# result shape: {3 4}, values: uninitialized
```

#### Named Parameter Syntax
```tcl
# Create reference tensor
set input [torch::ones {2 5}]

# Create empty tensor with same shape using named parameters
set result [torch::empty_like -input $input]
```

#### CamelCase Alias
```tcl
# Same functionality with camelCase
set input [torch::ones {4 3}]
set result [torch::emptyLike -input $input]
```

### Customizing Tensor Properties

#### Different Data Type
```tcl
# Input tensor is float32, create empty tensor as float64
set input [torch::ones {3 3}]
set result [torch::empty_like -input $input -dtype float64]

# Positional syntax
set result [torch::empty_like $input float64]
```

#### Different Device
```tcl
# Create empty tensor on specific device
set input [torch::ones {2 4}]
set result [torch::empty_like -input $input -device cpu]

# Positional syntax
set result [torch::empty_like $input float32 cpu]
```

#### Enable Gradient Computation
```tcl
# Create empty tensor with gradient computation enabled
set input [torch::ones {3 2}]
set result [torch::empty_like -input $input -requiresGrad true]
```

### Advanced Usage

#### All Parameters
```tcl
set input [torch::ones {5 5}]
set result [torch::empty_like -input $input -dtype float64 -device cpu -requiresGrad true]
```

#### Parameter Order Flexibility
```tcl
# Named parameters can be in any order
set input [torch::ones {3 3}]
set result [torch::empty_like -dtype float32 -requiresGrad false -input $input -device cpu]
```

### Working with Different Tensor Shapes

#### 1D Tensors
```tcl
set input [torch::ones {10}]
set result [torch::empty_like $input]
# result shape: {10}
```

#### 3D Tensors
```tcl
set input [torch::ones {2 3 4}]
set result [torch::empty_like $input]
# result shape: {2 3 4}
```

#### Large Tensors
```tcl
set input [torch::ones {1000 1000}]
set result [torch::empty_like $input]
# Memory-efficient: no initialization overhead
```

## Use Cases

### 1. Pre-allocating Output Tensors
```tcl
# Prepare output tensor for computation
set input [torch::randn {100 50}]
set weights [torch::randn {50 25}]

# Pre-allocate output tensor
set output [torch::empty_like $input]

# Perform computation (example)
# ... fill output with computed values ...
```

### 2. Memory-Efficient Tensor Creation
```tcl
# When you need a tensor with specific shape but will fill it immediately
set reference [torch::ones {1000 1000}]
set workspace [torch::empty_like $reference]

# workspace is ready for use without initialization overhead
```

### 3. Gradient-Enabled Tensors
```tcl
# Create tensor for gradient computation
set input [torch::randn {10 10}]
set output [torch::empty_like -input $input -requiresGrad true]

# Now output can participate in gradient computation
```

### 4. Cross-Device Tensor Creation
```tcl
# Create tensor on different device with same shape
set cpu_tensor [torch::ones {5 5}]
set gpu_tensor [torch::empty_like -input $cpu_tensor -device cuda]
```

## Performance Considerations

- **Memory Efficiency**: No initialization overhead compared to `zeros_like` or `ones_like`
- **Speed**: Fastest tensor creation method for known shapes
- **Memory Usage**: Only allocates memory, doesn't initialize values
- **Device Transfer**: Efficient for creating tensors on different devices

## Important Notes

### Uninitialized Values
- The tensor contains arbitrary values (whatever was in memory)
- Values are **not** guaranteed to be zero or any specific value
- Always fill the tensor before using values

```tcl
# DON'T do this - values are undefined
set empty_tensor [torch::empty_like $input]
set sum [torch::sum $empty_tensor]  # Result is undefined!

# DO this - fill before use
set empty_tensor [torch::empty_like $input]
# ... fill empty_tensor with computed values ...
set sum [torch::sum $empty_tensor]  # Now it's safe
```

## Error Handling

The function will raise an error if:
- Input tensor name is invalid or doesn't exist
- Required parameters are missing
- Unknown parameters are provided
- Invalid parameter values are given

```tcl
# Error handling example
if {[catch {torch::empty_like -input invalid_tensor} result]} {
    puts "Error: $result"
} else {
    puts "Success: $result"
}
```

## Related Functions

- `torch::zeros_like` - Creates tensor filled with zeros (same shape)
- `torch::ones_like` - Creates tensor filled with ones (same shape)
- `torch::full_like` - Creates tensor filled with specific value (same shape)
- `torch::rand_like` - Creates tensor with random values (same shape)
- `torch::randn_like` - Creates tensor with random normal values (same shape)
- `torch::empty` - Creates uninitialized tensor with specified shape

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax (still supported)
set result [torch::empty_like $input]
set result [torch::empty_like $input float64]
set result [torch::empty_like $input float32 cpu]

# New named parameter syntax
set result [torch::empty_like -input $input]
set result [torch::empty_like -input $input -dtype float64]
set result [torch::empty_like -input $input -dtype float32 -device cpu]

# Both produce identical results
```

### Benefits of Named Parameters

1. **Clarity**: Parameter purpose is explicit
2. **Flexibility**: Parameters can be provided in any order
3. **Extensibility**: Easy to add new parameters like `-requiresGrad`
4. **Consistency**: Matches modern TCL conventions

## Best Practices

### Memory Management
```tcl
# Use empty_like for output tensors that will be filled
proc matrix_multiply {a b} {
    # Pre-allocate result tensor
    set rows [lindex [torch::tensor_shape $a] 0]
    set cols [lindex [torch::tensor_shape $b] 1]
    set result [torch::empty_like $a]  # Will be overwritten anyway
    
    # Perform computation
    set result [torch::matmul $a $b]
    return $result
}
```

### Type Safety
```tcl
# Always specify dtype when precision matters
set input [torch::ones {100 100}]
set output [torch::empty_like -input $input -dtype float64]  # Explicit precision
```

### Device Management
```tcl
# Ensure tensors are on the right device
set cpu_input [torch::ones {50 50}]
set gpu_output [torch::empty_like -input $cpu_input -device cuda]
```

## Technical Notes

- Implements PyTorch's `torch.empty_like()` function
- Preserves tensor properties (device, dtype) from input unless overridden
- Memory allocation only, no value initialization
- Thread-safe operation
- Supports all PyTorch data types and devices

## Version History

- Added dual syntax support in refactoring phase
- Original positional syntax maintained for backward compatibility
- Named parameter syntax added for modern TCL conventions
- CamelCase alias (emptyLike) provided for consistency
- Added `-requiresGrad` parameter for gradient computation support 