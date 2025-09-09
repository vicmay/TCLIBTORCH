# torch::cummin

Compute the cumulative minimum of a tensor along a specified dimension.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::cummin -input <tensor> -dim <dimension>
torch::cumMin -input <tensor> -dim <dimension>  # camelCase alias
```

### Positional Syntax (Legacy)
```tcl
torch::cummin <tensor> <dimension>
torch::cumMin <tensor> <dimension>  # camelCase alias
```

## Parameters

### Named Parameters
- **`-input`** (required): Input tensor handle
- **`-dim`** (optional): Dimension along which to compute cumulative minimum (default: 0)

### Positional Parameters
1. **`tensor`** (required): Input tensor handle
2. **`dim`** (required): Dimension along which to compute cumulative minimum

## Returns

Returns a tensor handle containing the cumulative minimum values along the specified dimension.

## Description

The `torch::cummin` command computes the cumulative minimum of elements in a tensor along a specified dimension. At each position, the result contains the minimum of all elements from the beginning of that dimension up to the current position.

The function returns only the values tensor (the first element of PyTorch's cummin tuple). The indices tensor is not currently exposed.

## Examples

### Basic Usage

#### Named Parameter Syntax
```tcl
# Create a test tensor
set input [torch::tensor_create -data {3.0 1.0 4.0 2.0 5.0 0.0} -shape {2 3} -dtype float32]

# Compute cumulative minimum along dimension 0
set result [torch::cummin -input $input -dim 0]

# Compute cumulative minimum along dimension 1
set result [torch::cummin -input $input -dim 1]

# Using camelCase alias
set result [torch::cumMin -input $input -dim 0]
```

#### Positional Syntax (Backward Compatible)
```tcl
# Create a test tensor
set input [torch::tensor_create -data {3.0 1.0 4.0 2.0 5.0 0.0} -shape {2 3} -dtype float32]

# Compute cumulative minimum along dimension 0
set result [torch::cummin $input 0]

# Compute cumulative minimum along dimension 1
set result [torch::cummin $input 1]

# Using camelCase alias
set result [torch::cumMin $input 0]
```

### Advanced Examples

#### 1D Tensor
```tcl
# Create 1D tensor: [5.0, 2.0, 8.0, 1.0, 3.0]
set input [torch::tensor_create -data {5.0 2.0 8.0 1.0 3.0} -shape {5} -dtype float32]
set result [torch::cummin -input $input -dim 0]
# Result: [5.0, 2.0, 2.0, 1.0, 1.0]
```

#### 2D Tensor
```tcl
# Create 2D tensor: [[3.0, 1.0, 4.0],
#                    [2.0, 5.0, 0.0]]
set input [torch::tensor_create -data {3.0 1.0 4.0 2.0 5.0 0.0} -shape {2 3} -dtype float32]

# Cumulative minimum along rows (dim 0)
set result [torch::cummin -input $input -dim 0]
# Result: [[3.0, 1.0, 4.0],
#          [2.0, 1.0, 0.0]]

# Cumulative minimum along columns (dim 1)
set result [torch::cummin -input $input -dim 1]
# Result: [[3.0, 1.0, 1.0],
#          [2.0, 2.0, 0.0]]
```

#### Negative Dimension
```tcl
# Use negative dimension to count from the end
set input [torch::tensor_create -data {3.0 1.0 4.0 2.0} -shape {2 2} -dtype float32]
set result [torch::cummin -input $input -dim -1]  # Same as dim 1 for 2D tensor
```

#### Different Data Types
```tcl
# Integer tensor
set input [torch::tensor_create -data {5 2 8 1 3} -shape {5} -dtype int32]
set result [torch::cummin -input $input -dim 0]

# Double precision
set input [torch::tensor_create -data {5.0 2.0 8.0 1.0 3.0} -shape {5} -dtype float64]
set result [torch::cummin -input $input -dim 0]
```

#### Parameter Order Flexibility (Named Syntax)
```tcl
# Parameters can be specified in any order
set input [torch::tensor_create -data {3.0 1.0 4.0} -shape {3} -dtype float32]
set result1 [torch::cummin -input $input -dim 0]
set result2 [torch::cummin -dim 0 -input $input]  # Same result
```

## Error Handling

The command performs comprehensive validation and provides clear error messages:

```tcl
# Invalid tensor handle
catch {torch::cummin invalid_tensor 0} error
# Error: "Invalid tensor name"

# Missing required parameters (positional)
catch {torch::cummin $tensor} error
# Error: "Wrong number of arguments for positional syntax. Expected: torch::cummin tensor dim"

# Missing required parameters (named)
catch {torch::cummin -dim 0} error
# Error: "Required parameter missing: -input"

# Unknown parameter
catch {torch::cummin -input $tensor -unknown value} error
# Error: "Unknown parameter: -unknown"

# Invalid dimension value
catch {torch::cummin $tensor invalid_dim} error
# Error: "Invalid dim value. Expected integer."

# Missing value for parameter
catch {torch::cummin -input $tensor -dim} error
# Error: "Missing value for parameter"
```

## Data Type Support

The `torch::cummin` command supports all numeric tensor data types:

- **Floating point**: `float32`, `float64`, `float16`
- **Integer**: `int8`, `int16`, `int32`, `int64`
- **Unsigned integer**: `uint8`, `uint16`, `uint32`, `uint64`
- **Boolean**: `bool`

## Performance Notes

- The operation is performed natively in PyTorch/LibTorch for optimal performance
- Memory usage is proportional to the input tensor size
- GPU acceleration is automatically used when input tensors are on CUDA devices
- No significant performance difference between positional and named parameter syntax

## Mathematical Background

For a tensor **x** along dimension **d**, the cumulative minimum is computed as:

```
cummin(x)[i₀, i₁, ..., iₐ, ..., iₙ] = min(x[i₀, i₁, ..., 0:iₐ+1, ..., iₙ])
```

Where the minimum is taken over all indices from 0 to iₐ (inclusive) in dimension d.

## Migration Guide

### From Positional to Named Parameters

**Old (Positional) Syntax:**
```tcl
set result [torch::cummin $tensor 0]
set result [torch::cummin $tensor 1]
```

**New (Named Parameter) Syntax:**
```tcl
set result [torch::cummin -input $tensor -dim 0]
set result [torch::cummin -input $tensor -dim 1]
```

### Benefits of Named Parameter Syntax

1. **Self-documenting**: Parameter names make code more readable
2. **Flexible ordering**: Parameters can be specified in any order
3. **Future-proof**: Easier to extend with additional parameters
4. **Error prevention**: Reduces mistakes from parameter misplacement

### Backward Compatibility

- **All existing code continues to work** - no breaking changes
- Both syntaxes produce identical results
- Performance is equivalent between syntaxes
- Error messages are improved in both syntaxes

## Related Commands

- **torch::cummax**: Cumulative maximum along a dimension
- **torch::cumsum**: Cumulative sum along a dimension
- **torch::cumprod**: Cumulative product along a dimension
- **torch::tensor_min**: Minimum value(s) along a dimension
- **torch::tensor_max**: Maximum value(s) along a dimension

## See Also

- [torch::cummax](cummax.md) - Cumulative maximum
- [torch::cumsum](cumsum.md) - Cumulative sum
- [torch::cumprod](cumprod.md) - Cumulative product
- [PyTorch cummin documentation](https://pytorch.org/docs/stable/generated/torch.cummin.html)

---

**API Version**: LibTorch TCL Extension v2.0+  
**Last Updated**: 2024  
**Compatibility**: Backward compatible with all previous versions 