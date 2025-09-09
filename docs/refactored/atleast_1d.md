# torch::atleast_1d

Ensures that the input tensor has at least 1 dimension. If the input is a scalar (0D tensor), it is converted to a 1D tensor with shape `{1}`. Higher-dimensional tensors are returned unchanged.

## Syntax

### Current Syntax
```tcl
torch::atleast_1d tensor
```

### Named Parameter Syntax  
```tcl
torch::atleast_1d -input tensor
torch::atleast_1d -tensor tensor
```

### CamelCase Alias
```tcl
torch::atleast1d tensor
torch::atleast1d -input tensor
```

All syntaxes are fully supported and equivalent.

## Parameters

### Named Parameters
- `-input` (required): Input tensor name
- `-tensor` (required): Alias for `-input` parameter

### Positional Parameters
1. `tensor` (required): Input tensor name

## Description

The `torch::atleast_1d` function ensures that the input tensor has at least 1 dimension. This is useful for standardizing tensor shapes in operations that require minimum dimensionality. The function follows these transformation rules:

- **Scalar (0D)**: Converted to 1D tensor with shape `{1}`
- **1D or higher**: Returned unchanged

This function is commonly used in preprocessing pipelines where consistent dimensionality is required.

## Transformation Rules

| Input Shape | Output Shape | Description |
|-------------|--------------|-------------|
| `{}` (scalar) | `{1}` | Scalar converted to 1D |
| `{n}` | `{n}` | 1D tensor unchanged |
| `{m, n}` | `{m, n}` | 2D tensor unchanged |
| `{l, m, n}` | `{l, m, n}` | 3D tensor unchanged |
| Higher dimensions | Unchanged | Preserved as-is |

## Examples

### Basic Usage

#### Positional Syntax
```tcl
# Scalar tensor transformation
set scalar [torch::tensor_create {42.0}]
set result [torch::atleast_1d $scalar]
set shape [torch::tensor_shape $result]
# shape = {1}

# 1D tensor (unchanged)
set vector [torch::zeros {5}]
set result [torch::atleast_1d $vector]
set shape [torch::tensor_shape $result]
# shape = {5}

# 2D tensor (unchanged)
set matrix [torch::ones {3 4}]
set result [torch::atleast_1d $matrix]
set shape [torch::tensor_shape $result]
# shape = {3 4}
```

#### Named Parameter Syntax
```tcl
# Same operations using named parameters
set scalar [torch::tensor_create {42.0}]
set result [torch::atleast_1d -input $scalar]

# Alternative parameter name
set result [torch::atleast_1d -tensor $scalar]
```

#### CamelCase Alias
```tcl
# Using camelCase alias
set scalar [torch::tensor_create {3.14}]
set result [torch::atleast1d $scalar]
set result [torch::atleast1d -input $scalar]
```

### Shape Standardization Pipeline

```tcl
# Standardize a list of tensors to at least 1D
set scalar [torch::tensor_create {1.0}]
set vector [torch::zeros {3}]
set matrix [torch::ones {2 3}]

set standardized_scalar [torch::atleast_1d -input $scalar]    # {1}
set standardized_vector [torch::atleast_1d -input $vector]    # {3}
set standardized_matrix [torch::atleast_1d -input $matrix]    # {2 3}
```

### Working with Data Processing

```tcl
# Process mixed-dimension data
proc process_tensor {tensor_name} {
    # Ensure at least 1D for consistent processing
    set normalized [torch::atleast_1d -input $tensor_name]
    
    # Now we can safely apply 1D+ operations
    return $normalized
}

set data1 [torch::tensor_create {5.0}]     # scalar
set data2 [torch::zeros {10}]              # 1D
set data3 [torch::ones {3 3}]              # 2D

set proc1 [process_tensor $data1]          # shape {1}
set proc2 [process_tensor $data2]          # shape {10}
set proc3 [process_tensor $data3]          # shape {3 3}
```

### Batch Processing Example

```tcl
# Ensure all inputs have consistent minimum dimensionality
set inputs [list \
    [torch::tensor_create {1.0}] \
    [torch::zeros {5}] \
    [torch::ones {2 3}] \
    [torch::zeros {1 1 1}]]

set standardized_inputs {}
foreach input $inputs {
    set standard [torch::atleast_1d -input $input]
    lappend standardized_inputs $standard
}
# All tensors now have at least 1 dimension
```

## Input Requirements

- **Data Type**: Any tensor data type (float32, float64, int32, etc.)
- **Shape**: Any tensor shape (including scalars)
- **Device**: CPU and CUDA tensors supported
- **Memory**: Efficient operation with minimal memory overhead

## Output

Returns a tensor that:
- Has the same data type as input
- Has at least 1 dimension
- Preserves all data values
- Uses minimal memory (view when possible)

## Error Handling

The function will raise an error if:
- Input tensor name is invalid or doesn't exist
- Required parameters are missing
- Unknown parameters are provided
- Too many positional arguments are provided

## Performance Considerations

- **Memory Efficient**: Creates views when possible (no data copying for already 1D+ tensors)
- **Fast Operation**: Minimal computational overhead
- **GPU Support**: Works efficiently on CUDA tensors
- **Batch Processing**: Suitable for processing lists of tensors
- **No Data Movement**: Preserves tensor device and memory layout

## Common Use Cases

1. **Data Preprocessing**: Standardizing input shapes for machine learning models
2. **API Consistency**: Ensuring functions receive minimum dimensionality
3. **Broadcasting Setup**: Preparing tensors for broadcasting operations
4. **Pipeline Normalization**: Standardizing tensor shapes in processing pipelines
5. **Legacy Code Support**: Adapting old code to handle scalar inputs
6. **Tensor Validation**: Ensuring tensors meet minimum dimension requirements

## Related Functions

- `torch::atleast_2d` - Ensures tensor has at least 2 dimensions
- `torch::atleast_3d` - Ensures tensor has at least 3 dimensions
- `torch::tensor_reshape` - General tensor reshaping
- `torch::tensor_squeeze` - Removes dimensions of size 1
- `torch::tensor_unsqueeze` - Adds dimensions of size 1
- `torch::tensor_expand` - Expands tensor dimensions

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax (still supported)
set result [torch::atleast_1d $input_tensor]

# New named parameter syntax
set result [torch::atleast_1d -input $input_tensor]

# Alternative parameter name
set result [torch::atleast_1d -tensor $input_tensor]

# CamelCase alias
set result [torch::atleast1d $input_tensor]
set result [torch::atleast1d -input $input_tensor]

# All produce identical results
```

### Benefits of Named Parameters

1. **Clarity**: Parameter purpose is explicit
2. **Consistency**: Matches modern TCL conventions
3. **Flexibility**: Parameters can be provided in any order
4. **Future-Proof**: Easy to add optional parameters
5. **Multiple Aliases**: Both `-input` and `-tensor` are supported
6. **CamelCase Support**: Modern naming convention available

## Technical Notes

- Implements PyTorch's `torch.atleast_1d()` function
- Preserves tensor properties (device, requires_grad, dtype, etc.)
- Creates views when possible to minimize memory usage
- Thread-safe operation
- Supports automatic differentiation when `requires_grad=true`
- Maintains tensor memory layout when possible

## Mathematical Background

The `atleast_1d` function performs a shape transformation:
- **Input shape**: `S = [d₁, d₂, ..., dₙ]` where n ≥ 0
- **Output shape**: `S'` where:
  - If n = 0 (scalar): `S' = [1]`
  - If n ≥ 1: `S' = S` (unchanged)

This ensures the mathematical property that `dim(output) ≥ 1`.

## Comparison with Related Functions

| Function | Minimum Dimensions | Scalar Transform | Example Transform |
|----------|-------------------|------------------|-------------------|
| `atleast_1d` | 1 | `{} → {1}` | Scalar to vector |
| `atleast_2d` | 2 | `{} → {1,1}` | Scalar to matrix |
| `atleast_3d` | 3 | `{} → {1,1,1}` | Scalar to 3D tensor |
| `unsqueeze` | +1 | `{n} → {1,n}` | Adds specific dimension |

## Edge Cases

```tcl
# Empty tensor (valid 1D tensor)
set empty [torch::zeros {0}]
set result [torch::atleast_1d $empty]
# Output shape: {0}

# Large scalar
set big_scalar [torch::tensor_create {1e6}]
set result [torch::atleast_1d $big_scalar]
# Output shape: {1}

# High-dimensional tensor
set high_dim [torch::ones {1 2 3 4 5}]
set result [torch::atleast_1d $high_dim]
# Output shape: {1 2 3 4 5} (unchanged)
```

## Best Practices

1. **Use for Input Validation**: Ensure functions receive minimum dimensionality
2. **Pipeline Standardization**: Apply early in processing pipelines
3. **Batch Processing**: Process lists of mixed-dimension tensors
4. **API Design**: Use in functions that need consistent input shapes
5. **Performance**: Prefer this over manual reshaping for dimension guarantees

## Version History

- Added dual syntax support in refactoring phase
- Original positional syntax maintained for backward compatibility
- Named parameter syntax added for modern TCL conventions
- Added camelCase alias (`atleast1d`) for modern naming
- Enhanced documentation with transformation examples and use cases 