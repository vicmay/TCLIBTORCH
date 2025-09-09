# torch::vector_norm

Computes the vector norm of a tensor.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::vector_norm input ?ord? ?dim? ?keepdim?
```

### Named Parameter Syntax  
```tcl
torch::vector_norm -input tensor ?-ord double? ?-dim list? ?-keepdim bool?
```

### CamelCase Alias
```tcl
torch::vectorNorm input ?ord? ?dim? ?keepdim?
torch::vectorNorm -input tensor ?-ord double? ?-dim list? ?-keepdim bool?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | - | **Required.** Name of the input tensor |
| `ord` | double | 2.0 | Order of the norm. Can be positive number, 'inf', '-inf', or 'fro' |
| `dim` | list of ints | none | Dimensions to compute the norm over. If not specified, computes norm over all dimensions |
| `keepdim` | bool | false | Whether to keep the dimension that was reduced |

## Returns

Returns a new tensor handle containing the computed vector norm.

## Description

The `torch::vector_norm` command computes the vector norm along the specified dimensions of the input tensor. This is useful for:

- Computing L1, L2, or other norms of vectors/tensors
- Normalizing data for machine learning
- Measuring distances in vector spaces
- Statistical analysis and data preprocessing

The function uses PyTorch's `torch.linalg.vector_norm` function internally.

## Examples

### Basic Usage
```tcl
# Create a simple vector
set vec [torch::tensor_create {3.0 4.0}]

# Compute L2 norm (default)
set l2_norm [torch::vector_norm $vec]
puts "L2 norm: [torch::tensor_to_list $l2_norm]"  ;# Output: 5.0

# Compute L1 norm
set l1_norm [torch::vector_norm $vec 1.0]
puts "L1 norm: [torch::tensor_to_list $l1_norm]"  ;# Output: 7.0
```

### Named Parameter Syntax
```tcl
# Create a 2D tensor
set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]

# Compute norm along dimension 0
set norm [torch::vector_norm -input $tensor -dim {0}]
puts "Norm along dim 0: [torch::tensor_to_list $norm]"

# Compute norm with keepdim
set norm_keepdim [torch::vector_norm -input $tensor -dim {0} -keepdim 1]
puts "Shape with keepdim: [torch::tensor_shape $norm_keepdim]"
```

### CamelCase Alias
```tcl
# Using camelCase alias
set vec [torch::tensor_create {1.0 2.0 3.0}]
set norm [torch::vectorNorm $vec]
puts "Vector norm: [torch::tensor_to_list $norm]"

# With named parameters
set norm2 [torch::vectorNorm -input $vec -ord 1.0]
puts "L1 norm: [torch::tensor_to_list $norm2]"
```

### Different Norm Orders
```tcl
set vec [torch::tensor_create {3.0 -4.0 2.0}]

# L2 norm (default)
set l2 [torch::vector_norm $vec 2.0]
puts "L2: [torch::tensor_to_list $l2]"  ;# ≈ 5.385

# L1 norm  
set l1 [torch::vector_norm $vec 1.0]
puts "L1: [torch::tensor_to_list $l1]"  ;# 9.0

# Infinity norm
set linf [torch::vector_norm $vec inf]
puts "L∞: [torch::tensor_to_list $linf]"  ;# 4.0
```

### Multi-dimensional Examples
```tcl
# 3D tensor
set tensor3d [torch::tensor_create {{{1.0 2.0} {3.0 4.0}} {{5.0 6.0} {7.0 8.0}}}]

# Norm along last dimension
set norm_last [torch::vector_norm $tensor3d 2.0 {2}]
puts "Shape: [torch::tensor_shape $norm_last]"

# Norm along multiple dimensions
set norm_multi [torch::vector_norm $tensor3d 2.0 {0 1}]
puts "Multi-dim norm: [torch::tensor_to_list $norm_multi]"
```

## Error Handling

The command will raise an error in the following cases:

- **Invalid tensor**: If the input tensor name doesn't exist
- **Invalid ord**: If the ord parameter is not a valid number or string
- **Invalid keepdim**: If keepdim is not a boolean value
- **Missing required parameter**: If input tensor is not provided in named syntax
- **Unknown parameter**: If an unrecognized parameter name is used

### Error Examples
```tcl
# Error: Invalid tensor
catch {torch::vector_norm "nonexistent"} error
puts $error  ;# "Invalid input tensor"

# Error: Missing required parameter  
catch {torch::vector_norm -ord 2.0} error
puts $error  ;# "Required parameter missing: -input"

# Error: Invalid ord value
set vec [torch::tensor_create {1.0 2.0}]
catch {torch::vector_norm -input $vec -ord "invalid"} error
puts $error  ;# "Invalid ord value"
```

## Mathematical Details

The vector norm is computed as:

- **L1 norm** (ord=1): `||x||₁ = Σ|xᵢ|`
- **L2 norm** (ord=2): `||x||₂ = √(Σxᵢ²)`  
- **Lp norm** (ord=p): `||x||_p = (Σ|xᵢ|^p)^(1/p)`
- **L∞ norm** (ord=inf): `||x||_∞ = max|xᵢ|`

## Performance Notes

- Vector norms are computed efficiently using optimized BLAS operations
- For large tensors, consider specifying specific dimensions to reduce computation
- The `keepdim` parameter can be useful for broadcasting operations

## See Also

- [`torch::tensor_norm`](tensor_norm.md) - General tensor norm function
- [`torch::matrix_norm`](matrix_norm.md) - Matrix norm computation
- [`torch::tensor_normalize`](tensor_normalize.md) - Tensor normalization

## Migration Guide

### From Positional to Named Parameters

**Old positional syntax:**
```tcl
set result [torch::vector_norm $tensor 2.0 {0} 1]
```

**New named parameter syntax:**
```tcl
set result [torch::vector_norm -input $tensor -ord 2.0 -dim {0} -keepdim 1]
```

**CamelCase alias:**
```tcl
set result [torch::vectorNorm -input $tensor -ord 2.0 -dim {0} -keepdim 1]
```

Both syntaxes are fully supported and can be used interchangeably. The named parameter syntax is recommended for new code as it's more readable and self-documenting. 