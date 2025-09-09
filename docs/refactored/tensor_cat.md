# torch::tensor_cat

Concatenates a sequence of tensors along a specified dimension.

## Description

The `torch::tensor_cat` command concatenates a sequence of tensors along a specified dimension. This operation is useful for combining multiple tensors into a single tensor, commonly used in data preprocessing and neural network operations.

**Note**: All input tensors must have the same shape except for the dimension being concatenated.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_cat tensor_list dim
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_cat -tensors tensor_list -dim dimension
```

### CamelCase Alias
```tcl
torch::tensorCat -tensors tensor_list -dim dimension
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensors` | list | Yes | List of tensor names to concatenate |
| `dim` | integer | Yes | Dimension along which to concatenate |

## Return Value

Returns a string containing the handle name of the resulting tensor.

## Examples

### Basic Usage

```tcl
# Create tensors to concatenate
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
set tensor_list [list $a $b]

# Using positional syntax
set result1 [torch::tensor_cat $tensor_list 0]

# Using named parameter syntax
set result2 [torch::tensor_cat -tensors $tensor_list -dim 0]

# Using camelCase alias
set result3 [torch::tensorCat -tensors $tensor_list -dim 0]
```

### 2D Tensor Concatenation

```tcl
# Create 2D tensors
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
set a2d [torch::tensor_reshape $a {2 2}]

set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
set b2d [torch::tensor_reshape $b {2 2}]

set tensor_list [list $a2d $b2d]

# Concatenate along dimension 0 (rows)
set result_dim0 [torch::tensor_cat -tensors $tensor_list -dim 0]

# Concatenate along dimension 1 (columns)
set result_dim1 [torch::tensor_cat -tensors $tensor_list -dim 1]
```

### Multiple Tensor Concatenation

```tcl
# Create multiple tensors
set a [torch::tensor_create -data {1.0 2.0} -dtype float32 -device cpu]
set b [torch::tensor_create -data {3.0 4.0} -dtype float32 -device cpu]
set c [torch::tensor_create -data {5.0 6.0} -dtype float32 -device cpu]

set tensor_list [list $a $b $c]
set result [torch::tensor_cat -tensors $tensor_list -dim 0]
```

### Different Data Types

```tcl
# Float32 tensors
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
set tensor_list [list $a $b]
set result [torch::tensor_cat -tensors $tensor_list -dim 0]

# Float64 tensors
set a64 [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float64 -device cpu]
set b64 [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float64 -device cpu]
set tensor_list64 [list $a64 $b64]
set result64 [torch::tensor_cat -tensors $tensor_list64 -dim 0]
```

## Error Handling

### Invalid Tensor Names
```tcl
# This will throw an error
set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
set tensor_list [list invalid_tensor $b]
catch {torch::tensor_cat $tensor_list 0} result
puts $result
# Output: Invalid tensor name: invalid_tensor
```

### Missing Parameters
```tcl
# This will throw an error
catch {torch::tensor_cat} result
puts $result
# Output: Required parameters missing: at least 2 tensors and dimension
```

### Unknown Parameters
```tcl
# This will throw an error
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
set tensor_list [list $a $b]
catch {torch::tensor_cat -tensors $tensor_list -dim 0 -unknown_param value} result
puts $result
# Output: Unknown parameter: -unknown_param
```

### Single Tensor (Edge Case)
```tcl
# This will throw an error - need at least 2 tensors
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
set tensor_list [list $a]
catch {torch::tensor_cat $tensor_list 0} result
puts $result
# Output: Required parameters missing: at least 2 tensors and dimension
```

## Migration Guide

### From Old Syntax to New Syntax

**Before (Positional Only):**
```tcl
set tensor_list [list $a $b]
set result [torch::tensor_cat $tensor_list 0]
```

**After (Named Parameters):**
```tcl
set tensor_list [list $a $b]
set result [torch::tensor_cat -tensors $tensor_list -dim 0]
```

**After (CamelCase):**
```tcl
set tensor_list [list $a $b]
set result [torch::tensorCat -tensors $tensor_list -dim 0]
```

### Benefits of New Syntax

1. **Clarity**: Parameter names make the code more readable
2. **Maintainability**: Easier to understand and modify
3. **Consistency**: Follows modern API design patterns
4. **Backward Compatibility**: Old syntax still works

## Technical Notes

- **Tensor Requirements**: All input tensors must have compatible shapes
- **Dimension Compatibility**: Tensors must have the same shape except for the concatenation dimension
- **Memory Usage**: The operation creates a new tensor for the result
- **Performance**: Concatenation along the last dimension is typically fastest

## Related Commands

- `torch::tensor_stack` - Stack tensors along a new dimension
- `torch::tensor_reshape` - Reshape tensors for concatenation
- `torch::tensor_permute` - Permute dimensions before concatenation 