# torch::cummax

Compute the cumulative maximum of elements along a given dimension, returning the cumulative maximum values.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::cummax tensor dim
```

### Named Parameter Syntax  
```tcl
torch::cummax -input tensor -dim dim
```

### CamelCase Alias
```tcl
torch::cumMax tensor dim
torch::cumMax -input tensor -dim dim
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| input | string | - | Name of the input tensor |
| dim | int | 0 | Dimension along which to compute cumulative maximum |

## Returns

Returns the name of a new tensor containing the cumulative maximum values along the specified dimension.

**Note**: PyTorch's `cummax` operation returns a tuple of (values, indices), but this command returns only the values tensor for simplicity and consistency with typical usage.

## Error Handling

- **Invalid Tensor**: Returns error if the input tensor name doesn't exist
- **Invalid Dimension**: Returns error for dimension types that cannot be converted to integer
- **Dimension Out of Range**: PyTorch handles out-of-range dimensions according to its own rules
- **Invalid Parameters**: Returns error for unknown parameters or missing required values

## Examples

### Basic Usage
```tcl
# Create a 1D tensor
set t1 [torch::tensor_create [list [list 3 1 4 1 5 9 2 6 5 3]] -dtype float32]

# Compute cumulative maximum using positional syntax
set result [torch::cummax $t1 0]
puts [torch::tensor_data $result]
# Output: [[3.0 3.0 4.0 4.0 5.0 9.0 9.0 9.0 9.0 9.0]]
```

### 2D Tensor Operations
```tcl
# Create a 2D tensor
set t2 [torch::tensor_create [list [list 1 4 7] [list 2 1 8] [list 3 6 2]] -dtype float32]

# Cumulative maximum along rows (dim 0)
set result_rows [torch::cummax $t2 0]
puts [torch::tensor_data $result_rows]
# Output: [[1.0 4.0 7.0] [2.0 4.0 8.0] [3.0 6.0 8.0]]

# Cumulative maximum along columns (dim 1) 
set result_cols [torch::cummax $t2 1]
puts [torch::tensor_data $result_cols]
# Output: [[1.0 4.0 7.0] [2.0 2.0 8.0] [3.0 6.0 6.0]]
```

### Named Parameter Syntax
```tcl
# Using named parameters
set t3 [torch::tensor_create [list [list 5 2 8 1 9]] -dtype float32]
set result [torch::cummax -input $t3 -dim 0]
puts [torch::tensor_data $result]
# Output: [[5.0 5.0 8.0 8.0 9.0]]

# Parameter order independence
set result2 [torch::cummax -dim 0 -input $t3]
# Same result as above
```

### CamelCase Alias
```tcl
# Using camelCase alias
set t4 [torch::tensor_create [list [list 2 7 1 8]] -dtype float32]
set result [torch::cumMax $t4 0]
puts [torch::tensor_data $result]
# Output: [[2.0 7.0 7.0 8.0]]

# CamelCase with named parameters
set result2 [torch::cumMax -input $t4 -dim 0]
```

### Working with Different Data Types
```tcl
# Integer tensors
set int_tensor [torch::tensor_create [list [list 3 1 4 1 5]] -dtype int32]
set result [torch::cummax $int_tensor 0]

# Float64 tensors  
set float64_tensor [torch::tensor_create [list [list 3.5 1.2 4.7 1.9 5.1]] -dtype float64]
set result [torch::cummax $float64_tensor 0]
```

### Negative Values
```tcl
# Cumulative maximum with negative numbers
set neg_tensor [torch::tensor_create [list [list -3 -1 -4 -1 -5]] -dtype float32]
set result [torch::cummax $neg_tensor 0]
puts [torch::tensor_data $result]
# Output: [[-3.0 -1.0 -1.0 -1.0 -1.0]]
```

### Error Handling
```tcl
# Handle invalid tensor name
if {[catch {torch::cummax "nonexistent_tensor" 0} error]} {
    puts "Error: $error"
    # Output: Error: Invalid tensor name
}

# Handle missing parameters
if {[catch {torch::cummax -dim 0} error]} {
    puts "Error: $error"
    # Output: Error: Required parameter missing: -input
}
```

## Mathematical Details

### Cumulative Maximum Definition
For a 1D tensor `[x₀, x₁, x₂, ..., xₙ]`, the cumulative maximum is:
```
cummax[i] = max(x₀, x₁, ..., xᵢ)
```

### Multi-dimensional Behavior
- **dim=0**: Cumulative maximum along rows (down columns)
- **dim=1**: Cumulative maximum along columns (across rows)
- **dim=-1**: Cumulative maximum along last dimension

### Examples of Cumulative Maximum
```
Input:  [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
Output: [3, 3, 4, 4, 5, 9, 9, 9, 9, 9]

Input 2D:
[[1, 4, 7],
 [2, 1, 8], 
 [3, 6, 2]]

Cummax dim=0 (along rows):
[[1, 4, 7],
 [2, 4, 8],
 [3, 6, 8]]

Cummax dim=1 (along columns):
[[1, 4, 7],
 [2, 2, 8],
 [3, 6, 6]]
```

## Migration from Positional to Named Parameters

### Old Style (Positional)
```tcl
set result [torch::cummax $tensor 0]
```

### New Style (Named Parameters)
```tcl
set result [torch::cummax -input $tensor -dim 0]
```

### Using CamelCase
```tcl
set result [torch::cumMax -input $tensor -dim 0]
```

## Related Commands

- **torch::cummin**: Cumulative minimum along a dimension
- **torch::cumsum**: Cumulative sum along a dimension  
- **torch::cumprod**: Cumulative product along a dimension
- **torch::max**: Maximum value (with optional dimension)
- **torch::argmax**: Indices of maximum values

## Technical Notes

### Dimension Handling
- Dimensions are 0-indexed
- Negative dimensions are supported (e.g., -1 for last dimension)
- Invalid dimensions are handled by PyTorch's error handling

### Memory Considerations
- Creates a new tensor with same shape as input
- Memory usage is identical to input tensor
- No in-place operation available

### Performance Characteristics
- Linear time complexity O(n) where n is the number of elements
- Efficient implementation using PyTorch's optimized kernels
- CUDA acceleration available for GPU tensors

## Implementation Details

This command uses PyTorch's `tensor.cummax(dim)` function which returns a tuple of (values, indices). For simplicity and typical usage patterns, only the values tensor is returned.

The dual syntax implementation maintains full backward compatibility while providing modern named parameter access with improved readability and flexibility. 