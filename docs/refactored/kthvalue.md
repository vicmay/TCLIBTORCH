# torch::kthvalue

Returns the k-th smallest element of a tensor along a specified dimension. This operation is particularly useful for finding order statistics like median, quartiles, or any specific rank within the data.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::kthvalue tensor k dim ?keepdim?
```

### Named Parameter Syntax
```tcl
torch::kthvalue -input tensor -k k -dim dim -keepdim keepdim
```

### CamelCase Alias
```tcl
torch::kthValue tensor k dim ?keepdim?
torch::kthValue -input tensor -k k -dim dim -keepdim keepdim
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor` / `-input` | Tensor | Yes | Input tensor to find the k-th smallest element in |
| `k` / `-k` | Integer | Yes | The k-th smallest element to find (1-indexed) |
| `dim` / `-dim` | Integer | Yes | The dimension along which to find the k-th smallest element |
| `keepdim` / `-keepdim` | Boolean | No | Whether to keep the dimension (default: false) |

## Returns

Returns a tensor handle containing the k-th smallest elements along the specified dimension.

## Mathematical Foundation

The kthvalue operation finds the k-th smallest element along a specified dimension. For a tensor of shape `[..., n, ...]` along dimension `dim`, the operation sorts the values along that dimension and returns the element at position `k` (1-indexed).

### Mathematical Properties

1. **Ordering**: For dimension with values `[v₁, v₂, ..., vₙ]`, when sorted: `[s₁ ≤ s₂ ≤ ... ≤ sₙ]`
2. **K-th Selection**: Returns `sₖ` (the k-th smallest value)
3. **Dimension Reduction**: The result has dimension `dim` removed unless `keepdim=true`

### Examples of K-th Values

For array `[5, 1, 8, 3, 2]`:
- k=1: Returns `1` (smallest)
- k=2: Returns `2` (second smallest)
- k=3: Returns `3` (median)
- k=4: Returns `5` (second largest)
- k=5: Returns `8` (largest)

## Examples

### Basic Usage

```tcl
# Create a 1D tensor
set data [torch::tensor_create -data {5.0 3.0 8.0 1.0 9.0} -dtype float32 -shape {5}]

# Find 2nd smallest element (positional syntax)
set result1 [torch::kthvalue $data 2 0]
# Returns: tensor containing 3.0

# Find 3rd smallest element (named parameter syntax)
set result2 [torch::kthvalue -input $data -k 3 -dim 0]
# Returns: tensor containing 5.0

# Find 1st smallest element (CamelCase alias)
set result3 [torch::kthValue $data 1 0]
# Returns: tensor containing 1.0
```

### 2D Tensor Operations

```tcl
# Create a 2D tensor [[1, 5], [3, 2]]
set matrix [torch::tensor_create -data {1.0 5.0 3.0 2.0} -dtype float32 -shape {2 2}]

# Find 1st smallest along dimension 0 (column-wise)
set result1 [torch::kthvalue $matrix 1 0]
# Returns: tensor with shape [2] containing [1.0, 2.0]

# Find 2nd smallest along dimension 1 (row-wise)
set result2 [torch::kthvalue $matrix 2 1]
# Returns: tensor with shape [2] containing [5.0, 3.0]
```

### Using keepdim Parameter

```tcl
# Create a 2D tensor [[4, 7], [2, 9]]
set data [torch::tensor_create -data {4.0 7.0 2.0 9.0} -dtype float32 -shape {2 2}]

# Without keepdim (default)
set result1 [torch::kthvalue $data 1 0]
# Returns: tensor with shape [2]

# With keepdim=true
set result2 [torch::kthvalue $data 1 0 true]
# Returns: tensor with shape [1, 2]

# Named parameter syntax with keepdim
set result3 [torch::kthvalue -input $data -k 1 -dim 0 -keepdim true]
# Returns: tensor with shape [1, 2]
```

### Statistical Applications

```tcl
# Find median of a dataset
set data [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0} -dtype float32 -shape {9}]
set median [torch::kthvalue $data 5 0]  # k=5 for 9 elements
# Returns: tensor containing 5.0 (median)

# Find quartiles
set q1 [torch::kthvalue $data 3 0]  # 1st quartile (approximately)
set q3 [torch::kthvalue $data 7 0]  # 3rd quartile (approximately)
```

### Multi-dimensional Analysis

```tcl
# 3D tensor analysis
set data3d [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -shape {2 2 2}]

# Find 1st smallest along the last dimension
set result [torch::kthvalue $data3d 1 2]
# Returns: tensor with shape [2, 2]

# Find 2nd smallest along the first dimension with keepdim
set result_keepdim [torch::kthvalue -input $data3d -k 2 -dim 0 -keepdim true]
# Returns: tensor with shape [1, 2, 2]
```

## Use Cases

### 1. Median Calculation
```tcl
# Calculate median for odd number of elements
set data [torch::tensor_create -data {10.0 5.0 15.0 20.0 3.0} -dtype float32 -shape {5}]
set median [torch::kthvalue $data 3 0]  # k=3 for 5 elements
# Returns: median value
```

### 2. Outlier Detection
```tcl
# Find values at specific percentiles
set data [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0} -dtype float32 -shape {10}]
set p90 [torch::kthvalue $data 9 0]    # 90th percentile
set p10 [torch::kthvalue $data 1 0]    # 10th percentile
```

### 3. Data Analysis
```tcl
# Analyze multiple samples
set samples [torch::tensor_create -data {1.0 5.0 3.0 2.0 8.0 1.0 4.0 7.0 6.0 9.0 2.0 3.0} -dtype float32 -shape {3 4}]

# Find median of each sample (row-wise)
set medians [torch::kthvalue $samples 2 1]  # k=2 for 4 elements per row
# Returns: tensor with shape [3] containing median of each row
```

### 4. Robust Statistics
```tcl
# Calculate trimmed mean (excluding extreme values)
set data [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 100.0} -dtype float32 -shape {10}]

# Get 2nd smallest and 9th smallest to exclude extremes
set lower_bound [torch::kthvalue $data 2 0]
set upper_bound [torch::kthvalue $data 9 0]
# Use these bounds for trimmed statistics
```

## Data Type Support

```tcl
# Float32 tensors
set f32_data [torch::tensor_create -data {1.5 3.2 0.8 2.1} -dtype float32 -shape {4}]
set f32_result [torch::kthvalue $f32_data 2 0]

# Float64 tensors
set f64_data [torch::tensor_create -data {2.7 1.1 4.3 0.9} -dtype float64 -shape {4}]
set f64_result [torch::kthvalue $f64_data 3 0]

# Integer tensors
set int_data [torch::tensor_create -data {10 5 20 15} -dtype int64 -shape {4}]
set int_result [torch::kthvalue $int_data 2 0]
```

## Mathematical Properties

### Order Statistics
```tcl
# For sorted array [a₁ ≤ a₂ ≤ ... ≤ aₙ]
# kthvalue(tensor, k, dim) returns aₖ
```

### Relationship to Other Functions
```tcl
# Special cases
set data [torch::tensor_create -data {5.0 1.0 8.0 3.0} -dtype float32 -shape {4}]

# k=1 is equivalent to min
set min_val [torch::kthvalue $data 1 0]
set min_check [torch::tensor_min $data 0]  # Should be equivalent

# k=n is equivalent to max (where n is size of dimension)
set max_val [torch::kthvalue $data 4 0]
set max_check [torch::tensor_max $data 0]  # Should be equivalent
```

### Dimension Behavior
```tcl
# Input shape: [a, b, c, d]
# kthvalue along dim=1: output shape [a, c, d] (if keepdim=false)
# kthvalue along dim=1: output shape [a, 1, c, d] (if keepdim=true)
```

## Performance Considerations

- **Time Complexity**: O(n log n) where n is the size of the dimension being processed
- **Memory Usage**: Minimal additional memory required
- **GPU Acceleration**: Fully supported on CUDA tensors

## Error Handling

The function will raise errors for:
- Invalid tensor handles
- k values outside valid range [1, dim_size]
- Invalid dimension indices
- Missing required parameters

## Migration Guide

### From Positional to Named Parameters
```tcl
# Old positional syntax
torch::kthvalue $tensor 3 0 true

# New named parameter syntax
torch::kthvalue -input $tensor -k 3 -dim 0 -keepdim true

# CamelCase alias
torch::kthValue -input $tensor -k 3 -dim 0 -keepdim true
```

### Parameter Mapping
- `tensor` → `-input`
- `k` → `-k`
- `dim` → `-dim`
- `keepdim` → `-keepdim`

## See Also

- [torch::tensor_min](tensor_min.md) - Find minimum values
- [torch::tensor_max](tensor_max.md) - Find maximum values
- [torch::tensor_median](tensor_median.md) - Find median values
- [torch::tensor_sort](tensor_sort.md) - Sort tensor values
- [torch::tensor_topk](tensor_topk.md) - Find top-k values 