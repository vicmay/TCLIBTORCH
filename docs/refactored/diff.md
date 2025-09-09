# torch::diff

Computes the n-th discrete difference along a given dimension.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::diff tensor ?n? ?dim?
```

### Named Parameter Syntax  
```tcl
torch::diff -input tensor ?-n n? ?-dim dim?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tensor` / `-input` | string | Yes | - | Handle to input tensor |
| `n` / `-n` | integer | No | 1 | Number of times to apply the difference operation |
| `dim` / `-dim` | integer | No | -1 | Dimension along which to compute the difference |

## Return Value

Returns a string handle to the result tensor containing the computed differences.

## Description

The `torch::diff` command computes the n-th discrete difference along a specified dimension. It calculates the difference between consecutive elements:

```
diff[i] = input[i+1] - input[i]
```

For higher-order differences (n > 1), the operation is applied recursively.

### Key Features:
- **Flexible parameters**: Configurable number of differences and dimension
- **Shape reduction**: Output size is reduced by `n` along the specified dimension
- **Multi-dimensional support**: Works with tensors of any number of dimensions
- **Data type preservation**: Maintains the input tensor's data type
- **Dual syntax support**: Supports both positional and named parameter syntax

### Mathematical Operation

For a 1D tensor `[a, b, c, d]`:
- `n=1`: `[b-a, c-b, d-c]` (size reduced from 4 to 3)
- `n=2`: `[(c-b)-(b-a), (d-c)-(c-b)]` = `[c-2b+a, d-2c+b]` (size reduced from 4 to 2)

### Dimension Handling

- `dim=-1` (default): Uses the last dimension
- `dim=0`: Computes differences along the first dimension
- `dim=1`: Computes differences along the second dimension
- And so on...

## Examples

### Basic Usage

```tcl
# Create a simple sequence
set tensor [torch::tensor_create -data {1.0 4.0 7.0 10.0} -shape {4} -dtype float32]

# Positional syntax - first difference
set diff1 [torch::diff $tensor]
# Result: [3.0, 3.0, 3.0] (shape: [3])

# Named parameter syntax
set diff1 [torch::diff -input $tensor]
# Same result: [3.0, 3.0, 3.0]
```

### Higher-Order Differences

```tcl
# Second difference (difference of differences)
set diff2 [torch::diff $tensor 2]
# Or using named parameters
set diff2 [torch::diff -input $tensor -n 2]
# Result: [0.0, 0.0] (shape: [2])

# Third difference
set diff3 [torch::diff -input $tensor -n 3]
# Result: [0.0] (shape: [1])
```

### Multi-Dimensional Tensors

```tcl
# Create a 2D tensor
set matrix [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
# Matrix looks like:
# [[1.0, 2.0, 3.0],
#  [4.0, 5.0, 6.0]]

# Difference along dimension 0 (rows)
set diff_rows [torch::diff $matrix 1 0]
# Result shape: [1 3], content: [[3.0, 3.0, 3.0]]

# Difference along dimension 1 (columns)  
set diff_cols [torch::diff $matrix 1 1]
# Result shape: [2 2], content: [[1.0, 1.0], [1.0, 1.0]]

# Using named parameters
set diff_cols [torch::diff -input $matrix -n 1 -dim 1]
```

### Signal Processing Applications

```tcl
# Compute velocity from position data
proc compute_velocity {position_data dt} {
    set velocity_raw [torch::diff $position_data]
    # Scale by time step to get actual velocity
    set dt_tensor [torch::tensor_create -data [list $dt] -shape {1} -dtype float32]
    return [torch::div $velocity_raw $dt_tensor]
}

# Compute acceleration from velocity data
proc compute_acceleration {velocity_data dt} {
    set accel_raw [torch::diff $velocity_data]
    set dt_tensor [torch::tensor_create -data [list $dt] -shape {1} -dtype float32]
    return [torch::div $accel_raw $dt_tensor]
}

# Usage
set positions [torch::tensor_create -data {0.0 1.0 4.0 9.0 16.0} -shape {5} -dtype float32]
set dt 1.0
set velocities [compute_velocity $positions $dt]
set accelerations [compute_acceleration $velocities $dt]
```

### Image Processing - Edge Detection

```tcl
# Simple edge detection using differences
proc detect_edges {image_tensor} {
    # Compute differences along width (horizontal edges)
    set h_edges [torch::diff $image_tensor 1 -1]
    
    # Compute differences along height (vertical edges) 
    set v_edges [torch::diff $image_tensor 1 -2]
    
    return [list $h_edges $v_edges]
}

# 3D tensor representing a grayscale image [height, width]
set image [torch::tensor_create -data {1.0 1.0 5.0 1.0 1.0 5.0} -shape {2 3} -dtype float32]
set edges [detect_edges $image]
set horizontal_edges [lindex $edges 0]
set vertical_edges [lindex $edges 1]
```

### Time Series Analysis

```tcl
# Analyze trends in time series data
proc analyze_trends {time_series} {
    # First difference - velocity/rate of change
    set first_diff [torch::diff $time_series]
    
    # Second difference - acceleration/curvature
    set second_diff [torch::diff $time_series 2]
    
    return [list $first_diff $second_diff]
}

# Stock price analysis
set prices [torch::tensor_create -data {100.0 102.0 101.0 105.0 108.0 106.0} -shape {6} -dtype float32]
set analysis [analyze_trends $prices]
set price_changes [lindex $analysis 0]    # Daily changes
set momentum [lindex $analysis 1]         # Change in changes
```

### 3D Data Processing

```tcl
# Process 3D volumetric data
set volume [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 2 2} -dtype float32]

# Differences along each dimension
set diff_x [torch::diff $volume 1 0]  # Along first dimension
set diff_y [torch::diff $volume 1 1]  # Along second dimension  
set diff_z [torch::diff $volume 1 2]  # Along third dimension

# Check resulting shapes
puts "Original shape: [torch::tensor_shape $volume]"      # 2 2 2
puts "Diff X shape: [torch::tensor_shape $diff_x]"        # 1 2 2
puts "Diff Y shape: [torch::tensor_shape $diff_y]"        # 2 1 2
puts "Diff Z shape: [torch::tensor_shape $diff_z]"        # 2 2 1
```

## Technical Details

### Shape Transformation

For an input tensor with shape `[d0, d1, ..., dk, ..., dn]`:
- Applying `torch::diff` with `n=1` along dimension `k` results in shape `[d0, d1, ..., dk-1, ..., dn]`
- Applying `torch::diff` with `n=m` along dimension `k` results in shape `[d0, d1, ..., dk-m, ..., dn]`

### Data Type Behavior

- **Integer types**: Preserved, but may overflow for large differences
- **Floating-point types**: Preserved with full precision
- **Mixed operations**: Follow PyTorch's standard type promotion rules

### Memory Considerations

- **Input size**: N elements along difference dimension
- **Output size**: N-n elements along difference dimension  
- **Memory usage**: Proportional to output tensor size
- **In-place**: Not supported (always creates new tensor)

### Performance Characteristics

- **Time complexity**: O(N) where N is total number of elements
- **Space complexity**: O(M) where M is output tensor size
- **Parallelization**: Automatically parallelized for large tensors
- **GPU support**: Inherits from underlying PyTorch tensor operations

## Error Handling

The command provides comprehensive error checking:

```tcl
# Invalid tensor handle
catch {torch::diff invalid_tensor} error
# Error: "Invalid tensor name"

# Missing required parameters
catch {torch::diff} error
# Error: "Required parameter missing: -input"

# Invalid parameter names
catch {torch::diff -invalid $tensor} error  
# Error: "Unknown parameter: -invalid"

# Invalid parameter values
catch {torch::diff $tensor invalid_n} error
# Error: "Invalid n value. Expected integer."

catch {torch::diff $tensor 1 invalid_dim} error
# Error: "Invalid dim value. Expected integer."
```

## Edge Cases and Special Behaviors

### Dimension Size Constraints

```tcl
# If n >= dimension_size, may result in empty tensor or error
set small_tensor [torch::tensor_create -data {1.0 2.0} -shape {2} -dtype float32]
catch {torch::diff $small_tensor 3} result
# May fail because n=3 > size=2
```

### Single Element Tensors

```tcl
set single [torch::tensor_create -data {5.0} -shape {1} -dtype float32]
catch {torch::diff $single} result
# Typically fails because no difference can be computed
```

### Zero and Negative n Values

```tcl
# n=0 might return original tensor (implementation dependent)
catch {torch::diff $tensor 0} result

# Negative n values are typically invalid
catch {torch::diff $tensor -1} result
```

## Comparison with Related Operations

| Operation | Purpose | Output Size | Use Case |
|-----------|---------|-------------|----------|
| `torch::diff` | Discrete differences | N-n | Derivatives, changes |
| `torch::gradient` | Numerical gradient | N | Continuous derivatives |
| `torch::cumsum` | Cumulative sum | N | Running totals |
| `torch::roll` | Circular shift | N | Data shifting |

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set result [torch::diff $tensor]
set result [torch::diff $tensor 2]
set result [torch::diff $tensor 1 0]

# New named parameter syntax  
set result [torch::diff -input $tensor]
set result [torch::diff -input $tensor -n 2]
set result [torch::diff -input $tensor -n 1 -dim 0]

# Mixed parameter order (named syntax advantage)
set result [torch::diff -dim 0 -input $tensor -n 1]
```

### Benefits of Named Parameters

- **Self-documenting**: Parameter purpose is explicit
- **Flexible ordering**: Parameters can be specified in any order  
- **Optional clarity**: Optional parameters are clearly identified
- **Future-proof**: New parameters can be added without breaking existing code
- **Error reduction**: Less prone to parameter position mistakes

## See Also

- [torch::gradient](gradient.md) - Numerical gradient computation
- [torch::cumsum](cumsum.md) - Cumulative sum along dimension
- [torch::cumprod](cumprod.md) - Cumulative product along dimension
- [torch::roll](roll.md) - Roll tensor elements along dimension
- [torch::tensor_shape](tensor_shape.md) - Get tensor dimensions 