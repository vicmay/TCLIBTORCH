# torch::combinations

## Overview
Compute r-length permutations of elements from the input tensor. Generates all combinations of the specified length r from the given input tensor, with optional replacement.

## Syntax

### Original Positional Syntax
```tcl
torch::combinations input ?r? ?with_replacement?
```

### Named Parameter Syntax
```tcl
torch::combinations -input|-tensor tensor -r r -with_replacement|-replacement with_replacement
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input`/`tensor` | string | Input tensor handle | Required |
| `r` | integer | Length of each combination | 2 |
| `with_replacement`/`replacement` | boolean | Allow replacement in combinations | false (0) |

## Returns
Returns a tensor handle containing all r-length combinations. The output tensor has shape `[num_combinations, r]` where `num_combinations` depends on the input size and whether replacement is allowed.

## Description

The `torch::combinations` function generates all possible combinations of length `r` from the input tensor:

- **Without replacement**: Standard combinations C(n,r) = n!/(r!(n-r)!)
- **With replacement**: Combinations with repetition = C(n+r-1,r)

The input tensor is treated as a 1-D sequence regardless of its original shape, and combinations are formed using the tensor elements directly.

## Examples

### Basic Usage Without Replacement

```tcl
# Create input tensor
set input [torch::tensor_create {0 1 2 3} int32]

# Generate all 2-combinations (default)
set result [torch::combinations $input]
# Result: [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
# Shape: {6 2}

# Generate all 3-combinations
set result [torch::combinations $input 3]
# Result: [[0,1,2], [0,1,3], [0,2,3], [1,2,3]]  
# Shape: {4 3}
```

### Using Named Parameters

```tcl
# Same as above using named syntax
set input [torch::tensor_create {0 1 2 3} int32]

# Basic named syntax
set result [torch::combinations -input $input -r 2]

# Using alternative parameter names
set result [torch::combinations -tensor $input -r 3]

# Parameter order flexibility
set result [torch::combinations -r 2 -input $input -with_replacement 0]
```

### Combinations With Replacement

```tcl
# Create input tensor
set input [torch::tensor_create {0 1 2} int32]

# Without replacement: C(3,2) = 3 combinations
set without_replacement [torch::combinations $input 2 0]
# Result: [[0,1], [0,2], [1,2]]
# Shape: {3 2}

# With replacement: C(3+2-1,2) = C(4,2) = 6 combinations  
set with_replacement [torch::combinations $input 2 1]
# Result: [[0,0], [0,1], [0,2], [1,1], [1,2], [2,2]]
# Shape: {6 2}

# Named syntax with replacement
set result [torch::combinations -input $input -r 2 -with_replacement 1]
set result [torch::combinations -input $input -r 2 -replacement 1]  # Alternative
```

### Different Data Types

```tcl
# Float tensor
set float_input [torch::tensor_create {1.5 2.5 3.5} float32]
set result [torch::combinations -input $float_input -r 2]

# String-like indices (using integer representation)
set indices [torch::tensor_create {10 20 30 40} int64]
set result [torch::combinations $indices 3]
```

### Single Element Combinations

```tcl
# r=1 returns all individual elements
set input [torch::tensor_create {5 10 15 20} int32]
set singles [torch::combinations $input 1]
# Result: [[5], [10], [15], [20]]
# Shape: {4 1}

# r=0 returns empty combinations
set empty [torch::combinations $input 0]
# Result: empty tensor with shape {1 0}
```

## Mathematical Properties

### Combination Counts

For an input tensor with `n` elements:

| Type | Formula | Example (n=4, r=2) |
|------|---------|-------------------|
| Without replacement | C(n,r) = n!/(r!(n-r)!) | C(4,2) = 6 |
| With replacement | C(n+r-1,r) = (n+r-1)!/(r!(n-1)!) | C(5,2) = 10 |

### Edge Cases

```tcl
# r=0: Single empty combination
set input [torch::tensor_create {1 2 3} int32]
set result [torch::combinations $input 0]  # Shape: {1 0}

# r equals input size: Single combination of all elements
set result [torch::combinations $input 3]  # Shape: {1 3}

# r > input size (without replacement): Empty result
set result [torch::combinations $input 5]  # May error or return empty

# Single element input
set single [torch::tensor_create {42} int32]
set result [torch::combinations $single 1]  # Shape: {1 1}
```

## Applications

### Data Analysis and Sampling

```tcl
# Generate all possible pairs from a dataset
set features [torch::tensor_create {0 1 2 3 4} int32]
set feature_pairs [torch::combinations $features 2]

# Sample different subsets for cross-validation
set data_indices [torch::arange 0 100 1 int32]
set test_combinations [torch::combinations $data_indices 10]
```

### Combinatorial Optimization

```tcl
# Generate all possible team combinations
set players [torch::tensor_create {1 2 3 4 5 6} int32]
set teams_of_3 [torch::combinations $players 3]

# Investment portfolio combinations
set assets [torch::tensor_create {0 1 2 3 4} int32]  # Asset IDs
set portfolios [torch::combinations $assets 3]  # Select 3 assets
```

### Experiment Design

```tcl
# All possible factor combinations for DOE
set factor_levels [torch::tensor_create {0 1 2} int32]
set experiments [torch::combinations -input $factor_levels -r 2 -with_replacement 1]

# A/B testing combinations
set variants [torch::tensor_create {0 1} int32]
set test_pairs [torch::combinations $variants 2 1]  # With replacement
```

## Performance Considerations

- **Memory usage**: Output size grows rapidly with input size and r
- **Computation time**: Without replacement is generally faster than with replacement
- **Large inputs**: Consider chunking for very large combination sets

```tcl
# Performance example
set large_input [torch::arange 0 20 1 int32]

# This generates C(20,3) = 1140 combinations
set start [clock clicks -milliseconds]
set result [torch::combinations $large_input 3]
set end [clock clicks -milliseconds]
puts "Generated [lindex [torch::tensor_shape $result] 0] combinations in [expr {$end - $start}]ms"
```

## Error Handling

### Common Errors

```tcl
# Missing input parameter
catch {torch::combinations -r 2} error
# Error: "Required parameter -input missing"

# Invalid tensor handle
catch {torch::combinations invalid_tensor} error  
# Error: "Invalid input tensor"

# Unknown parameter
catch {torch::combinations -input $tensor -invalid_param value} error
# Error: "Unknown parameter: -invalid_param"

# Invalid r parameter
catch {torch::combinations -input $tensor -r invalid} error
# Error: "Invalid -r parameter"

# Negative r (may error depending on PyTorch version)
catch {torch::combinations $tensor -1} error
# Error: tensor operation error
```

## Comparison with Related Functions

| Function | Purpose | Output Shape | Use Case |
|----------|---------|--------------|----------|
| `torch::combinations` | r-combinations | `{C(n,r), r}` | Fixed-size subsets |
| `torch::permutations` | r-permutations | `{P(n,r), r}` | Ordered arrangements |
| `torch::cartesian_prod` | Cartesian product | Variable | All paired combinations |
| `torch::meshgrid` | Grid combinations | Variable | Coordinate grids |

## Best Practices

1. **Choose appropriate r**: Consider memory limitations for large combinations
   ```tcl
   # Good: manageable combination count
   set result [torch::combinations $input 3]
   
   # Caution: very large result for big inputs
   set huge_result [torch::combinations $big_input 10]
   ```

2. **Use named syntax for clarity**:
   ```tcl
   # Clear and self-documenting
   set result [torch::combinations -input $data -r 2 -with_replacement 0]
   ```

3. **Validate input sizes** before generating combinations:
   ```tcl
   set input_size [lindex [torch::tensor_shape $input] 0]
   if {$input_size > 20 && $r > 5} {
       puts "Warning: Large combination count expected"
   }
   ```

4. **Consider data types** for the intended use:
   ```tcl
   # Use appropriate integer type for indices
   set indices [torch::tensor_create $index_list int64]
   set combinations [torch::combinations $indices $r]
   ```

## See Also
- [`torch::permutations`](permutations.md) - Generate permutations instead of combinations
- [`torch::cartesian_prod`](cartesian_prod.md) - Cartesian product of tensors
- [`torch::stack`](stack.md) - Stack tensors along new dimension
- [`torch::cat`](cat.md) - Concatenate tensors along existing dimension 