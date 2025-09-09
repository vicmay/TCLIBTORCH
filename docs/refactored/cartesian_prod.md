# torch::cartesian_prod

Computes the Cartesian product of a sequence of tensors. This is equivalent to applying a nested loop over all input tensors.

## Syntax

### Positional (Backward Compatible)
```tcl
torch::cartesian_prod tensor1 tensor2 [tensor3 ...]
```

### Named Parameters  
```tcl
torch::cartesian_prod -tensors {tensor1 tensor2 ...}
torch::cartesian_prod -tensors tensor1  # Single tensor
```

### CamelCase Alias
```tcl
torch::cartesianProd tensor1 tensor2 [tensor3 ...]
torch::cartesianProd -tensors {tensor1 tensor2 ...}
```

## Parameters

### Positional Parameters
- `tensor1`, `tensor2`, ... - Variable number of 1D tensor handles to compute Cartesian product

### Named Parameters
- `-tensors` - List of 1D tensor handles or single tensor handle

## Return Value

Returns a single tensor handle containing the Cartesian product. The output tensor has shape `(N, k)` where:
- `N` is the total number of combinations (product of all input tensor sizes)
- `k` is the number of input tensors

Each row in the output tensor represents one combination from the Cartesian product.

## Mathematical Details

The Cartesian product of sets A₁ × A₂ × ... × Aₖ creates all possible k-tuples where each element comes from the corresponding set.

For tensors:
- Input: `k` tensors of sizes `(n₁,), (n₂,), ..., (nₖ,)`
- Output: Tensor of shape `(n₁ × n₂ × ... × nₖ, k)`

## Examples

### Basic Two Tensor Product
```tcl
# Create input tensors
set x [torch::tensor_create {1 2} float32]    # Size: [2]
set y [torch::tensor_create {3 4} float32]    # Size: [2]

# Compute Cartesian product - all syntaxes work
set result1 [torch::cartesian_prod $x $y]
set result2 [torch::cartesian_prod -tensors [list $x $y]]
set result3 [torch::cartesianProd $x $y]

# Result shape: [4, 2] containing:
# [[1, 3], [1, 4], [2, 3], [2, 4]]
```

### Three Tensor Product
```tcl
# Create three input tensors
set a [torch::tensor_create {1 2} float32]    # Size: [2]
set b [torch::tensor_create {3} float32]      # Size: [1]
set c [torch::tensor_create {4 5} float32]    # Size: [2]

# Compute Cartesian product
set result [torch::cartesian_prod $a $b $c]

# Result shape: [4, 3] containing:
# [[1, 3, 4], [1, 3, 5], [2, 3, 4], [2, 3, 5]]
```

### Different Vector Sizes
```tcl
# Tensors with different sizes
set colors [torch::tensor_create {1 2 3} float32]     # 3 colors
set sizes [torch::tensor_create {10 20} float32]      # 2 sizes

# All combinations: 3 × 2 = 6 combinations
set combinations [torch::cartesian_prod $colors $sizes]

# Result shape: [6, 2] containing all color-size pairs
```

### Named Parameter with List
```tcl
# Create tensor list
set tensor_list [list \
    [torch::tensor_create {1 2} float32] \
    [torch::tensor_create {3 4} float32] \
    [torch::tensor_create {5} float32]   \
]

# Compute using named parameter syntax
set result [torch::cartesian_prod -tensors $tensor_list]

# Result shape: [4, 3] - all combinations of (1,2) × (3,4) × (5)
```

### Single Tensor Case
```tcl
# Single tensor input
set items [torch::tensor_create {1 2 3} float32]
set result [torch::cartesian_prod $items]

# Result shape: [3, 1] containing: [[1], [2], [3]]
```

## Common Use Cases

1. **Grid Generation**: Create coordinate grids for meshgrid-like operations
2. **Parameter Combinations**: Generate all parameter combinations for hyperparameter search
3. **Combinatorial Problems**: Enumerate all possible combinations
4. **Database Joins**: Cross-product operations similar to SQL cross joins
5. **Neural Networks**: Generate all possible input combinations for certain architectures

## Performance Considerations

- **Memory Usage**: Output size grows exponentially with number of inputs
- **Large Inputs**: Be careful with large input tensors - memory usage = ∏(input_sizes) × k
- **Alternative Approaches**: Consider `torch::meshgrid` for coordinate grids

### Memory Estimation
For input tensors of sizes n₁, n₂, ..., nₖ:
- Output elements: n₁ × n₂ × ... × nₖ × k
- Memory scales multiplicatively with input sizes

## Error Conditions

- **No tensors provided**: Must provide at least one tensor
- **Invalid tensor handles**: All tensor names must exist in tensor storage
- **Non-1D tensors**: All input tensors should be 1-dimensional
- **Missing parameter values**: Named parameters must have values

## Mathematical Equivalence

This operation is equivalent to:
```python
# PyTorch equivalent
import torch
result = torch.cartesian_prod(tensor1, tensor2, ...)
```

## Examples with Expected Output

### Simple 2×2 Case
```tcl
set x [torch::tensor_create {0 1} float32]
set y [torch::tensor_create {0 1} float32]
set result [torch::cartesian_prod $x $y]

# Output tensor contains:
# [[0, 0], [0, 1], [1, 0], [1, 1]]
```

### Coordinate Grid
```tcl
# Create 2D coordinate grid
set x_coords [torch::tensor_create {0 1 2} float32]
set y_coords [torch::tensor_create {0 1} float32]
set grid_points [torch::cartesian_prod $x_coords $y_coords]

# Result: All (x,y) coordinate pairs
# [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
```

## See Also

- `torch::meshgrid` - Create coordinate grids (different output format)
- `torch::combinations` - Generate combinations without replacement
- `torch::tensor_expand` - Expand tensor to specific shape
- `torch::broadcast_tensors` - Broadcast tensors to common shape

## Implementation Status

- ✅ Dual syntax support (positional + named parameters)
- ✅ CamelCase alias (`torch::cartesianProd`)
- ✅ Comprehensive test coverage (17 test cases)
- ✅ Error handling and validation
- ✅ Complete documentation
- ✅ Memory-efficient implementation using PyTorch backend 