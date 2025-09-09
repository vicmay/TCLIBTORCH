# torch::hstack

Stack tensors horizontally (concatenate along dimension 1) to create a larger tensor.

## Syntax

### Current Syntax
```tcl
torch::hstack tensor1 tensor2 [tensor3 ...]           ;# Multiple tensor arguments
torch::hstack tensor_list                             ;# Single list of tensors
torch::hStack tensor1 tensor2 [tensor3 ...]           ;# camelCase alias
torch::hStack tensor_list                             ;# camelCase alias
```

### Named Parameters Syntax  
```tcl
torch::hstack -tensors tensor_list
torch::hstack -inputs tensor_list                     ;# alternative parameter name
torch::hStack -tensors tensor_list                    ;# camelCase alias
```

## Parameters

### Required Parameters
- **tensors** (list): List of tensor handles to stack horizontally

### Parameter Aliases
- **-tensors** or **-inputs**: Input tensors parameter

### Positional Arguments
The command also accepts tensors as individual positional arguments:
- **tensor1, tensor2, ...**: Individual tensor handles to stack

## Description

The `torch::hstack` command stacks tensors horizontally (along dimension 1) to create a larger tensor. It is equivalent to concatenation along the second dimension for 2D+ tensors, or along the first (and only) dimension for 1D tensors.

Key behaviors:
- **1D tensors**: Concatenated along dimension 0 (creating a longer 1D tensor)
- **2D+ tensors**: Concatenated along dimension 1 (width/column dimension)
- All input tensors must have compatible shapes except along the stacking dimension
- The result tensor has the same number of dimensions as the input tensors

## Return Value

Returns a tensor handle representing the horizontally stacked result.

## Examples

### Basic Usage

#### Two Tensor Stacking
```tcl
# Create two 2D tensors
set tensor1 [torch::ones -shape {2 3}]  ;# Shape: [2, 3]
set tensor2 [torch::ones -shape {2 4}]  ;# Shape: [2, 4]

# Stack horizontally (positional syntax)
set result [torch::hstack $tensor1 $tensor2]
set shape [torch::tensor_shape $result]  ;# Returns: {2 7}

# Stack horizontally (named parameters)
set result [torch::hstack -tensors [list $tensor1 $tensor2]]
set shape [torch::tensor_shape $result]  ;# Returns: {2 7}

# Using camelCase alias
set result [torch::hStack -tensors [list $tensor1 $tensor2]]
set shape [torch::tensor_shape $result]  ;# Returns: {2 7}
```

#### Multiple Tensor Stacking
```tcl
# Create three tensors with compatible shapes
set tensor1 [torch::ones -shape {3 2}]  ;# Shape: [3, 2]
set tensor2 [torch::ones -shape {3 3}]  ;# Shape: [3, 3]
set tensor3 [torch::ones -shape {3 1}]  ;# Shape: [3, 1]

# Stack all three (positional syntax)
set result [torch::hstack $tensor1 $tensor2 $tensor3]
set shape [torch::tensor_shape $result]  ;# Returns: {3 6}

# Stack all three (named parameters)
set tensor_list [list $tensor1 $tensor2 $tensor3]
set result [torch::hstack -tensors $tensor_list]
set shape [torch::tensor_shape $result]  ;# Returns: {3 6}
```

### Advanced Usage

#### 1D Tensor Stacking
```tcl
# Create 1D tensors
set tensor1 [torch::ones -shape {3}]  ;# Shape: [3]
set tensor2 [torch::ones -shape {4}]  ;# Shape: [4]

# Stack horizontally (concatenated along dim 0 for 1D)
set result [torch::hstack -tensors [list $tensor1 $tensor2]]
set shape [torch::tensor_shape $result]  ;# Returns: {7}
```

#### 3D Tensor Stacking
```tcl
# Create 3D tensors
set tensor1 [torch::ones -shape {2 3 4}]  ;# Shape: [2, 3, 4]
set tensor2 [torch::ones -shape {2 5 4}]  ;# Shape: [2, 5, 4]

# Stack along dimension 1 (width)
set result [torch::hstack -tensors [list $tensor1 $tensor2]]
set shape [torch::tensor_shape $result]  ;# Returns: {2 8 4}
```

#### Different Data Types
```tcl
# Float32 tensors
set tensor1 [torch::ones -shape {2 3} -dtype float32]
set tensor2 [torch::ones -shape {2 4} -dtype float32]
set result [torch::hstack -tensors [list $tensor1 $tensor2]]

# Int64 tensors
set tensor1 [torch::ones -shape {2 3} -dtype int64]
set tensor2 [torch::ones -shape {2 4} -dtype int64]
set result [torch::hstack -tensors [list $tensor1 $tensor2]]
```

#### Alternative Parameter Names
```tcl
set tensor1 [torch::ones -shape {2 3}]
set tensor2 [torch::ones -shape {2 4}]

# Using -inputs instead of -tensors
set result [torch::hstack -inputs [list $tensor1 $tensor2]]
```

#### Data Processing Pipeline
```tcl
# Create feature matrices for machine learning
set features1 [torch::randn -shape {100 10}]  ;# 100 samples, 10 features
set features2 [torch::randn -shape {100 5}]   ;# 100 samples, 5 features
set features3 [torch::randn -shape {100 8}]   ;# 100 samples, 8 features

# Combine all features horizontally
set combined_features [torch::hstack $features1 $features2 $features3]
set shape [torch::tensor_shape $combined_features]  ;# Returns: {100 23}

puts "Combined feature matrix shape: $shape"
```

## Error Handling

The command will throw an error in the following cases:

### Missing Required Parameters
```tcl
# Error: No tensors provided
torch::hstack

# Error: Missing value for parameter
torch::hstack -tensors
```

### Invalid Parameter Names
```tcl
# Error: Invalid parameter name
set tensor1 [torch::ones -shape {2 3}]
torch::hstack -invalid [list $tensor1]
```

### Incompatible Tensor Shapes
```tcl
# Error: Different first dimensions
set tensor1 [torch::ones -shape {2 3}]  ;# Shape: [2, 3]
set tensor2 [torch::ones -shape {3 4}]  ;# Shape: [3, 4]
torch::hstack $tensor1 $tensor2  ;# Error: 2 != 3
```

### Empty Tensor List
```tcl
# Error: Empty tensor list
torch::hstack -tensors {}
```

## Implementation Details

### Dual Syntax Support
The command supports both the original positional syntax (for backward compatibility) and the new named parameter syntax:

```tcl
# Original syntax (still supported)
torch::hstack $tensor1 $tensor2 $tensor3

# List-based syntax (still supported)
torch::hstack [list $tensor1 $tensor2 $tensor3]

# New named parameter syntax
torch::hstack -tensors [list $tensor1 $tensor2 $tensor3]

# All produce identical results
```

### camelCase Alias
The command provides a camelCase alias for modern coding conventions:

```tcl
# snake_case (original)
torch::hstack -tensors [list $tensor1 $tensor2]

# camelCase (alias)
torch::hStack -tensors [list $tensor1 $tensor2]
```

### Parameter Validation
- Validates that tensors parameter is provided and not empty
- Checks for unknown parameter names
- Ensures proper argument count for positional syntax
- Verifies tensor shape compatibility

### Stacking Behavior by Dimension

| Input Tensors | Stacking Dimension | Result |
|---------------|-------------------|--------|
| 1D: `[A]`, `[B]` | 0 | `[A+B]` |
| 2D: `[H, W1]`, `[H, W2]` | 1 | `[H, W1+W2]` |
| 3D: `[D, H, W1]`, `[D, H, W2]` | 1 | `[D, H, W1+W2]` |
| ND: `[..., W1]`, `[..., W2]` | -2 | `[..., W1+W2]` |

## Performance Considerations

- The stacking operation is memory-efficient when possible
- All output tensors preserve the data type of the input tensors
- Memory usage scales with the total size of all input tensors
- GPU tensors are stacked on the same device

## Related Commands

- **torch::vstack**: Stack tensors vertically (along dimension 0)
- **torch::dstack**: Stack tensors along depth dimension (dimension 2)
- **torch::cat**: Generic concatenation along any dimension
- **torch::column_stack**: Stack 1D tensors as columns
- **torch::row_stack**: Alternative name for vertical stacking

## Mathematical Background

Horizontal stacking concatenates tensors along dimension 1:

```
Input tensors:
A: [batch_size, width_A, height, ...]
B: [batch_size, width_B, height, ...]

Result:
C: [batch_size, width_A + width_B, height, ...]
```

For 1D tensors, stacking occurs along dimension 0:
```
Input tensors:
A: [length_A]
B: [length_B]

Result:
C: [length_A + length_B]
```

## Use Cases

### Machine Learning
```tcl
# Combine different feature sets
set demographic_features [torch::randn -shape {1000 5}]
set behavioral_features [torch::randn -shape {1000 10}]
set interaction_features [torch::randn -shape {1000 3}]

set all_features [torch::hstack $demographic_features $behavioral_features $interaction_features]
# Shape: [1000, 18]
```

### Image Processing
```tcl
# Combine image channels or regions
set left_half [torch::randn -shape {3 256 128}]   ;# RGB, height, width/2
set right_half [torch::randn -shape {3 256 128}]  ;# RGB, height, width/2

set full_image [torch::hstack -tensors [list $left_half $right_half]]
# Shape: [3, 256, 256]
```

### Time Series Analysis
```tcl
# Combine multiple time series signals
set signal1 [torch::randn -shape {1000 1}]  ;# 1000 timesteps, 1 feature
set signal2 [torch::randn -shape {1000 1}]  ;# 1000 timesteps, 1 feature
set signal3 [torch::randn -shape {1000 1}]  ;# 1000 timesteps, 1 feature

set multi_signal [torch::hstack $signal1 $signal2 $signal3]
# Shape: [1000, 3]
```

## Version History

| Version | Change |
|---------|--------|
| 1.0.0   | Initial implementation with positional syntax |
| 2.0.0   | Added dual syntax support with named parameters |
| 2.0.0   | Added camelCase alias (`torch::hStack`) |
| 2.0.0   | Added parameter validation and error handling |
| 2.0.0   | Added alternative parameter names (`-inputs`) |
| 2.0.0   | Enhanced error messages for better debugging |

## See Also

- [torch::vstack](vstack.md) - Vertical tensor stacking
- [torch::dstack](dstack.md) - Depth tensor stacking  
- [torch::cat](cat.md) - Generic tensor concatenation
- [torch::column_stack](column_stack.md) - Column-based stacking
- [torch::split](split.md) - Tensor splitting (inverse operation)
- [Tensor Manipulation Guide](../tensor_manipulation.md) 