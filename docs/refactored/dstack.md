# torch::dstack

Stacks tensors along a new depth dimension (dimension 2). This operation concatenates tensors along the depth (third) dimension, creating a new dimension if the input tensors are 1D or 2D.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::dstack tensor_list
torch::dstack tensor1 tensor2 [tensor3 ...]
```

### Named Parameter Syntax (Recommended)
```tcl
torch::dstack -tensors tensor_list
torch::dstack -inputs tensor_list
```

### CamelCase Alias
```tcl
torch::dStack -tensors tensor_list
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| tensor_list/-tensors/-inputs | list/multiple | List of tensors or multiple tensor arguments | Yes |

## Description

The `torch::dstack` command stacks tensors along the depth dimension (dimension 2). This is equivalent to concatenation along the third dimension. Input tensors are automatically expanded to at least 3D before stacking:

- **1D tensors**: Shape `[N]` becomes `[1, N, 1]` then stacked along dim 2
- **2D tensors**: Shape `[H, W]` becomes `[H, W, 1]` then stacked along dim 2  
- **3D+ tensors**: Stacked directly along dimension 2

## Mathematical Details

For input tensors with shapes:
- **1D**: `[N]` → Output: `[1, N, num_tensors]`
- **2D**: `[H, W]` → Output: `[H, W, num_tensors]`
- **3D**: `[H, W, D]` → Output: `[H, W, D + D + ...]`

The operation preserves the first two dimensions and concatenates along the third dimension.

## Examples

### Basic Usage - 2D Tensors

```tcl
# Create 2D tensors
set t1 [torch::zeros {3 4} float32 cpu false]
set t2 [torch::ones {3 4} float32 cpu false]

# Positional syntax - tensor list
set result [torch::dstack [list $t1 $t2]]
# Result shape: [3, 4, 2]

# Positional syntax - multiple arguments
set result [torch::dstack $t1 $t2]
# Result shape: [3, 4, 2]

# Named parameter syntax
set result [torch::dstack -tensors [list $t1 $t2]]
# Result shape: [3, 4, 2]
```

### Advanced Usage - Multiple Tensors

```tcl
# Create multiple tensors
set t1 [torch::zeros {2 3} float32 cpu false]
set t2 [torch::ones {2 3} float32 cpu false]
set t3 [torch::full {2 3} 2.0 float32 cpu false]

# Stack three tensors
set result [torch::dstack -tensors [list $t1 $t2 $t3]]
# Result shape: [2, 3, 3]

# Using inputs alias
set result [torch::dstack -inputs [list $t1 $t2 $t3]]
# Result shape: [2, 3, 3]
```

### 1D Tensor Stacking

```tcl
# Create 1D tensors
set t1 [torch::zeros {5} float32 cpu false]
set t2 [torch::ones {5} float32 cpu false]

# Stack 1D tensors
set result [torch::dstack -tensors [list $t1 $t2]]
# Result shape: [1, 5, 2]
```

### CamelCase Syntax

```tcl
# Using camelCase alias
set t1 [torch::zeros {2 2} float32 cpu false]
set t2 [torch::ones {2 2} float32 cpu false]
set result [torch::dStack -tensors [list $t1 $t2]]
# Result shape: [2, 2, 2]
```

## Applications

### Computer Vision - Channel Stacking
```tcl
# Stack RGB color channels
set red_channel [torch::zeros {224 224} float32 cpu false]
set green_channel [torch::ones {224 224} float32 cpu false]
set blue_channel [torch::full {224 224} 0.5 float32 cpu false]

set rgb_image [torch::dstack -tensors [list $red_channel $green_channel $blue_channel]]
# Creates RGB image with shape [224, 224, 3]
```

### Feature Map Concatenation
```tcl
# Combine multiple feature maps
set feature1 [torch::randn {16 16} float32 cpu false]
set feature2 [torch::randn {16 16} float32 cpu false]
set feature3 [torch::randn {16 16} float32 cpu false]

set combined_features [torch::dstack -inputs [list $feature1 $feature2 $feature3]]
# Result shape: [16, 16, 3]
```

### Time Series Data
```tcl
# Stack time series data
set day1_data [torch::randn {24} float32 cpu false]  # 24 hours
set day2_data [torch::randn {24} float32 cpu false]  
set day3_data [torch::randn {24} float32 cpu false]

set weekly_data [torch::dstack -tensors [list $day1_data $day2_data $day3_data]]
# Result shape: [1, 24, 3]
```

## Error Handling

```tcl
# Missing tensors parameter
catch {torch::dstack} error
puts "Error: $error"
# Error: tensor_list or -tensors tensor_list

# Missing value for named parameter
catch {torch::dstack -tensors} error
puts "Error: $error"
# Error: Missing value for parameter: -tensors

# Unknown parameter
set t1 [torch::zeros {2 3} float32 cpu false]
catch {torch::dstack -unknown [list $t1]} error
puts "Error: $error"
# Error: Unknown parameter: -unknown

# Empty tensor list
catch {torch::dstack -tensors [list]} error
puts "Error: $error"
# Error: Missing required parameter: tensors

# Incompatible tensor shapes (first two dimensions must match)
set t1 [torch::zeros {2 3} float32 cpu false]
set t2 [torch::ones {4 5} float32 cpu false]
catch {torch::dstack -tensors [list $t1 $t2]} error
puts "Error: $error"
# Error: Tensor shapes incompatible for stacking
```

## Performance Considerations

- **Memory Efficient**: Creates a new tensor without copying input data when possible
- **Shape Broadcasting**: Automatically handles dimension expansion for 1D and 2D tensors
- **Device Consistency**: All input tensors should be on the same device
- **Dtype Preservation**: Output dtype matches input tensor dtypes

## Notes

1. **Dimension Expansion**: 1D and 2D tensors are automatically expanded to 3D
2. **Shape Compatibility**: First two dimensions must match across all input tensors
3. **Memory Layout**: Result tensor has contiguous memory layout
4. **Device Placement**: All tensors must be on the same device

## Comparison with Related Operations

| Operation | Dimension | Use Case |
|-----------|-----------|----------|
| `torch::vstack` | Vertical (dim 0) | Stack rows |
| `torch::hstack` | Horizontal (dim 1) | Stack columns |
| `torch::dstack` | Depth (dim 2) | Stack along depth |
| `torch::cat` | Any dimension | General concatenation |

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax - tensor list
set result [torch::dstack [list $t1 $t2 $t3]]

# New named parameter syntax
set result [torch::dstack -tensors [list $t1 $t2 $t3]]

# Alternative with inputs alias
set result [torch::dstack -inputs [list $t1 $t2 $t3]]
```

### From Multiple Arguments to Named Parameters

```tcl
# Old positional syntax - multiple arguments
set result [torch::dstack $t1 $t2 $t3]

# New named parameter syntax
set result [torch::dstack -tensors [list $t1 $t2 $t3]]
```

### Parameter Aliases

- `-tensors` and `-inputs` are interchangeable for the tensor list parameter
- Both accept a Tcl list of tensor handles

## Return Value

Returns a Tcl list containing a single tensor handle representing the stacked result. The tensor has the same dtype as the input tensors and is placed on the same device.

## See Also

- `torch::vstack` - Stack tensors vertically (dimension 0)
- `torch::hstack` - Stack tensors horizontally (dimension 1)
- `torch::cat` - Concatenate tensors along specified dimension
- `torch::stack` - Stack tensors along new dimension
- `torch::column_stack` - Stack 1D tensors as columns
- `torch::row_stack` - Stack tensors as rows 