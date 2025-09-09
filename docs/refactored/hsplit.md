# torch::hsplit

Split a tensor horizontally (along dimension 1) into multiple smaller tensors.

## Syntax

### Current Syntax
```tcl
torch::hsplit tensor sections_or_indices
torch::hSplit tensor sections_or_indices                    ;# camelCase alias
```

### Named Parameters Syntax  
```tcl
torch::hsplit -tensor tensor -sections sections_or_indices
torch::hsplit -input tensor -sections sections_or_indices   ;# alternative parameter names
torch::hsplit -tensor tensor -indices indices_list
torch::hSplit -tensor tensor -sections sections_or_indices  ;# camelCase alias
```

## Parameters

### Required Parameters
- **tensor** (string): Handle to the input tensor to split
- **sections_or_indices** (int or list): Either:
  - An integer specifying the number of equal sections to split into
  - A list of integers specifying the indices at which to split

### Parameter Aliases
- **-tensor** or **-input**: Input tensor parameter
- **-sections** or **-indices**: Sections/indices parameter

## Description

The `torch::hsplit` command horizontally splits a tensor into multiple smaller tensors. It is equivalent to calling `torch::split` with `dim=1` (the width dimension).

The function supports two modes:
1. **Equal sections**: Split into N equal parts (sections must evenly divide the tensor dimension)
2. **Specified indices**: Split at specific indices along dimension 1

## Return Value

Returns a list of tensor handles, each representing a split portion of the original tensor.

## Examples

### Basic Usage

#### Equal Sections Split
```tcl
# Create a tensor with shape [2, 6, 3]
set tensor [torch::ones -shape {2 6 3}]

# Split into 2 equal sections (positional syntax)
set result [torch::hsplit $tensor 2]
;# Returns 2 tensors, each with shape [2, 3, 3]

# Split into 2 equal sections (named parameters)
set result [torch::hsplit -tensor $tensor -sections 2]
;# Returns 2 tensors, each with shape [2, 3, 3]

# Using camelCase alias
set result [torch::hSplit -tensor $tensor -sections 2]
;# Returns 2 tensors, each with shape [2, 3, 3]
```

#### Split at Specific Indices
```tcl
# Create a tensor with shape [2, 8, 3]
set tensor [torch::ones -shape {2 8 3}]

# Split at indices 2, 4, 6 (positional syntax)
set result [torch::hsplit $tensor {2 4 6}]
;# Returns 4 tensors with shapes [2, 2, 3], [2, 2, 3], [2, 2, 3], [2, 2, 3]

# Split at indices 2, 4, 6 (named parameters)
set result [torch::hsplit -tensor $tensor -indices {2 4 6}]
;# Returns 4 tensors with shapes [2, 2, 3], [2, 2, 3], [2, 2, 3], [2, 2, 3]
```

### Advanced Usage

#### Working with Different Data Types
```tcl
# Float32 tensor
set tensor_f32 [torch::ones -shape {2 6 3} -dtype float32]
set result [torch::hsplit -tensor $tensor_f32 -sections 3]

# Int64 tensor
set tensor_i64 [torch::ones -shape {2 6 3} -dtype int64]
set result [torch::hsplit -tensor $tensor_i64 -sections 3]
```

#### Alternative Parameter Names
```tcl
set tensor [torch::ones -shape {2 6 3}]

# Using -input instead of -tensor
set result [torch::hsplit -input $tensor -sections 2]

# Using -indices instead of -sections
set result [torch::hsplit -input $tensor -indices {2 4}]
```

#### Data Processing Pipeline
```tcl
# Create sequential data
set tensor [torch::arange -start 0 -end 24 -dtype float32]
set reshaped [torch::reshape $tensor -shape {2 12}]

# Split into 3 equal sections for processing
set splits [torch::hsplit -tensor $reshaped -sections 3]

# Process each split
foreach split $splits {
    set shape [torch::tensor_shape $split]
    puts "Split shape: $shape"
}
```

## Error Handling

The command will throw an error in the following cases:

### Missing Required Parameters
```tcl
# Error: Missing tensor parameter
torch::hsplit -sections 2

# Error: Missing sections parameter  
torch::hsplit -tensor $tensor
```

### Invalid Parameter Names
```tcl
# Error: Invalid parameter name
torch::hsplit -tensor $tensor -invalid 2
```

### Incompatible Split Size
```tcl
# Error: Cannot split size 5 into 2 equal parts
set tensor [torch::ones -shape {2 5 3}]
torch::hsplit $tensor 2
```

### Wrong Number of Arguments
```tcl
# Error: Wrong number of positional arguments
torch::hsplit $tensor
```

## Implementation Details

### Dual Syntax Support
The command supports both the original positional syntax (for backward compatibility) and the new named parameter syntax:

```tcl
# Original syntax (still supported)
torch::hsplit $tensor 2

# New named parameter syntax
torch::hsplit -tensor $tensor -sections 2

# Both produce identical results
```

### camelCase Alias
The command provides a camelCase alias for modern coding conventions:

```tcl
# snake_case (original)
torch::hsplit -tensor $tensor -sections 2

# camelCase (alias)
torch::hSplit -tensor $tensor -sections 2
```

### Parameter Validation
- Validates that both tensor and sections/indices parameters are provided
- Checks for unknown parameter names
- Ensures proper argument count for positional syntax

## Performance Considerations

- The split operation is memory-efficient and doesn't copy data unnecessarily
- All output tensors share the same data type as the input tensor
- Memory usage scales with the number of splits requested

## Related Commands

- **torch::vsplit**: Split tensor vertically (along dimension 0)
- **torch::dsplit**: Split tensor along depth dimension (dimension 2)
- **torch::split**: Generic tensor split along any dimension
- **torch::tensor_split**: Advanced tensor splitting with more options
- **torch::chunk**: Split tensor into approximately equal chunks

## Mathematical Background

Horizontal splitting divides a tensor along its second dimension (width):

```
Input tensor shape: [batch_size, width, height, ...]
Split along dimension 1 (width)
Output: Multiple tensors with shape [batch_size, width/N, height, ...]
```

For a tensor with shape `[A, B, C, D]` split into N sections:
- Each section has shape `[A, B/N, C, D]`
- Total elements preserved: `A × B × C × D`

## Version History

| Version | Change |
|---------|--------|
| 1.0.0   | Initial implementation with positional syntax |
| 2.0.0   | Added dual syntax support with named parameters |
| 2.0.0   | Added camelCase alias (`torch::hSplit`) |
| 2.0.0   | Added parameter validation and error handling |
| 2.0.0   | Added alternative parameter names (`-input`, `-indices`) |

## See Also

- [torch::vsplit](vsplit.md) - Vertical tensor splitting
- [torch::dsplit](dsplit.md) - Depth tensor splitting  
- [torch::split](split.md) - Generic tensor splitting
- [torch::chunk](chunk.md) - Tensor chunking
- [Tensor Manipulation Guide](../tensor_manipulation.md) 