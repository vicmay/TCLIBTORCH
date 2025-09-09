# torch::tensordot

Computes the tensor dot product along specified dimensions.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::tensordot a b dims
```

### Named Parameters (New)
```tcl
torch::tensordot -a tensor_a -b tensor_b -dims dimensions
```

### CamelCase Alias
```tcl
torch::tensorDot -a tensor_a -b tensor_b -dims dimensions
```

## Parameters

| Parameter   | Type          | Required | Description                                 |
|-------------|---------------|----------|---------------------------------------------|
| a           | tensor_handle | Yes      | First input tensor                          |
| b           | tensor_handle | Yes      | Second input tensor                         |
| dims        | list          | Yes      | Dimensions to contract (must be compatible) |

## Return Value

Returns a tensor handle containing the result of the tensor dot product operation.

## Description

The `torch::tensordot` function computes the tensor dot product between two tensors along specified dimensions. The dimensions specified in the `dims` parameter must be compatible between the two tensors (i.e., the sizes of the contracted dimensions must match).

**Important Notes:**
- Dimensions in the `dims` list must be unique (no duplicates)
- The contracted dimensions must have matching sizes in both tensors
- For 1D tensors, use a single dimension (e.g., `{0}`) instead of duplicate dimensions

## Examples

### Basic Usage

```tcl
# Create 2x2 tensors
set a [torch::tensor_create -data {{1 2} {3 4}} -dtype float32]
set b [torch::tensor_create -data {{5 6} {7 8}} -dtype float32]

# Compute tensordot using positional syntax
set result [torch::tensordot $a $b {0 1}]
puts [torch::tensor_to_list $result]
# Output: 70.0

# Compute tensordot using named syntax
set result [torch::tensordot -a $a -b $b -dims {0 1}]
puts [torch::tensor_to_list $result]
# Output: 70.0

# Compute tensordot using camelCase alias
set result [torch::tensorDot -a $a -b $b -dims {0 1}]
puts [torch::tensor_to_list $result]
# Output: 70.0
```

### Different Tensor Shapes

```tcl
# Create 2x3 and 3x2 tensors
set a [torch::tensor_create -data {{1 2 3} {4 5 6}} -dtype float32]
set b [torch::tensor_create -data {{7 8 9} {10 11 12}} -dtype float32]

# Contract along compatible dimensions
set result [torch::tensordot $a $b {1 0}]
puts [torch::tensor_to_list $result]
# Output: 217.0
```

### 1D Tensor Dot Product

```tcl
# Create 1D tensors
set a [torch::tensor_create -data {1 2 3} -dtype float32]
set b [torch::tensor_create -data {4 5 6} -dtype float32]

# Use single dimension (not duplicate)
set result [torch::tensordot $a $b {0}]
puts [torch::tensor_to_list $result]
# Output: 32.0
```

### Edge Cases

```tcl
# Zero tensor
set a [torch::tensor_create -data {{0 0} {0 0}} -dtype float32]
set b [torch::tensor_create -data {{1 2} {3 4}} -dtype float32]
set result [torch::tensordot $a $b {0 1}]
puts [torch::tensor_to_list $result]
# Output: 0.0

# Identity matrix
set a [torch::tensor_create -data {{1 0} {0 1}} -dtype float32]
set b [torch::tensor_create -data {{1 0} {0 1}} -dtype float32]
set result [torch::tensordot $a $b {0 1}]
puts [torch::tensor_to_list $result]
# Output: 2.0
```

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
torch::tensordot tensor_a tensor_b {0 1}
```

**New (Named):**
```tcl
torch::tensordot -a tensor_a -b tensor_b -dims {0 1}
```

**New (CamelCase):**
```tcl
torch::tensorDot -a tensor_a -b tensor_b -dims {0 1}
```

### Benefits of Named Parameters

1. **Clarity**: Parameter names make the code more readable
2. **Flexibility**: Parameters can be specified in any order
3. **Maintainability**: Easier to understand and modify
4. **Consistency**: Follows modern API design patterns

## Error Handling

The function will throw an error in the following cases:

- **Missing parameters**: If any required parameter is missing
- **Invalid tensor handles**: If the specified tensors don't exist
- **Invalid dimensions**: If dimensions are not valid integers
- **Duplicate dimensions**: If the same dimension appears multiple times in the dims list
- **Incompatible dimensions**: If the contracted dimensions don't have matching sizes
- **Unknown parameters**: If an unrecognized parameter is provided

### Error Examples

```tcl
# Missing parameters
torch::tensordot
# Error: Required parameters missing: a, b, and dims required

# Invalid tensor
torch::tensordot invalid_tensor $b {0 1}
# Error: Invalid tensor a

# Duplicate dimensions
torch::tensordot $a $b {0 0}
# Error: dim 0 appears multiple times in the list of dims

# Incompatible dimensions
set a [torch::tensor_create -data {{1 2 3} {4 5 6}} -dtype float32]
set b [torch::tensor_create -data {{7 8} {9 10}} -dtype float32]
torch::tensordot $a $b {1 0}
# Error: contracted dimensions need to match, but first has size 3 in dim 1 and second has size 2 in dim 1
```

## Notes

- The function maintains backward compatibility with the original positional parameter syntax
- The camelCase alias `torch::tensorDot` provides an alternative naming convention
- Tensor memory is managed automatically by the extension
- The function supports tensors of different data types (float32, int32, etc.)
- For complex tensor operations, ensure the contracted dimensions have compatible sizes
- The result tensor will have the same data type as the input tensors 