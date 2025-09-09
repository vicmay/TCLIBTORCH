# torch::column_stack / torch::columnStack

## Description
Stacks a sequence of 1D or 2D tensors horizontally (column wise) to make a single tensor. This is equivalent to concatenation along the last dimension after ensuring all tensors have at least 2 dimensions.

## Syntax

### Original Syntax (Snake Case)
```tcl
torch::column_stack tensor_list
torch::column_stack tensor1 tensor2 [tensor3 ...]
```

### New Syntax (CamelCase Alias)
```tcl
torch::columnStack tensor_list  
torch::columnStack tensor1 tensor2 [tensor3 ...]
```

## Parameters

### List Format
- **tensor_list**: A list of tensor handles to stack horizontally

### Multiple Arguments Format  
- **tensor1, tensor2, ...**: Individual tensor handles to stack horizontally

## Return Value
Returns a new tensor handle containing the horizontally stacked tensors.

## Behavior

### Input Handling
- **1D tensors**: Treated as column vectors (reshaped to Nx1)
- **2D tensors**: Stacked along the last dimension (columns)
- **Empty tensors**: Not supported, will raise an error

### Shape Requirements
- All input tensors must have the same number of rows (first dimension)
- The number of columns can vary between tensors
- Minimum 1 tensor required

### Output Shape
For tensors with shapes `[H, W1]`, `[H, W2]`, ..., `[H, Wn]`:
- Result shape: `[H, W1 + W2 + ... + Wn]`

## Examples

### Basic Usage
```tcl
# Create test matrices
set mat1 [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
set mat1 [torch::tensor_reshape $mat1 {2 2}]  # 2x2 matrix

set mat2 [torch::tensor_create {5.0 6.0 7.0 8.0} float32]  
set mat2 [torch::tensor_reshape $mat2 {2 2}]  # 2x2 matrix

# Stack horizontally using list syntax
set result [torch::column_stack [list $mat1 $mat2]]
# Result shape: [2, 4] - same height, combined width

# Stack horizontally using multiple arguments
set result [torch::column_stack $mat1 $mat2]
# Same result as above
```

### CamelCase Alias
```tcl
# Using camelCase alias with list syntax
set result [torch::columnStack [list $mat1 $mat2]]

# Using camelCase alias with multiple arguments
set result [torch::columnStack $mat1 $mat2]
```

### 1D Tensor Handling
```tcl
# Create 1D tensors
set vec1 [torch::tensor_create {1.0 2.0 3.0} float32]
set vec2 [torch::tensor_create {4.0 5.0 6.0} float32]

# Column stack treats 1D tensors as column vectors
set result [torch::column_stack $vec1 $vec2]
# Result shape: [3, 2] - each vector becomes a column
```

### Column Vectors
```tcl
# Create explicit column vectors
set col1 [torch::tensor_create {1.0 2.0 3.0} float32]
set col1 [torch::tensor_reshape $col1 {3 1}]  # 3x1 column

set col2 [torch::tensor_create {4.0 5.0 6.0} float32]
set col2 [torch::tensor_reshape $col2 {3 1}]  # 3x1 column

set result [torch::column_stack $col1 $col2]
# Result shape: [3, 2] - side-by-side columns
```

### Multiple Tensors
```tcl
# Stack multiple tensors at once
set t1 [torch::tensor_create {1.0 2.0} float32]
set t1 [torch::tensor_reshape $t1 {2 1}]  # 2x1

set t2 [torch::tensor_create {3.0 4.0} float32]
set t2 [torch::tensor_reshape $t2 {2 1}]  # 2x1

set t3 [torch::tensor_create {5.0 6.0} float32]
set t3 [torch::tensor_reshape $t3 {2 1}]  # 2x1

set result [torch::column_stack [list $t1 $t2 $t3]]
# Result shape: [2, 3] - three columns side by side
```

### Different Column Widths
```tcl
# Tensors with different number of columns
set narrow [torch::tensor_create {1.0 2.0} float32]
set narrow [torch::tensor_reshape $narrow {2 1}]  # 2x1

set wide [torch::tensor_create {3.0 4.0 5.0 6.0} float32]
set wide [torch::tensor_reshape $wide {2 2}]  # 2x2

set result [torch::column_stack $narrow $wide]
# Result shape: [2, 3] - different widths are accommodated
```

## Mathematical Operations

### Matrix Concatenation
```tcl
# Creating larger matrices by combining smaller ones
set A [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
set A [torch::tensor_reshape $A {2 2}]

set B [torch::tensor_create {5.0 6.0 7.0 8.0} float32]
set B [torch::tensor_reshape $B {2 2}]

set combined [torch::column_stack $A $B]
# Result: [[1, 2, 5, 6], [3, 4, 7, 8]]
```

### Data Assembly
```tcl
# Assembling feature matrices from separate feature vectors
set feature1 [torch::tensor_create {0.1 0.2 0.3} float32]  # Feature 1
set feature2 [torch::tensor_create {0.4 0.5 0.6} float32]  # Feature 2
set feature3 [torch::tensor_create {0.7 0.8 0.9} float32]  # Feature 3

set features [torch::column_stack $feature1 $feature2 $feature3]
# Result: 3x3 feature matrix with each column representing one feature
```

## Error Handling

### Empty Tensor List
```tcl
# This will raise an error
catch {torch::column_stack {}} result
puts $result  # Error: column_stack expects a non-empty TensorList
```

### No Arguments
```tcl
# This will raise an error
catch {torch::column_stack} result
puts $result  # Error: wrong # args
```

### Invalid Tensor Names
```tcl
# This will raise an error
catch {torch::column_stack invalid_tensor} result
puts $result  # Error: Invalid tensor name
```

### Incompatible Shapes
```tcl
# This will raise an error for mismatched heights
set t1 [torch::zeros {2 1}]  # Height: 2
set t2 [torch::zeros {3 1}]  # Height: 3 (incompatible)

catch {torch::column_stack $t1 $t2} result
puts $result  # Error: incompatible tensor sizes
```

## Data Type Compatibility
The function supports automatic type promotion when stacking tensors of different data types:

```tcl
# Mixed data types
set float_tensor [torch::tensor_create {1.0 2.0} float32]
set double_tensor [torch::tensor_create {3.0 4.0} float64]

set result [torch::column_stack $float_tensor $double_tensor]
# Result will be promoted to the higher precision type (float64)
```

## Performance Notes

- **Memory efficiency**: Creates a new tensor without modifying inputs
- **Type promotion**: Automatic promotion to common data type
- **Batch processing**: Both syntax variants have similar performance
- **Large tensors**: Efficient for large-scale matrix operations

## Use Cases

1. **Feature matrix assembly**: Combining different feature vectors into a matrix
2. **Matrix augmentation**: Adding columns to existing matrices
3. **Data preprocessing**: Organizing data for machine learning workflows
4. **Scientific computing**: Matrix operations and transformations
5. **Neural networks**: Preparing input features for linear layers

## Relationship to Other Functions

- **torch::row_stack**: Vertical (row-wise) stacking equivalent
- **torch::cat**: More general concatenation along any dimension
- **torch::hstack**: Alternative name for horizontal stacking
- **torch::vstack**: Alternative name for vertical stacking

## Migration Guide

### From Row-wise to Column-wise
```tcl
# Old: row stacking
set result [torch::row_stack $tensor1 $tensor2]

# New: column stacking (different operation)
set result [torch::column_stack $tensor1 $tensor2]
```

### Syntax Modernization
```tcl
# Old style (still supported)
set result [torch::column_stack $tensors]

# New style (recommended)
set result [torch::columnStack $tensors]
```

## See Also
- [torch::row_stack](row_stack.md) - Stack tensors vertically
- [torch::cat](cat.md) - General tensor concatenation
- [torch::hstack](hstack.md) - Alternative horizontal stacking
- [torch::vstack](vstack.md) - Alternative vertical stacking 