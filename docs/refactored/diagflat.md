# torch::diagflat

Creates a diagonal matrix from a flattened tensor.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::diagflat input ?offset?
```

### Named Parameter Syntax (Recommended)
```tcl
torch::diagflat -input input ?-offset offset?
```

### CamelCase Alias
```tcl
torch::diagFlat ...
```

## Parameters

### Required Parameters
- **input** (tensor): Input tensor of any shape (will be flattened)

### Optional Parameters
- **offset** (integer, default: 0): Diagonal offset where to place the flattened values
  - 0: Main diagonal
  - Positive values: Upper diagonals 
  - Negative values: Lower diagonals

## Returns

Returns a tensor handle containing a 2-D square matrix with the flattened input values placed on the specified diagonal and zeros elsewhere.

## Description

The `torch::diagflat` command creates a diagonal matrix from any input tensor by first flattening it into a 1-D vector and then placing those values on the specified diagonal of a square matrix.

Key differences from `torch::diag`:
- **Always creates a matrix**: Never extracts diagonal elements
- **Flattens input first**: Any tensor shape becomes a 1-D vector before diagonal placement
- **Always square output**: The resulting matrix is always square

The process is:
1. **Flatten** the input tensor into a 1-D vector
2. **Create** a square matrix with zeros
3. **Place** the flattened values on the specified diagonal

The size of the output matrix depends on:
- **Number of elements** in flattened input
- **Diagonal offset** (larger offsets require larger matrices)

## Examples

### Basic Usage

#### Vector Input (1-D)
```tcl
# Create a vector
set vector [torch::tensor_create -data {1.0 2.0 3.0} -shape {3} -dtype float32]

# Create diagonal matrix using positional syntax
set diag_matrix [torch::diagflat $vector]
# Result: 3x3 matrix with [1,2,3] on main diagonal

# Create diagonal matrix using named parameters
set diag_matrix [torch::diagflat -input $vector]
```

#### Matrix Input (2-D)
```tcl
# Create a 2x3 matrix
set matrix [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]

# Create diagonal matrix from flattened values
set diag_matrix [torch::diagflat $matrix]
# Result: 6x6 matrix with [1,2,3,4,5,6] on main diagonal

# Using named parameters
set diag_matrix [torch::diagflat -input $matrix]
```

#### Higher Dimensional Input
```tcl
# Create a 2x2x2 tensor
set tensor3d [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 2 2} -dtype float32]

# Create diagonal matrix from flattened 3D tensor
set diag_matrix [torch::diagflat $tensor3d]
# Result: 8x8 matrix with [1,2,3,4,5,6,7,8] on main diagonal
```

### Diagonal Offsets

#### Upper Diagonal (Positive Offset)
```tcl
set vector [torch::tensor_create -data {1.0 2.0 3.0} -shape {3} -dtype float32]

# Place values on upper diagonal (offset +1)
set upper_matrix [torch::diagflat $vector 1]
# Result: 4x4 matrix with [1,2,3] on first upper diagonal

# Using named parameters
set upper_matrix [torch::diagflat -input $vector -offset 1]
```

#### Lower Diagonal (Negative Offset)
```tcl
# Place values on lower diagonal (offset -1)
set lower_matrix [torch::diagflat $vector -1]
# Result: 4x4 matrix with [1,2,3] on first lower diagonal

# Using named parameters
set lower_matrix [torch::diagflat -input $vector -offset -1]
```

#### Large Offsets
```tcl
set vector [torch::tensor_create -data {1.0 2.0} -shape {2} -dtype float32]

# Large positive offset
set matrix [torch::diagflat $vector 5]
# Result: 7x7 matrix with [1,2] on the 5th upper diagonal

# Large negative offset
set matrix [torch::diagflat $vector -3]
# Result: 5x5 matrix with [1,2] on the 3rd lower diagonal
```

### Flattening Behavior

#### Matrix Flattening (Row-Major Order)
```tcl
# Create 2x2 matrix
set matrix [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]

# Diagonal matrix from flattened values
set result [torch::diagflat $matrix]
# Input flattened: [1,2,3,4] (row-major order)
# Result: 4x4 diagonal matrix with [1,2,3,4] on main diagonal
```

#### 3D Tensor Flattening
```tcl
# Create 2x1x3 tensor
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 1 3} -dtype float32]

# Diagonal matrix from flattened 3D tensor
set result [torch::diagflat $tensor]
# Input flattened: [1,2,3,4,5,6]
# Result: 6x6 diagonal matrix
```

### Working with Different Data Types

#### Integer Tensors
```tcl
set int_matrix [torch::tensor_create -data {1 2 3 4} -shape {2 2} -dtype int32]
set int_diag [torch::diagflat $int_matrix]
# Creates integer diagonal matrix
```

#### Double Precision
```tcl
set double_tensor [torch::tensor_create -data {1.0 2.0 3.0} -shape {3} -dtype float64]
set double_diag [torch::diagflat $double_tensor]
# Creates double precision diagonal matrix
```

### CamelCase Alias Usage
```tcl
set vector [torch::tensor_create -data {1.0 2.0} -shape {2} -dtype float32]

# Using camelCase alias
set result1 [torch::diagFlat $vector]
set result2 [torch::diagFlat -input $vector -offset 1]

# Both produce the same results as torch::diagflat
```

### Parameter Order Flexibility
```tcl
set matrix [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]

# Named parameters can be in any order
set result1 [torch::diagflat -input $matrix -offset 2]
set result2 [torch::diagflat -offset 2 -input $matrix]
# Both produce identical results
```

## Mathematical Properties

### Matrix Size Calculation
For input with `n` elements and offset `k`:
- **Matrix size**: `(n + |k|) × (n + |k|)`
- **Diagonal length**: `n` (number of flattened elements)
- **Zero elements**: Total elements minus diagonal elements

### Diagonal Positioning
- **k = 0**: Main diagonal (elements at positions [0,0], [1,1], ..., [n-1,n-1])
- **k > 0**: Upper diagonal (elements at positions [0,k], [1,k+1], ..., [n-1,n-1+k])
- **k < 0**: Lower diagonal (elements at positions [|k|,0], [|k|+1,1], ..., [|k|+n-1,n-1])

### Flattening Order
PyTorch uses **row-major (C-style)** flattening:
- 2D matrix `[[a,b], [c,d]]` becomes `[a,b,c,d]`
- 3D tensor shape `[2,2,2]` flattens in order `[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]`

## Common Use Cases

### Creating Test Matrices
```tcl
# Create a simple diagonal matrix for testing
set test_values [torch::tensor_create -data {1.0 2.0 3.0} -shape {3} -dtype float32]
set test_matrix [torch::diagflat $test_values]
# Result: 3x3 diagonal matrix for linear algebra tests
```

### Scaling Matrices
```tcl
# Create scaling matrix from multi-dimensional scale factors
set scale_tensor [torch::tensor_create -data {0.5 1.0 1.5 2.0} -shape {2 2} -dtype float32]
set scale_matrix [torch::diagflat $scale_tensor]
# Result: 4x4 diagonal scaling matrix
```

### Converting Arrays to Diagonal Form
```tcl
# Convert any tensor to diagonal matrix representation
set complex_tensor [create_some_tensor]  # Any shape
set diagonal_form [torch::diagflat $complex_tensor]
# Useful for diagonal matrix operations on arbitrary data
```

### Identity-like Matrices
```tcl
# Create identity matrix of specific size
set size 5
set ones [torch::tensor_create -data [lrepeat $size 1.0] -shape [list $size] -dtype float32]
set identity [torch::diagflat $ones]
# Result: 5x5 identity matrix
```

### Sparse Matrix Representation
```tcl
# Create sparse-like diagonal matrices for efficient computation
set sparse_data [torch::tensor_create -data {1.0 0.0 3.0 0.0 5.0} -shape {5} -dtype float32]
set sparse_matrix [torch::diagflat $sparse_data]
# Diagonal representation of sparse data
```

## Error Handling

The command performs validation and provides clear error messages:

```tcl
# Invalid tensor names
catch {torch::diagflat invalid_tensor} error
# Error: "Invalid input tensor"

# Missing required parameters
catch {torch::diagflat} error
# Error: Usage message

# Invalid parameter names
catch {torch::diagflat -invalid_param $tensor} error
# Error: "Unknown parameter: -invalid_param"

# Invalid offset value
catch {torch::diagflat $tensor invalid_offset} error
# Error: "Invalid offset value" or integer conversion error

# Missing parameter value
catch {torch::diagflat -input} error
# Error: "Missing value for parameter"
```

## Performance Notes

- **Memory usage**: Scales quadratically with input size: O((n + |offset|)²)
- **Computation speed**: Linear with input size for flattening, quadratic for matrix creation
- **Large tensors**: Consider memory limitations for very large input tensors
- **Sparse representation**: Most output matrix elements are zeros (sparse structure)

## Technical Details

### Memory Layout
- **Input flattening**: Temporary 1-D view of input data
- **Matrix creation**: New square matrix filled with zeros except diagonal
- **Device compatibility**: Works on both CPU and GPU tensors
- **Data type preservation**: Output maintains input tensor data type

### Flattening Mechanics
- **Contiguous memory**: Ensures efficient flattening operation
- **View vs Copy**: May create view or copy depending on input layout
- **Dimension collapse**: All input dimensions reduced to single dimension

### Matrix Construction
- **Zero initialization**: Output matrix starts as zeros
- **Diagonal assignment**: Only diagonal elements are set to input values
- **Memory efficiency**: Considers sparse structure for large matrices

## Comparison with Related Functions

### torch::diagflat vs torch::diag
| Feature | torch::diagflat | torch::diag |
|---------|----------------|-------------|
| Input processing | Always flattens first | Uses input as-is |
| Output | Always creates matrix | Matrix creation OR extraction |
| 1D input | Creates diagonal matrix | Creates diagonal matrix |
| 2D input | Flattens then creates matrix | Extracts diagonal elements |
| Use case | Matrix creation from any data | Dual purpose (create/extract) |

### torch::diagflat vs torch::eye
| Feature | torch::diagflat | torch::eye |
|---------|----------------|-------------|
| Values | Custom values from input | Always 1s and 0s |
| Flexibility | Any diagonal values | Identity matrices only |
| Input | Tensor required | Size specification |
| Offset support | Yes | No (main diagonal only) |

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
set result [torch::diagflat $input]
set result [torch::diagflat $input 2]
```

**New (Named Parameters):**
```tcl
set result [torch::diagflat -input $input]
set result [torch::diagflat -input $input -offset 2]
```

**Benefits of Named Parameters:**
- Self-documenting code
- Parameter order independence
- Easier to extend with new optional parameters
- Reduced errors from parameter misplacement

## Related Commands

- **torch::diag**: Extract diagonal or create diagonal matrix (dual purpose)
- **torch::eye**: Create identity matrices
- **torch::zeros**: Create zero matrices
- **torch::flatten**: Flatten tensors without diagonal placement
- **torch::tensor_create**: Create input tensors

## See Also

- [torch::diag](diag.md) - Diagonal operations (extraction and creation)
- [torch::eye](eye.md) - Identity matrix creation
- [torch::zeros](zeros.md) - Zero matrix creation
- [torch::flatten](flatten.md) - Tensor flattening operations
- [Linear Algebra in PyTorch](https://pytorch.org/docs/stable/torch.html#linear-algebra)
- [Matrix Operations](https://pytorch.org/docs/stable/torch.html#matrix-operations) 