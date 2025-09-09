# torch::diag

Extracts diagonal elements from a matrix or creates a diagonal matrix from a vector.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::diag input ?diagonal?
```

### Named Parameter Syntax (Recommended)
```tcl
torch::diag -input input ?-diagonal diagonal?
```

### CamelCase Alias
The command `torch::diag` is already in camelCase format (no underscores).

## Parameters

### Required Parameters
- **input** (tensor): Input tensor (1-D for diagonal matrix creation, 2-D or higher for diagonal extraction)

### Optional Parameters
- **diagonal** (integer, default: 0): Which diagonal to extract or create
  - 0: Main diagonal
  - Positive values: Upper diagonals 
  - Negative values: Lower diagonals

## Returns

Returns a tensor handle containing:
- If **input is 1-D**: A 2-D diagonal matrix with input values on the specified diagonal
- If **input is 2-D or higher**: A 1-D tensor containing the diagonal elements

## Description

The `torch::diag` command has dual functionality depending on the input tensor dimensions:

### 1. Diagonal Matrix Creation (1-D Input)
When given a 1-D tensor, creates a 2-D square matrix with the input values placed on the specified diagonal and zeros elsewhere.

### 2. Diagonal Extraction (2-D+ Input)
When given a 2-D or higher-dimensional tensor, extracts the diagonal elements along the specified diagonal offset.

The `diagonal` parameter controls which diagonal to work with:
- **0** (default): Main diagonal (top-left to bottom-right)
- **Positive values**: Upper diagonals (above main diagonal)
- **Negative values**: Lower diagonals (below main diagonal)

## Examples

### Basic Usage

#### Diagonal Matrix Creation from Vector
```tcl
# Create a vector
set vector [torch::tensor_create -data {1.0 2.0 3.0} -shape {3} -dtype float32]

# Create diagonal matrix using positional syntax
set diag_matrix [torch::diag $vector]
# Result: 3x3 matrix with 1,2,3 on main diagonal

# Create diagonal matrix using named parameters
set diag_matrix [torch::diag -input $vector]
```

#### Diagonal Extraction from Matrix
```tcl
# Create a 3x3 matrix
set matrix [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0} -shape {3 3} -dtype float32]

# Extract main diagonal using positional syntax
set diagonal [torch::diag $matrix]
# Result: 1-D tensor with values {1.0, 5.0, 9.0}

# Extract main diagonal using named parameters
set diagonal [torch::diag -input $matrix]
```

### Diagonal Offsets

#### Upper Diagonal (Positive Offset)
```tcl
set matrix [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0} -shape {3 3} -dtype float32]

# Extract upper diagonal (offset +1)
set upper_diag [torch::diag $matrix 1]
# Result: 1-D tensor with values {2.0, 6.0} (elements [0,1] and [1,2])

# Using named parameters
set upper_diag [torch::diag -input $matrix -diagonal 1]
```

#### Lower Diagonal (Negative Offset)
```tcl
# Extract lower diagonal (offset -1)
set lower_diag [torch::diag $matrix -1]
# Result: 1-D tensor with values {4.0, 8.0} (elements [1,0] and [2,1])

# Using named parameters
set lower_diag [torch::diag -input $matrix -diagonal -1]
```

#### Creating Off-Diagonal Matrix
```tcl
set vector [torch::tensor_create -data {1.0 2.0} -shape {2} -dtype float32]

# Create matrix with values on upper diagonal (offset +1)
set matrix [torch::diag $vector 1]
# Result: 3x3 matrix with 1,2 on first upper diagonal

# Using named parameters
set matrix [torch::diag -input $vector -diagonal 1]
```

### Different Matrix Shapes

#### Non-Square Matrices
```tcl
# Create 2x3 matrix
set matrix [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]

# Extract main diagonal (limited by min(rows, cols))
set diagonal [torch::diag $matrix]
# Result: 1-D tensor with 2 elements {1.0, 5.0}
```

#### Rectangular Matrices
```tcl
# Create 3x2 matrix  
set matrix [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {3 2} -dtype float32]

# Extract main diagonal
set diagonal [torch::diag $matrix]
# Result: 1-D tensor with 2 elements {1.0, 4.0}
```

### Identity Matrix Creation
```tcl
# Create identity-like matrix
set ones [torch::tensor_create -data {1.0 1.0 1.0} -shape {3} -dtype float32]
set identity [torch::diag $ones]
# Result: 3x3 identity matrix
```

### Working with Different Data Types

#### Integer Tensors
```tcl
set int_vector [torch::tensor_create -data {1 2 3} -shape {3} -dtype int32]
set int_matrix [torch::diag $int_vector]
# Creates integer diagonal matrix
```

#### Double Precision
```tcl
set double_vector [torch::tensor_create -data {1.0 2.0 3.0} -shape {3} -dtype float64]
set double_matrix [torch::diag $double_vector]
# Creates double precision diagonal matrix
```

### Parameter Order Flexibility
```tcl
set matrix [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]

# Named parameters can be in any order
set result1 [torch::diag -input $matrix -diagonal 1]
set result2 [torch::diag -diagonal 1 -input $matrix]
# Both produce the same result
```

## Mathematical Properties

### Diagonal Matrix Properties
- **Square**: Diagonal matrices created from 1-D input are always square
- **Sparse**: Only diagonal elements are non-zero
- **Symmetric**: When diagonal=0 and all diagonal elements are the same

### Diagonal Extraction Properties
- **Length**: For diagonal offset `k` in an `m×n` matrix:
  - If `k ≥ 0`: length = `min(m, n-k)`
  - If `k < 0`: length = `min(m+k, n)`
- **Range**: Diagonal offsets are bounded by matrix dimensions

## Common Use Cases

### Linear Algebra Operations
```tcl
# Create diagonal matrix for scaling
set scale_factors [torch::tensor_create -data {2.0 3.0 4.0} -shape {3} -dtype float32]
set scale_matrix [torch::diag $scale_factors]

# Extract eigenvalues from diagonal matrix
set eigenvalues [torch::diag $diagonal_matrix]
```

### Identity Matrix Generation
```tcl
# Create identity matrix of desired size
set size 5
set ones [torch::tensor_create -data [lrepeat $size 1.0] -shape [list $size] -dtype float32]
set identity [torch::diag $ones]
```

### Matrix Analysis
```tcl
# Check if matrix is diagonal by comparing with diag(diag(matrix))
set matrix [create_some_matrix]
set extracted_diag [torch::diag $matrix]
set reconstructed [torch::diag $extracted_diag]
# Compare $matrix with $reconstructed to check if matrix is diagonal
```

### Trace Calculation
```tcl
# Extract main diagonal for trace calculation
set matrix [create_square_matrix]
set main_diagonal [torch::diag $matrix]
# Sum elements of main_diagonal to get trace
```

## Error Handling

The command performs validation and provides clear error messages:

```tcl
# Invalid tensor names
catch {torch::diag invalid_tensor} error
# Error: "Invalid input tensor"

# Missing required parameters
catch {torch::diag} error
# Error: Usage message

# Invalid parameter names
catch {torch::diag -invalid_param $tensor} error
# Error: "Unknown parameter: -invalid_param"

# Invalid diagonal offset
catch {torch::diag $matrix invalid_offset} error
# Error: "Invalid diagonal value" or integer conversion error
```

## Performance Notes

- **Memory efficiency**: Diagonal matrices store only non-zero elements internally
- **Computation speed**: Diagonal extraction is O(min(m,n)) for m×n matrices
- **Large matrices**: Performance scales linearly with matrix size for extraction
- **Diagonal creation**: Performance scales with the square of vector length

## Technical Details

### Diagonal Indexing
For a matrix with shape `[m, n]` and diagonal offset `k`:
- **Valid range**: `-m < k < n`
- **Element positions**: `(i, i+k)` for upper diagonals, `(i-k, i)` for lower diagonals
- **Boundary handling**: Automatically clips to valid matrix bounds

### Memory Layout
- **Input preservation**: Original tensor remains unchanged
- **Output format**: New tensor with appropriate dimensions
- **Data type preservation**: Output maintains input data type
- **Device compatibility**: Works with both CPU and GPU tensors

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
set result [torch::diag $input]
set result [torch::diag $input 1]
```

**New (Named Parameters):**
```tcl
set result [torch::diag -input $input]
set result [torch::diag -input $input -diagonal 1]
```

**Benefits of Named Parameters:**
- Self-documenting code
- Parameter order independence
- Easier to extend with new optional parameters
- Reduced errors from parameter misplacement

## Related Commands

- **torch::diagflat**: Create diagonal matrix from flattened input
- **torch::tril**: Extract lower triangular matrix
- **torch::triu**: Extract upper triangular matrix
- **torch::trace**: Calculate matrix trace
- **torch::tensor_create**: Create input tensors

## See Also

- [torch::diagflat](diagflat.md) - Diagonal matrix from flattened tensor
- [torch::tril](tril.md) - Lower triangular matrix operations
- [torch::triu](triu.md) - Upper triangular matrix operations
- [torch::trace](trace.md) - Matrix trace calculation
- [Linear Algebra in PyTorch](https://pytorch.org/docs/stable/torch.html#linear-algebra)
- [Matrix Operations](https://pytorch.org/docs/stable/torch.html#matrix-operations) 