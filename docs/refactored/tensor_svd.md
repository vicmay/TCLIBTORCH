# torch::tensor_svd / torch::tensorSvd

Compute the Singular Value Decomposition (SVD) of a tensor.

## Description

The `torch::tensor_svd` command computes the Singular Value Decomposition of a matrix tensor. It supports both positional and named parameter syntax, with full backward compatibility.

**Alias**: `torch::tensorSvd` (camelCase)

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_svd tensor
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_svd -input tensor
torch::tensor_svd -tensor tensor
```

### CamelCase Alias
```tcl
torch::tensorSvd tensor
torch::tensorSvd -input tensor
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor` / `-input` / `-tensor` | string | Yes | Tensor handle to compute SVD |

## Return Value

Returns a dictionary-like string containing three tensor handles:
- `U`: Left singular vectors
- `S`: Singular values
- `V`: Right singular vectors

Format: `{U u_handle S s_handle V v_handle}`

## Examples

### Basic Usage

**Positional syntax:**
```tcl
set tensor [torch::tensor_create {{1 2} {3 4}}]
set result [torch::tensor_svd $tensor]
puts "SVD result: $result"
```

**Named parameter syntax:**
```tcl
set tensor [torch::tensor_create {{1 2} {3 4}}]
set result [torch::tensor_svd -input $tensor]
puts "SVD result: $result"
```

**CamelCase alias:**
```tcl
set tensor [torch::tensor_create {{1 2} {3 4}}]
set result [torch::tensorSvd -input $tensor]
puts "SVD result: $result"
```

### Extracting Individual Components

```tcl
set tensor [torch::tensor_create {{1 2 3} {4 5 6}}]
set svd_result [torch::tensor_svd $tensor]

;# Parse the result to get individual components
set u_handle [lindex $svd_result 1]  ;# U matrix
set s_handle [lindex $svd_result 3]  ;# S vector
set v_handle [lindex $svd_result 5]  ;# V matrix

puts "U matrix: $u_handle"
puts "S values: $s_handle"
puts "V matrix: $v_handle"
```

### Different Matrix Shapes

```tcl
;# Square matrix
set square [torch::tensor_create {{1 2} {3 4}}]
set result1 [torch::tensor_svd $square]

;# Rectangular matrix (2x3)
set rect1 [torch::tensor_create {{1 2 3} {4 5 6}}]
set result2 [torch::tensor_svd $rect1]

;# Tall matrix (3x2)
set rect2 [torch::tensor_create {{1 2} {3 4} {5 6}}]
set result3 [torch::tensor_svd $rect2]

puts "Square SVD: $result1"
puts "Rectangular SVD: $result2"
puts "Tall SVD: $result3"
```

### Different Data Types

```tcl
set tensor1 [torch::tensor_create {{1.5 2.5} {3.5 4.5}} float32]
set tensor2 [torch::tensor_create {{1.5 2.5} {3.5 4.5}} float64]
set result1 [torch::tensor_svd $tensor1]
set result2 [torch::tensor_svd $tensor2]
puts "Float32 SVD: $result1"
puts "Float64 SVD: $result2"
```

### Special Matrices

```tcl
;# Identity matrix
set identity [torch::tensor_create {{1 0} {0 1}}]
set result [torch::tensor_svd $identity]
puts "Identity SVD: $result"

;# Zero matrix
set zero [torch::tensor_create {{0 0} {0 0}}]
set result [torch::tensor_svd $zero]
puts "Zero SVD: $result"

;# Symmetric matrix
set symmetric [torch::tensor_create {{2 1} {1 2}}]
set result [torch::tensor_svd $symmetric]
puts "Symmetric SVD: $result"
```

## Error Handling

The command provides clear error messages for various error conditions:

```tcl
# Invalid tensor name
catch {torch::tensor_svd invalid_tensor} result
puts $result  ;# Output: Invalid tensor name

# Missing tensor parameter
catch {torch::tensor_svd} result
puts $result  ;# Output: Required input parameter missing

# Unknown named parameter
catch {torch::tensor_svd -invalid tensor} result
puts $result  ;# Output: Unknown parameter: -invalid

# Missing parameter value
catch {torch::tensor_svd -input} result
puts $result  ;# Output: Missing value for parameter

# Single element tensor (not supported)
set tensor [torch::tensor_create {{5}}]
catch {torch::tensor_svd $tensor} result
puts $result  ;# Output: linalg.svd: input should have at least 2 dimensions

# Integer tensor (not supported)
set tensor [torch::tensor_create {{1 2} {3 4}} int32]
catch {torch::tensor_svd $tensor} result
puts $result  ;# Output: linalg.svd: Expected a floating point or complex tensor
```

## Migration Guide

### From Positional to Named Parameters

**Old code:**
```tcl
set result [torch::tensor_svd $tensor]
```

**New code (equivalent):**
```tcl
set result [torch::tensor_svd -input $tensor]
```

### Using CamelCase Alias

**Old code:**
```tcl
set result [torch::tensor_svd $tensor]
```

**New code (equivalent):**
```tcl
set result [torch::tensorSvd $tensor]
```

## Notes

- The input tensor must be a 2D matrix (at least 2 dimensions)
- The input tensor must be floating point (float32, float64) or complex
- For an m×n matrix, the SVD returns:
  - U: m×m orthogonal matrix (left singular vectors)
  - S: min(m,n) singular values (diagonal elements)
  - V: n×n orthogonal matrix (right singular vectors)
- The singular values are returned in descending order
- The command supports tensors of various floating point data types
- The result format is consistent: `{U u_handle S s_handle V v_handle}`
- SVD is useful for matrix factorization, dimensionality reduction, and solving linear systems

## Mathematical Properties

The SVD decomposition satisfies: A = U × Σ × V^T

Where:
- A is the input matrix
- U contains the left singular vectors
- Σ (S) contains the singular values on the diagonal
- V^T is the transpose of V (right singular vectors)

## See Also

- `torch::tensor_eigen` - Eigenvalue decomposition
- `torch::tensor_qr` - QR decomposition
- `torch::tensor_cholesky` - Cholesky decomposition
- `torch::tensor_pinv` - Moore-Penrose pseudoinverse
- `torch::tensor_matrix_exp` - Matrix exponential 