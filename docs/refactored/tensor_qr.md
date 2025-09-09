# torch::tensor_qr

Computes the QR decomposition of a matrix.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_qr tensor
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_qr -tensor tensor
```

### CamelCase Alias
```tcl
torch::tensorQr tensor
torch::tensorQr -tensor tensor
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor` | string | required | Name of the input matrix tensor |

## Description

The `torch::tensor_qr` command computes the QR decomposition of a matrix. The QR decomposition factors a matrix A into the product of an orthogonal matrix Q and an upper triangular matrix R, such that A = Q × R.

The command returns a result in the format `{Q tensor_name R tensor_name}` where:
- `Q` is the orthogonal matrix (Q tensor)
- `R` is the upper triangular matrix (R tensor)

**Note**: This command uses the deprecated `torch::qr` function. PyTorch recommends using `torch::linalg::qr` instead, but this command maintains backward compatibility.

## Examples

### Basic QR Decomposition
```tcl
# Create a 2x2 matrix
set matrix [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]

# Compute QR decomposition
set result [torch::tensor_qr $matrix]

# Parse result to get Q and R tensors
set result_str [string trim $result "{}"]
set parts [split $result_str]
set q_tensor [lindex $parts 1]
set r_tensor [lindex $parts 3]

# Get tensor data
set q_data [torch::tensor_to_list $q_tensor]
set r_data [torch::tensor_to_list $r_tensor]

puts "Q matrix: $q_data"
puts "R matrix: $r_data"
# Output:
# Q matrix: -0.3162277936935425 -0.9486833214759827 -0.9486833214759827 0.3162277638912201
# R matrix: -3.1622776985168457 -4.427188396453857 0.0 -0.6324553489685059
```

### Using Named Parameters
```tcl
# Create a 2x2 matrix
set matrix [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]

# Compute QR decomposition with named parameters
set result [torch::tensor_qr -tensor $matrix]

# Parse result
set result_str [string trim $result "{}"]
set parts [split $result_str]
set q_tensor [lindex $parts 1]
set r_tensor [lindex $parts 3]

puts "Q tensor: $q_tensor"
puts "R tensor: $r_tensor"
```

### Using CamelCase Alias
```tcl
# Create a 2x2 matrix
set matrix [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]

# Compute QR decomposition using camelCase alias
set result [torch::tensorQr $matrix]

# Parse result
set result_str [string trim $result "{}"]
set parts [split $result_str]
set q_tensor [lindex $parts 1]
set r_tensor [lindex $parts 3]

puts "Q tensor: $q_tensor"
puts "R tensor: $r_tensor"
```

### 3x3 Matrix QR Decomposition
```tcl
# Create a 3x3 matrix
set matrix [torch::tensor_create {{1.0 2.0 3.0} {4.0 5.0 6.0} {7.0 8.0 9.0}} float32 cpu true]

# Compute QR decomposition
set result [torch::tensor_qr $matrix]

# Parse result
set result_str [string trim $result "{}"]
set parts [split $result_str]
set q_tensor [lindex $parts 1]
set r_tensor [lindex $parts 3]

# Get tensor data
set q_data [torch::tensor_to_list $q_tensor]
set r_data [torch::tensor_to_list $r_tensor]

puts "Q matrix size: [llength $q_data]"
puts "R matrix size: [llength $r_data]"
# Output: Q matrix size: 9, R matrix size: 9
```

### Identity Matrix QR Decomposition
```tcl
# Create identity matrix
set matrix [torch::tensor_create {{1.0 0.0} {0.0 1.0}} float32 cpu true]

# Compute QR decomposition
set result [torch::tensor_qr $matrix]

# Parse result
set result_str [string trim $result "{}"]
set parts [split $result_str]
set q_tensor [lindex $parts 1]
set r_tensor [lindex $parts 3]

# Get tensor data
set q_data [torch::tensor_to_list $q_tensor]
set r_data [torch::tensor_to_list $r_tensor]

puts "Q matrix: $q_data"
puts "R matrix: $r_data"
# Output:
# Q matrix: 1.0 0.0 -0.0 1.0
# R matrix: 1.0 0.0 0.0 1.0
```

### Zero Matrix QR Decomposition
```tcl
# Create zero matrix
set matrix [torch::tensor_create {{0.0 0.0} {0.0 0.0}} float32 cpu true]

# Compute QR decomposition
set result [torch::tensor_qr $matrix]

# Parse result
set result_str [string trim $result "{}"]
set parts [split $result_str]
set q_tensor [lindex $parts 1]
set r_tensor [lindex $parts 3]

# Get tensor data
set q_data [torch::tensor_to_list $q_tensor]
set r_data [torch::tensor_to_list $r_tensor]

puts "Q matrix: $q_data"
puts "R matrix: $r_data"
# Output:
# Q matrix: 1.0 0.0 -0.0 1.0
# R matrix: 0.0 0.0 0.0 0.0
```

## Error Handling

### Missing Tensor
```tcl
set result [catch {torch::tensor_qr nonexistent_tensor} error]
puts $error
# Output: Invalid tensor name
```

### Invalid Arguments
```tcl
# No arguments
set result [catch {torch::tensor_qr} error]
puts $error
# Output: Usage: torch::tensor_qr tensor | torch::tensor_qr -tensor tensor

# Too many arguments
set matrix [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
set result [catch {torch::tensor_qr $matrix extra_arg} error]
puts $error
# Output: Usage: torch::tensor_qr tensor

# Unknown named parameter
set result [catch {torch::tensor_qr -tensor $matrix -unknown param} error]
puts $error
# Output: Unknown parameter: -unknown. Valid parameters are: -tensor

# Missing value for parameter
set result [catch {torch::tensor_qr -tensor} error]
puts $error
# Output: Missing value for parameter
```

## Migration Guide

### From Old Positional Syntax
The old syntax is still fully supported for backward compatibility:

```tcl
# Old syntax (still works)
set result [torch::tensor_qr $matrix]

# New named parameter syntax (recommended)
set result [torch::tensor_qr -tensor $matrix]

# CamelCase alias (also available)
set result [torch::tensorQr -tensor $matrix]
```

### Benefits of Named Parameters
1. **Clarity**: Parameter names make the code more readable
2. **Flexibility**: Parameters can be specified in any order
3. **Maintainability**: Easier to understand and modify
4. **Documentation**: Self-documenting code

## Mathematical Background

The QR decomposition of a matrix A is:
```
A = Q × R
```

Where:
- **Q** is an orthogonal matrix (Q^T × Q = I)
- **R** is an upper triangular matrix

The QR decomposition is useful for:
- Solving linear systems
- Computing eigenvalues
- Least squares problems
- Matrix factorization

## Notes

- The command returns tensor names in the format `{Q tensor_name R tensor_name}`
- The Q matrix is orthogonal (Q^T × Q = I)
- The R matrix is upper triangular
- The decomposition is not unique (signs may vary)
- This command uses the deprecated `torch::qr` function
- For new code, consider using `torch::linalg::qr` instead
- All numeric types are supported
- The command works on both CPU and CUDA tensors

## See Also

- `torch::tensor_svd` - Singular Value Decomposition
- `torch::tensor_eigen` - Eigenvalue decomposition
- `torch::tensor_cholesky` - Cholesky decomposition
- `torch::tensor_create` - Create a new tensor 