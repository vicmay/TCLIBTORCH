# torch::tensor_eigen

Computes the eigenvalues and eigenvectors of a real symmetric matrix (or batch of matrices) using the symmetric/Hermitian eigendecomposition (linalg.eigh).

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_eigen tensor
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_eigen -input tensor
torch::tensor_eigen -tensor tensor
```

### CamelCase Alias
```tcl
torch::tensorEigen tensor
torch::tensorEigen -input tensor
torch::tensorEigen -tensor tensor
```

## Parameters

| Parameter | Type   | Required | Description                                 |
|-----------|--------|----------|---------------------------------------------|
| tensor    | string | Yes      | Name of the input tensor (must be square)   |

## Return Value

Returns a Tcl list of the form:
```
{eigenvalues <handle> eigenvectors <handle>}
```
- `<handle>` is a tensor handle for the eigenvalues (1D tensor) or eigenvectors (2D tensor).

## Description

- Computes the eigenvalues and eigenvectors of a real symmetric matrix (or batch of matrices).
- Uses the lower triangle of the matrix.
- The input tensor must be square and symmetric.
- The eigenvalues are returned in ascending order.

## Examples

### Basic Usage
```tcl
set tensor [torch::tensor_create -data {4 1 1 4} -shape {2 2}]
set result [torch::tensor_eigen $tensor]
set eigenvalues [lindex $result 1]
set eigenvectors [lindex $result 3]
set vals [torch::tensor_to_list -input $eigenvalues]
set vecs [torch::tensor_to_list -input $eigenvectors]
puts "Eigenvalues: $vals"
puts "Eigenvectors: $vecs"
```

### Named Parameter Syntax
```tcl
set tensor [torch::tensor_create -data {2 1 1 2} -shape {2 2}]
set result [torch::tensor_eigen -input $tensor]
```

### CamelCase Alias
```tcl
set tensor [torch::tensor_create -data {1 0 0 1} -shape {2 2}]
set result [torch::tensorEigen $tensor]
```

## Error Handling

- If the input tensor is not found:
  ```tcl
  catch {torch::tensor_eigen invalid_tensor} msg
  puts $msg  ;# Output: Invalid tensor name
  ```
- If the input tensor is not square:
  ```tcl
  set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
  catch {torch::tensor_eigen $tensor} msg
  puts $msg  ;# Output: linalg.eigh: A must be batches of square matrices, ...
  ```
- If required parameters are missing:
  ```tcl
  catch {torch::tensor_eigen} msg
  puts $msg  ;# Output: Required parameter missing: -input
  ```
- If an unknown parameter is provided:
  ```tcl
  set tensor [torch::tensor_create -data {1 2 3 4} -shape {2 2}]
  catch {torch::tensor_eigen -unknown $tensor} msg
  puts $msg  ;# Output: Unknown parameter: -unknown
  ```

## Migration Guide

**Old Code:**
```tcl
set result [torch::tensor_eigen $tensor]
```
**New Code (Optional):**
```tcl
set result [torch::tensor_eigen -input $tensor]
```
**CamelCase:**
```tcl
set result [torch::tensorEigen $tensor]
```

## Related Commands
- `torch::tensor_cholesky` — Cholesky decomposition
- `torch::tensor_qr` — QR decomposition
- `torch::tensor_matrix_exp` — Matrix exponential
- `torch::tensor_create` — Create a tensor
- `torch::tensor_to_list` — Convert tensor to Tcl list

## Notes
- Only real symmetric (or Hermitian) matrices are supported.
- The returned eigenvectors are orthonormal.
- The returned tensor handles are new tensors in the storage.
- The operation is performed using PyTorch's `torch.linalg.eigh`. 