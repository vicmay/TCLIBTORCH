# torch::tensor_pinv

Compute the pseudo-inverse (Moore-Penrose inverse) of a tensor.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_pinv tensor_handle ?rcond?
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_pinv -input tensor_handle ?-rcond rcond?
torch::tensor_pinv -tensor tensor_handle ?-rcond rcond?
```

### CamelCase Alias
```tcl
torch::tensorPinv tensor_handle ?rcond?
torch::tensorPinv -input tensor_handle ?-rcond rcond?
torch::tensorPinv -tensor tensor_handle ?-rcond rcond?
```

## Parameters

| Parameter         | Type   | Required | Description                                 |
|------------------|--------|----------|---------------------------------------------|
| `tensor_handle`  | string | Yes      | The handle of the tensor to compute pseudo-inverse |
| `-rcond`         | double | No       | Cutoff for small singular values (default: 1e-15) |

## Return Value

Returns a tensor handle containing the pseudo-inverse of the input tensor.

- For a 2×2 matrix, returns a 2×2 pseudo-inverse
- For a 2×3 matrix, returns a 3×2 pseudo-inverse
- The result is always a 2D tensor

## Examples

### Basic Usage

```tcl
set t [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
set pinv_result [torch::tensor_pinv $t]
puts "Pseudo-inverse: [torch::tensor_to_list $pinv_result]"
```

### With rcond Parameter

```tcl
set t [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
set pinv_result [torch::tensor_pinv $t 1e-10]
puts "Pseudo-inverse: [torch::tensor_to_list $pinv_result]"
```

### Rectangular Matrix

```tcl
set t [torch::tensor_create {{1.0 2.0 3.0} {4.0 5.0 6.0}} float32 cpu true]
set pinv_result [torch::tensor_pinv $t]
puts "Pseudo-inverse: [torch::tensor_to_list $pinv_result]"
```

### Named Parameter Syntax

```tcl
set t [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
set pinv_result [torch::tensor_pinv -input $t]
puts "Pseudo-inverse: [torch::tensor_to_list $pinv_result]"

set t2 [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
set pinv_result2 [torch::tensor_pinv -input $t2 -rcond 1e-10]
puts "Pseudo-inverse: [torch::tensor_to_list $pinv_result2]"
```

### CamelCase Alias

```tcl
set t [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
set pinv_result [torch::tensorPinv $t]
puts "Pseudo-inverse: [torch::tensor_to_list $pinv_result]"
```

### Error Handling

```tcl
catch {torch::tensor_pinv invalid_tensor} result
puts "Error: $result"  ;# Output: Error: Invalid tensor name

catch {torch::tensor_pinv} result
puts "Error: $result"  ;# Output: Error: Required input parameter missing

set t [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
catch {torch::tensor_pinv -unknown $t} result
puts "Error: $result"  ;# Output: Error: Unknown parameter: -unknown

;# Single element tensor (requires 2+ dimensions)
set t [torch::tensor_create {1.0} float32 cpu true]
catch {torch::tensor_pinv $t} result
puts "Error: $result"  ;# Output: Error: expected a tensor with 2 or more dimensions
```

## Migration Guide

### From Old Syntax to New Syntax

**Old (Positional Only):**
```tcl
set pinv_result [torch::tensor_pinv $tensor]
set pinv_result [torch::tensor_pinv $tensor 1e-10]
```

**New (Named Parameters):**
```tcl
set pinv_result [torch::tensor_pinv -input $tensor]
set pinv_result [torch::tensor_pinv -input $tensor -rcond 1e-10]
```

**New (CamelCase):**
```tcl
set pinv_result [torch::tensorPinv $tensor]
set pinv_result [torch::tensorPinv -input $tensor -rcond 1e-10]
```

### Backward Compatibility

The old positional syntax continues to work:
```tcl
set pinv_result [torch::tensor_pinv $tensor]
```

## Notes

- Both snake_case and camelCase versions are functionally identical
- Works with tensors of 2 or more dimensions
- Returns the Moore-Penrose pseudo-inverse
- The `rcond` parameter controls the cutoff for small singular values
- For square invertible matrices, the pseudo-inverse equals the regular inverse
- For rectangular matrices, the pseudo-inverse provides the best least-squares solution

## Mathematical Background

The pseudo-inverse A⁺ of a matrix A satisfies:
- AA⁺A = A
- A⁺AA⁺ = A⁺
- (AA⁺)ᵀ = AA⁺
- (A⁺A)ᵀ = A⁺A

## Related Commands

- `torch::tensor_inverse` - Matrix inverse (for square matrices)
- `torch::tensor_svd` - Singular value decomposition
- `torch::tensor_qr` - QR decomposition
- `torch::tensor_cholesky` - Cholesky decomposition
- `torch::tensor_create` - Create tensors 