# torch::sparse_coalesce

Coalesces a sparse tensor by summing duplicate coordinates and sorting them into a canonical order.

## Syntax

### New Syntax (Named Parameters)
```tcl
torch::sparse_coalesce -input TENSOR
torch::sparseCoalesce -input TENSOR
```

### Legacy Syntax (Positional Parameters) 
```tcl
torch::sparse_coalesce tensor
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| input | Tensor | Required | Input sparse tensor to coalesce |

## Description

The sparse coalesce operation performs two main tasks:
1. Combines duplicate coordinates by summing their corresponding values
2. Sorts the coordinates into a canonical order

This operation is useful for:
- Ensuring a unique representation of sparse tensors
- Optimizing memory usage by combining duplicates
- Preparing tensors for operations that require coalesced inputs

A sparse tensor is considered "coalesced" when:
- It has no duplicate coordinates
- The coordinates are sorted in row-major order
- The values corresponding to the coordinates are combined appropriately

## Examples

### Basic Usage
```tcl
# Create a sparse tensor
set values [torch::tensor_randn -shape {5} -dtype float32]
set indices [torch::tensor_create {0 1 2 3 4} {5} int64]
set sparse_tensor [torch::sparse_coo -values $values -indices $indices -size {10}]

# Named parameter syntax
set coalesced [torch::sparse_coalesce -input $sparse_tensor]

# Legacy positional syntax  
set coalesced [torch::sparse_coalesce $sparse_tensor]

# camelCase alias
set coalesced [torch::sparseCoalesce -input $sparse_tensor]
```

### Working with Duplicate Coordinates
```tcl
# Create a sparse tensor with duplicate coordinates
set values [torch::tensor_create {1.0 2.0 3.0} {3} float32]
set indices [torch::tensor_create {0 0 1} {3} int64]
set sparse_tensor [torch::sparse_coo -values $values -indices $indices -size {5}]

# Coalesce will combine the duplicate coordinates (0) by summing values
set coalesced [torch::sparse_coalesce -input $sparse_tensor]

puts "Coalesced tensor: $coalesced"
```

### Chaining Operations
```tcl
# Create and manipulate sparse tensor
set tensor [torch::sparse_coo -values $values -indices $indices -size {10}]
set transposed [torch::sparse_transpose $tensor 0 1]

# Coalesce after operations that might create duplicates
set result [torch::sparse_coalesce -input $transposed]

puts "Final result: $result"
```

## Return Value

Returns a new coalesced sparse tensor with:
- No duplicate coordinates
- Sorted coordinates in row-major order
- Combined values for any previously duplicate coordinates

## Notes

- **Idempotent**: Coalescing an already coalesced tensor is a no-op
- **Memory**: May temporarily use extra memory during the coalescing process
- **Performance**: Required by some sparse operations for optimal performance
- **Ordering**: Coordinates are sorted in row-major order
- **Values**: Values at duplicate coordinates are summed together

## Error Handling

The function validates:
- Input tensor must exist and be valid
- Input tensor must be a sparse tensor

## Compatibility

✅ **Backward Compatible**: All existing code using positional parameters continues to work  
✅ **Named Parameters**: New code can use clearer `-parameter value` syntax  
✅ **camelCase**: Modern `torch::sparseCoalesce` alias available  

## Migration Guide

```tcl
# Old style → New style
torch::sparse_coalesce $t → torch::sparse_coalesce -input $t

# Modern camelCase
torch::sparse_coalesce $t → torch::sparseCoalesce -input $t
```

## See Also

- `torch::sparse_coo` - Create sparse COO tensor
- `torch::sparse_transpose` - Transpose sparse tensor
- `torch::sparse_add` - Add sparse tensors
- `torch::sparse_to_dense` - Convert sparse to dense tensor
- `torch::sparse_reshape` - Reshape sparse tensor 