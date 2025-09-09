# torch::sparse_to_dense / torch::sparseToDense

Converts a sparse tensor to a dense tensor, preserving all values and dimensions.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::sparse_to_dense sparse_tensor
```

### Named Parameter Syntax
```tcl
torch::sparse_to_dense -input sparse_tensor
torch::sparseToDense -input sparse_tensor  ;# camelCase alias
```

## Parameters

- **sparse_tensor** (tensor, required): The input sparse tensor to convert to dense format.

## Return Value

Returns a new dense tensor containing the same values as the input sparse tensor, with zeros in positions not specified in the sparse tensor.

## Examples

### Basic Usage
```tcl
# Create a sparse tensor
set indices [torch::tensor_create {{0 0} {1 1} {2 2}} -dtype "int64"]
set values [torch::tensor_create {1.0 2.0 3.0}]
set sparse [torch::sparse_coo_tensor $indices $values {3 3}]

# Convert to dense using positional syntax
set dense1 [torch::sparse_to_dense $sparse]
;# Result: tensor([[1.0, 0.0, 0.0],
;#                [0.0, 2.0, 0.0],
;#                [0.0, 0.0, 3.0]])

# Convert to dense using named parameter syntax
set dense2 [torch::sparse_to_dense -input $sparse]

# Using camelCase alias
set dense3 [torch::sparseToDense -input $sparse]
```

### Error Handling
```tcl
# Invalid tensor handle
if {[catch {torch::sparse_to_dense invalid_tensor} err]} {
    puts "Error: $err"  ;# Prints: Error: Invalid sparse tensor
}

# Missing input parameter
if {[catch {torch::sparse_to_dense} err]} {
    puts "Error: $err"  ;# Prints usage message
}
```

## See Also

- `torch::sparse_coo_tensor` - Create a sparse COO tensor
- `torch::sparse_tensor_dense` - Alternative command for dense conversion
- `torch::sparse_reshape` - Reshape a sparse tensor
- `torch::sparse_mask` - Apply a mask to a sparse tensor 