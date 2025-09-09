# torch::broadcast_tensors

Broadcasts a list of tensors to a common shape according to PyTorch broadcasting rules.

## Syntax

### Positional (Backward Compatible)
```tcl
torch::broadcast_tensors tensor1 tensor2 [tensor3 ...]
```

### Named Parameters  
```tcl
torch::broadcast_tensors -tensors {tensor1 tensor2 ...}
torch::broadcast_tensors -tensors tensor1  # Single tensor
```

### CamelCase Alias
```tcl
torch::broadcastTensors tensor1 tensor2 [tensor3 ...]
torch::broadcastTensors -tensors {tensor1 tensor2 ...}
```

## Parameters

### Positional Parameters
- `tensor1`, `tensor2`, ... - Variable number of tensor handles to broadcast

### Named Parameters
- `-tensors` - List of tensor handles or single tensor handle to broadcast

## Return Value

Returns a Tcl list of tensor handles representing the broadcasted tensors. The number of returned tensors equals the number of input tensors, and all have the same shape after broadcasting.

## Broadcasting Rules

The broadcasting follows PyTorch rules:
1. Tensors are aligned from the rightmost dimension
2. Dimensions of size 1 can be expanded to any size
3. Missing dimensions are treated as size 1
4. All dimensions must be compatible (equal or one of them is 1)

## Examples

### Basic Broadcasting
```tcl
# Create tensors with different shapes
set t1 [torch::tensor_create {1 2 3} float32]    # Shape: [3]
set t2 [torch::tensor_create {5} float32]        # Shape: [1]

# Broadcast tensors - both syntaxes work
set result1 [torch::broadcast_tensors $t1 $t2]
set result2 [torch::broadcast_tensors -tensors [list $t1 $t2]]
set result3 [torch::broadcastTensors $t1 $t2]

# Extract broadcasted tensors
set broadcasted_t1 [lindex $result1 0]  # Shape: [3], values: [1, 2, 3]
set broadcasted_t2 [lindex $result1 1]  # Shape: [3], values: [5, 5, 5]
```

### Multiple Tensor Broadcasting
```tcl
# Create tensors with compatible shapes
set t1 [torch::tensor_create {1} float32]        # Shape: [1]
set t2 [torch::tensor_create {2} float32]        # Shape: [1]  
set t3 [torch::tensor_create {3} float32]        # Shape: [1]

# Broadcast all three tensors
set result [torch::broadcast_tensors $t1 $t2 $t3]

# All tensors now have the same shape [1]
puts "Number of output tensors: [llength $result]"  # Output: 3
```

### Named Parameter with List
```tcl
# Create multiple tensors
set tensors [list \
    [torch::tensor_create {1 2} float32] \
    [torch::tensor_create {3} float32]   \
    [torch::tensor_create {4} float32]   \
]

# Broadcast using named parameter syntax
set result [torch::broadcast_tensors -tensors $tensors]

# All output tensors will have shape [2]
foreach tensor $result {
    puts "Tensor handle: $tensor"
}
```

### Error Handling
```tcl
# This will fail - incompatible shapes for broadcasting
catch {
    set t1 [torch::tensor_create {1 2} float32]    # Shape: [2]
    set t2 [torch::tensor_create {1 2 3} float32]  # Shape: [3]
    torch::broadcast_tensors $t1 $t2
} error
puts "Error: $error"
```

## Mathematical Notes

- Broadcasting creates views of the original tensors when possible (memory efficient)
- The operation is equivalent to PyTorch's `torch.broadcast_tensors()` function
- Scalar tensors (shape []) can be broadcast to any shape
- Broadcasting is a fundamental operation used internally by many PyTorch operations

## Common Use Cases

1. **Element-wise Operations**: Prepare tensors for operations like addition, multiplication
2. **Neural Networks**: Align tensors for layer computations
3. **Data Processing**: Expand smaller tensors to match batch dimensions
4. **Mathematical Operations**: Prepare tensors for matrix operations

## Error Conditions

- **No tensors provided**: Must provide at least one tensor
- **Invalid tensor handles**: All tensor names must exist in tensor storage
- **Incompatible shapes**: Tensors must follow broadcasting rules
- **Missing parameter values**: Named parameters must have values

## Performance Notes

- Broadcasting is memory efficient when possible (creates views, not copies)
- Large tensors with many broadcast dimensions may use significant memory
- Consider the final tensor sizes when broadcasting large tensors

## See Also

- `torch::tensor_expand` - Expand tensor to specific shape
- `torch::tensor_repeat` - Repeat tensor along dimensions  
- `torch::tensor_add` - Element-wise addition with broadcasting
- `torch::tensor_mul` - Element-wise multiplication with broadcasting

## Implementation Status

- ✅ Dual syntax support (positional + named parameters)
- ✅ CamelCase alias (`torch::broadcastTensors`)
- ✅ Comprehensive test coverage (15 test cases)
- ✅ Error handling and validation
- ✅ Complete documentation 