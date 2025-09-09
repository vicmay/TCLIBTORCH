# torch::distributed_send

Point-to-point send operation for distributed computing environments.

## Syntax

### Modern Syntax (Named Parameters)
```tcl
torch::distributed_send -tensor tensor_handle -dst destination_rank ?-tag tag_value?
torch::distributedSend -tensor tensor_handle -dst destination_rank ?-tag tag_value?  # camelCase alias
```

### Legacy Syntax (Positional Parameters)
```tcl
torch::distributed_send tensor_handle destination_rank ?tag_value?
torch::distributedSend tensor_handle destination_rank ?tag_value?  # camelCase alias
```

## Parameters

### Required Parameters
- **tensor** (`string`): Handle to the tensor to be sent
- **dst** (`integer`): Destination rank (process ID) to send the tensor to. Must be ≥ 0.

### Optional Parameters  
- **tag** (`integer`): Message tag for matching with corresponding receive operations. Default: 0

## Return Value

Returns `"send_completed"` when the send operation completes successfully.

## Description

The `torch::distributed_send` command performs a blocking point-to-point send operation, transmitting a tensor from the current process to a specified destination process in a distributed computing environment. This is a fundamental communication primitive for distributed training and inference.

The command supports both modern named parameter syntax and legacy positional syntax for backward compatibility. A camelCase alias `torch::distributedSend` is also available for consistency with modern TCL naming conventions.

## Key Features

- **Blocking Operation**: The command blocks until the send operation is completed
- **Type Safety**: Automatically handles different tensor data types (float32, float64, int32, int64, etc.)
- **Shape Agnostic**: Works with tensors of any shape and dimensionality
- **Message Tagging**: Optional tag parameter for message matching
- **Error Validation**: Comprehensive parameter validation and error reporting

## Examples

### Basic Usage (Modern Syntax)

```tcl
# Create a tensor to send
set data [torch::ones -shape {3 4} -dtype float32 -device cpu]

# Send to rank 1 with default tag
torch::distributed_send -tensor $data -dst 1

# Send to rank 2 with specific tag
torch::distributed_send -tensor $data -dst 2 -tag 42
```

### CamelCase Alias

```tcl
# Using camelCase alias - identical functionality
set data [torch::randn -shape {10 10} -dtype float32 -device cpu]
torch::distributedSend -tensor $data -dst 3 -tag 100
```

### Legacy Syntax (Backward Compatibility)

```tcl
# Legacy positional syntax still supported
set data [torch::zeros -shape {5 5} -dtype int32 -device cpu]
torch::distributed_send $data 1        # Send to rank 1, tag 0
torch::distributed_send $data 2 50     # Send to rank 2, tag 50
```

### Parameter Order Independence (Modern Syntax)

```tcl
# Parameters can be specified in any order with named syntax
set data [torch::full -shape {2 3} -value 5.0 -dtype float64 -device cpu]

torch::distributed_send -dst 1 -tensor $data -tag 25
torch::distributed_send -tag 30 -tensor $data -dst 2
torch::distributed_send -tensor $data -dst 3  # tag defaults to 0
```

### Different Data Types

```tcl
# Float32 tensor
set f32_data [torch::ones -shape {4 4} -dtype float32 -device cpu]
torch::distributed_send -tensor $f32_data -dst 1

# Float64 tensor  
set f64_data [torch::ones -shape {4 4} -dtype float64 -device cpu]
torch::distributed_send -tensor $f64_data -dst 1

# Integer tensors
set int32_data [torch::ones -shape {4 4} -dtype int32 -device cpu]
set int64_data [torch::ones -shape {4 4} -dtype int64 -device cpu]
torch::distributed_send -tensor $int32_data -dst 1
torch::distributed_send -tensor $int64_data -dst 1
```

### Complex Scenarios

```tcl
# Multiple sends with different tags
set tensor1 [torch::ones -shape {3 3} -dtype float32 -device cpu]
set tensor2 [torch::zeros -shape {5 5} -dtype float64 -device cpu]

torch::distributed_send -tensor $tensor1 -dst 1 -tag 1
torch::distributed_send -tensor $tensor2 -dst 1 -tag 2

# Large tensor send
set large_tensor [torch::randn -shape {1000 1000} -dtype float32 -device cpu]
torch::distributed_send -tensor $large_tensor -dst 2 -tag 999

# High-dimensional tensor
set multi_dim [torch::ones -shape {2 3 4 5} -dtype float32 -device cpu]
torch::distributed_send -tensor $multi_dim -dst 3
```

## Error Handling

### Parameter Validation Errors

```tcl
# Missing required parameters
torch::distributed_send -dst 1
# Error: Required parameters missing or invalid: -tensor and -dst are required

# Invalid tensor handle
torch::distributed_send -tensor "invalid_handle" -dst 1  
# Error: Invalid tensor handle: invalid_handle

# Invalid destination rank
torch::distributed_send -tensor $data -dst "invalid"
# Error: Invalid -dst parameter. Must be an integer.

# Negative destination rank
torch::distributed_send -tensor $data -dst -1
# Error: Required parameters missing or invalid: -tensor and -dst are required
```

### Syntax Errors

```tcl
# Unknown parameter
torch::distributed_send -tensor $data -dst 1 -unknown param
# Error: Unknown parameter: -unknown

# Missing parameter value
torch::distributed_send -tensor $data -dst
# Error: Missing value for parameter

# Wrong number of arguments (positional syntax)
torch::distributed_send $data
# Error: Wrong number of arguments for positional syntax. Expected: torch::distributed_send tensor dst ?tag?
```

## Migration Guide

### From Legacy to Modern Syntax

```tcl
# OLD (Legacy Positional Syntax)
torch::distributed_send $tensor 1
torch::distributed_send $tensor 1 42

# NEW (Modern Named Parameter Syntax)  
torch::distributed_send -tensor $tensor -dst 1
torch::distributed_send -tensor $tensor -dst 1 -tag 42

# ALTERNATIVE (CamelCase Alias)
torch::distributedSend -tensor $tensor -dst 1
torch::distributedSend -tensor $tensor -dst 1 -tag 42
```

### Benefits of Migration

1. **Self-Documenting**: Parameter names make code more readable
2. **Order Independence**: Parameters can be specified in any order
3. **IDE Support**: Better autocomplete and error detection
4. **Future-Proof**: Easier to extend with new optional parameters
5. **Consistency**: Matches other modern LibTorch TCL commands

### Backward Compatibility

The legacy positional syntax continues to work unchanged:

```tcl
# This syntax will continue to work indefinitely
torch::distributed_send $tensor 1 42
```

## Integration with Other Commands

### With Distributed Receive

```tcl
# Sender process (rank 0)
set data [torch::ones -shape {3 3} -dtype float32 -device cpu]
torch::distributed_send -tensor $data -dst 1 -tag 100

# Receiver process (rank 1)  
set received [torch::distributed_recv -shape {3 3} -src 0 -tag 100]
```

### With Non-blocking Operations

```tcl
# Use with non-blocking operations for advanced patterns
set handle [torch::distributed_isend -tensor $data -dst 1 -tag 50]
torch::distributed_wait $handle
```

## Performance Considerations

1. **Tensor Size**: Larger tensors take longer to transmit
2. **Network Topology**: Communication patterns affect performance
3. **Blocking Nature**: Consider using `torch::distributed_isend` for non-blocking operations
4. **Data Type**: Different data types have different transfer costs

## Best Practices

1. **Use Named Parameters**: Improves code readability and maintainability
2. **Consistent Tagging**: Use meaningful tag values for message matching
3. **Error Handling**: Always wrap distributed operations in try-catch blocks
4. **Documentation**: Document communication patterns in distributed code
5. **Testing**: Test with multiple processes to verify distributed correctness

## Related Commands

- [`torch::distributed_recv`](distributed_recv.md) - Blocking point-to-point receive
- [`torch::distributed_isend`](distributed_isend.md) - Non-blocking point-to-point send  
- [`torch::distributed_irecv`](distributed_irecv.md) - Non-blocking point-to-point receive
- [`torch::distributed_wait`](distributed_wait.md) - Wait for non-blocking operations
- [`torch::distributed_test`](distributed_test.md) - Test completion of non-blocking operations
- [`torch::distributed_gather`](distributed_gather.md) - Gather tensors from all processes
- [`torch::distributed_scatter`](distributed_scatter.md) - Scatter tensor to all processes
- [`torch::distributed_all_reduce`](../distributed_all_reduce.md) - All-reduce operation
- [`torch::distributed_broadcast`](../distributed_broadcast.md) - Broadcast operation

## Technical Notes

### Implementation Details

- The command performs blocking send operations
- Tensor data is automatically serialized for network transmission
- All tensor data types (float32, float64, int32, int64, etc.) are supported
- The operation returns only after the send buffer is safe to reuse

### Thread Safety

The command is thread-safe when used with proper distributed initialization.

### Memory Management

- Tensor handles are managed automatically
- No explicit cleanup is required for sent tensors
- The send operation does not modify the source tensor

---

**Note**: This command requires proper distributed initialization via `torch::distributed_init` for multi-process execution. In single-process mode, it provides a simulation interface for testing and development. 