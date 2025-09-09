# torch::distributed_isend

## Overview
Initiates a non-blocking point-to-point send operation to another process in a distributed setting. This function immediately returns a handle that can be used to wait for the completion of the send operation, allowing for asynchronous communication patterns.

## Syntax

### Positional Syntax (Backward Compatibility)
```tcl
torch::distributed_isend tensor dst ?tag?
```

### Named Parameter Syntax (Recommended)
```tcl
torch::distributed_isend -tensor tensor -dst dst [-tag tag]
```

### camelCase Alias
```tcl
torch::distributedIsend -tensor tensor -dst dst [-tag tag]
```

## Parameters

### Required Parameters
- **tensor** (string): Tensor handle containing the data to send
- **dst** (integer): Destination process rank (must be ≥ 0)

### Optional Parameters
- **tag** (integer): Communication tag for message identification. Default: `0`

## Return Value
Returns a handle string that can be used with `torch::distributed_wait` or `torch::distributed_test` to check the completion status of the send operation.

## Examples

### Basic Usage
```tcl
# Create tensor to send
set my_tensor [torch::randn -shape {3 4}]

# Basic non-blocking send
set handle [torch::distributed_isend $my_tensor 1]

# Wait for completion
torch::distributed_wait $handle
```

### Named Parameter Syntax
```tcl
# Using named parameters
set handle [torch::distributed_isend -tensor $my_tensor -dst 1]

# With communication tag
set handle [torch::distributed_isend -tensor $my_tensor -dst 2 -tag 42]
```

### camelCase Alias
```tcl
# Using camelCase alias
set handle [torch::distributedIsend -tensor $my_tensor -dst 1]

# With all parameters
set handle [torch::distributedIsend -tensor $my_tensor -dst 3 -tag 100]
```

### Positional Syntax
```tcl
# Basic positional syntax
set handle [torch::distributed_isend $my_tensor 1]

# With tag
set handle [torch::distributed_isend $my_tensor 2 42]
```

## Communication Patterns

### 1. Simple Send-Receive Pair
```tcl
# Sender process (rank 0)
set data [torch::randn -shape {10 10}]
set send_handle [torch::distributed_isend $data 1]
torch::distributed_wait $send_handle

# Receiver process (rank 1)
set recv_handle [torch::distributed_irecv -shape {10 10} -src 0]
set received_data [torch::distributed_wait $recv_handle]
```

### 2. Multiple Sends with Tags
```tcl
# Send different data to same destination with tags
set data1 [torch::randn -shape {5 5}]
set data2 [torch::randn -shape {3 3}]

set handle1 [torch::distributed_isend $data1 1 10]
set handle2 [torch::distributed_isend $data2 1 20]

# Wait for both to complete
torch::distributed_wait $handle1
torch::distributed_wait $handle2
```

### 3. Ring Communication
```tcl
# Send to next process in ring
set my_rank [torch::distributed_get_rank]
set world_size [torch::distributed_get_world_size]
set next_rank [expr {($my_rank + 1) % $world_size}]

set data [torch::randn -shape {8 8}]
set handle [torch::distributed_isend $data $next_rank]
torch::distributed_wait $handle
```

## Handle Format

The returned handle encodes the destination and tag information:
- Format: `isend_handle_dst{dst}_tag{tag}`
- Example: `isend_handle_dst1_tag42` for dst=1, tag=42

## Error Handling

The function validates all parameters and provides clear error messages:

```tcl
# Invalid tensor handle
catch {torch::distributed_isend "invalid_tensor" 1} error
# Error: Invalid tensor handle: invalid_tensor

# Invalid destination
catch {torch::distributed_isend $tensor "invalid_dst"} error  
# Error: Invalid dst parameter. Must be an integer.

# Invalid tag
catch {torch::distributed_isend $tensor 1 "invalid_tag"} error
# Error: Invalid tag parameter. Must be an integer.

# Missing required parameters
catch {torch::distributed_isend -tensor $tensor} error
# Error: Required parameters missing or invalid: -tensor and -dst are required
```

## Performance Considerations

### Non-blocking Operations
- `torch::distributed_isend` returns immediately without waiting for completion
- Use `torch::distributed_wait` to ensure completion before proceeding
- Multiple non-blocking operations can be overlapped for better performance

### Memory Management
- The tensor data must remain valid until the send operation completes
- Use `torch::distributed_wait` to synchronize before tensor cleanup

### Tag Usage
- Tags allow multiple messages between same processes to be distinguished
- Use consistent tag values between sender and receiver
- Tags can be any integer value (including negative)

## Synchronization

### Waiting for Completion
```tcl
# Start send operation
set handle [torch::distributed_isend $tensor 1]

# Do other work...
set other_result [compute_something]

# Wait for send to complete
torch::distributed_wait $handle
```

### Testing for Completion
```tcl
# Start send operation
set handle [torch::distributed_isend $tensor 1]

# Check if completed (non-blocking)
set is_complete [torch::distributed_test $handle]
if {$is_complete} {
    puts "Send operation completed"
}
```

## Common Use Cases

### 1. Distributed Training
```tcl
# Send gradients to parameter server
set gradients [compute_gradients $model]
set handle [torch::distributed_isend $gradients 0 $layer_id]
torch::distributed_wait $handle
```

### 2. Data Pipeline
```tcl
# Send processed data to next stage
set processed_data [process_batch $input_data]
set handle [torch::distributed_isend $processed_data $next_stage]
torch::distributed_wait $handle
```

### 3. Collective Operations
```tcl
# Send data for all-reduce operation
set local_data [compute_local_result]
set handle [torch::distributed_isend $local_data $coordinator]
torch::distributed_wait $handle
```

## Integration with Other Distributed Operations

### With Receive Operations
```tcl
# Coordinated send-receive
set send_handle [torch::distributed_isend $send_data 1]
set recv_handle [torch::distributed_irecv -shape {10 10} -src 1]

set received_data [torch::distributed_wait $recv_handle]
torch::distributed_wait $send_handle
```

### With Collective Operations
```tcl
# Send data before collective operation
set handle [torch::distributed_isend $data 0]
torch::distributed_wait $handle
torch::distributed_barrier
```

## Data Type Support

Supports all standard PyTorch tensor data types:
- Float: `float32`, `float64`, `float16`
- Integer: `int32`, `int64`, `int16`, `int8`
- Unsigned: `uint8`
- Boolean: `bool`
- Complex: `complex64`, `complex128`

## Device Compatibility

- **CPU**: Full support
- **CUDA**: Full support with GPU-to-GPU communication
- **Mixed**: Automatic device handling based on tensor device

## Limitations

- This is a simplified implementation for demonstration purposes
- In a real distributed setting, this would integrate with MPI, NCCL, or Gloo backends
- The actual behavior depends on the underlying distributed communication library

## See Also

- [`torch::distributed_irecv`](distributed_irecv.md) - Non-blocking receive operation
- [`torch::distributed_send`](distributed_send.md) - Blocking send operation
- [`torch::distributed_recv`](distributed_recv.md) - Blocking receive operation
- [`torch::distributed_wait`](distributed_wait.md) - Wait for non-blocking operations
- [`torch::distributed_test`](distributed_test.md) - Test completion of non-blocking operations
- [`torch::distributed_barrier`](distributed_barrier.md) - Synchronization barrier

---

*This documentation covers both the legacy positional syntax and the new named parameter syntax. The named parameter syntax is recommended for new code due to improved readability and maintainability.* 