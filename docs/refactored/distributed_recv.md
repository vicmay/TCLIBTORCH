# torch::distributed_recv

## Overview
Performs a blocking point-to-point receive operation from another process in a distributed setting. This function blocks until a matching send operation is completed, making it suitable for synchronous communication patterns.

## Syntax

### Positional Syntax (Backward Compatibility)
```tcl
torch::distributed_recv shape src ?tag?
```

### Named Parameter Syntax (Recommended)
```tcl
torch::distributed_recv -shape shape -src src [-tag tag]
```

### camelCase Alias
```tcl
torch::distributedRecv -shape shape -src src [-tag tag]
```

## Parameters

### Required Parameters
- **shape** (list): Shape of the tensor to receive as a list of integers (e.g., `{2 3}`)
- **src** (integer): Source process rank to receive from (must be ≥ 0)

### Optional Parameters
- **tag** (integer): Communication tag for message identification. Default: `0`

## Return Value
Returns a tensor handle containing the received data. The tensor is initialized with zeros in this simplified implementation.

## Examples

### Basic Usage
```tcl
# Receive a 2x3 tensor from process 0
set received_tensor [torch::distributed_recv {2 3} 0]

# Process the received data
set shape [torch::tensor_shape $received_tensor]
puts "Received tensor shape: $shape"
```

### Named Parameter Syntax
```tcl
# Using named parameters
set tensor [torch::distributed_recv -shape {2 3} -src 0]

# With communication tag
set tensor [torch::distributed_recv -shape {4 4} -src 1 -tag 42]
```

### camelCase Alias
```tcl
# Using camelCase alias
set tensor [torch::distributedRecv -shape {2 3} -src 0]

# With all parameters
set tensor [torch::distributedRecv -shape {5 5} -src 2 -tag 100]
```

### Positional Syntax
```tcl
# Basic positional syntax
set tensor [torch::distributed_recv {2 3} 0]

# With tag
set tensor [torch::distributed_recv {4 4} 1 42]
```

## Communication Patterns

### 1. Simple Send-Receive Pair
```tcl
# Receiver process (rank 1)
set received_data [torch::distributed_recv {10 10} 0]
puts "Received data from rank 0"

# Sender process (rank 0)
set data [torch::randn -shape {10 10}]
torch::distributed_send $data 1
```

### 2. Multiple Receives with Tags
```tcl
# Receive different data from same source with tags
set data1 [torch::distributed_recv {5 5} 0 10]
set data2 [torch::distributed_recv {3 3} 0 20]

puts "Received both tensors with different tags"
```

### 3. Ring Communication
```tcl
# Receive from previous process in ring
set my_rank [torch::distributed_get_rank]
set world_size [torch::distributed_get_world_size]
set prev_rank [expr {($my_rank - 1 + $world_size) % $world_size}]

set received_data [torch::distributed_recv {8 8} $prev_rank]
puts "Received data in ring pattern"
```

## Blocking vs Non-blocking

### Blocking Receive (this function)
```tcl
# This blocks until data is received
set data [torch::distributed_recv {5 5} 0]
puts "Data received - continuing execution"
```

### Non-blocking Alternative
```tcl
# For non-blocking receive, use torch::distributed_irecv
set handle [torch::distributed_irecv -shape {5 5} -src 0]
# Do other work...
set data [torch::distributed_wait $handle]
```

## Error Handling

The function validates all parameters and provides clear error messages:

```tcl
# Invalid shape format
catch {torch::distributed_recv "invalid_shape" 0} error
# Error: Invalid shape format

# Invalid source rank
catch {torch::distributed_recv {2 3} "invalid_src"} error
# Error: Invalid src parameter. Must be an integer.

# Invalid tag
catch {torch::distributed_recv {2 3} 0 "invalid_tag"} error
# Error: Invalid tag parameter. Must be an integer.

# Missing required parameters
catch {torch::distributed_recv -shape {2 3}} error
# Error: Required parameters missing or invalid: -shape and -src are required
```

## Tensor Properties

### Shape Specification
```tcl
# 1D tensor
set tensor1d [torch::distributed_recv {100} 0]

# 2D tensor
set tensor2d [torch::distributed_recv {10 20} 0]

# 3D tensor
set tensor3d [torch::distributed_recv {5 10 15} 0]

# 4D tensor (e.g., batch of images)
set tensor4d [torch::distributed_recv {32 3 224 224} 0]
```

### Data Types
```tcl
# Default data type is float32
set float_tensor [torch::distributed_recv {10 10} 0]

# The received tensor will match the sender's data type
# This is handled automatically by the communication backend
```

## Synchronization Considerations

### Deadlock Prevention
```tcl
# Avoid deadlock - make sure send/recv operations are properly ordered
# BAD: Both processes try to receive first
if {$rank == 0} {
    set data [torch::distributed_recv {5 5} 1]  # Deadlock!
    torch::distributed_send $my_data 1
} else {
    set data [torch::distributed_recv {5 5} 0]  # Deadlock!
    torch::distributed_send $my_data 0
}

# GOOD: One sends first, other receives first
if {$rank == 0} {
    torch::distributed_send $my_data 1
    set data [torch::distributed_recv {5 5} 1]
} else {
    set data [torch::distributed_recv {5 5} 0]
    torch::distributed_send $my_data 0
}
```

### Tag Matching
```tcl
# Tags must match between sender and receiver
# Sender (rank 0):
torch::distributed_send $data 1 42

# Receiver (rank 1):
set data [torch::distributed_recv {10 10} 0 42]  # Tag must match
```

## Performance Considerations

### Message Size
- Larger tensors take more time to transfer
- Consider using non-blocking operations for large messages
- Network bandwidth affects transfer time

### Memory Allocation
- The tensor is allocated before receiving data
- Ensure sufficient memory is available for the specified shape
- Pre-allocated tensor shape must match sender's tensor shape

### Network Topology
- Direct connections between nodes are fastest
- Multi-hop communication adds latency
- Consider network topology when designing communication patterns

## Common Use Cases

### 1. Parameter Server Pattern
```tcl
# Worker receives updated parameters from server
if {$rank != 0} {
    set updated_params [torch::distributed_recv {1000 784} 0]
    # Update local model with received parameters
}
```

### 2. Data Pipeline
```tcl
# Receive processed data from previous stage
if {$stage > 0} {
    set input_data [torch::distributed_recv {batch_size 512} [expr {$stage - 1}]]
    # Process the received data
}
```

### 3. Allreduce Implementation
```tcl
# Receive partial results for reduction
set partial_result [torch::distributed_recv {model_size} $other_rank]
set combined [torch::tensor_add $local_result $partial_result]
```

## Integration with Other Operations

### With Send Operations
```tcl
# Coordinated communication
torch::distributed_send $my_data $partner_rank
set received_data [torch::distributed_recv {expected_shape} $partner_rank]
```

### With Collective Operations
```tcl
# Receive data before collective operation
set data [torch::distributed_recv {10 10} 0]
set result [torch::distributed_all_reduce $data]
```

### With Barriers
```tcl
# Synchronize before receiving
torch::distributed_barrier
set data [torch::distributed_recv {5 5} 0]
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
- **Mixed**: Automatic device handling based on sender's tensor device

## Limitations

- This is a simplified implementation for demonstration purposes
- In a real distributed setting, this would integrate with MPI, NCCL, or Gloo backends
- The actual behavior depends on the underlying distributed communication library
- Current implementation returns a zeros tensor regardless of sender data

## See Also

- [`torch::distributed_send`](distributed_send.md) - Blocking send operation
- [`torch::distributed_irecv`](distributed_irecv.md) - Non-blocking receive operation
- [`torch::distributed_isend`](distributed_isend.md) - Non-blocking send operation
- [`torch::distributed_wait`](distributed_wait.md) - Wait for non-blocking operations
- [`torch::distributed_test`](distributed_test.md) - Test completion of non-blocking operations
- [`torch::distributed_barrier`](distributed_barrier.md) - Synchronization barrier

---

*This documentation covers both the legacy positional syntax and the new named parameter syntax. The named parameter syntax is recommended for new code due to improved readability and maintainability.* 