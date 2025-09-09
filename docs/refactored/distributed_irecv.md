# torch::distributed_irecv

Initiate a non-blocking receive operation for distributed training communication.

## Syntax

### snake_case (Original)
```tcl
torch::distributed_irecv shape src ?tag?
```

### camelCase (New)
```tcl
torch::distributedIrecv -shape shape -src src ?-tag tag?
```

## Parameters

### Required Parameters
- **shape** (list of integers): The shape of the tensor to receive
- **src** (integer): The rank of the source process to receive from

### Optional Parameters
- **tag** (integer): Message tag for communication (default: 0)

## Parameter Details

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| shape | list of integers | Yes | - | Tensor shape to receive (e.g., {3 4} for 3x4 tensor) |
| src | integer | Yes | - | Source process rank (must be >= 0) |
| tag | integer | No | 0 | Message tag for matching sends/receives |

## Return Value

Returns a handle string (e.g., "irecv_handle_1") that can be used with `torch::distributed_wait` or `torch::distributed_test` to check completion status or retrieve the received tensor.

## Examples

### Basic Usage

**Positional Syntax:**
```tcl
set handle [torch::distributed_irecv {10} 0]
# Returns: "irecv_handle_1"
```

**Named Parameter Syntax:**
```tcl
set handle [torch::distributed_irecv -shape {10} -src 0]
# Returns: "irecv_handle_1"
```

**camelCase Alias:**
```tcl
set handle [torch::distributedIrecv -shape {10} -src 0]
# Returns: "irecv_handle_1"
```

### Multi-Dimensional Tensors

**2D Tensor:**
```tcl
# Positional
set handle [torch::distributed_irecv {3 4} 1]

# Named parameter
set handle [torch::distributed_irecv -shape {3 4} -src 1]
```

**3D Tensor:**
```tcl
# Positional
set handle [torch::distributed_irecv {2 3 4} 2]

# Named parameter  
set handle [torch::distributed_irecv -shape {2 3 4} -src 2]
```

**High-Dimensional Tensor:**
```tcl
set handle [torch::distributed_irecv -shape {32 3 224 224} -src 0]
set handle [torch::distributed_irecv -shape {64 100 300} -src 1]
```

### With Message Tags

**Positional Syntax:**
```tcl
set handle [torch::distributed_irecv {5 5} 1 100]
```

**Named Parameter Syntax:**
```tcl
set handle [torch::distributed_irecv -shape {5 5} -src 1 -tag 100]
```

### Multiple Simultaneous Receives

```tcl
# Start multiple non-blocking receives
set handle1 [torch::distributed_irecv -shape {10 10} -src 0 -tag 1]
set handle2 [torch::distributed_irecv -shape {20 20} -src 1 -tag 2]
set handle3 [torch::distributed_irecv -shape {30 30} -src 2 -tag 3]

# Later, wait for completion
set tensor1 [torch::distributed_wait $handle1]
set tensor2 [torch::distributed_wait $handle2]
set tensor3 [torch::distributed_wait $handle3]
```

### Common Machine Learning Shapes

```tcl
# Image batch (batch_size, channels, height, width)
set handle [torch::distributed_irecv -shape {32 3 224 224} -src 0]

# Feature matrix (samples, features)
set handle [torch::distributed_irecv -shape {128 512} -src 0]

# Sequence data (batch_size, sequence_length, embedding_dim)
set handle [torch::distributed_irecv -shape {64 100 300} -src 0]

# Transformer hidden states (batch_size, seq_length, hidden_size)
set handle [torch::distributed_irecv -shape {16 512 512} -src 0]
```

## Usage Patterns

### Point-to-Point Communication

```tcl
# Rank 0 receives from rank 1
if {$rank == 0} {
    set handle [torch::distributed_irecv -shape {100 100} -src 1 -tag 42]
    set tensor [torch::distributed_wait $handle]
    puts "Received tensor: $tensor"
}

# Rank 1 sends to rank 0 (using distributed_isend)
if {$rank == 1} {
    set handle [torch::distributed_isend $my_tensor 0 42]
    torch::distributed_wait $handle
    puts "Send completed"
}
```

### Pipeline Communication

```tcl
# Receive from previous stage, process, send to next stage
set recv_handle [torch::distributed_irecv -shape {batch_size feature_dim} -src [expr {$rank - 1}]]
set input_tensor [torch::distributed_wait $recv_handle]

# Process the tensor
set output_tensor [my_processing_function $input_tensor]

# Send to next stage
set send_handle [torch::distributed_isend $output_tensor [expr {$rank + 1}]]
torch::distributed_wait $send_handle
```

### Overlapping Communication and Computation

```tcl
# Start receiving next batch while processing current batch
set recv_handle [torch::distributed_irecv -shape {32 784} -src $data_rank -tag $next_batch_id]

# Process current batch
set result [my_model_forward $current_batch]

# Wait for next batch to arrive
set next_batch [torch::distributed_wait $recv_handle]
```

## Implementation Details

### Non-Blocking Operation

The `distributed_irecv` command returns immediately with a handle, allowing the program to continue execution while the receive operation completes in the background. This enables overlapping of communication and computation for better performance.

### Handle Management

Each `distributed_irecv` call returns a unique handle that must be used with `torch::distributed_wait` or `torch::distributed_test` to:
- Check if the operation has completed (`torch::distributed_test`)
- Wait for completion and retrieve the tensor (`torch::distributed_wait`)

### Message Matching

Receives are matched with sends based on:
1. **Source rank**: Must match the sender's rank
2. **Tag**: Must match the sender's tag
3. **Communicator**: Must be in the same distributed group

### Tensor Shape

The shape parameter specifies the expected dimensions of the incoming tensor. The actual received tensor will have this exact shape, filled with data from the sender.

## Error Handling

The command validates all parameters and provides clear error messages:

### Missing Required Parameters
```tcl
# Error: Missing source
catch {torch::distributed_irecv -shape {3 3}} error
puts $error
# Output: "Required parameters missing or invalid: -shape and -src are required"
```

### Invalid Parameter Types
```tcl
# Error: Invalid source type
catch {torch::distributed_irecv -shape {3 3} -src "not_a_number"} error
puts $error
# Output: "Invalid -src parameter. Must be an integer."

# Error: Invalid tag type
catch {torch::distributed_irecv -shape {3 3} -src 0 -tag "not_a_number"} error
puts $error
# Output: "Invalid -tag parameter. Must be an integer."
```

### Invalid Parameter Values
```tcl
# Error: Negative source rank
catch {torch::distributed_irecv -shape {3 3} -src -1} error
puts $error
# Output: "Required parameters missing or invalid: -shape and -src are required"

# Error: Empty shape
catch {torch::distributed_irecv -shape {} -src 0} error
puts $error
# Output: "Required parameters missing or invalid: -shape and -src are required"
```

### Unknown Parameters
```tcl
# Error: Unknown parameter
catch {torch::distributed_irecv -shape {3 3} -src 0 -unknown value} error
puts $error
# Output: "Unknown parameter: -unknown"
```

### Wrong Argument Count (Positional)
```tcl
# Error: Too few arguments
catch {torch::distributed_irecv {3 3}} error
puts $error
# Output: "Wrong number of arguments for positional syntax. Expected: torch::distributed_irecv shape src ?tag?"
```

## Migration Guide

### From Positional to Named Parameters

**Before (Positional):**
```tcl
set handle [torch::distributed_irecv {128 256} 1 42]
```

**After (Named Parameters):**
```tcl
set handle [torch::distributed_irecv -shape {128 256} -src 1 -tag 42]
```

**After (camelCase):**
```tcl
set handle [torch::distributedIrecv -shape {128 256} -src 1 -tag 42]
```

### Benefits of Named Parameters

1. **Self-documenting**: Parameter names make the code more readable
2. **Flexible ordering**: Parameters can be specified in any order
3. **Optional clarity**: Easy to omit optional parameters
4. **Error prevention**: Less likely to mix up parameter positions

## Best Practices

1. **Use meaningful tags** to distinguish different types of messages
2. **Match tensor shapes** exactly between sender and receiver
3. **Handle communication errors** appropriately in production code
4. **Use non-blocking operations** to overlap communication with computation
5. **Pair with distributed_wait** to retrieve the actual tensor data

## Performance Considerations

1. **Non-blocking advantage**: Allows computation while communication happens
2. **Memory allocation**: Shape parameter allows pre-allocation of receive buffer
3. **Tag overhead**: Simple integer tags have minimal overhead
4. **Handle storage**: Store handles efficiently for batch operations

## Related Commands

- `torch::distributed_isend` - Non-blocking send operation
- `torch::distributed_wait` - Wait for non-blocking operation completion
- `torch::distributed_test` - Test if non-blocking operation is complete
- `torch::distributed_recv` - Blocking receive operation
- `torch::distributed_send` - Blocking send operation

## Compatibility

- **Backward Compatible**: All existing positional syntax continues to work
- **Cross-Platform**: Works on Linux, Windows, and macOS
- **Thread-Safe**: Safe to call from multiple threads
- **Version**: Available in LibTorch TCL Extension 1.0+

## Notes

- This command returns immediately; use `torch::distributed_wait` to get the actual tensor
- The shape parameter determines the size of the receive buffer
- For production multi-GPU training, ensure NCCL libraries are properly installed
- The emulated implementation is suitable for testing distributed training logic
- Message matching is based on source rank and tag parameters 