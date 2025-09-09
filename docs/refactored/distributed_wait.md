# torch::distributed_wait

Wait for a non-blocking distributed operation to complete and retrieve the result.

## Syntax

### Modern Syntax (Recommended)
```tcl
torch::distributed_wait -handle handle_string
torch::distributedWait -handle handle_string
```

### Legacy Syntax (Backward Compatibility)
```tcl
torch::distributed_wait handle_string
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| handle | string | Yes | Handle string returned by a non-blocking operation |

### Parameter Details

- **handle**: The handle string returned by non-blocking operations such as `torch::distributed_isend` or `torch::distributed_irecv`. This handle is used to identify the specific operation to wait for.

## Return Values

The return value depends on the type of operation the handle represents:

### For Send Operations (handles containing "isend")
- Returns: `"send_completed"` - String indicating the send operation has completed

### For Receive Operations (handles containing "irecv")
- Returns: Tensor handle (e.g., `"tensor0"`) - The received tensor data

### For Other Operations
- Returns: `"operation_completed"` - String indicating the operation has completed

## Examples

### Basic Usage

```tcl
# Modern syntax
set result [torch::distributed_wait -handle "isend_handle_1"]
puts "Send operation result: $result"

# Legacy syntax (backward compatibility)
set result [torch::distributed_wait "isend_handle_1"]
puts "Send operation result: $result"

# camelCase alias
set result [torch::distributedWait -handle "isend_handle_1"]
puts "Send operation result: $result"
```

### Waiting for Send Operations

```tcl
# Start a non-blocking send operation
set tensor [torch::tensor_create -data [list 1.0 2.0 3.0 4.0] -shape [list 2 2]]
set send_handle [torch::distributed_isend -tensor $tensor -dst 1 -tag 0]

# Wait for the send operation to complete
set result [torch::distributed_wait -handle $send_handle]
puts "Send completed: $result"  # Output: "send_completed"
```

### Waiting for Receive Operations

```tcl
# Start a non-blocking receive operation
set recv_handle [torch::distributed_irecv -shape [list 2 2] -src 0 -tag 0]

# Wait for the receive operation to complete and get the tensor
set received_tensor [torch::distributed_wait -handle $recv_handle]
puts "Received tensor: $received_tensor"  # Output: tensor handle like "tensor0"

# Use the received tensor
set tensor_data [torch::tensor_get -tensor $received_tensor -data]
puts "Tensor data: $tensor_data"
```

### Waiting for Multiple Operations

```tcl
# Start multiple non-blocking operations
set tensor1 [torch::tensor_create -data [list 1.0 2.0] -shape [list 2]]
set tensor2 [torch::tensor_create -data [list 3.0 4.0] -shape [list 2]]

set send_handle1 [torch::distributed_isend -tensor $tensor1 -dst 1 -tag 0]
set send_handle2 [torch::distributed_isend -tensor $tensor2 -dst 2 -tag 0]
set recv_handle [torch::distributed_irecv -shape [list 2] -src 0 -tag 0]

# Wait for all operations to complete
set send_result1 [torch::distributed_wait -handle $send_handle1]
set send_result2 [torch::distributed_wait -handle $send_handle2]
set received_tensor [torch::distributed_wait -handle $recv_handle]

puts "Send 1: $send_result1"       # "send_completed"
puts "Send 2: $send_result2"       # "send_completed"
puts "Received: $received_tensor"  # tensor handle
```

### Error Handling

```tcl
# Wait with error handling
try {
    set result [torch::distributed_wait -handle "invalid_handle"]
    puts "Operation result: $result"
} on error {err} {
    puts "Error waiting for operation: $err"
}
```

## Error Handling

The function validates inputs and provides clear error messages:

```tcl
# Missing handle parameter
catch {torch::distributed_wait} msg
puts $msg  ;# "Wrong number of arguments..."

# Invalid parameter name
catch {torch::distributed_wait -invalid_param "value"} msg
puts $msg  ;# "Unknown parameter: -invalid_param"

# Missing value for parameter
catch {torch::distributed_wait -handle} msg
puts $msg  ;# "Missing value for parameter"
```

## Migration Guide

### From Legacy to Modern Syntax

```tcl
# Legacy syntax
set result [torch::distributed_wait "my_handle"]

# Modern equivalent
set result [torch::distributed_wait -handle "my_handle"]
```

### Benefits of Modern Syntax

1. **Explicit Parameters**: Clear parameter names improve code readability
2. **IDE Support**: Better autocomplete and parameter hints
3. **Validation**: Enhanced parameter validation and error messages
4. **Consistency**: Uniform API across all LibTorch TCL commands

## Handle Type Detection

The function automatically detects the operation type based on the handle string:

```tcl
# Send operations (handles containing "isend")
set send_result [torch::distributed_wait "isend_handle_dst1_tag0"]
# Returns: "send_completed"

# Receive operations (handles containing "irecv")
set recv_result [torch::distributed_wait "irecv_handle_src0_tag0"]
# Returns: tensor handle like "tensor0"

# Other operations
set other_result [torch::distributed_wait "generic_handle"]
# Returns: "operation_completed"
```

## Common Use Cases

### 1. Synchronous Communication Pattern

```tcl
# Send data and wait for completion
set tensor [torch::tensor_create -data [list 1.0 2.0 3.0] -shape [list 3]]
set send_handle [torch::distributed_isend -tensor $tensor -dst 1]
set result [torch::distributed_wait -handle $send_handle]
puts "Data sent successfully: $result"
```

### 2. Asynchronous Receive Pattern

```tcl
# Start receive and wait for data
set recv_handle [torch::distributed_irecv -shape [list 3] -src 0]
set received_tensor [torch::distributed_wait -handle $recv_handle]
puts "Received tensor: $received_tensor"
```

### 3. Batch Operations

```tcl
# Wait for multiple operations in sequence
set handles [list "isend_handle_1" "isend_handle_2" "irecv_handle_1"]
set results [list]

foreach handle $handles {
    set result [torch::distributed_wait -handle $handle]
    lappend results $result
}

puts "All operations completed: $results"
```

### 4. Pipeline Processing

```tcl
# Pipeline with overlapping operations
set recv_handle [torch::distributed_irecv -shape [list 2 2] -src 0]
set input_tensor [torch::distributed_wait -handle $recv_handle]

# Process the data
set processed_tensor [torch::tensor_mul -tensor1 $input_tensor -tensor2 2.0]

# Send the result
set send_handle [torch::distributed_isend -tensor $processed_tensor -dst 1]
set send_result [torch::distributed_wait -handle $send_handle]
```

## Performance Considerations

- **Blocking Operation**: This function blocks until the operation completes
- **Use with distributed_test**: For non-blocking checks, use `torch::distributed_test` first
- **Resource Management**: Automatically handles cleanup of completed operations
- **Memory Efficiency**: Receives operations return tensor handles for efficient memory use

## Implementation Notes

- The current implementation provides simulation results for testing purposes
- In a real distributed environment, this would wait for actual network operations
- Handle strings are case-sensitive for operation type detection
- The function is thread-safe and can be called from any thread

## Integration Examples

### With torch::distributed_test

```tcl
# Test first, then wait if needed
set handle [torch::distributed_isend -tensor $tensor -dst 1]
set is_complete [torch::distributed_test -handle $handle]
if {$is_complete ne "true"} {
    # Not complete yet, wait for it
    set result [torch::distributed_wait -handle $handle]
    puts "Operation completed: $result"
}
```

### With torch::distributed_isend

```tcl
# Complete send workflow
set tensor [torch::tensor_create -data [list 1.0 2.0 3.0] -shape [list 3]]
set handle [torch::distributed_isend -tensor $tensor -dst 1 -tag 5]
set result [torch::distributed_wait -handle $handle]
puts "Send completed: $result"
```

### With torch::distributed_irecv

```tcl
# Complete receive workflow
set handle [torch::distributed_irecv -shape [list 2 2] -src 0 -tag 5]
set received_tensor [torch::distributed_wait -handle $handle]
puts "Received tensor: $received_tensor"

# Use the received tensor
set data [torch::tensor_get -tensor $received_tensor -data]
puts "Tensor data: $data"
```

## Troubleshooting

### Common Issues

1. **Wrong Return Type**: Make sure to handle different return types based on operation
2. **Handle Validation**: Ensure handles are valid strings from non-blocking operations
3. **Case Sensitivity**: Handle strings are case-sensitive for operation detection
4. **Resource Cleanup**: Received tensors should be properly managed

### Debug Examples

```tcl
# Check handle type
set handle "isend_handle_1"
if {[string match "*isend*" $handle]} {
    puts "This is a send operation"
    set result [torch::distributed_wait -handle $handle]
    # Expect: "send_completed"
} elseif {[string match "*irecv*" $handle]} {
    puts "This is a receive operation"
    set result [torch::distributed_wait -handle $handle]
    # Expect: tensor handle
}
```

## Related Commands

- [`torch::distributed_test`](distributed_test.md) - Test if non-blocking operation is complete
- [`torch::distributed_isend`](distributed_isend.md) - Non-blocking send operation
- [`torch::distributed_irecv`](distributed_irecv.md) - Non-blocking receive operation
- [`torch::distributed_send`](distributed_send.md) - Blocking send operation
- [`torch::distributed_recv`](distributed_recv.md) - Blocking receive operation

## See Also

- [Distributed Operations Guide](../distributed_operations.md)
- [Non-blocking Operations Tutorial](../tutorials/non_blocking_operations.md)
- [Error Handling Best Practices](../error_handling.md)
- [Tensor Management Guide](../tensor_management.md) 