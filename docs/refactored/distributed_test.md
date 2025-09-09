# torch::distributed_test

Test if a non-blocking distributed operation has completed.

## Syntax

### Modern Syntax (Recommended)
```tcl
torch::distributed_test -handle handle_string
torch::distributedTest -handle handle_string
```

### Legacy Syntax (Backward Compatibility)
```tcl
torch::distributed_test handle_string
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| handle | string | Yes | Handle string returned by a non-blocking operation |

### Parameter Details

- **handle**: The handle string returned by non-blocking operations such as `torch::distributed_isend` or `torch::distributed_irecv`. This handle is used to identify the specific operation to test.

## Return Values

Returns a string indicating the completion status:
- `"true"` - Operation has completed
- `"false"` - Operation is still in progress (in actual distributed environments)

**Note**: In the current implementation, this function always returns `"true"` as a simulation of completed operations.

## Examples

### Basic Usage

```tcl
# Modern syntax
set is_complete [torch::distributed_test -handle "isend_handle_1"]
puts "Operation completed: $is_complete"

# Legacy syntax (backward compatibility)
set is_complete [torch::distributed_test "isend_handle_1"]
puts "Operation completed: $is_complete"

# camelCase alias
set is_complete [torch::distributedTest -handle "isend_handle_1"]
puts "Operation completed: $is_complete"
```

### Integration with Non-blocking Operations

```tcl
# Start a non-blocking send operation
set tensor [torch::tensor_create -data [list 1.0 2.0 3.0 4.0] -shape [list 2 2]]
set send_handle [torch::distributed_isend -tensor $tensor -dst 1 -tag 0]

# Test if the send operation is complete
set is_complete [torch::distributed_test -handle $send_handle]
if {$is_complete eq "true"} {
    puts "Send operation completed successfully"
} else {
    puts "Send operation still in progress"
}
```

### Polling for Completion

```tcl
# Start a non-blocking receive operation
set recv_handle [torch::distributed_irecv -shape [list 2 2] -src 0 -tag 0]

# Poll for completion
set max_attempts 100
set attempt 0
while {$attempt < $max_attempts} {
    set is_complete [torch::distributed_test -handle $recv_handle]
    if {$is_complete eq "true"} {
        puts "Receive operation completed after $attempt attempts"
        break
    }
    incr attempt
    after 10  ;# Wait 10ms before next check
}

if {$attempt == $max_attempts} {
    puts "Receive operation timed out"
}
```

### Error Handling

```tcl
# Test with error handling
try {
    set is_complete [torch::distributed_test -handle "invalid_handle"]
    puts "Test result: $is_complete"
} on error {err} {
    puts "Error testing operation: $err"
}
```

## Error Handling

The function validates inputs and provides clear error messages:

```tcl
# Missing handle parameter
catch {torch::distributed_test} msg
puts $msg  ;# "Wrong number of arguments..."

# Invalid parameter name
catch {torch::distributed_test -invalid_param "value"} msg
puts $msg  ;# "Unknown parameter: -invalid_param"

# Missing value for parameter
catch {torch::distributed_test -handle} msg
puts $msg  ;# "Missing value for parameter"
```

## Migration Guide

### From Legacy to Modern Syntax

```tcl
# Legacy syntax
set is_complete [torch::distributed_test "my_handle"]

# Modern equivalent
set is_complete [torch::distributed_test -handle "my_handle"]
```

### Benefits of Modern Syntax

1. **Explicit Parameters**: Clear parameter names improve code readability
2. **IDE Support**: Better autocomplete and parameter hints
3. **Validation**: Enhanced parameter validation and error messages
4. **Consistency**: Uniform API across all LibTorch TCL commands

## Common Use Cases

### 1. Testing Send Operations

```tcl
# Non-blocking send with completion test
set tensor [torch::tensor_create -data [list 1.0 2.0 3.0 4.0] -shape [list 2 2]]
set handle [torch::distributed_isend -tensor $tensor -dst 1]
set is_complete [torch::distributed_test -handle $handle]
```

### 2. Testing Receive Operations

```tcl
# Non-blocking receive with completion test
set handle [torch::distributed_irecv -shape [list 2 2] -src 0]
set is_complete [torch::distributed_test -handle $handle]
```

### 3. Batch Testing Multiple Operations

```tcl
# Test multiple operations
set handles [list "handle1" "handle2" "handle3"]
set all_complete true
foreach handle $handles {
    set is_complete [torch::distributed_test -handle $handle]
    if {$is_complete ne "true"} {
        set all_complete false
        break
    }
}
puts "All operations complete: $all_complete"
```

## Performance Considerations

- **Lightweight Operation**: Testing completion is a fast, non-blocking operation
- **Frequent Polling**: Safe to call frequently in polling loops
- **No Side Effects**: Testing does not affect the underlying operation

## Implementation Notes

- The current implementation always returns `"true"` for simulation purposes
- In a real distributed environment, this would check actual operation status
- The function is thread-safe and can be called from any thread
- Handles are case-sensitive strings

## Integration Examples

### With torch::distributed_wait

```tcl
# Test first, then wait if needed
set handle [torch::distributed_isend -tensor $tensor -dst 1]
set is_complete [torch::distributed_test -handle $handle]
if {$is_complete ne "true"} {
    # Wait for completion
    torch::distributed_wait $handle
}
```

### With torch::distributed_isend

```tcl
# Complete send workflow
set tensor [torch::tensor_create -data [list 1.0 2.0] -shape [list 2]]
set handle [torch::distributed_isend -tensor $tensor -dst 1 -tag 5]
set is_complete [torch::distributed_test -handle $handle]
puts "Send completed: $is_complete"
```

### With torch::distributed_irecv

```tcl
# Complete receive workflow
set handle [torch::distributed_irecv -shape [list 2 2] -src 0 -tag 5]
set is_complete [torch::distributed_test -handle $handle]
if {$is_complete eq "true"} {
    set result [torch::distributed_wait $handle]
    puts "Received tensor: $result"
}
```

## Related Commands

- [`torch::distributed_wait`](distributed_wait.md) - Wait for completion and retrieve results
- [`torch::distributed_isend`](distributed_isend.md) - Non-blocking send operation
- [`torch::distributed_irecv`](distributed_irecv.md) - Non-blocking receive operation
- [`torch::distributed_send`](distributed_send.md) - Blocking send operation
- [`torch::distributed_recv`](distributed_recv.md) - Blocking receive operation

## See Also

- [Distributed Operations Guide](../distributed_operations.md)
- [Non-blocking Operations Tutorial](../tutorials/non_blocking_operations.md)
- [Error Handling Best Practices](../error_handling.md) 