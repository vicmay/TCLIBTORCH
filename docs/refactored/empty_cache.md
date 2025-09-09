# torch::empty_cache

Releases all unoccupied cached memory currently held by the caching allocator so that it can be used in other GPU applications.

## Syntax

### Current Syntax
```tcl
torch::empty_cache ?device?
```

### Named Parameter Syntax  
```tcl
torch::empty_cache -device device
```

### CamelCase Alias
```tcl
torch::emptyCache ?device?
torch::emptyCache -device device
```

All syntaxes are fully supported and equivalent.

## Parameters

### Named Parameters
- `-device` (optional): Device specification (e.g., "cpu", "cuda", "cuda:0")

### Positional Parameters
1. `device` (optional): Device specification

## Description

The `torch::empty_cache` function releases all unoccupied cached memory currently held by the CUDA caching allocator. This can be useful for reducing GPU memory usage when memory is at a premium, or when you want to ensure that cached memory is available for other GPU applications.

This command primarily affects CUDA devices. For CPU operations, it has minimal effect since CPU memory management is handled differently.

## Return Values

The function returns one of the following strings:
- `"cache_cleared"`: Cache was successfully cleared
- `"cuda_not_available"`: CUDA is not available on the system
- `"cache_clear_attempted"`: Cache clearing was attempted but may not have completed fully

## Examples

### Basic Usage

#### Positional Syntax
```tcl
# Clear cache for all devices
set result [torch::empty_cache]
puts $result  # Outputs: cache_cleared, cuda_not_available, or cache_clear_attempted

# Clear cache for specific device
set result [torch::empty_cache cuda]
```

#### Named Parameter Syntax
```tcl
# Clear cache for all devices
set result [torch::empty_cache]

# Clear cache for specific device
set result [torch::empty_cache -device cuda]
set result [torch::empty_cache -device cuda:0]
```

#### CamelCase Alias
```tcl
# Same functionality with camelCase
set result [torch::emptyCache]
set result [torch::emptyCache -device cuda]
```

### Memory Management Workflow

```tcl
# Create large tensors
set large_tensor1 [torch::randn -shape {10000 10000}]
set large_tensor2 [torch::randn -shape {5000 5000}]

# Perform operations
set result [torch::matmul $large_tensor1 $large_tensor2]

# Clear tensors (remove references)
unset large_tensor1 large_tensor2 result

# Empty cache to release GPU memory
set status [torch::empty_cache]
puts "Cache status: $status"
```

### Device-Specific Cache Clearing

```tcl
# Clear cache for specific CUDA device
set result [torch::empty_cache -device cuda:0]

# Clear cache for all CUDA devices
set result [torch::empty_cache -device cuda]

# CPU cache clearing (limited effect)
set result [torch::empty_cache -device cpu]
```

### Integration with Memory Monitoring

```tcl
# Check memory before clearing
set before_stats [torch::memory_stats]

# Clear cache
set clear_result [torch::empty_cache]

# Check memory after clearing
set after_stats [torch::memory_stats]

puts "Cache clear result: $clear_result"
puts "Memory before: $before_stats"
puts "Memory after: $after_stats"
```

## When to Use

### Recommended Use Cases

1. **Memory Pressure**: When GPU memory is at capacity and you need to free up space
2. **Between Training Phases**: After completing a training epoch or experiment
3. **Application Transitions**: When switching between different models or tasks
4. **Memory Debugging**: To ensure accurate memory usage measurements
5. **Resource Sharing**: Before yielding GPU resources to other applications

### Not Recommended

1. **During Active Training**: Don't call frequently during training as it can hurt performance
2. **Small Memory Operations**: Overhead may not be worth the small memory gains
3. **CPU-Only Workloads**: Limited benefit for CPU-only operations

## Performance Considerations

- **No Performance Impact**: The operation itself is fast
- **Allocation Overhead**: Future allocations may be slower initially as cache rebuilds
- **Fragmentation**: Can help with memory fragmentation issues
- **Frequency**: Avoid calling too frequently as it negates cache benefits

## Error Handling

The function handles errors gracefully:
- Invalid device specifications will still attempt to clear general cache
- CUDA unavailability is reported as a status, not an error
- Unknown parameters will raise an error

```tcl
# Error handling example
if {[catch {torch::empty_cache -device invalid} result]} {
    puts "Error: $result"
} else {
    puts "Result: $result"
}
```

## System Requirements

- **CUDA Support**: Most effective with CUDA-enabled builds
- **GPU Memory**: Only affects GPU memory, not system RAM
- **Permissions**: No special permissions required

## Related Functions

- `torch::memory_stats` - Get memory usage statistics
- `torch::memory_summary` - Get memory usage summary
- `torch::memory_snapshot` - Take memory usage snapshot
- `torch::synchronize` - Synchronize CUDA operations

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax (still supported)
set result [torch::empty_cache]
set result [torch::empty_cache cuda]

# New named parameter syntax
set result [torch::empty_cache]
set result [torch::empty_cache -device cuda]

# Both produce identical results
```

### Benefits of Named Parameters

1. **Clarity**: Parameter purpose is explicit
2. **Flexibility**: Parameters can be provided in any order  
3. **Extensibility**: Easy to add optional parameters in the future
4. **Consistency**: Matches modern TCL conventions

## Best Practices

### Memory Management Strategy

```tcl
proc cleanup_gpu_memory {} {
    # 1. Remove tensor references
    # (This should be done by your application logic)
    
    # 2. Synchronize to ensure operations complete
    torch::synchronize
    
    # 3. Clear cache
    set result [torch::empty_cache]
    
    return $result
}
```

### Monitoring Memory Usage

```tcl
proc monitor_memory_usage {operation_name} {
    set before [torch::memory_stats]
    
    # Perform your operations here
    eval $operation_name
    
    set after_ops [torch::memory_stats]
    
    # Clear cache and check again
    torch::empty_cache
    set after_clear [torch::memory_stats]
    
    puts "Memory usage for $operation_name:"
    puts "  Before: $before"
    puts "  After ops: $after_ops" 
    puts "  After clear: $after_clear"
}
```

## Technical Notes

- Implements CUDA caching allocator empty_cache functionality
- Thread-safe operation
- Does not affect tensors currently in use
- Only releases unoccupied cached memory
- May improve memory fragmentation in some cases

## Version History

- Added dual syntax support in refactoring phase
- Original positional syntax maintained for backward compatibility
- Named parameter syntax added for modern TCL conventions
- CamelCase alias (emptyCache) provided for consistency 