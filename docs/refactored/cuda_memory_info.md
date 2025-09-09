# torch::cuda_memory_info

Get detailed memory information for a CUDA device including used, free, and total memory.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::cuda_memory_info [device_id]
```

### Named Parameter Syntax  
```tcl
torch::cuda_memory_info [-device_id device_id]
```

### CamelCase Alias
```tcl
torch::cudaMemoryInfo [device_id]
torch::cudaMemoryInfo [-device_id device_id]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| device_id | int | 0 | The CUDA device ID to query (0-based indexing) |

## Returns

Returns a string containing memory information in the format:
```
Device <id> Memory: Used=<used>MB Free=<free>MB Total=<total>MB
```

Where:
- `<id>`: The device ID number
- `<used>`: Amount of memory currently in use (in megabytes)
- `<free>`: Amount of available memory (in megabytes)  
- `<total>`: Total memory capacity of the device (in megabytes)

## Error Handling

- **CUDA Not Available**: Returns error "CUDA not available" if CUDA is not installed or available
- **Invalid Device ID**: Returns error "Invalid device ID" if the specified device ID doesn't exist
- **Invalid Parameters**: Returns error for invalid parameter types or unknown parameters

## Examples

### Basic Usage
```tcl
# Get memory info for default device (device 0)
set info [torch::cuda_memory_info]
puts $info
# Output: Device 0 Memory: Used=1024MB Free=7168MB Total=8192MB
```

### Specific Device
```tcl
# Get memory info for device 1 using positional syntax
set info [torch::cuda_memory_info 1]
puts $info

# Same using named parameter syntax  
set info [torch::cuda_memory_info -device_id 1]
puts $info

# Same using camelCase alias
set info [torch::cudaMemoryInfo 1]
puts $info
```

### Error Handling
```tcl
# Handle case when CUDA is not available
if {[catch {torch::cuda_memory_info} result]} {
    puts "Error: $result"
    # Output: Error: CUDA not available
} else {
    puts "Memory info: $result"
}

# Handle invalid device ID
if {[catch {torch::cuda_memory_info 999} result]} {
    puts "Error: $result"
    # Output: Error: Invalid device ID
}
```

### Memory Monitoring
```tcl
# Monitor memory usage across all devices
set device_count [torch::cuda_device_count]
for {set i 0} {$i < $device_count} {incr i} {
    set info [torch::cuda_memory_info $i]
    puts "Device $i: $info"
}

# Parse memory values for calculations
set info [torch::cuda_memory_info]
regexp {Used=(\d+)MB.*Free=(\d+)MB.*Total=(\d+)MB} $info match used free total
puts "Memory utilization: [expr {($used * 100.0) / $total}]%"
```

### Memory Tracking
```tcl
# Check memory before and after operations
proc check_memory_usage {device_id operation} {
    set before [torch::cuda_memory_info $device_id]
    regexp {Used=(\d+)MB} $before match used_before
    
    # Execute operation
    eval $operation
    
    set after [torch::cuda_memory_info $device_id]
    regexp {Used=(\d+)MB} $after match used_after
    
    set diff [expr {$used_after - $used_before}]
    puts "Memory change: ${diff}MB"
}

# Usage example
check_memory_usage 0 {
    set tensor [torch::tensor_create [list [list 1000 1000]] -dtype float32]
}
```

## Memory Information Details

The returned memory information includes:

1. **Device Number**: Zero-based index of the device
2. **Used Memory**: Currently allocated memory in megabytes
3. **Free Memory**: Available memory for allocation in megabytes
4. **Total Memory**: Total device memory capacity in megabytes

### Memory Calculation
- **Used** + **Free** = **Total** (approximately)
- Small discrepancies may occur due to:
  - Memory fragmentation
  - Driver overhead
  - Rounding to megabytes

### Memory Units
All memory values are reported in megabytes (MB) for consistency and readability.

## Migration from Positional to Named Parameters

### Old Style (Positional)
```tcl
set info [torch::cuda_memory_info 0]
```

### New Style (Named Parameters)
```tcl
set info [torch::cuda_memory_info -device_id 0]
```

### Using CamelCase
```tcl
set info [torch::cudaMemoryInfo -device_id 0]
```

## Related Commands

- **torch::cuda_is_available**: Check if CUDA is available
- **torch::cuda_device_count**: Get the number of available CUDA devices  
- **torch::cuda_device_info**: Get device information including name and compute capability
- **torch::tensor_to**: Move tensors to specific CUDA devices

## Technical Notes

### Device ID Validation
- Device IDs are zero-based (0, 1, 2, ...)
- Must be within the range [0, device_count-1]
- Negative device IDs are invalid

### Memory Synchronization
- The command synchronizes the device before querying memory
- This ensures accurate memory reporting
- May cause slight performance overhead for frequent calls

### CUDA Driver Dependencies
- Requires CUDA runtime to be properly installed
- Uses `cudaMemGetInfo()` for accurate memory reporting
- Memory values come directly from CUDA driver

### Performance Considerations
- Memory queries involve device synchronization
- Suitable for monitoring but avoid in performance-critical loops
- Consider caching results for frequently accessed information

## Implementation Details

This command uses the following CUDA Runtime API functions:

1. **torch::cuda::synchronize()**: Ensures device operations are complete
2. **cudaSetDevice()**: Sets the active device for memory queries
3. **cudaMemGetInfo()**: Retrieves free and total memory information

The dual syntax implementation maintains full backward compatibility while providing modern named parameter access.

## Memory Management Best Practices

### Monitoring Memory Usage
```tcl
# Regular memory checking
proc monitor_gpu_memory {} {
    set device_count [torch::cuda_device_count]
    for {set i 0} {$i < $device_count} {incr i} {
        set info [torch::cuda_memory_info $i]
        regexp {Used=(\d+)MB.*Total=(\d+)MB} $info match used total
        set usage [expr {($used * 100.0) / $total}]
        
        if {$usage > 90.0} {
            puts "Warning: Device $i memory usage is ${usage}%"
        }
    }
}
```

### Memory Leak Detection
```tcl
# Compare memory before and after operations
proc detect_memory_leaks {operations} {
    set before_info [torch::cuda_memory_info]
    regexp {Used=(\d+)MB} $before_info match used_before
    
    foreach op $operations {
        eval $op
    }
    
    set after_info [torch::cuda_memory_info]
    regexp {Used=(\d+)MB} $after_info match used_after
    
    set leak [expr {$used_after - $used_before}]
    if {$leak > 0} {
        puts "Potential memory leak detected: ${leak}MB"
    }
}
``` 