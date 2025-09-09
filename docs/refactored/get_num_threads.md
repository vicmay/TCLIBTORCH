# torch::get_num_threads

Get the current number of threads used by PyTorch/LibTorch operations.

## Syntax

### Standard Syntax
```tcl
torch::get_num_threads
```

### CamelCase Alias
```tcl
torch::getNumThreads
```

## Parameters

This command takes no parameters.

## Return Value

Returns the current number of threads as an integer.

## Description

The `torch::get_num_threads` command retrieves the current number of threads that PyTorch/LibTorch uses for parallelizing CPU operations. This is useful for:

- Monitoring system configuration
- Debugging performance issues
- Understanding parallel execution behavior
- Verifying thread settings in training scripts

The number of threads affects the performance of CPU-based tensor operations, linear algebra computations, and other CPU-intensive tasks. By default, PyTorch typically uses all available CPU cores, but this can be controlled using `torch::set_num_threads`.

## Examples

### Basic Usage
```tcl
# Get current number of threads
set num_threads [torch::get_num_threads]
puts "Current number of threads: $num_threads"
;# Output: Current number of threads: 8 (example value)
```

### Using CamelCase Alias
```tcl
# Get current number of threads using camelCase
set num_threads [torch::getNumThreads]
puts "Current number of threads: $num_threads"
;# Output: Current number of threads: 8 (example value)
```

### Monitoring Thread Configuration
```tcl
# Check current configuration
puts "Initial thread count: [torch::get_num_threads]"

# Create and perform some operations
set tensor1 [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -dtype float32]
set tensor2 [torch::tensorCreate -data {5.0 6.0 7.0 8.0} -dtype float32]
set result [torch::tensorAdd -input $tensor1 -other $tensor2]

puts "Thread count after operations: [torch::get_num_threads]"
;# Thread count remains the same unless explicitly changed
```

### Thread Configuration Management
```tcl
# Save current thread configuration
set original_threads [torch::get_num_threads]
puts "Original threads: $original_threads"

# Change thread count
torch::set_num_threads 4
puts "After setting to 4: [torch::get_num_threads]"

# Restore original configuration
torch::set_num_threads $original_threads
puts "Restored threads: [torch::get_num_threads]"
```

### Performance Testing with Different Thread Counts
```tcl
# Test performance with different thread counts
set original_threads [torch::get_num_threads]

foreach thread_count {1 2 4 8} {
    torch::set_num_threads $thread_count
    set current_threads [torch::get_num_threads]
    puts "Testing with $current_threads threads"
    
    # Perform some computation
    set start_time [clock microseconds]
    set tensor [torch::zeros -shape {1000 1000} -dtype float32]
    set result [torch::tensorAdd -input $tensor -other $tensor]
    set end_time [clock microseconds]
    
    set duration [expr {($end_time - $start_time) / 1000.0}]
    puts "Time with $thread_count threads: ${duration}ms"
}

# Restore original setting
torch::set_num_threads $original_threads
```

### System Information Display
```tcl
# Display comprehensive system information
puts "=== PyTorch Thread Configuration ==="
puts "Number of threads: [torch::get_num_threads]"
puts "CPU device available: [torch::device_is_available cpu]"

if {[torch::device_is_available cuda]} {
    puts "CUDA device available: Yes"
    puts "CUDA device count: [torch::cuda_device_count]"
} else {
    puts "CUDA device available: No"
}
```

### Training Script Thread Management
```tcl
# Example training script with thread management
proc setup_threading {target_threads} {
    set current_threads [torch::get_num_threads]
    if {$current_threads != $target_threads} {
        puts "Adjusting threads from $current_threads to $target_threads"
        torch::set_num_threads $target_threads
        puts "New thread count: [torch::get_num_threads]"
    } else {
        puts "Thread count already optimal: $current_threads"
    }
}

# Setup for training
setup_threading 8

# Training loop with thread monitoring
for {set epoch 0} {$epoch < 5} {incr epoch} {
    puts "Epoch $epoch - Threads: [torch::get_num_threads]"
    # ... training logic ...
}
```

### Consistency Verification
```tcl
# Verify thread count consistency
set results {}
for {set i 0} {$i < 10} {incr i} {
    lappend results [torch::get_num_threads]
}

set first [lindex $results 0]
set all_consistent 1
foreach result $results {
    if {$result != $first} {
        set all_consistent 0
        break
    }
}

if {$all_consistent} {
    puts "Thread count is consistent: $first"
} else {
    puts "Warning: Thread count is inconsistent: $results"
}
```

## Notes

- The command takes no parameters and returns an integer
- The number of threads is typically set to the number of CPU cores by default
- Thread count affects CPU-bound operations but not GPU operations
- The value remains constant unless explicitly changed with `torch::set_num_threads`
- Both `torch::get_num_threads` and `torch::getNumThreads` return identical results
- Thread count should be positive and typically ranges from 1 to the number of CPU cores
- Setting too many threads can actually decrease performance due to overhead

## Performance Considerations

- **Single-threaded (1 thread)**: May be slower but uses less memory and has predictable behavior
- **Multi-threaded (2-8 threads)**: Usually optimal for most workloads
- **Many threads (>CPU cores)**: May cause overhead and performance degradation
- **Default setting**: Usually matches the number of CPU cores and provides good performance

## See Also

- [torch::set_num_threads](set_num_threads.md) - Set the number of threads
- [torch::device_is_available](device_is_available.md) - Check device availability
- [torch::cuda_device_count](cuda_device_count.md) - Get CUDA device count
- [torch::benchmark](benchmark.md) - Performance benchmarking tools 