# torch::get_world_size

Get the total number of processes in distributed training.

## Syntax

### Standard Syntax
```tcl
torch::get_world_size
```

### CamelCase Alias
```tcl
torch::getWorldSize
```

## Parameters

This command takes no parameters.

## Return Value

Returns the total number of processes (world size) as an integer.

## Description

The `torch::get_world_size` command retrieves the total number of processes in a distributed training setup. This is useful for:

- Determining the total number of processes in multi-GPU/multi-node training
- Calculating per-process data sharding
- Implementing distributed algorithms that need to know the total process count
- Coordinating work distribution across all processes

In distributed training, the world size represents the total number of processes participating in the training. This is typically equal to the number of GPUs or nodes in the distributed setup.

In single-process (non-distributed) mode, the world size is always 1.

## Examples

### Basic Usage
```tcl
# Get total number of processes
set world_size [torch::get_world_size]
puts "Total processes: $world_size"
;# Output: Total processes: 1 (in single-process mode)
```

### Using CamelCase Alias
```tcl
# Get world size using camelCase
set world_size [torch::getWorldSize]
puts "World size: $world_size"
;# Output: World size: 1 (in single-process mode)
```

### Distributed Training Setup
```tcl
# Initialize distributed training with 4 processes
torch::distributed_init -rank 0 -worldSize 4 -masterAddr "127.0.0.1" -masterPort 29500

# Get rank and world size
set rank [torch::get_rank]
set world_size [torch::get_world_size]

puts "Process $rank of $world_size"
;# Output: Process 0 of 4
```

### Data Sharding Based on World Size
```tcl
# Distribute data across all processes
set world_size [torch::get_world_size]
set rank [torch::get_rank]

# Calculate data range for each process
set total_samples 10000
set samples_per_process [expr {$total_samples / $world_size}]
set start_idx [expr {$rank * $samples_per_process}]
set end_idx [expr {($rank + 1) * $samples_per_process}]

puts "World size: $world_size"
puts "Rank $rank processing samples $start_idx to [expr {$end_idx - 1}]"
puts "Each process handles $samples_per_process samples"
```

### Gradient Averaging in Distributed Training
```tcl
# Average gradients across all processes
set world_size [torch::get_world_size]

# Sum gradients across all processes
torch::distributed_all_reduce $gradients -operation sum

# Average by dividing by world size
set averaged_gradients [torch::tensor_div $gradients $world_size]

puts "Averaged gradients across $world_size processes"
```

### Dynamic Batch Size Based on World Size
```tcl
# Adjust batch size based on number of processes
set world_size [torch::get_world_size]
set base_batch_size 32
set effective_batch_size [expr {$base_batch_size * $world_size}]

puts "Base batch size: $base_batch_size"
puts "World size: $world_size"
puts "Effective batch size: $effective_batch_size"

# Each process handles base_batch_size samples
# Total effective batch size is base_batch_size * world_size
```

### Process Validation
```tcl
# Validate distributed setup
set rank [torch::get_rank]
set world_size [torch::get_world_size]

puts "=== Distributed Setup Validation ==="
puts "Rank: $rank"
puts "World Size: $world_size"

# Check if rank is valid
if {$rank >= 0 && $rank < $world_size} {
    puts "✓ Valid rank: $rank ∈ [0, $world_size)"
} else {
    puts "✗ Invalid rank: $rank not in [0, $world_size)"
}

# Check if world size is reasonable
if {$world_size > 0 && $world_size <= 1000} {
    puts "✓ Valid world size: $world_size"
} else {
    puts "✗ Invalid world size: $world_size"
}
```

### Training Loop with World Size Awareness
```tcl
# Training loop that scales with world size
set world_size [torch::get_world_size]
set rank [torch::get_rank]

# Scale learning rate with world size
set base_lr 0.01
set scaled_lr [expr {$base_lr * $world_size}]

puts "Base learning rate: $base_lr"
puts "Scaled learning rate: $scaled_lr (for $world_size processes)"

for {set epoch 0} {$epoch < 10} {incr epoch} {
    # Only rank 0 prints progress
    if {$rank == 0} {
        puts "Epoch $epoch: Training with $world_size processes"
    }
    
    # All ranks do training
    # ... training logic with scaled_lr ...
    
    # Synchronize gradients across all processes
    torch::distributed_all_reduce $gradients
    
    # Average gradients
    set averaged_gradients [torch::tensor_div $gradients $world_size]
}
```

### Multi-GPU Resource Allocation
```tcl
# Allocate resources based on world size
set world_size [torch::get_world_size]
set rank [torch::get_rank]

if {$world_size > 1} {
    puts "Multi-GPU distributed training:"
    puts "  Total GPUs: $world_size"
    puts "  Current GPU: $rank"
    puts "  Device: cuda:$rank"
    
    # Each process uses a different GPU
    set device "cuda:$rank"
    
    # Calculate memory usage per GPU
    set total_memory_gb 16
    set memory_per_gpu [expr {$total_memory_gb / $world_size}]
    puts "  Memory per GPU: ${memory_per_gpu}GB"
} else {
    puts "Single-process training:"
    puts "  Processes: $world_size"
    puts "  Device: cpu"
    
    set device "cpu"
}
```

### Collective Operations Setup
```tcl
# Setup collective operations based on world size
set world_size [torch::get_world_size]
set rank [torch::get_rank]

puts "Setting up collective operations for $world_size processes"

# All-reduce operation
if {$world_size > 1} {
    puts "Rank $rank: Performing all-reduce across $world_size processes"
    torch::distributed_all_reduce $tensor
} else {
    puts "Single process: Skipping all-reduce"
}

# Broadcast from rank 0
if {$world_size > 1} {
    puts "Broadcasting from rank 0 to $world_size processes"
    torch::distributed_broadcast $tensor -root 0
} else {
    puts "Single process: Skipping broadcast"
}
```

### Performance Monitoring
```tcl
# Monitor performance across all processes
set world_size [torch::get_world_size]
set rank [torch::get_rank]

# Measure training time per process
set start_time [clock milliseconds]

# ... training code ...

set end_time [clock milliseconds]
set duration [expr {$end_time - $start_time}]

puts "Rank $rank: Training time: ${duration}ms"

# Synchronize all processes
torch::distributed_barrier

if {$rank == 0} {
    puts "All $world_size processes completed training"
}
```

### Error Handling in Distributed Context
```tcl
# Robust error handling for distributed training
set world_size [torch::get_world_size]
set rank [torch::get_rank]

try {
    # Distributed training logic
    puts "Rank $rank of $world_size starting training"
    
    # ... training code ...
    
} catch {error_msg} {
    puts "Error on rank $rank of $world_size: $error_msg"
    
    # Notify other processes of error
    if {$world_size > 1} {
        puts "Rank $rank: Notifying other processes of error"
        # ... error notification logic ...
    }
}
```

## Related Commands

- [`torch::get_rank`](get_rank.md) - Get current process rank
- [`torch::is_distributed`](is_distributed.md) - Check if distributed training is active
- [`torch::distributed_init`](distributed_init.md) - Initialize distributed training
- [`torch::distributed_barrier`](distributed_barrier.md) - Synchronize all processes
- [`torch::distributed_all_reduce`](distributed_all_reduce.md) - Reduce tensors across all processes

## Notes

- In single-process mode, world size is always 1
- World size must be positive and typically ranges from 1 to 1000
- World size should match the number of processes launched in your distributed setup
- The rank of any process is always less than the world size: `rank < world_size`
- World size is constant throughout the training session once distributed training is initialized 