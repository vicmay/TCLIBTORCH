# torch::get_rank

Get the current process rank in distributed training.

## Syntax

### Standard Syntax
```tcl
torch::get_rank
```

### CamelCase Alias
```tcl
torch::getRank
```

## Parameters

This command takes no parameters.

## Return Value

Returns the current process rank as an integer (0-based).

## Description

The `torch::get_rank` command retrieves the current process rank in a distributed training setup. This is useful for:

- Identifying the current process in multi-GPU/multi-node training
- Implementing rank-specific logic in distributed training scripts
- Debugging distributed training issues
- Coordinating work distribution across processes

In distributed training, each process has a unique rank ranging from 0 to `world_size - 1`. The rank identifies which process is running and is used for:
- Process coordination
- Data sharding
- Gradient synchronization
- Output control (typically only rank 0 prints logs)

In single-process (non-distributed) mode, the rank is always 0.

## Examples

### Basic Usage
```tcl
# Get current process rank
set rank [torch::get_rank]
puts "Current process rank: $rank"
;# Output: Current process rank: 0 (in single-process mode)
```

### Using CamelCase Alias
```tcl
# Get current process rank using camelCase
set rank [torch::getRank]
puts "Current process rank: $rank"
;# Output: Current process rank: 0 (in single-process mode)
```

### Distributed Training Setup
```tcl
# Initialize distributed training
torch::distributed_init -rank 0 -worldSize 4 -masterAddr "127.0.0.1" -masterPort 29500

# Get rank and world size
set rank [torch::get_rank]
set world_size [torch::get_world_size]

puts "Process $rank of $world_size"
;# Output: Process 0 of 4
```

### Rank-Specific Logic
```tcl
# Get current rank
set rank [torch::get_rank]

if {$rank == 0} {
    puts "This is the master process (rank 0)"
    ;# Master process logic - logging, checkpointing, etc.
} else {
    puts "This is worker process rank $rank"
    ;# Worker process logic
}
```

### Distributed Data Loading
```tcl
# Setup distributed data loading based on rank
set rank [torch::get_rank]
set world_size [torch::get_world_size]

# Calculate data range for this process
set total_samples 10000
set samples_per_rank [expr {$total_samples / $world_size}]
set start_idx [expr {$rank * $samples_per_rank}]
set end_idx [expr {($rank + 1) * $samples_per_rank}]

puts "Rank $rank processing samples $start_idx to [expr {$end_idx - 1}]"
```

### Process Coordination
```tcl
# Coordinate processes based on rank
set rank [torch::get_rank]
set world_size [torch::get_world_size]

# Only rank 0 initializes the model
if {$rank == 0} {
    puts "Rank 0: Initializing model..."
    set model [torch::sequential \
        [torch::linear -inFeatures 784 -outFeatures 128] \
        [torch::linear -inFeatures 128 -outFeatures 10]]
    puts "Rank 0: Model initialized"
}

# Synchronize all processes
torch::distributed_barrier
puts "Rank $rank: All processes synchronized"
```

### Training Loop with Rank Awareness
```tcl
# Training loop with rank-specific behavior
set rank [torch::get_rank]
set world_size [torch::get_world_size]

for {set epoch 0} {$epoch < 10} {incr epoch} {
    # Only rank 0 prints progress
    if {$rank == 0} {
        puts "Epoch $epoch starting..."
    }
    
    # All ranks do training
    # ... training logic ...
    
    # Synchronize gradients across all ranks
    torch::distributed_all_reduce $gradients
    
    # Only rank 0 saves checkpoints
    if {$rank == 0 && $epoch % 5 == 0} {
        puts "Rank 0: Saving checkpoint at epoch $epoch"
        # ... checkpoint saving logic ...
    }
}
```

### Multi-GPU Training Setup
```tcl
# Check if running in distributed mode
set is_distributed [torch::is_distributed]
set rank [torch::get_rank]
set world_size [torch::get_world_size]

if {$is_distributed} {
    puts "Running distributed training:"
    puts "  Rank: $rank"
    puts "  World Size: $world_size"
    puts "  Device: cuda:$rank"
    
    # Set device based on rank
    set device "cuda:$rank"
} else {
    puts "Running single-process training:"
    puts "  Rank: $rank"
    puts "  Device: cpu"
    
    set device "cpu"
}
```

### Debugging Distributed Setup
```tcl
# Display comprehensive distributed information
puts "=== Distributed Training Status ==="
puts "Rank: [torch::get_rank]"
puts "World Size: [torch::get_world_size]"
puts "Is Distributed: [torch::is_distributed]"

# Verify rank is valid
set rank [torch::get_rank]
set world_size [torch::get_world_size]

if {$rank >= 0 && $rank < $world_size} {
    puts "✓ Rank is valid: $rank ∈ [0, $world_size)"
} else {
    puts "✗ Invalid rank: $rank not in [0, $world_size)"
}
```

### Process Communication Example
```tcl
# Example of rank-based communication pattern
set rank [torch::get_rank]
set world_size [torch::get_world_size]

# Create data on rank 0
if {$rank == 0} {
    set data [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -dtype float32]
    puts "Rank 0: Created data tensor"
} else {
    # Other ranks create empty tensor
    set data [torch::zeros -shape {4} -dtype float32]
    puts "Rank $rank: Created empty tensor"
}

# Broadcast data from rank 0 to all ranks
set result [torch::distributed_broadcast $data -root 0]
puts "Rank $rank: Received broadcasted data"
```

## Notes

- The command takes no parameters and returns an integer
- Rank is always non-negative and ranges from 0 to `world_size - 1`
- In single-process mode, rank is always 0
- Rank remains constant throughout the process lifetime
- Both `torch::get_rank` and `torch::getRank` return identical results
- Rank is set during distributed initialization and cannot be changed
- Master process is typically rank 0

## Distributed Training Concepts

- **Rank 0 (Master)**: Usually responsible for logging, checkpointing, and coordination
- **Worker Ranks**: All ranks including rank 0 participate in training
- **World Size**: Total number of processes in distributed training
- **Process Group**: All processes with ranks 0 to `world_size - 1`

## See Also

- [torch::get_world_size](get_world_size.md) - Get total number of processes
- [torch::is_distributed](is_distributed.md) - Check if distributed training is active
- [torch::distributed_init](distributed_init.md) - Initialize distributed training
- [torch::distributed_barrier](distributed_barrier.md) - Synchronize all processes
- [torch::distributed_all_reduce](distributed_all_reduce.md) - All-reduce operation
- [torch::distributed_broadcast](distributed_broadcast.md) - Broadcast operation 