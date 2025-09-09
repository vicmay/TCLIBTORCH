# torch::distributed_barrier - Distributed Training Barrier Synchronization

## Overview
Implements a distributed training barrier that synchronizes all processes in a distributed training setup. This is a critical collective communication primitive that ensures all processes reach the same execution point before continuing.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::distributed_barrier
```

### Named Parameter Syntax
```tcl
torch::distributed_barrier
```

### CamelCase Alias
```tcl
torch::distributedBarrier
```

## Parameters
This command takes no parameters.

## Return Value
Returns a string message indicating the synchronization status:
- `"Distributed not initialized"` - If distributed training is not initialized
- `"Barrier synchronized (single GPU)"` - If distributed training is in single GPU mode
- `"Barrier synchronized (simulated multi-GPU)"` - If distributed training is in multi-GPU simulation mode

## Examples

### Basic Usage
```tcl
# Load the extension
load ./libtorchtcl.so

# Without initialization
puts [torch::distributed_barrier]
# Output: "Distributed not initialized"

# Initialize single GPU mode
torch::distributed_init 0 1 "gloo"
puts [torch::distributed_barrier]
# Output: "Barrier synchronized (single GPU)"

# Initialize multi-GPU simulation mode
torch::distributed_init 0 4 "nccl"
puts [torch::distributed_barrier]
# Output: "Barrier synchronized (simulated multi-GPU)"
```

### CamelCase Usage
```tcl
# Using camelCase alias
torch::distributedBarrier
```

### Training Loop Synchronization
```tcl
# Initialize distributed training
torch::distributed_init 0 4 "nccl"

# Training loop with barrier synchronization
for {set epoch 0} {$epoch < 10} {incr epoch} {
    # Forward pass, backward pass, optimizer step
    # ...
    
    # Synchronize all processes before next epoch
    torch::distributedBarrier
    puts "Epoch $epoch completed on all processes"
}
```

### Multi-Process Data Loading
```tcl
# Synchronize after data loading
proc load_distributed_data {rank world_size} {
    # Load data shard for this rank
    # ...
    
    # Wait for all processes to finish loading
    torch::distributedBarrier
    puts "All processes have finished loading data"
}
```

## Technical Details

### Barrier Semantics
- **Blocking Operation**: All processes block until every process in the group reaches the barrier
- **Collective Communication**: Requires coordination across all processes in the distributed group
- **Synchronization Point**: Ensures consistent execution state across processes

### Implementation Details
- **Single GPU Mode**: Returns immediately with confirmation message
- **Multi-GPU Simulation**: Simulates barrier behavior for testing purposes
- **Error Handling**: Returns error if arguments are provided (barrier takes no parameters)

### Performance Characteristics
- **Latency**: O(log(world_size)) in typical tree-based implementations
- **Scalability**: Performance depends on underlying communication backend
- **Network Usage**: Minimal data transfer (typically just synchronization signals)

## Error Handling

### Common Errors
```tcl
# Too many arguments
catch {torch::distributed_barrier extra_arg} error
puts $error
# Output: "Wrong number of arguments. Expected: torch::distributed_barrier"
```

### Error Cases
1. **Extra Arguments**: Providing any arguments to the barrier command
2. **Uninitialized State**: Calling barrier before distributed initialization (returns status message, not error)

## Migration Guide

### From Other Frameworks
```python
# PyTorch equivalent
import torch.distributed as dist
dist.barrier()

# TCL LibTorch equivalent
torch::distributed_barrier
```

### Backward Compatibility
- All existing code using `torch::distributed_barrier` continues to work unchanged
- The command has always taken no parameters, so no syntax changes needed

## Integration Examples

### Training Synchronization
```tcl
proc distributed_training {epochs} {
    torch::distributed_init 0 4 "nccl"
    
    for {set epoch 0} {$epoch < $epochs} {incr epoch} {
        # Training step
        train_one_epoch $epoch
        
        # Synchronize before validation
        torch::distributedBarrier
        
        # Validation step (on all processes)
        validate_model $epoch
        
        # Synchronize before next epoch
        torch::distributedBarrier
    }
}
```

### Checkpoint Coordination
```tcl
proc save_checkpoint {model_file rank} {
    # Only rank 0 saves the model
    if {$rank == 0} {
        save_model $model_file
    }
    
    # All processes wait for save to complete
    torch::distributedBarrier
    puts "Checkpoint saved and synchronized"
}
```

### Debugging and Profiling
```tcl
proc profile_training_step {} {
    set start_time [clock milliseconds]
    
    # Training computation
    train_step
    
    # Measure time before barrier
    set compute_time [expr {[clock milliseconds] - $start_time}]
    
    # Synchronize and measure communication overhead
    set barrier_start [clock milliseconds]
    torch::distributedBarrier
    set barrier_time [expr {[clock milliseconds] - $barrier_start}]
    
    puts "Compute: ${compute_time}ms, Barrier: ${barrier_time}ms"
}
```

## Best Practices

### When to Use Barriers
1. **Epoch Boundaries**: Synchronize before starting new training epochs
2. **Validation Points**: Ensure all processes finish training before validation
3. **Checkpoint Coordination**: Synchronize model saving/loading across processes
4. **Resource Cleanup**: Coordinate cleanup operations

### When NOT to Use Barriers
1. **Every Training Step**: Can significantly impact performance
2. **Independent Operations**: When processes don't need coordination
3. **Data Loading**: Unless specifically required for consistency

### Performance Optimization
- Use barriers sparingly to minimize communication overhead
- Group multiple operations between barriers when possible
- Consider asynchronous alternatives for non-critical synchronization

## See Also
- `torch::distributed_init` - Initialize distributed training
- `torch::distributed_all_reduce` - Collective reduction operations
- `torch::distributed_all_to_all` - All-to-all communication
- `torch::get_rank` - Get current process rank
- `torch::get_world_size` - Get total number of processes 