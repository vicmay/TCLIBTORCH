# torch::is_distributed

Check if distributed training is enabled.

## Syntax

### Standard Syntax
```tcl
torch::is_distributed
```

### CamelCase Alias
```tcl
torch::isDistributed
```

## Parameters

This command takes no parameters.

## Return Value

Returns a boolean value:
- `0` (false) - Distributed training is not enabled (single-process mode)
- `1` (true) - Distributed training is enabled (multi-process mode)

## Description

The `torch::is_distributed` command checks whether distributed training is currently enabled. This is useful for:

- Determining if the current process is part of a distributed training setup
- Implementing conditional logic for distributed vs. single-process training
- Validating distributed training configuration
- Debugging distributed training issues

The command returns `true` if and only if distributed training has been initialized with a world size greater than 1. In single-process mode or when distributed training is not initialized, it returns `false`.

## Examples

### Basic Usage
```tcl
# Check if distributed training is enabled
set is_distributed [torch::is_distributed]
puts "Distributed training enabled: $is_distributed"
;# Output: Distributed training enabled: 0 (in single-process mode)
```

### Using CamelCase Alias
```tcl
# Check using camelCase alias
set is_distributed [torch::isDistributed]
puts "Distributed training enabled: $is_distributed"
;# Output: Distributed training enabled: 0 (in single-process mode)
```

### Distributed Training Detection
```tcl
# Initialize distributed training
torch::distributed_init 0 4 "nccl"

# Check if distributed training is enabled
set is_distributed [torch::is_distributed]
puts "Distributed training enabled: $is_distributed"
;# Output: Distributed training enabled: 1
```

### Conditional Logic Based on Distributed Mode
```tcl
# Check if running in distributed mode
set is_distributed [torch::is_distributed]

if {$is_distributed} {
    puts "Running in distributed mode"
    set rank [torch::get_rank]
    set world_size [torch::get_world_size]
    puts "Process rank: $rank, World size: $world_size"
    
    # Distributed-specific logic
    set device "cuda:$rank"
    set batch_size [expr {32 / $world_size}]  ;# Adjust batch size per process
} else {
    puts "Running in single-process mode"
    
    # Single-process logic
    set device "cpu"
    set batch_size 32
}
```

### Training Mode Configuration
```tcl
# Configure training based on distributed mode
set is_distributed [torch::is_distributed]
set rank [torch::get_rank]

if {$is_distributed} {
    # Distributed training configuration
    puts "Setting up distributed training..."
    
    # Only rank 0 handles logging and checkpointing
    set enable_logging [expr {$rank == 0}]
    set enable_checkpointing [expr {$rank == 0}]
    
    # Adjust learning rate for distributed training
    set base_lr 0.01
    set lr [expr {$base_lr * [torch::get_world_size]}]
    
    puts "Distributed mode: lr=$lr, logging=$enable_logging, checkpointing=$enable_checkpointing"
} else {
    # Single-process training configuration
    puts "Setting up single-process training..."
    
    set enable_logging 1
    set enable_checkpointing 1
    set lr 0.01
    
    puts "Single-process mode: lr=$lr, logging=$enable_logging, checkpointing=$enable_checkpointing"
}
```

### Model Synchronization
```tcl
# Synchronize model initialization based on distributed mode
set is_distributed [torch::is_distributed]
set rank [torch::get_rank]

# Initialize model
set model [torch::sequential \
    [torch::linear -inFeatures 784 -outFeatures 128] \
    [torch::linear -inFeatures 128 -outFeatures 10]]

if {$is_distributed} {
    puts "Rank $rank: Initialized model in distributed mode"
    
    # Synchronize model parameters across all processes
    torch::distributed_barrier
    puts "Rank $rank: Model synchronized across all processes"
} else {
    puts "Initialized model in single-process mode"
}
```

### Data Loading Strategy
```tcl
# Configure data loading based on distributed mode
set is_distributed [torch::is_distributed]

if {$is_distributed} {
    # Distributed data loading
    set rank [torch::get_rank]
    set world_size [torch::get_world_size]
    
    # Calculate data shard for this process
    set total_samples 10000
    set samples_per_rank [expr {$total_samples / $world_size}]
    set start_idx [expr {$rank * $samples_per_rank}]
    set end_idx [expr {($rank + 1) * $samples_per_rank}]
    
    puts "Rank $rank: Loading samples $start_idx to [expr {$end_idx - 1}]"
    
    # Load data shard
    # ... data loading logic ...
} else {
    # Single-process data loading
    puts "Loading all samples (single-process mode)"
    
    # Load all data
    # ... data loading logic ...
}
```

### Debugging and Validation
```tcl
# Comprehensive distributed training status
puts "=== Distributed Training Status ==="
puts "Is Distributed: [torch::is_distributed]"
puts "Rank: [torch::get_rank]"
puts "World Size: [torch::get_world_size]"

# Validation logic
set is_distributed [torch::is_distributed]
set rank [torch::get_rank]
set world_size [torch::get_world_size]

if {$is_distributed} {
    if {$world_size > 1} {
        puts "✓ Valid distributed setup: world_size=$world_size > 1"
    } else {
        puts "✗ Invalid distributed setup: world_size=$world_size should be > 1"
    }
} else {
    if {$world_size == 1} {
        puts "✓ Valid single-process setup: world_size=$world_size == 1"
    } else {
        puts "✗ Inconsistent setup: is_distributed=false but world_size=$world_size"
    }
}
```

### Integration with Other Distributed Functions
```tcl
# Check distributed state before using distributed operations
set is_distributed [torch::is_distributed]

if {$is_distributed} {
    puts "Distributed training enabled - using distributed operations"
    
    # Safe to use distributed operations
    torch::distributed_barrier
    
    # Example: all-reduce operation
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result [torch::distributed_all_reduce $tensor]
    puts "All-reduce completed"
} else {
    puts "Single-process mode - skipping distributed operations"
    
    # Skip distributed operations
    puts "Continuing with single-process training..."
}
```

### Performance Monitoring
```tcl
# Monitor distributed training performance
set is_distributed [torch::is_distributed]

if {$is_distributed} {
    set rank [torch::get_rank]
    set world_size [torch::get_world_size]
    
    # Measure communication overhead
    set start_time [clock milliseconds]
    torch::distributed_barrier
    set barrier_time [expr {[clock milliseconds] - $start_time}]
    
    puts "Rank $rank: Barrier synchronization took ${barrier_time}ms"
    
    # Performance metrics for distributed training
    puts "Distributed training metrics:"
    puts "  Processes: $world_size"
    puts "  Communication latency: ${barrier_time}ms"
} else {
    puts "Single-process training - no communication overhead"
}
```

### Error Handling
```tcl
# Robust distributed training setup
proc setup_training {} {
    set is_distributed [torch::is_distributed]
    
    if {$is_distributed} {
        # Validate distributed setup
        set rank [torch::get_rank]
        set world_size [torch::get_world_size]
        
        if {$rank < 0 || $rank >= $world_size} {
            error "Invalid rank: $rank not in range [0, $world_size)"
        }
        
        puts "Distributed training setup valid"
        return "distributed"
    } else {
        puts "Single-process training setup"
        return "single"
    }
}

# Use the setup function
set training_mode [setup_training]
puts "Training mode: $training_mode"
```

## Technical Details

### Distributed Training Logic
The `torch::is_distributed` command returns `true` if and only if:
1. Distributed training has been initialized with `torch::distributed_init`
2. The world size is greater than 1

### Implementation Details
- **Single-Process Mode**: Returns `false` when world_size ≤ 1
- **Multi-Process Mode**: Returns `true` when world_size > 1
- **Uninitialized State**: Returns `false` when distributed training is not initialized

### Performance Characteristics
- **Constant Time**: O(1) operation - simple state check
- **No Communication**: Does not involve network communication
- **Lightweight**: Minimal computational overhead

## Best Practices

### When to Use
1. **Conditional Logic**: Before using distributed-specific operations
2. **Configuration**: Setting up different parameters for distributed vs. single-process training
3. **Validation**: Ensuring distributed setup is correct
4. **Debugging**: Diagnosing distributed training issues

### Common Patterns
```tcl
# Pattern 1: Conditional distributed operations
if {[torch::is_distributed]} {
    # Distributed-specific code
} else {
    # Single-process code
}

# Pattern 2: Configuration based on mode
set is_distributed [torch::is_distributed]
set config [expr {$is_distributed ? "distributed" : "single"}]

# Pattern 3: Early validation
if {[torch::is_distributed] && [torch::get_world_size] <= 1} {
    error "Inconsistent distributed state"
}
```

## Migration Guide

### From Other Frameworks
```python
# PyTorch equivalent
import torch.distributed as dist
is_distributed = dist.is_initialized() and dist.get_world_size() > 1

# TCL LibTorch equivalent
set is_distributed [torch::is_distributed]
```

### Backward Compatibility
- All existing code using `torch::is_distributed` continues to work unchanged
- The command has always taken no parameters and returned a boolean value
- No breaking changes in behavior or syntax

## Related Commands

- `torch::distributed_init` - Initialize distributed training
- `torch::get_rank` - Get current process rank
- `torch::get_world_size` - Get total number of processes
- `torch::distributed_barrier` - Synchronize all processes 