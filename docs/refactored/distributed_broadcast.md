# torch::distributed_broadcast - Distributed Training Broadcast Operation

## Overview
Implements distributed broadcast operation that sends a tensor from one process (root) to all other processes in a distributed training setup. This is a fundamental collective communication primitive used to synchronize model parameters, gradients, or other data across all processes.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::distributed_broadcast tensor ?root?
```

### Named Parameter Syntax
```tcl
torch::distributed_broadcast -tensor <tensor_name> ?-root <rank>?
```

### CamelCase Alias
```tcl
torch::distributedBroadcast tensor ?root?
torch::distributedBroadcast -tensor <tensor_name> ?-root <rank>?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor` | string | required | Name of the tensor to broadcast |
| `root` | integer | 0 | Rank of the process that owns the source tensor |

## Return Value
Returns a new tensor handle containing the broadcasted tensor data. In simulation mode, returns a copy of the input tensor.

## Examples

### Basic Usage
```tcl
# Load the extension
load ./libtorchtcl.so

# Create a tensor to broadcast
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2}]

# Positional syntax - broadcast from rank 0 (default)
set result1 [torch::distributed_broadcast $tensor]

# Positional syntax - broadcast from rank 2
set result2 [torch::distributed_broadcast $tensor 2]

# Named parameter syntax
set result3 [torch::distributed_broadcast -tensor $tensor]
set result4 [torch::distributed_broadcast -tensor $tensor -root 1]
```

### CamelCase Usage
```tcl
# Using camelCase alias
set result1 [torch::distributedBroadcast $tensor]
set result2 [torch::distributedBroadcast $tensor 3]
set result3 [torch::distributedBroadcast -tensor $tensor -root 0]
```

### Parameter Server Architecture
```tcl
proc parameter_server_broadcast {model_params world_size} {
    # Initialize distributed training
    torch::distributed_init 0 $world_size "nccl"
    
    # Broadcast parameters from server (rank 0) to all workers
    set broadcasted_params {}
    foreach param $model_params {
        set result [torch::distributedBroadcast $param 0]
        lappend broadcasted_params $result
    }
    
    return $broadcasted_params
}
```

### Model Initialization Synchronization
```tcl
proc sync_model_parameters {model_tensors root_rank} {
    set synced_model {}
    
    foreach {layer_name tensor} $model_tensors {
        # Broadcast each layer's parameters from root rank
        set synced_tensor [torch::distributed_broadcast $tensor $root_rank]
        dict set synced_model $layer_name $synced_tensor
    }
    
    return $synced_model
}
```

### Gradient Aggregation with Broadcast
```tcl
proc distributed_gradient_step {gradients learning_rate world_size} {
    # Aggregate gradients (simplified example)
    set aggregated_grads {}
    
    foreach grad $gradients {
        # In real implementation, would do all-reduce first
        # Then broadcast the averaged gradient
        set avg_grad [torch::distributed_broadcast $grad 0]
        lappend aggregated_grads $avg_grad
    }
    
    return $aggregated_grads
}
```

## Technical Details

### Broadcast Semantics
- **One-to-Many Communication**: Data flows from one source (root) to all processes
- **Collective Operation**: All processes must participate in the broadcast
- **Synchronous**: All processes block until the operation completes
- **Data Integrity**: Ensures all processes receive identical data

### Implementation Details
- **Single GPU Mode**: Returns a copy of the input tensor
- **Multi-GPU Simulation**: Simulates broadcast by copying the tensor
- **Root Rank**: Specifies which process provides the source data
- **Memory Management**: Creates new tensor handles for results

### Performance Characteristics
- **Latency**: O(log(world_size)) in tree-based implementations
- **Bandwidth**: O(data_size) per process
- **Scalability**: Efficient for large world sizes with proper algorithms
- **Memory Usage**: Creates copies of tensors during broadcast

## Error Handling

### Common Errors
```tcl
# Missing required tensor argument
catch {torch::distributed_broadcast} error
puts $error
# Output: "wrong # args: should be \"distributed_broadcast tensor ?root?\""

# Invalid tensor handle
catch {torch::distributed_broadcast invalid_tensor} error
puts $error
# Output: "Tensor not found"

# Invalid root rank type
catch {torch::distributed_broadcast $tensor "not_number"} error
puts $error
# Output: "root must be an integer"

# Negative root rank
catch {torch::distributed_broadcast $tensor -1} error
puts $error
# Output: "Invalid arguments: tensor required and root must be >= 0"
```

### Error Cases
1. **Missing Arguments**: Not providing required tensor parameter
2. **Invalid Tensor**: Using non-existent tensor handles
3. **Invalid Root Type**: Providing non-integer root rank
4. **Negative Root**: Using negative root rank values
5. **Parameter Mismatch**: Incorrect named parameter usage

## Migration Guide

### From Other Frameworks
```python
# PyTorch equivalent
import torch.distributed as dist
dist.broadcast(tensor, src=0)

# TensorFlow equivalent
import tensorflow as tf
tf.distribute.broadcast(tensor, root_rank=0)

# TCL LibTorch equivalent
torch::distributed_broadcast $tensor 0
```

### Backward Compatibility
- All existing positional syntax code continues to work unchanged
- New named parameter syntax provides enhanced clarity
- CamelCase aliases support modern naming conventions

## Integration Examples

### Distributed Training Loop
```tcl
proc distributed_training_epoch {model_params data_loader} {
    # Broadcast initial model parameters
    set synced_params {}
    foreach param $model_params {
        set synced [torch::distributedBroadcast $param 0]
        lappend synced_params $synced
    }
    
    # Training computations with synchronized parameters
    # ... forward pass, backward pass ...
    
    return $synced_params
}
```

### Configuration Broadcasting
```tcl
proc broadcast_config {config_tensor master_rank} {
    # Ensure all processes have the same configuration
    set shared_config [torch::distributed_broadcast $config_tensor $master_rank]
    
    puts "Configuration synchronized across all processes"
    return $shared_config
}
```

### Checkpoint Loading
```tcl
proc load_distributed_checkpoint {checkpoint_file world_size} {
    torch::distributed_init 0 $world_size "gloo"
    
    # Only rank 0 loads the checkpoint
    if {[torch::get_rank] == 0} {
        set checkpoint [load_checkpoint $checkpoint_file]
    } else {
        set checkpoint {}
    }
    
    # Broadcast checkpoint to all processes
    foreach {param_name tensor} $checkpoint {
        set broadcasted [torch::distributedBroadcast $tensor 0]
        dict set distributed_checkpoint $param_name $broadcasted
    }
    
    return $distributed_checkpoint
}
```

## Best Practices

### When to Use Broadcast
1. **Model Initialization**: Synchronize initial parameters across processes
2. **Configuration Sharing**: Distribute hyperparameters and settings
3. **Checkpoint Loading**: Share loaded model states
4. **Global State Updates**: Propagate global training information

### Performance Optimization
- **Batch Operations**: Broadcast multiple tensors together when possible
- **Data Layout**: Use contiguous memory layouts for better performance
- **Size Considerations**: Large tensors may benefit from chunked broadcasting
- **Network Topology**: Consider network bandwidth and latency

### Memory Management
- Clean up intermediate tensors after broadcast operations
- Monitor memory usage in multi-GPU setups
- Use appropriate data types to minimize transfer overhead

## See Also
- `torch::distributed_init` - Initialize distributed training
- `torch::distributed_all_reduce` - All-reduce collective operations
- `torch::distributed_all_to_all` - All-to-all communication
- `torch::distributed_barrier` - Process synchronization
- `torch::get_rank` - Get current process rank
- `torch::get_world_size` - Get total number of processes 