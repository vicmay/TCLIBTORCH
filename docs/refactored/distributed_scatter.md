# torch::distributed_scatter

Scatters a tensor from a source process to all processes in a distributed computation environment. This operation distributes chunks of the input tensor to different processes, enabling efficient data distribution across multiple nodes.

## Syntax

### Snake_case (Original)
```tcl
torch::distributed_scatter tensor ?src? ?group?
```

### CamelCase (Refactored)
```tcl
torch::distributedScatter tensor ?src? ?group?
```

### Named Parameters
```tcl
torch::distributed_scatter -tensor tensor ?-src src? ?-group group?
torch::distributedScatter -tensor tensor ?-src src? ?-group group?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tensor` | string | Yes | - | Handle to the input tensor to scatter |
| `src` | integer | No | 0 | Source process rank that holds the tensor to scatter |
| `group` | string | No | "" | Process group for the operation |

## Return Value

Returns a tensor handle containing the scattered chunk for the current process.

## Examples

### Basic Usage

```tcl
# Create a tensor
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]

# Positional syntax - default source rank 0
set result1 [torch::distributed_scatter $tensor]

# Named syntax - explicit source rank
set result2 [torch::distributed_scatter -tensor $tensor -src 0]

# CamelCase alias
set result3 [torch::distributedScatter $tensor 0]
```

### Different Source Ranks

```tcl
# Scatter from rank 1
set result1 [torch::distributed_scatter $tensor 1]

# Scatter from rank 2 with named syntax
set result2 [torch::distributed_scatter -tensor $tensor -src 2]

# Scatter from high rank number
set result3 [torch::distributedScatter $tensor 100]
```

### With Process Groups

```tcl
# Specify process group
set result [torch::distributed_scatter $tensor 0 "workers"]

# Named syntax with group
set result [torch::distributed_scatter -tensor $tensor -src 1 -group "data_parallel"]
```

## Use Cases

### 1. Data Distribution
```tcl
# Distribute large dataset across workers
set full_dataset [torch::tensor_create -data $large_data -shape $data_shape -dtype float32]
set local_chunk [torch::distributed_scatter $full_dataset 0]
```

### 2. Model Parallel Training
```tcl
# Distribute model parameters across devices
set model_params [torch::tensor_create -data $param_data -shape $param_shape -dtype float32]
set local_params [torch::distributed_scatter $model_params 0 "model_parallel"]
```

### 3. Workload Distribution
```tcl
# Distribute computation tasks
set task_tensor [torch::tensor_create -data $task_data -shape $task_shape -dtype float32]
set local_tasks [torch::distributed_scatter -tensor $task_tensor -src 0]
```

## Distributed Computing Patterns

### Scatter vs Broadcast
```tcl
# Broadcast: All processes get the full tensor
set broadcasted [torch::distributed_broadcast $tensor 0]

# Scatter: Each process gets a different chunk
set scattered_chunk [torch::distributed_scatter $tensor 0]
```

### Data Parallel Pattern
```tcl
# Scatter training data across workers
set batch_data [create_large_batch]
set local_batch [torch::distributed_scatter $batch_data 0 "data_workers"]

# Each worker processes their chunk
set local_result [process_batch $local_batch]

# Gather results back
set all_results [torch::distributed_gather $local_result 0]
```

### Pipeline Parallel Pattern
```tcl
# Scatter different model layers to different devices
set layer_params [get_layer_parameters]
set my_layer [torch::distributed_scatter $layer_params 0 "pipeline_group"]
```

## Advanced Examples

### Multi-dimensional Tensors
```tcl
# 3D tensor scatter
set tensor_3d [torch::tensor_create -data $data_3d -shape {4 3 2} -dtype float32]
set chunk_3d [torch::distributed_scatter $tensor_3d 0]
```

### Dynamic Source Selection
```tcl
# Select source based on data availability
set data_source [get_data_source_rank]
set distributed_data [torch::distributed_scatter $tensor $data_source]
```

### Combining with Other Operations
```tcl
# Scatter then reduce
set data [torch::tensor_create -data $input_data -shape $shape -dtype float32]
set local_chunk [torch::distributed_scatter $data 0]
set processed [torch::tensor_mul $local_chunk 2.0]
set result [torch::distributed_all_reduce $processed "sum"]
```

### Error Handling
```tcl
# Handle invalid source ranks
if {[catch {torch::distributed_scatter $tensor -1} error]} {
    puts "Error: $error"
}

# Validate tensor handle
if {[catch {torch::distributed_scatter "invalid_tensor" 0} error]} {
    puts "Error: $error"
}
```

## Performance Considerations

### Memory Efficiency
- Scatter reduces memory usage per process compared to broadcasting
- Each process only holds its portion of the data
- Enables processing of datasets larger than single-node memory

### Communication Patterns
```tcl
# Efficient for large tensor distribution
set large_tensor [create_large_tensor $size]
set my_portion [torch::distributed_scatter $large_tensor 0]
```

### Source Process Optimization
```tcl
# Choose source process strategically
set source_rank [get_optimal_source_rank]
set distributed_data [torch::distributed_scatter $tensor $source_rank]
```

## Integration with Training Loops

### Data Parallel Training
```tcl
proc distributed_data_loading {full_dataset} {
    # Scatter dataset across workers
    set local_data [torch::distributed_scatter $full_dataset 0]
    return $local_data
}

proc training_step {model local_batch} {
    # Each worker processes their data chunk
    set outputs [torch::model_forward $model $local_batch]
    set loss [torch::loss_function $outputs $targets]
    
    # Backward pass on local data
    torch::backward $loss
    
    return $loss
}
```

### Model Parallel Training
```tcl
proc distribute_model_layers {model_params} {
    # Scatter different layers to different devices
    set my_layers [torch::distributed_scatter $model_params 0 "model_group"]
    return $my_layers
}
```

## Communication Topology

### Ring Topology
```tcl
# Scatter in ring pattern
set my_rank [torch::get_rank]
set source [expr {($my_rank - 1) % $world_size}]
set data_chunk [torch::distributed_scatter $tensor $source "ring_group"]
```

### Tree Topology
```tcl
# Hierarchical scatter
set level1_chunk [torch::distributed_scatter $tensor 0 "level1_group"]
set level2_chunk [torch::distributed_scatter $level1_chunk $local_leader "level2_group"]
```

## Error Handling

### Common Errors
- **Invalid tensor handle**: Tensor doesn't exist
- **Invalid source rank**: Source rank is negative
- **Missing parameters**: Required tensor parameter not provided
- **Parameter type errors**: Invalid parameter types

### Error Messages
```tcl
# Missing tensor parameter
torch::distributed_scatter -src 0
# Error: Required parameter missing: -tensor

# Invalid source rank
torch::distributed_scatter $tensor -1
# Error: Required parameter missing: -tensor, or invalid src parameter

# Invalid tensor handle
torch::distributed_scatter "nonexistent_tensor" 0
# Error: Invalid tensor handle
```

## Best Practices

1. **Choose appropriate source**: Select source rank based on data locality
2. **Handle errors gracefully**: Always wrap calls in error handling
3. **Consider memory usage**: Scatter reduces per-process memory requirements
4. **Process group management**: Use appropriate groups for your topology
5. **Load balancing**: Ensure even distribution of data chunks

## Syntax Migration Guide

### From Positional to Named Parameters
```tcl
# Old positional syntax
set result [torch::distributed_scatter $tensor 1 "group1"]

# New named syntax  
set result [torch::distributed_scatter -tensor $tensor -src 1 -group "group1"]
```

### From Snake_case to CamelCase
```tcl
# Snake_case (still supported)
set result [torch::distributed_scatter $tensor 0]

# CamelCase (new)
set result [torch::distributedScatter $tensor 0]
```

## Implementation Details

### Single Process Mode
In single-process environments, scatter returns a clone of the input tensor, simulating the behavior of receiving the entire tensor as a "chunk."

### Simulation Behavior
```tcl
# In simulation mode (single process)
set original [torch::tensor_create -data {1 2 3 4} -shape {2 2} -dtype float32]
set scattered [torch::distributed_scatter $original 0]
# scattered contains a copy of the original tensor
```

## Related Commands

- `torch::distributed_gather` - Gather tensors from all processes
- `torch::distributed_broadcast` - Broadcast tensor to all processes
- `torch::distributed_all_to_all` - All-to-all communication
- `torch::distributed_reduce_scatter` - Reduce and scatter in one operation
- `torch::distributed_all_reduce` - Reduce across all processes

## Communication Patterns

### Scatter-Gather Pattern
```tcl
# 1. Scatter data from source
set local_chunk [torch::distributed_scatter $data 0]

# 2. Process locally
set processed_chunk [local_processing $local_chunk]

# 3. Gather results
set final_result [torch::distributed_gather $processed_chunk 0]
```

### Scatter-AllReduce Pattern
```tcl
# 1. Scatter data
set local_data [torch::distributed_scatter $data 0]

# 2. Compute local gradients
set local_gradients [compute_gradients $local_data]

# 3. All-reduce gradients
set averaged_gradients [torch::distributed_all_reduce $local_gradients "mean"]
```

## See Also

- [Distributed Training Guide](../guides/distributed_training.md)
- [Process Groups](../guides/process_groups.md)
- [Communication Patterns](../guides/communication_patterns.md)
- [Memory Management](../guides/memory_management.md) 