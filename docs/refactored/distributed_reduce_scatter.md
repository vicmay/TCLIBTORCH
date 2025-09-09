# torch::distributed_reduce_scatter

Performs a reduce-scatter operation on a tensor across distributed processes. This operation combines reduction and scattering in a single step, reducing the tensor according to the specified operation and distributing the result chunks across processes.

## Syntax

### Snake_case (Original)
```tcl
torch::distributed_reduce_scatter tensor ?op? ?group?
```

### CamelCase (Refactored)
```tcl
torch::distributedReduceScatter tensor ?op? ?group?
```

### Named Parameters
```tcl
torch::distributed_reduce_scatter -tensor tensor ?-op op? ?-group group?
torch::distributedReduceScatter -tensor tensor ?-op op? ?-group group?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tensor` | string | Yes | - | Handle to the input tensor to reduce and scatter |
| `op` | string | No | "sum" | Reduction operation to perform |
| `group` | string | No | "" | Process group for the operation |

### Valid Operations

- **sum**: Element-wise addition across processes
- **mean**: Element-wise average across processes  
- **max**: Element-wise maximum across processes
- **min**: Element-wise minimum across processes
- **product**: Element-wise multiplication across processes

## Return Value

Returns a tensor handle containing the reduced and scattered result.

## Examples

### Basic Usage

```tcl
# Create a tensor
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]

# Positional syntax - default sum operation
set result1 [torch::distributed_reduce_scatter $tensor]

# Named syntax - explicit sum operation
set result2 [torch::distributed_reduce_scatter -tensor $tensor -op "sum"]

# CamelCase alias
set result3 [torch::distributedReduceScatter $tensor "sum"]
```

### Different Operations

```tcl
# Mean operation
set mean_result [torch::distributed_reduce_scatter $tensor "mean"]

# Max operation
set max_result [torch::distributed_reduce_scatter -tensor $tensor -op "max"]

# Min operation  
set min_result [torch::distributedReduceScatter -tensor $tensor -op "min"]

# Product operation
set product_result [torch::distributed_reduce_scatter $tensor "product"]
```

### With Process Groups

```tcl
# Specify process group
set result [torch::distributed_reduce_scatter $tensor "sum" "workers"]

# Named syntax with group
set result [torch::distributed_reduce_scatter -tensor $tensor -op "mean" -group "workers"]
```

## Use Cases

### 1. Gradient Aggregation
```tcl
# Aggregate gradients across workers in distributed training
set gradients [torch::tensor_create -data {0.1 0.2 0.3 0.4} -shape {2 2} -dtype float32]
set aggregated [torch::distributed_reduce_scatter $gradients "sum"]
```

### 2. Distributed Statistics
```tcl
# Compute distributed statistics
set local_stats [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]

# Sum across processes
set total_sum [torch::distributed_reduce_scatter $local_stats "sum"]

# Average across processes
set average [torch::distributed_reduce_scatter $local_stats "mean"]
```

### 3. Parallel Reduction
```tcl
# Parallel reduction with scattering
set data [torch::tensor_create -data {5.0 10.0 15.0 20.0} -shape {2 2} -dtype float32]
set reduced [torch::distributed_reduce_scatter -tensor $data -op "max"]
```

## Distributed Computing Patterns

### AllReduce vs ReduceScatter
```tcl
# AllReduce: All processes get the full reduced result
set all_reduced [torch::distributed_all_reduce $tensor "sum"]

# ReduceScatter: Each process gets a chunk of the reduced result
set scatter_reduced [torch::distributed_reduce_scatter $tensor "sum"]
```

### Efficient Bandwidth Usage
```tcl
# ReduceScatter uses less bandwidth than AllReduce + Scatter
# Single operation instead of two separate operations
set efficient_result [torch::distributed_reduce_scatter $tensor "sum"]
```

## Advanced Examples

### Multi-dimensional Tensors
```tcl
# 3D tensor reduce-scatter
set tensor_3d [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 2 2} -dtype float32]
set result_3d [torch::distributed_reduce_scatter $tensor_3d "mean"]
```

### Combining with Other Operations
```tcl
# Reduce-scatter followed by local computation
set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
set reduced [torch::distributed_reduce_scatter $input "sum"]
set normalized [torch::tensor_div $reduced 2.0]
```

### Error Handling
```tcl
# Handle invalid operations
if {[catch {torch::distributed_reduce_scatter $tensor "invalid_op"} error]} {
    puts "Error: $error"
}

# Validate tensor handle
if {[catch {torch::distributed_reduce_scatter "invalid_tensor" "sum"} error]} {
    puts "Error: $error"
}
```

## Performance Considerations

### Memory Efficiency
- ReduceScatter is more memory efficient than AllReduce for large tensors
- Each process only receives its portion of the result
- Reduces memory requirements in distributed training

### Communication Patterns
```tcl
# Efficient for gradient aggregation in data parallel training
set gradients [get_local_gradients]
set aggregated_chunk [torch::distributed_reduce_scatter $gradients "sum"]
```

### Process Group Optimization
```tcl
# Use specific process groups for better performance
set worker_group "data_parallel_workers"
set result [torch::distributed_reduce_scatter -tensor $tensor -op "sum" -group $worker_group]
```

## Integration with Training Loops

### Distributed Training Integration
```tcl
proc distributed_training_step {model inputs targets} {
    # Forward pass
    set outputs [torch::model_forward $model $inputs]
    set loss [torch::loss_function $outputs $targets]
    
    # Backward pass
    torch::backward $loss
    
    # Get gradients
    set gradients [torch::model_gradients $model]
    
    # Reduce-scatter gradients
    set aggregated_gradients [torch::distributed_reduce_scatter $gradients "sum"]
    
    # Update model
    torch::optimizer_step $optimizer $aggregated_gradients
}
```

## Error Handling

### Common Errors
- **Invalid tensor handle**: Tensor doesn't exist
- **Invalid operation**: Operation not in {sum, mean, max, min, product}
- **Missing parameters**: Required tensor parameter not provided
- **Parameter type errors**: Invalid parameter types

### Error Messages
```tcl
# Missing tensor parameter
torch::distributed_reduce_scatter -op "sum"
# Error: Required parameter missing: -tensor

# Invalid operation
torch::distributed_reduce_scatter $tensor "invalid"
# Error: Required parameter missing: -tensor, or invalid operation. Valid operations: sum, mean, max, min, product

# Invalid tensor handle
torch::distributed_reduce_scatter "nonexistent_tensor" "sum"
# Error: Invalid tensor handle
```

## Best Practices

1. **Use appropriate operations**: Choose the reduction operation that matches your use case
2. **Handle errors gracefully**: Always wrap calls in error handling
3. **Consider memory usage**: ReduceScatter is more memory efficient than AllReduce
4. **Process group management**: Use appropriate process groups for your topology
5. **Performance monitoring**: Monitor communication overhead in distributed settings

## Syntax Migration Guide

### From Positional to Named Parameters
```tcl
# Old positional syntax
set result [torch::distributed_reduce_scatter $tensor "sum" "group1"]

# New named syntax  
set result [torch::distributed_reduce_scatter -tensor $tensor -op "sum" -group "group1"]
```

### From Snake_case to CamelCase
```tcl
# Snake_case (still supported)
set result [torch::distributed_reduce_scatter $tensor "sum"]

# CamelCase (new)
set result [torch::distributedReduceScatter $tensor "sum"]
```

## Related Commands

- `torch::distributed_all_reduce` - Reduce across all processes
- `torch::distributed_scatter` - Scatter tensor to processes  
- `torch::distributed_gather` - Gather tensors from processes
- `torch::distributed_all_to_all` - All-to-all communication
- `torch::distributed_broadcast` - Broadcast tensor to all processes

## See Also

- [Distributed Training Guide](../guides/distributed_training.md)
- [Process Groups](../guides/process_groups.md)
- [Communication Patterns](../guides/communication_patterns.md) 