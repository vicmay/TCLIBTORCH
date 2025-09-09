# torch::distributed_all_reduce

Performs an all-reduce operation on tensors across distributed processes, combining values from all processes and distributing the result to all processes.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::distributed_all_reduce tensor ?operation?
```

### Named Parameter Syntax  
```tcl
torch::distributed_all_reduce -tensor tensor ?-operation operation?
```

### CamelCase Alias
```tcl
torch::distributedAllReduce tensor ?operation?
torch::distributedAllReduce -tensor tensor ?-operation operation?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tensor` / `-tensor` | string | Yes | - | Handle to the input tensor to reduce |
| `operation` / `-operation` | string | No | "sum" | Reduction operation: "sum", "mean", "max", or "min" |

## Return Value

Returns a string handle to the result tensor containing the reduced values.

## Description

The `torch::distributed_all_reduce` command performs a distributed all-reduce operation that combines tensors from all processes using a specified reduction operation and distributes the result to all processes. This is a fundamental operation in distributed training and parallel computing.

### Key Features:
- **Multiple reduction operations**: Supports sum, mean, max, and min operations
- **Distributed coordination**: Synchronizes across all processes in the distributed group
- **Shape preservation**: Output tensor has the same shape as input tensor
- **Data type preservation**: Maintains the input tensor's data type
- **Dual syntax support**: Supports both positional and named parameter syntax
- **Simulation mode**: Works in single-GPU environments for testing

### Reduction Operations

- **sum**: Element-wise sum across all processes
- **mean**: Element-wise average across all processes  
- **max**: Element-wise maximum across all processes
- **min**: Element-wise minimum across all processes

### Distributed Behavior

- **Multi-GPU mode**: Performs actual all-reduce across distributed processes
- **Single-GPU mode**: Returns processed tensor (for testing/development)
- **Simulation**: In development environments, simulates distributed behavior

## Examples

### Basic Usage

```tcl
# Create a tensor for reduction
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4} -dtype float32]

# Positional syntax - default sum operation
set result [torch::distributed_all_reduce $tensor]

# Named parameter syntax
set result [torch::distributed_all_reduce -tensor $tensor]

# Explicit sum operation
set result [torch::distributed_all_reduce $tensor sum]
set result [torch::distributed_all_reduce -tensor $tensor -operation sum]
```

### Different Reduction Operations

```tcl
set tensor [torch::tensor_create -data {1.0 5.0 3.0 7.0} -shape {4} -dtype float32]

# Sum reduction (default)
set sum_result [torch::distributed_all_reduce $tensor sum]

# Mean reduction
set mean_result [torch::distributed_all_reduce $tensor mean]  

# Maximum reduction
set max_result [torch::distributed_all_reduce $tensor max]

# Minimum reduction  
set min_result [torch::distributed_all_reduce $tensor min]

# Using named parameters
set mean_result [torch::distributed_all_reduce -tensor $tensor -operation mean]
```

### CamelCase Alias Usage

```tcl
set tensor [torch::tensor_create -data {2.0 4.0 6.0 8.0} -shape {4} -dtype float32]

# CamelCase alias with positional syntax
set result [torch::distributedAllReduce $tensor mean]

# CamelCase alias with named parameters
set result [torch::distributedAllReduce -tensor $tensor -operation sum]
```

### Multi-Dimensional Tensors

```tcl
# 2D tensor reduction
set matrix [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
set reduced_matrix [torch::distributed_all_reduce $matrix sum]

# 3D tensor reduction
set volume [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 2 2} -dtype float32]
set reduced_volume [torch::distributed_all_reduce $volume mean]

# Shape is preserved
puts "Original shape: [torch::tensor_shape $matrix]"     # 2 3
puts "Reduced shape: [torch::tensor_shape $reduced_matrix]"  # 2 3
```

### Distributed Training Example

```tcl
# Gradient synchronization in distributed training
proc synchronize_gradients {model_gradients} {
    set synchronized_gradients {}
    
    foreach grad $model_gradients {
        # All-reduce gradients across all processes
        set sync_grad [torch::distributed_all_reduce $grad mean]
        lappend synchronized_gradients $sync_grad
    }
    
    return $synchronized_gradients
}

# Usage in training loop
set gradients [list $grad1 $grad2 $grad3]
set sync_gradients [synchronize_gradients $gradients]

# Apply synchronized gradients to model
foreach param $model_params sync_grad $sync_gradients {
    set updated_param [torch::sub $param $sync_grad]
    # Update model parameter
}
```

### Parameter Server Simulation

```tcl
# Simulate parameter server with all-reduce
proc update_global_parameters {local_updates} {
    set global_updates {}
    
    foreach update $local_updates {
        # Aggregate updates from all workers
        set aggregated [torch::distributed_all_reduce $update sum]
        lappend global_updates $aggregated
    }
    
    return $global_updates
}

# Worker sends parameter updates
set worker_updates [list $update1 $update2 $update3]
set global_params [update_global_parameters $worker_updates]
```

### Collective Communication Pattern

```tcl
# Ring all-reduce simulation for large tensors
proc ring_all_reduce {tensor operation} {
    # In actual distributed setting, this would use multiple processes
    # Here we simulate the operation
    
    switch $operation {
        "sum" {
            return [torch::distributed_all_reduce $tensor sum]
        }
        "mean" {
            return [torch::distributed_all_reduce $tensor mean]  
        }
        "max" {
            return [torch::distributed_all_reduce $tensor max]
        }
        "min" {
            return [torch::distributed_all_reduce $tensor min]
        }
        default {
            error "Unsupported operation: $operation"
        }
    }
}

# Large tensor processing
set large_tensor [torch::tensor_create -data [lrepeat 10000 1.5] -shape {10000} -dtype float32]
set result [ring_all_reduce $large_tensor "mean"]
```

### Error Handling and Validation

```tcl
# Safe all-reduce with error handling
proc safe_all_reduce {tensor_name operation} {
    # Validate tensor exists
    if {[catch {torch::tensor_shape $tensor_name}]} {
        error "Invalid tensor: $tensor_name"
    }
    
    # Validate operation
    set valid_ops {sum mean max min}
    if {$operation ni $valid_ops} {
        error "Invalid operation: $operation. Must be one of: [join $valid_ops {, }]"
    }
    
    # Perform all-reduce with error handling
    if {[catch {torch::distributed_all_reduce $tensor_name $operation} result]} {
        error "All-reduce failed: $result"
    }
    
    return $result
}

# Usage
set tensor [torch::tensor_create -data {1.0 2.0 3.0} -shape {3} -dtype float32]
set result [safe_all_reduce $tensor "mean"]
```

### Mixed Data Types

```tcl
# Integer tensor reduction
set int_tensor [torch::tensor_create -data {1 2 3 4 5} -shape {5} -dtype int32]
set int_result [torch::distributed_all_reduce $int_tensor sum]

# Float tensor reduction  
set float_tensor [torch::tensor_create -data {1.5 2.5 3.5 4.5} -shape {4} -dtype float32]
set float_result [torch::distributed_all_reduce $float_tensor mean]

# Check data types are preserved
puts "Integer result type: [torch::tensor_dtype $int_result]"
puts "Float result type: [torch::tensor_dtype $float_result]"
```

## Technical Details

### Distributed Computing Context

All-reduce is a fundamental collective operation in distributed computing where:
1. Each process contributes a tensor
2. A reduction operation is applied element-wise across all processes
3. The result is distributed to all processes

### Mathematical Operations

For N processes with tensors T₁, T₂, ..., Tₙ:

- **Sum**: `Result[i] = T₁[i] + T₂[i] + ... + Tₙ[i]`
- **Mean**: `Result[i] = (T₁[i] + T₂[i] + ... + Tₙ[i]) / N`
- **Max**: `Result[i] = max(T₁[i], T₂[i], ..., Tₙ[i])`
- **Min**: `Result[i] = min(T₁[i], T₂[i], ..., Tₙ[i])`

### Implementation Details

- **Backend**: Uses PyTorch's distributed communication backend
- **Algorithms**: Optimized ring all-reduce or tree reduction depending on size
- **Memory**: In-place operations where possible to minimize memory usage
- **Synchronization**: Blocking operation that synchronizes all processes

### Performance Characteristics

- **Time complexity**: O(N) where N is tensor size
- **Communication complexity**: Optimized based on network topology
- **Memory usage**: Minimal additional memory overhead
- **Scalability**: Efficient scaling to hundreds or thousands of processes

### Single GPU Mode Behavior

In development/testing environments without distributed setup:
- **Sum**: Returns input tensor unchanged  
- **Mean**: May apply scaling factor simulation
- **Max/Min**: Returns input tensor unchanged
- **Shape**: Always preserved
- **Type**: Always preserved

## Error Handling

The command provides comprehensive error checking:

```tcl
# Invalid tensor handle
catch {torch::distributed_all_reduce invalid_tensor} error
# Error: "Tensor not found"

# Missing required parameters  
catch {torch::distributed_all_reduce} error
# Error: "wrong # args" or "Invalid arguments"

# Invalid operation
catch {torch::distributed_all_reduce $tensor invalid_op} error
# Error: "Invalid arguments: tensor required and operation must be sum/mean/max/min"

# Invalid parameter names
catch {torch::distributed_all_reduce -invalid $tensor} error  
# Error: "unknown option: -invalid"

# Missing parameter value
catch {torch::distributed_all_reduce -tensor} error
# Error: "missing value for option: -tensor"
```

## Common Use Cases

### 1. Gradient Synchronization
```tcl
# Synchronize gradients across workers in distributed training
set sync_grad [torch::distributed_all_reduce $local_gradient mean]
```

### 2. Model Parameter Averaging
```tcl
# Average model parameters across processes
set avg_param [torch::distributed_all_reduce $local_param mean]
```

### 3. Global Statistics Computation
```tcl
# Compute global maximum across all processes
set global_max [torch::distributed_all_reduce $local_max max]

# Compute global sum for loss aggregation
set total_loss [torch::distributed_all_reduce $local_loss sum]
```

### 4. Consensus and Voting
```tcl
# Vote aggregation (using sum for counting)
set vote_count [torch::distributed_all_reduce $local_vote sum]

# Decision making (using max for consensus)
set consensus [torch::distributed_all_reduce $local_decision max]
```

## Best Practices

### Performance Optimization
- Use appropriate data types (float32 vs float64)
- Consider tensor size for communication overhead
- Batch multiple small reductions when possible
- Use asynchronous operations when available

### Error Handling
- Always validate tensors before reduction
- Check for distributed initialization
- Handle communication failures gracefully
- Verify operation compatibility with data types

### Memory Management
- Be aware of temporary tensor creation
- Consider in-place operations where possible
- Monitor memory usage in large-scale distributed training
- Clean up intermediate results

## Comparison with Related Operations

| Operation | Scope | Result Distribution | Use Case |
|-----------|-------|-------------------|----------|
| `torch::distributed_all_reduce` | All processes | All processes get same result | Gradient sync, parameter averaging |
| `torch::distributed_reduce` | All processes | One process gets result | Loss aggregation, metric collection |
| `torch::distributed_broadcast` | One to all | All processes get same input | Parameter distribution |
| `torch::distributed_gather` | All to one | One process gets all inputs | Data collection, logging |

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set result [torch::distributed_all_reduce $tensor]
set result [torch::distributed_all_reduce $tensor sum]
set result [torch::distributed_all_reduce $tensor mean]

# New named parameter syntax  
set result [torch::distributed_all_reduce -tensor $tensor]
set result [torch::distributed_all_reduce -tensor $tensor -operation sum]
set result [torch::distributed_all_reduce -tensor $tensor -operation mean]

# Mixed parameter order (named syntax advantage)
set result [torch::distributed_all_reduce -operation mean -tensor $tensor]
```

### Benefits of Named Parameters

- **Self-documenting**: Parameter purpose is explicit
- **Flexible ordering**: Parameters can be specified in any order  
- **Optional clarity**: Optional parameters are clearly identified
- **Future-proof**: New parameters can be added without breaking existing code
- **Error reduction**: Less prone to parameter position mistakes

## See Also

- [torch::distributed_broadcast](distributed_broadcast.md) - Broadcast tensor from one process to all
- [torch::distributed_reduce](distributed_reduce.md) - Reduce tensors to one process
- [torch::distributed_gather](distributed_gather.md) - Gather tensors from all processes
- [torch::distributed_init](distributed_init.md) - Initialize distributed training
- [torch::distributed_barrier](distributed_barrier.md) - Synchronization barrier 