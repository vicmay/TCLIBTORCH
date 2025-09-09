# torch::distributed_gather

**Gather tensors from all processes in distributed training.**

## Overview

The `torch::distributed_gather` command implements a distributed tensor gathering operation that collects tensors from all participating processes. This is a simplified implementation that stacks the input tensor for demonstration purposes.

## Syntax Support

This command supports both legacy positional syntax and modern named parameter syntax, along with a camelCase alias.

### Named Parameter Syntax (Recommended)

```tcl
torch::distributed_gather -tensor <tensor_name> [-dst <dst_process>] [-group <group_name>]
torch::distributedGather -tensor <tensor_name> [-dst <dst_process>] [-group <group_name>]  # camelCase alias
```

### Positional Syntax (Legacy)

```tcl
torch::distributed_gather <tensor_name> [<dst_process>] [<group_name>]
torch::distributedGather <tensor_name> [<dst_process>] [<group_name>]  # camelCase alias
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tensor` | string | Yes | - | Name of the tensor to gather |
| `dst` | integer | No | 0 | Destination process rank |
| `group` | string | No | "" | Process group name |

## Return Value

Returns a tensor handle representing the gathered tensor with an additional dimension at the beginning.

## Shape Transformation

The output tensor has one additional dimension compared to the input:
- Input shape: `[D1, D2, ..., DN]`
- Output shape: `[1, D1, D2, ..., DN]`

## Examples

### Basic Usage

#### Named Parameter Syntax
```tcl
# Create a tensor
set input [torch::ones {2 3} float32]

# Gather tensor using named parameters
set result [torch::distributed_gather -tensor $input]
# result shape: {1 2 3}

# With destination process
set result [torch::distributed_gather -tensor $input -dst 1]

# With process group
set result [torch::distributed_gather -tensor $input -group "workers"]

# All parameters
set result [torch::distributed_gather -tensor $input -dst 0 -group "main"]
```

#### Positional Syntax (Legacy)
```tcl
# Create a tensor
set input [torch::ones {2 3} float32]

# Basic gather
set result [torch::distributed_gather $input]

# With destination process
set result [torch::distributed_gather $input 1]

# With destination and group
set result [torch::distributed_gather $input 0 "workers"]
```

#### CamelCase Alias
```tcl
# Using camelCase alias with named parameters
set result [torch::distributedGather -tensor $input -dst 1]

# Using camelCase alias with positional syntax
set result [torch::distributedGather $input 0]
```

### Advanced Examples

#### Different Data Types
```tcl
# Float64 tensor
set tensor_f64 [torch::zeros {3 3} float64]
set result [torch::distributed_gather -tensor $tensor_f64]

# Integer tensor
set tensor_int [torch::ones {2 2} int32]
set result [torch::distributed_gather -tensor $tensor_int]
```

#### Multiple Dimension Tensors
```tcl
# 1D tensor
set tensor_1d [torch::ones {5} float32]
set result [torch::distributed_gather -tensor $tensor_1d]
# result shape: {1 5}

# 3D tensor
set tensor_3d [torch::zeros {2 3 4} float32]
set result [torch::distributed_gather -tensor $tensor_3d]
# result shape: {1 2 3 4}

# 4D tensor
set tensor_4d [torch::ones {2 2 2 2} float32]
set result [torch::distributed_gather -tensor $tensor_4d]
# result shape: {1 2 2 2 2}
```

#### Different Process Ranks
```tcl
set tensor [torch::ones {2 2} float32]

# Gather to different processes
set result_0 [torch::distributed_gather -tensor $tensor -dst 0]
set result_1 [torch::distributed_gather -tensor $tensor -dst 1]
set result_neg [torch::distributed_gather -tensor $tensor -dst -1]
```

#### With Process Groups
```tcl
set tensor [torch::ones {3 3} float32]

# Named groups
set result [torch::distributed_gather -tensor $tensor -group "workers"]
set result [torch::distributed_gather -tensor $tensor -group "data_parallel"]

# Empty group (default)
set result [torch::distributed_gather -tensor $tensor -group ""]

# Group with spaces
set result [torch::distributed_gather -tensor $tensor -group "model parallel"]
```

### Workflow Integration

#### Distributed Training Workflow
```tcl
# Create model output
set model_output [torch::ones {4 10} float32]

# Gather outputs from all processes
set gathered_outputs [torch::distributed_gather -tensor $model_output -dst 0]

# Process the gathered results
set combined_shape [torch::tensor_shape $gathered_outputs]
puts "Gathered tensor shape: $combined_shape"
```

#### Mixed Syntax Usage
```tcl
set tensor [torch::ones {2 3} float32]

# Use both syntaxes in the same workflow
set result1 [torch::distributed_gather $tensor 0]                          # positional
set result2 [torch::distributedGather -tensor $tensor -dst 0]              # named camelCase
set result3 [torch::distributed_gather -tensor $tensor -dst 0 -group ""]   # named snake_case
```

## Error Handling

### Common Errors

#### Missing Required Parameters
```tcl
# This will fail - missing tensor
catch {torch::distributed_gather -dst 0} error
puts $error
# Output: Required parameter missing: -tensor
```

#### Invalid Tensor Names
```tcl
# This will fail - invalid tensor name
catch {torch::distributed_gather -tensor "nonexistent"} error
puts $error
# Output: Invalid tensor name
```

#### Invalid Parameter Types
```tcl
set tensor [torch::ones {2 2} float32]

# This will fail - dst must be integer
catch {torch::distributed_gather -tensor $tensor -dst "not_a_number"} error
puts $error
# Output: Invalid -dst parameter. Must be an integer.
```

#### Unknown Parameters
```tcl
set tensor [torch::ones {2 2} float32]

# This will fail - unknown parameter
catch {torch::distributed_gather -tensor $tensor -invalid_param value} error
puts $error
# Output: Unknown parameter: -invalid_param
```

#### Missing Parameter Values
```tcl
set tensor [torch::ones {2 2} float32]

# This will fail - missing value for -dst
catch {torch::distributed_gather -tensor $tensor -dst} error
puts $error
# Output: Missing value for parameter
```

#### Wrong Number of Positional Arguments
```tcl
# Too few arguments
catch {torch::distributed_gather} error
puts $error
# Output: Required parameter missing: -tensor

# Too many arguments
set tensor [torch::ones {2 2} float32]
catch {torch::distributed_gather $tensor 0 "group" extra_arg} error
puts $error
# Output: Wrong number of arguments for positional syntax
```

## Migration Guide

### From Positional to Named Parameters

#### Basic Migration
```tcl
# OLD (Positional)
set result [torch::distributed_gather $tensor]

# NEW (Named Parameters) 
set result [torch::distributed_gather -tensor $tensor]
```

#### With Destination
```tcl
# OLD (Positional)
set result [torch::distributed_gather $tensor 1]

# NEW (Named Parameters)
set result [torch::distributed_gather -tensor $tensor -dst 1]
```

#### With All Parameters
```tcl
# OLD (Positional)
set result [torch::distributed_gather $tensor 0 "workers"]

# NEW (Named Parameters)
set result [torch::distributed_gather -tensor $tensor -dst 0 -group "workers"]
```

#### Using CamelCase Alias
```tcl
# OLD (snake_case)
set result [torch::distributed_gather -tensor $tensor -dst 1]

# NEW (camelCase)
set result [torch::distributedGather -tensor $tensor -dst 1]
```

### Migration Benefits

1. **Readability**: Named parameters make code self-documenting
2. **Flexibility**: Parameters can be specified in any order
3. **Maintainability**: Easier to understand and modify
4. **Error Prevention**: Less prone to parameter ordering mistakes

## Implementation Notes

### Current Behavior
- This is a simplified implementation for demonstration purposes
- In a real distributed environment, this would gather tensors from all processes
- The current implementation stacks the input tensor, adding one dimension

### Real Distributed Usage
In a production distributed training environment:
- The command would collect tensors from all participating processes
- The first dimension of the output would equal the world size
- Synchronization barriers would ensure all processes participate

### Performance Considerations
- Gathering large tensors can be memory-intensive
- Network bandwidth affects performance in distributed settings
- Consider using reduce operations when full gathering isn't necessary

## Related Commands

- `torch::distributed_scatter` - Scatter tensors to all processes
- `torch::distributed_reduce` - Reduce tensors across processes
- `torch::distributed_all_reduce` - All-reduce operation
- `torch::distributed_all_to_all` - All-to-all communication
- `torch::distributed_broadcast` - Broadcast tensor to all processes

## See Also

- [Distributed Training Guide](../distributed_training.md)
- [Tensor Operations Reference](../tensor_operations.md)
- [API Migration Guide](../migration_guide.md) 