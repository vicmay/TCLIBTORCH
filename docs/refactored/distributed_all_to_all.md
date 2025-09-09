# torch::distributed_all_to_all

Performs an all-to-all communication operation between distributed processes, exchanging data between all processes.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::distributed_all_to_all tensor ?group?
```

### Named Parameter Syntax  
```tcl
torch::distributed_all_to_all -tensor tensor ?-group group?
```

### CamelCase Alias
```tcl
torch::distributedAllToAll tensor ?group?
torch::distributedAllToAll -tensor tensor ?-group group?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tensor` / `-tensor` | string | Yes | - | Handle to the input tensor |
| `group` / `-group` | string | No | "" | Process group identifier for the operation |

## Return Value

Returns a string handle to the result tensor after all-to-all communication.

## Description

The `torch::distributed_all_to_all` command performs distributed all-to-all communication where each process exchanges data with every other process in the group. This is fundamental for data redistribution in parallel computing.

### Key Features:
- **Data exchange**: Each process sends data to and receives data from all other processes
- **Group support**: Optional process group for targeted communication
- **Shape preservation**: Output tensor maintains input tensor shape
- **Simulation mode**: Works in single-GPU environments for testing

## Examples

### Basic Usage
```tcl
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4} -dtype float32]

# Positional syntax
set result [torch::distributed_all_to_all $tensor]

# Named parameter syntax
set result [torch::distributed_all_to_all -tensor $tensor]

# With group parameter
set result [torch::distributed_all_to_all $tensor "worker_group"]
set result [torch::distributed_all_to_all -tensor $tensor -group "worker_group"]
```

### CamelCase Alias
```tcl
set result [torch::distributedAllToAll $tensor]
set result [torch::distributedAllToAll -tensor $tensor -group "group1"]
```

## Technical Details

### Communication Pattern
In true distributed mode:
1. Each process contributes a tensor
2. Data is exchanged between all process pairs  
3. Result contains redistributed data from all processes

### Simulation Mode
In single-GPU development environments:
- Returns a clone of the input tensor
- Preserves shape and data type
- Enables testing without distributed setup

## Error Handling

```tcl
# Invalid tensor
catch {torch::distributed_all_to_all invalid_tensor} error
# Error: "Invalid tensor name"

# Missing parameters
catch {torch::distributed_all_to_all} error
# Error: "Wrong number of arguments" or "Required parameter missing"
```

## See Also

- [torch::distributed_all_reduce](distributed_all_reduce.md) - All-reduce operations
- [torch::distributed_broadcast](distributed_broadcast.md) - Broadcast operations  
- [torch::distributed_gather](distributed_gather.md) - Gather operations 