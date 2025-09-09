# torch::distributed_init

Initialize distributed training for multi-GPU or single-GPU setups.

## Syntax

### snake_case (Original)
```tcl
torch::distributed_init rank world_size master_addr ?master_port? ?backend?
```

### camelCase (New)
```tcl
torch::distributedInit -rank rank -worldSize world_size -masterAddr master_addr ?-masterPort master_port? ?-backend backend?
```

## Parameters

### Required Parameters
- **rank** (integer): The rank of the current process (0-based indexing)
- **world_size** (integer): Total number of processes in the distributed training
- **master_addr** (string): IP address or hostname of the master node

### Optional Parameters
- **master_port** (integer): Port number for the master node (default: 29500)
- **backend** (string): Communication backend to use (default: "nccl")

## Parameter Details

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| rank | integer | Yes | - | Process rank (must be >= 0 and < world_size) |
| world_size | integer | Yes | - | Total number of processes (must be > 0) |
| master_addr | string | Yes | - | Master node address (IP or hostname) |
| master_port | integer | No | 29500 | Master node port (must be > 0) |
| backend | string | No | "nccl" | Communication backend |

## Return Value

Returns a status string describing the initialization result:
- For single GPU (world_size=1): "Distributed training initialized (single GPU): rank=X, world_size=Y, backend=Z"
- For multi GPU (world_size>1): "Distributed training initialized (emulated multi-GPU): rank=X, world_size=Y, backend=emulated_Z (Note: Real multi-GPU requires NCCL headers)"

## Examples

### Basic Usage - Single GPU

**Positional Syntax:**
```tcl
set result [torch::distributed_init 0 1 "127.0.0.1"]
# Returns: "Distributed training initialized (single GPU): rank=0, world_size=1, backend=nccl"
```

**Named Parameter Syntax:**
```tcl
set result [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1"]
# Returns: "Distributed training initialized (single GPU): rank=0, world_size=1, backend=nccl"
```

**camelCase Alias:**
```tcl
set result [torch::distributedInit -rank 0 -worldSize 1 -masterAddr "127.0.0.1"]
# Returns: "Distributed training initialized (single GPU): rank=0, world_size=1, backend=nccl"
```

### Multi-GPU Setup (Emulated)

**Positional Syntax:**
```tcl
set result [torch::distributed_init 0 4 "192.168.1.100" 29500 "gloo"]
# Returns: "Distributed training initialized (emulated multi-GPU): rank=0, world_size=4, backend=emulated_gloo ..."
```

**Named Parameter Syntax:**
```tcl
set result [torch::distributed_init -rank 1 -worldSize 4 -masterAddr "192.168.1.100" -masterPort 29501 -backend "gloo"]
# Returns: "Distributed training initialized (emulated multi-GPU): rank=1, world_size=4, backend=emulated_gloo ..."
```

### Different Master Addresses

```tcl
# Localhost
torch::distributed_init -rank 0 -worldSize 1 -masterAddr "localhost"

# IPv4 address
torch::distributed_init -rank 0 -worldSize 1 -masterAddr "10.0.0.1"

# IPv6 address
torch::distributed_init -rank 0 -worldSize 1 -masterAddr "::1"

# Domain name
torch::distributed_init -rank 0 -worldSize 1 -masterAddr "master.cluster.com"
```

### Custom Backends

```tcl
# NCCL backend (default)
torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -backend "nccl"

# Gloo backend
torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -backend "gloo"

# MPI backend
torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -backend "mpi"

# Custom backend
torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -backend "custom_backend"
```

## Implementation Details

### Single vs Multi-GPU Behavior

- **Single GPU** (world_size=1): Uses optimized single-GPU semantics with rank forced to 0
- **Multi-GPU** (world_size>1): Uses emulated multi-GPU mode suitable for testing and development

### Backend Support

The command accepts any backend string but currently operates in emulated mode:
- **Single GPU**: Uses the specified backend directly
- **Multi-GPU**: Prepends "emulated_" to the backend name

### Initialization State

The command updates global distributed training state:
- Sets the current rank and world size
- Marks distributed training as initialized
- Affects subsequent distributed operations

## Error Handling

The command validates all parameters and provides clear error messages:

### Missing Required Parameters
```tcl
# Error: Missing world size
catch {torch::distributed_init -rank 0 -masterAddr "127.0.0.1"} error
puts $error
# Output: "Required parameters missing or invalid: -rank, -worldSize, and -masterAddr are required"
```

### Invalid Parameter Types
```tcl
# Error: Invalid rank type
catch {torch::distributed_init -rank "not_a_number" -worldSize 1 -masterAddr "127.0.0.1"} error
puts $error
# Output: "Invalid -rank parameter. Must be an integer."
```

### Invalid Parameter Values
```tcl
# Error: Negative rank
catch {torch::distributed_init -rank -1 -worldSize 1 -masterAddr "127.0.0.1"} error
puts $error
# Output: "Required parameters missing or invalid: -rank, -worldSize, and -masterAddr are required"

# Error: Zero world size
catch {torch::distributed_init -rank 0 -worldSize 0 -masterAddr "127.0.0.1"} error
puts $error
# Output: "Required parameters missing or invalid: -rank, -worldSize, and -masterAddr are required"

# Error: Empty master address
catch {torch::distributed_init -rank 0 -worldSize 1 -masterAddr ""} error
puts $error
# Output: "Required parameters missing or invalid: -rank, -worldSize, and -masterAddr are required"
```

### Unknown Parameters
```tcl
# Error: Unknown parameter
catch {torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -unknown value} error
puts $error
# Output: "Unknown parameter: -unknown"
```

### Wrong Argument Count (Positional)
```tcl
# Error: Too few arguments
catch {torch::distributed_init 0 1} error
puts $error
# Output: "Wrong number of arguments for positional syntax. Expected: torch::distributed_init rank world_size master_addr ?master_port? ?backend?"
```

## Migration Guide

### From Positional to Named Parameters

**Before (Positional):**
```tcl
set result [torch::distributed_init 0 4 "192.168.1.1" 29500 "gloo"]
```

**After (Named Parameters):**
```tcl
set result [torch::distributed_init -rank 0 -worldSize 4 -masterAddr "192.168.1.1" -masterPort 29500 -backend "gloo"]
```

**After (camelCase):**
```tcl
set result [torch::distributedInit -rank 0 -worldSize 4 -masterAddr "192.168.1.1" -masterPort 29500 -backend "gloo"]
```

### Benefits of Named Parameters

1. **Self-documenting**: Parameter names make code more readable
2. **Flexible ordering**: Parameters can be specified in any order
3. **Optional parameters**: Easy to specify only needed optional parameters
4. **Error prevention**: Less likely to mix up parameter positions

## Best Practices

1. **Use named parameters** for new code to improve readability
2. **Specify master port** explicitly to avoid conflicts
3. **Use appropriate backends** for your hardware setup
4. **Check return value** to verify successful initialization
5. **Handle errors** appropriately in production code

## Related Commands

- `torch::get_rank` - Get current process rank
- `torch::get_world_size` - Get total number of processes
- `torch::is_distributed` - Check if distributed training is initialized
- `torch::distributed_barrier` - Synchronize all processes
- `torch::distributed_broadcast` - Broadcast tensor to all processes
- `torch::all_reduce` - Reduce tensor across all processes

## Compatibility

- **Backward Compatible**: All existing positional syntax continues to work
- **Cross-Platform**: Works on Linux, Windows, and macOS
- **Thread-Safe**: Safe to call from multiple threads
- **Version**: Available in LibTorch TCL Extension 1.0+

## Notes

- This command sets up the distributed training environment but does not perform actual GPU communication
- For production multi-GPU training, ensure NCCL libraries are properly installed
- The emulated multi-GPU mode is suitable for testing distributed training logic
- Subsequent distributed operations will use the configuration set by this command 