# torch::synchronize

Synchronizes CUDA operations on the current or specified device, ensuring all queued operations are complete.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::synchronize ?device?
```

### Named Parameter Syntax (New)
```tcl
torch::synchronize -device device
```

### CamelCase Alias
```tcl
torch::synchronize ?device?
torch::synchronize -device device
```

## Parameters

| Parameter | Type   | Required | Description                                 |
|-----------|--------|----------|---------------------------------------------|
| device    | string | No       | Device to synchronize (e.g., "cpu", "cuda:0") |

## Returns

Returns the string `synchronized` if successful, or `cuda_not_available` if CUDA is not available.

## Description

The `torch::synchronize` command ensures that all queued CUDA operations on the specified device (or all devices if none is specified) are complete. This is useful for timing, debugging, and ensuring deterministic results in GPU computations.

- If CUDA is not available, returns `cuda_not_available`.
- If a device is specified, synchronizes that device (if it is a CUDA device).
- If no device is specified, synchronizes all CUDA devices.

## Examples

### Basic Usage
```tcl
# Synchronize all CUDA devices (or no-op if CUDA not available)
set result [torch::synchronize]
puts "Result: $result"
```

### Synchronize a Specific Device
```tcl
# Synchronize CUDA device 0
set result [torch::synchronize cuda:0]
puts "Result: $result"

# Named parameter syntax
set result [torch::synchronize -device cuda:0]
puts "Result: $result"
```

### Using the CamelCase Alias
```tcl
set result [torch::synchronize]
set result2 [torch::synchronize -device cpu]
```

### Error Handling
```tcl
# Too many positional arguments
catch {torch::synchronize cpu extra} result
puts "Error: $result"

# Named parameter without value
catch {torch::synchronize -device} result
puts "Error: $result"

# Unknown named parameter
catch {torch::synchronize -unknown foo} result
puts "Error: $result"
```

## Return Value

Returns the string `synchronized` if synchronization is successful, or `cuda_not_available` if CUDA is not available.

## Notes

- If CUDA is not available, the command is a no-op and returns `cuda_not_available`.
- Synchronization is important for accurate timing and debugging of GPU operations.
- The command is safe to call even if no CUDA device is present.

## Compatibility

✅ **Backward Compatible**: All existing code using positional parameters continues to work  
✅ **Named Parameters**: New code can use clearer `-parameter value` syntax  
✅ **camelCase**: Modern `torch::synchronize` alias available  

## Migration Guide

```tcl
# Old style → New style
torch::synchronize cuda:0 → torch::synchronize -device cuda:0
```

## See Also

- `torch::empty_cache` - Empty CUDA memory cache
- `torch::memory_stats` - Get memory statistics
- `torch::memory_summary` - Get memory summary 