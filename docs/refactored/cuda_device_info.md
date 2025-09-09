# torch::cuda_device_info

Get detailed information about a CUDA device including name and compute capability.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::cuda_device_info [device_id]
```

### Named Parameter Syntax  
```tcl
torch::cuda_device_info [-device_id device_id]
```

### CamelCase Alias
```tcl
torch::cudaDeviceInfo [device_id]
torch::cudaDeviceInfo [-device_id device_id]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| device_id | int | 0 | The CUDA device ID to query (0-based indexing) |

## Returns

Returns a string containing device information in the format:
```
Device <id>: <device_name> (Compute <major>.<minor>)
```

Where:
- `<id>`: The device ID number
- `<device_name>`: The name of the CUDA device (e.g., "GeForce RTX 3080")
- `<major>.<minor>`: The compute capability version (e.g., "8.6")

## Error Handling

- **CUDA Not Available**: Returns error "CUDA not available" if CUDA is not installed or available
- **Invalid Device ID**: Returns error "Invalid device ID" if the specified device ID doesn't exist
- **Invalid Parameters**: Returns error for invalid parameter types or unknown parameters

## Examples

### Basic Usage
```tcl
# Get info for default device (device 0)
set info [torch::cuda_device_info]
puts $info
# Output: Device 0: GeForce RTX 3080 (Compute 8.6)
```

### Specific Device
```tcl
# Get info for device 1 using positional syntax
set info [torch::cuda_device_info 1]
puts $info

# Same using named parameter syntax  
set info [torch::cuda_device_info -device_id 1]
puts $info

# Same using camelCase alias
set info [torch::cudaDeviceInfo 1]
puts $info
```

### Error Handling
```tcl
# Handle case when CUDA is not available
if {[catch {torch::cuda_device_info} result]} {
    puts "Error: $result"
    # Output: Error: CUDA not available
} else {
    puts "Device info: $result"
}

# Handle invalid device ID
if {[catch {torch::cuda_device_info 999} result]} {
    puts "Error: $result"
    # Output: Error: Invalid device ID
}
```

### Getting Multiple Device Info
```tcl
# Get info for all available devices
set device_count [torch::cuda_device_count]
for {set i 0} {$i < $device_count} {incr i} {
    set info [torch::cuda_device_info $i]
    puts "Device $i: $info"
}
```

## Device Information Format

The returned device information includes:

1. **Device Number**: Zero-based index of the device
2. **Device Name**: Marketing name of the GPU (from CUDA driver)
3. **Compute Capability**: Major and minor version indicating supported CUDA features

### Compute Capability Examples
- **7.5**: RTX 20 series (Turing architecture)
- **8.6**: RTX 30 series (Ampere architecture)  
- **8.9**: RTX 40 series (Ada Lovelace architecture)

## Migration from Positional to Named Parameters

### Old Style (Positional)
```tcl
set info [torch::cuda_device_info 0]
```

### New Style (Named Parameters)
```tcl
set info [torch::cuda_device_info -device_id 0]
```

### Using CamelCase
```tcl
set info [torch::cudaDeviceInfo -device_id 0]
```

## Related Commands

- **torch::cuda_is_available**: Check if CUDA is available
- **torch::cuda_device_count**: Get the number of available CUDA devices  
- **torch::cuda_memory_info**: Get memory information for a CUDA device
- **torch::tensor_to**: Move tensors to specific CUDA devices

## Technical Notes

### Device ID Validation
- Device IDs are zero-based (0, 1, 2, ...)
- Must be within the range [0, device_count-1]
- Negative device IDs are invalid

### CUDA Driver Dependencies
- Requires CUDA runtime to be properly installed
- Device information comes from CUDA driver properties
- Works with all CUDA-capable devices

### Performance Considerations
- Device queries are lightweight operations
- Information is cached by CUDA driver
- Safe to call repeatedly without performance concerns

## Implementation Details

This command uses the CUDA Runtime API's `cudaGetDeviceProperties()` function to retrieve comprehensive device information including:

- Device name from device properties
- Compute capability (major and minor versions)
- Device index validation against available device count

The dual syntax implementation maintains full backward compatibility while providing modern named parameter access. 