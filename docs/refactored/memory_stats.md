# torch::memory_stats / torch::memoryStats

**Category:** Memory Management  
**Aliases:** `torch::memory_stats`, `torch::memoryStats`

---

## Description

Returns statistics about the current memory usage in LibTorch, including CUDA device information if available. This command is useful for monitoring memory usage and diagnosing memory-related issues.

---

## Syntax

### Original Syntax
```tcl
torch::memory_stats ?device?
```
- `device` (optional): Device identifier (e.g., "cuda:0")

### CamelCase Alias
```tcl
torch::memoryStats ?device?
```
- `device` (optional): Device identifier (e.g., "cuda:0")

---

## Parameters

| Name   | Type   | Required | Default | Description                                 |
|--------|--------|----------|---------|---------------------------------------------|
| device | string | No       | -       | Device identifier (e.g., "cuda:0")          |

---

## Examples

### 1. Basic Usage
```tcl
# Get memory statistics for default device
set stats [torch::memory_stats]
puts $stats
```

### 2. Using CamelCase Alias
```tcl
# Get memory statistics using camelCase alias
set stats [torch::memoryStats]
puts $stats
```

### 3. Specifying Device
```tcl
# Get memory statistics for a specific CUDA device
if {[regexp {cuda_available: true} [torch::memory_stats]]} {
    set cuda_stats [torch::memoryStats "cuda:0"]
    puts "CUDA device 0 memory stats: $cuda_stats"
}
```

### 4. Parsing Memory Statistics
```tcl
# Get and parse memory statistics
set stats [torch::memoryStats]

# Check CUDA availability
if {[regexp {cuda_available: (true|false)} $stats -> cuda_available]} {
    if {$cuda_available eq "true"} {
        regexp {device_count: (\d+)} $stats -> device_count
        puts "CUDA is available with $device_count device(s)"
    } else {
        puts "CUDA is not available"
    }
}
```

---

## Error Handling

- If too many arguments are provided, the command will return an error with the message "wrong # args: should be torch::memory_stats ?device?".
- If an internal error occurs during statistics generation, an error message will be returned with details.

---

## Return Value

Returns a string containing memory statistics with the following information:
- `cuda_available`: Boolean indicating whether CUDA is available
- `device_count`: Number of CUDA devices (only if CUDA is available)

Example return value:
```
cuda_available: true device_count: 2
```

---

## See Also

- `torch::memory_snapshot` - Get a snapshot of current memory state
- `torch::memory_summary` - Get a summary of memory usage
- `torch::empty_cache` - Empty the CUDA memory cache
