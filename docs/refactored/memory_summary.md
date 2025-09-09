# torch::memory_summary / torch::memorySummary

**Category:** Memory Management  
**Aliases:** `torch::memory_summary`, `torch::memorySummary`

---

## Description

Returns a detailed summary of memory usage in LibTorch, including CUDA device information if available. This command provides a more comprehensive overview than `torch::memory_stats` and is useful for debugging memory-related issues.

---

## Syntax

### Original Syntax
```tcl
torch::memory_summary ?device?
```
- `device` (optional): Device identifier (e.g., "cuda:0")

### CamelCase Alias
```tcl
torch::memorySummary ?device?
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
# Get memory summary for default device
set summary [torch::memory_summary]
puts $summary
```

### 2. Using CamelCase Alias
```tcl
# Get memory summary using camelCase alias
set summary [torch::memorySummary]
puts $summary
```

### 3. Specifying Device
```tcl
# Get memory summary for a specific CUDA device
if {[regexp {CUDA Memory Summary} [torch::memory_summary]]} {
    set cuda_summary [torch::memorySummary "cuda:0"]
    puts "CUDA device 0 memory summary:\n$cuda_summary"
}
```

### 4. Memory Monitoring in an Application
```tcl
# Function to periodically check memory usage
proc monitor_memory_usage {interval_ms count} {
    for {set i 0} {$i < $count} {incr i} {
        puts "=== Memory Summary at [clock format [clock seconds]] ==="
        puts [torch::memorySummary]
        puts "================================================="
        after $interval_ms
    }
}

# Monitor memory every 5 seconds for 1 minute
monitor_memory_usage 5000 12
```

---

## Error Handling

- If too many arguments are provided, the command will return an error with the message "wrong # args: should be torch::memory_summary ?device?".
- If an internal error occurs during summary generation, an error message will be returned with details.

---

## Return Value

Returns a multi-line string containing a detailed memory summary with information such as:
- CUDA availability status
- Device count (if CUDA is available)
- Memory allocation information

Example return value:
```
CUDA Memory Summary:
Device Count: 2
```

---

## See Also

- `torch::memory_stats` - Get basic memory statistics
- `torch::memory_snapshot` - Get a snapshot of current memory state
- `torch::empty_cache` - Empty the CUDA memory cache
