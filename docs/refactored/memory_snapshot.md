# torch::memory_snapshot / torch::memorySnapshot

**Category:** Memory Management  
**Aliases:** `torch::memory_snapshot`, `torch::memorySnapshot`

---

## Description

Returns a snapshot of the current memory state, including timestamp and CUDA device information if available. This command is useful for debugging memory usage and tracking CUDA device availability.

---

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::memory_snapshot
```

### Named Parameter Syntax
This command doesn't accept any parameters, but the dual syntax parser will validate that no invalid parameters are provided.

### CamelCase Alias
```tcl
torch::memorySnapshot
```

---

## Parameters

This command takes no parameters.

---

## Examples

### 1. Basic Usage
```tcl
# Get current memory snapshot
set snapshot [torch::memory_snapshot]
puts $snapshot
```

### 2. Using CamelCase Alias
```tcl
# Get current memory snapshot using camelCase alias
set snapshot [torch::memorySnapshot]
puts $snapshot
```

### 3. Parsing Snapshot Information
```tcl
# Get and parse memory snapshot
set snapshot [torch::memorySnapshot]

# Extract timestamp
if {[regexp {timestamp: (\d+)} $snapshot -> timestamp]} {
    puts "Snapshot taken at: [clock format $timestamp]"
}

# Check CUDA availability
if {[regexp {cuda_available: (true|false)} $snapshot -> cuda_available]} {
    if {$cuda_available eq "true"} {
        regexp {device_count: (\d+)} $snapshot -> device_count
        puts "CUDA is available with $device_count device(s)"
    } else {
        puts "CUDA is not available"
    }
}
```

### 4. Monitoring Memory Over Time
```tcl
# Function to monitor memory at intervals
proc monitor_memory {count interval_ms} {
    for {set i 0} {$i < $count} {incr i} {
        puts "Memory snapshot $i: [torch::memorySnapshot]"
        after $interval_ms
    }
}

# Monitor memory every 5 seconds for 1 minute
monitor_memory 12 5000
```

---

## Error Handling

- If any arguments are provided, the command will return an error with the message "wrong # args: should be torch::memory_snapshot".
- If an internal error occurs during snapshot generation, an error message will be returned with details.

---

## Return Value

Returns a string containing the memory snapshot with the following information:
- `timestamp`: Unix timestamp when the snapshot was taken
- `cuda_available`: Boolean indicating whether CUDA is available
- `device_count`: Number of CUDA devices (only if CUDA is available)

Example return value:
```
timestamp: 1625612345 cuda_available: true device_count: 2
```

---

## See Also

- `torch::memory_stats` - Get detailed memory statistics
- `torch::memory_summary` - Get a summary of memory usage
- `torch::empty_cache` - Empty the CUDA memory cache
