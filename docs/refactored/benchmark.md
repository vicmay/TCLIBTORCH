# torch::benchmark

Benchmarks PyTorch operations to measure performance characteristics on different hardware configurations.

## Aliases
- `torch::benchmark`  (already in camelCase)

---

## Positional Syntax (back-compat)
```tcl
set time_microseconds [torch::benchmark <operation> ?iterations? ?size? ?dtype? ?device? ?verbose?]
```

## Named Parameter Syntax (recommended)
```tcl
set time_microseconds [torch::benchmark \
    -operation    <string>         ;# required: matmul, add, conv2d \
    -iterations   <int>            ;# default: 1 \
    -size         <string>         ;# default: 1000x1000 \
    -dtype        <string>         ;# default: float32 \
    -device       <string>         ;# default: cpu \
    -verbose      <0|1> ]          ;# default: 0
```
Parameter aliases:
* `-operation` / `-op`
* `-iterations` / `-iter`

### Parameters
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| **-operation** | string | — | Operation to benchmark: matmul/mm, add, conv2d |
| **-iterations** | int | 1 | Number of iterations to run (must be positive) |
| **-size** | string | 1000x1000 | Tensor dimensions: "NxM", "NxMxP", or single value |
| **-dtype** | string | float32 | Data type: float32, float64, double, float, int32, int64 |
| **-device** | string | cpu | Device: cpu, cuda |
| **-verbose** | bool (0/1) | 0 | Enable detailed output with operation info |

### Returns
- **Verbose=0**: Integer microseconds elapsed
- **Verbose=1**: Detailed string with operation, iterations, size, and timing

---

## Examples
```tcl
# Basic matrix multiplication benchmark
set time [torch::benchmark matmul]
puts "Time: $time microseconds"

# Positional (legacy) - multiple iterations
set time [torch::benchmark matmul 5 500x500 float64 cpu 0]

# Named (preferred) - detailed configuration
set time [torch::benchmark \
    -operation matmul \
    -iterations 3 \
    -size 1024x1024 \
    -dtype float64 \
    -device cpu \
    -verbose 1]

# Different operations
set add_time [torch::benchmark -operation add -size 2000x2000]
set conv_time [torch::benchmark -operation conv2d -size 1x64x128x128]

# Parameter aliases
set time [torch::benchmark -op mm -iter 2 -size 800x800]
```

---

## Supported Operations

### Matrix Multiplication (`matmul` or `mm`)
- **Input**: Square or rectangular matrices
- **Size format**: "NxM" (e.g., "1000x1000")
- **Operation**: `torch::mm(A, A)` where A is random tensor
- **Use case**: Dense linear algebra performance

### Element-wise Addition (`add`)
- **Input**: Same-sized tensors
- **Size format**: "NxM" or single dimension (e.g., "1000", "500x500")
- **Operation**: `A + B` where A, B are random tensors
- **Use case**: Element-wise operation performance

### 2D Convolution (`conv2d`)
- **Input**: 4D tensors (batch, channels, height, width)
- **Size format**: "NxCxHxW" (e.g., "1x64x224x224")
- **Operation**: Conv2d with 3x3 kernel, 32 output channels
- **Use case**: CNN layer performance

## Performance Notes
- **CUDA synchronization**: Automatically handles GPU timing accuracy
- **Multiple iterations**: More iterations provide more stable timing
- **Size scaling**: Larger tensors typically take longer (non-linear for matmul)
- **Data types**: float64 generally slower than float32
- **Memory bandwidth**: Some operations are memory-bound vs compute-bound

---

## Error Messages
| Situation | Message |
|-----------|---------|
| Missing operation (named) | `Missing required parameter: -operation` |
| Unknown operation | `Unknown operation: xyz (supported: matmul, add, conv2d)` |
| Invalid iterations | `Invalid iterations: must be positive integer` |
| Unknown parameter | `Unknown parameter: -foo` |
| Conv2d wrong dimensions | `conv2d requires 4D tensor size: NxCxHxW` |
| Invalid size format | `invalid_argument` (from std::stoi) |

---

## Migration Guide
Old:
```tcl
set time [torch::benchmark matmul]
```
New (same - already modern):
```tcl
set time [torch::benchmark -operation matmul]
```

Enhanced usage:
```tcl
# Old (limited)
set time [torch::benchmark matmul]

# New (full control)
set detailed [torch::benchmark \
    -operation matmul \
    -iterations 5 \
    -size 2048x2048 \
    -dtype float64 \
    -verbose 1]
```

---

## Performance Comparison Example
```tcl
# Compare operation performance
set matmul_time [torch::benchmark -op matmul -size 1000x1000]
set add_time [torch::benchmark -op add -size 1000x1000]
puts "Matrix multiply: $matmul_time μs"
puts "Addition: $add_time μs"
puts "Ratio: [expr {double($matmul_time)/$add_time}]x"

# Size scaling analysis
foreach size {100x100 200x200 400x400 800x800} {
    set time [torch::benchmark -op matmul -size $size]
    puts "Size $size: $time μs"
}

# Iteration stability
set time [torch::benchmark -op matmul -iter 10 -size 500x500 -verbose 1]
puts $time
```

---

## Tests
Validated by `tests/refactored/benchmark_test.tcl` (positional, named, operations, errors, performance characteristics).

## Compatibility
✅ Positional syntax retained • ✅ Named parameters added • ✅ Already camelCase 

## Related Commands

- **[torch::profiler_start](profiler_start.md)**: Start PyTorch profiler
- **[torch::profiler_stop](profiler_stop.md)**: Stop PyTorch profiler  
- **[torch::synchronize](synchronize.md)**: CUDA synchronization
- **[torch::get_num_threads](get_num_threads.md)**: Get thread count
- **[torch::set_num_threads](set_num_threads.md)**: Set thread count 