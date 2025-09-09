# torch::profiler_start

Starts the PyTorch profiler to collect performance metrics for subsequent operations.

## Syntax

### Legacy Syntax (Positional Parameters)
```tcl
torch::profiler_start ?config?
```

### Modern Syntax (Named Parameters)
```tcl
torch::profiler_start -config config
```

### CamelCase Alias
```tcl
torch::profilerStart ?config?
```

## Parameters

- `config` or `-config` (string, optional)
  - Configuration string for the profiler
  - Format: "key1=value1,key2=value2,..."
  - Common options:
    - `verbose=1` - Enable verbose output
    - `with_stack=1` - Include stack traces
    - `with_flops=1` - Include FLOPS calculations
    - `with_modules=1` - Include module information

## Return Value

Returns "profiler_started" on success.

## Description

The `profiler_start` command initiates profiling of PyTorch operations. Once started, it will collect performance metrics for all subsequent operations until `profiler_stop` is called.

## Examples

### Basic Usage
```tcl
torch::profiler_start
# ... perform operations to profile ...
torch::profiler_stop
```

### With Configuration
```tcl
torch::profiler_start "verbose=1,with_stack=1"
# ... perform operations to profile ...
torch::profiler_stop
```

### Using CamelCase Alias
```tcl
torch::profilerStart "verbose=1"
# ... perform operations to profile ...
torch::profilerStop
```

## See Also

- [torch::profiler_stop](profiler_stop.md) - Stop the profiler and collect results 