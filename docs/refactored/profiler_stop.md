# torch::profiler_stop

Stops the PyTorch profiler and collects the profiling results.

## Syntax

### Legacy Syntax (Positional Parameters)
```tcl
torch::profiler_stop
```

### Modern Syntax (Named Parameters)
```tcl
torch::profiler_stop
```

### CamelCase Alias
```tcl
torch::profilerStop
```

## Parameters

None. This command takes no parameters.

## Return Value

Returns "profiler_stopped" on success.

## Description

The `profiler_stop` command stops the active profiling session and collects the profiling results. It should be called after `profiler_start` to end the profiling session.

## Examples

### Basic Usage
```tcl
torch::profiler_start
# ... perform operations to profile ...
torch::profiler_stop
```

### Using CamelCase Alias
```tcl
torch::profilerStart
# ... perform operations to profile ...
torch::profilerStop
```

## See Also

- [torch::profiler_start](profiler_start.md) - Start the profiler 