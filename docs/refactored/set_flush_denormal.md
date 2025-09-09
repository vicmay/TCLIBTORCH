# torch::set_flush_denormal

Controls whether denormal floating-point numbers are flushed to zero during computation.

## Syntax

```tcl
# Positional syntax (backward compatible)
torch::set_flush_denormal enabled

# Named parameter syntax
torch::set_flush_denormal -enabled value

# CamelCase alias
torch::setFlushDenormal -enabled value
```

## Parameters

* `enabled` (integer): Whether to enable denormal flushing
  * `1`: Enable denormal flushing (flush denormals to zero)
  * `0`: Disable denormal flushing (preserve denormal numbers)

## Return Value

Returns a string indicating the new state:
* `denormal_flushing_enabled`: When flushing is enabled
* `denormal_flushing_disabled`: When flushing is disabled

## Description

The `set_flush_denormal` command controls how the system handles denormal floating-point numbers during computation. Denormal numbers (also called subnormal numbers) are very small numbers close to zero that require special handling by the CPU.

When denormal flushing is enabled:
- Denormal numbers are automatically rounded to zero
- This can improve performance on some CPU architectures
- May result in slight loss of precision for very small numbers

When denormal flushing is disabled:
- Denormal numbers are preserved and handled normally
- This provides maximum numerical precision
- May have performance impact on some architectures

## Examples

```tcl
# Enable denormal flushing (positional syntax)
torch::set_flush_denormal 1

# Disable denormal flushing (named parameter syntax)
torch::set_flush_denormal -enabled 0

# Enable using camelCase alias
torch::setFlushDenormal -enabled 1
```

## See Also

* [torch::get_num_threads](get_num_threads.md)
* [torch::set_num_threads](set_num_threads.md) 