# torch::template / torch::template

## Overview
Brief description of what this command does.

## Syntax

### Original Syntax (Backward Compatible)
```tcl
torch::template arg1 arg2 arg3
```

### New Syntax (Named Parameters)
```tcl
torch::template -param1 value1 -param2 value2 -param3 value3
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-param1` | type | required | Description of parameter 1 |
| `-param2` | type | default_value | Description of parameter 2 |
| `-param3` | type | default_value | Description of parameter 3 |

## Examples

### Basic Usage
```tcl
# Original syntax
set result [torch::template value1 value2 value3]

# New syntax
set result [torch::template -param1 value1 -param2 value2 -param3 value3]
```

### With Defaults
```tcl
# Only specify required parameters, use defaults for others
set result [torch::template -param1 value1]
```

### Advanced Usage
```tcl
# Example with specific parameter combinations
set result [torch::template -param1 value1 -param3 value3]
```

## Migration Guide

### Before (Old Syntax)
```tcl
torch::template arg1 arg2 arg3
```

### After (New Syntax)
```tcl
torch::template -param1 arg1 -param2 arg2 -param3 arg3
```

### Parameter Mapping
| Old Position | New Parameter | Description |
|--------------|---------------|-------------|
| 1st argument | `-param1` | Description |
| 2nd argument | `-param2` | Description |
| 3rd argument | `-param3` | Description |

## Return Value
Description of what the command returns.

## Error Conditions
- Invalid parameter names
- Missing required parameters
- Invalid parameter values
- Type mismatches

## Notes
- Both syntaxes are supported for backward compatibility
- Named parameters provide better readability and flexibility
- Default values reduce parameter verbosity
- Error messages are more descriptive with named parameters

## Related Commands
- `torch::related_command1` - Related functionality
- `torch::related_command2` - Related functionality

## See Also
- [API Reference](../api_reference.md)
- [Migration Guide](../migration_guide.md)
- [Examples](../examples.md) 