# torch::manual_seed

Sets the random number generator seed for reproducible results.

## Syntax

### Current Syntax (Positional)
```tcl
torch::manual_seed seed
```

### Named Parameter Syntax
```tcl
torch::manual_seed -seed value
```

### camelCase Alias
```tcl
torch::manualSeed seed
torch::manualSeed -seed value
```

All syntaxes are fully supported and equivalent.

## Parameters

### Positional Parameters
- `seed` (required): Non-negative integer seed value for the random number generator

### Named Parameters
- `-seed` (required): Non-negative integer seed value for the random number generator
- `-s` (alternative to `-seed`): Short form parameter name

## Return Value

Returns "ok" on successful completion.

## Description

Sets the seed for PyTorch's random number generators. This affects all random operations including:
- Random tensor creation (`torch::randn`, `torch::rand`, etc.)
- Random sampling operations (`torch::bernoulli`, `torch::multinomial`, etc.)
- Neural network initialization with random weights
- Dropout operations

Setting the same seed before random operations ensures reproducible results across multiple runs.

## Examples

### Basic Usage
```tcl
# Set seed using positional syntax
torch::manual_seed 42

# Set seed using named parameter syntax
torch::manual_seed -seed 42

# Set seed using alternative parameter name
torch::manual_seed -s 42

# Set seed using camelCase alias
torch::manualSeed 42
torch::manualSeed -seed 42
```

### Reproducibility Example
```tcl
# First run
torch::manual_seed 42
set tensor1 [torch::randn {2 3}]
puts [torch::tensor_print $tensor1]

# Second run with same seed - produces identical results
torch::manual_seed 42  
set tensor2 [torch::randn {2 3}]
puts [torch::tensor_print $tensor2]

# tensor1 and tensor2 will be identical
```

### Large Seed Values
```tcl
# Large seed values are supported
torch::manual_seed 2147483647

# Zero is a valid seed
torch::manual_seed 0
```

## Error Handling

```tcl
# Missing seed parameter
catch {torch::manual_seed} msg
puts $msg  ;# -> Error in manual_seed: Required parameters missing: seed value required

# Invalid seed parameter  
catch {torch::manual_seed abc} msg
puts $msg  ;# -> Error in manual_seed: expected integer but got "abc"

# Negative seed
catch {torch::manual_seed -1} msg
puts $msg  ;# -> Error in manual_seed: Seed must be non-negative

# Unknown parameter
catch {torch::manual_seed -unknown 42} msg
puts $msg  ;# -> Error in manual_seed: Unknown parameter: -unknown. Valid parameters are: -seed, -s

# Missing value for named parameter
catch {torch::manual_seed -seed} msg
puts $msg  ;# -> Error in manual_seed: Named parameters must come in pairs
```

## Supported Seed Range

- **Minimum**: 0 (zero is valid)
- **Maximum**: 2^63-1 (platform-dependent, but very large values are supported)
- **Type**: Non-negative integer

## Implementation Notes

- This command sets the seed for all PyTorch device types (CPU, CUDA, MPS, XPU) when available
- The seed affects the global random state, so it impacts all subsequent random operations
- For distributed training, each process should set a different seed for proper randomization
- The command uses PyTorch's `torch::manual_seed()` function internally

## Related Commands

- `torch::initial_seed` / `torch::initialSeed` – Get the initial random seed
- `torch::seed` – Generate and set a new random seed automatically
- `torch::get_rng_state` / `torch::getRngState` – Get current RNG state
- `torch::set_rng_state` – Set RNG state from tensor
- `torch::randn` – Create random normal tensor
- `torch::rand` – Create random uniform tensor
- `torch::bernoulli` – Random Bernoulli sampling

## Migration Guide

### From Positional to Named Parameters
```tcl
# Old style (still supported)
torch::manual_seed 42

# New style with named parameters
torch::manual_seed -seed 42

# camelCase style for modern APIs
torch::manualSeed -seed 42
```

### Best Practices
1. **Use named parameters** for better code readability in complex scripts
2. **Use camelCase aliases** for consistency with modern PyTorch naming conventions
3. **Set seeds early** in your script before any random operations
4. **Use different seeds** for different experiments or runs
5. **Document the seed value** used for reproducible research

## Version History

- **Original**: Positional syntax support
- **Refactored**: Added dual syntax support with named parameters and camelCase alias
- **Current**: Full backward compatibility maintained 