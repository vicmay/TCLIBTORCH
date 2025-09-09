# torch::seed / torch::seed

Generates and sets a new random seed for the random number generator.

## Syntax

```tcl
torch::seed
torch::seed  ;# camelCase alias
```

## Parameters

This command takes no parameters.

## Return Value

Returns a 64-bit integer representing the new seed value that was generated and set.

## Description

The `seed` command generates a new random seed using the system's high-resolution clock and sets it as the current seed for PyTorch's random number generator. This affects all subsequent random number generation operations.

The command:
1. Generates a new seed value using system time
2. Sets this seed as the current RNG seed using `torch::manual_seed`
3. Returns the generated seed value

This is useful when you want to:
- Initialize the random number generator with a fresh random seed
- Get a different sequence of random numbers each time your script runs
- Reset the random state in a non-deterministic way

## Examples

### Basic Usage
```tcl
set new_seed [torch::seed]
puts "Generated new seed: $new_seed"
```

### Using with Random Number Generation
```tcl
# Get current random numbers
set tensor1 [torch::normal -size {3} -mean 0.0 -std 1.0]

# Generate new seed and get different random numbers
torch::seed
set tensor2 [torch::normal -size {3} -mean 0.0 -std 1.0]
;# tensor2 will have different values than tensor1
```

### Using camelCase Alias
```tcl
set new_seed [torch::seed]
```

## Error Conditions

- Returns error if any arguments are provided (command takes no parameters)

## Related Commands

- `torch::manual_seed` - Set a specific seed value
- `torch::initial_seed` - Get the initial seed value
- `torch::get_rng_state` - Get current RNG state
- `torch::set_rng_state` - Set RNG state

## Migration Guide

This command has no positional syntax variant as it takes no parameters. The usage remains the same in both legacy and modern code. 