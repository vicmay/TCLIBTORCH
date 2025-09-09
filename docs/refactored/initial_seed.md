# torch::initial_seed

Returns the initial seed value used by the random number generator (RNG).

This seed is the starting point for PyTorch's random number generation in the current process.  Knowing the initial seed can be helpful for reproducibility when the default seed is acceptable.

## Syntax

```tcl
# snake_case (legacy)
set seed [torch::initial_seed]

# camelCase alias
set seed [torch::initialSeed]
```

There are **no parameters** for this command.  Any extra argument will raise an error.

## Returns

An integer representing the initial RNG seed.  In this implementation the value is the default PyTorch seed `2147483647`.

## Examples

```tcl
# Retrieve and print the initial seed
set seed [torch::initial_seed]
puts "Initial seed: $seed"  ;# -> 2147483647

# Using the camelCase alias
puts [torch::initialSeed]
```

## Error Handling

```tcl
# Passing any argument is invalid
catch {torch::initial_seed extra} msg
puts $msg  ;# -> Wrong # args: should be "torch::initial_seed"
```

## Related Commands

- `torch::manual_seed` – Manually set the RNG seed
- `torch::seed` – Generate and set a new random seed
- `torch::get_rng_state` – Get current RNG state
- `torch::set_rng_state` – Set RNG state

## Migration Guide

No migration is required because the command signature remains unchanged.  A new camelCase alias `torch::initialSeed` is provided for API consistency. 