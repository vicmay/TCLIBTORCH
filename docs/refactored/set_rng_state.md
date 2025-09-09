# torch::set_rng_state

Sets the state of the random number generator from a state tensor.

## Syntax

```tcl
# Positional syntax (backward compatible)
torch::set_rng_state state_tensor

# Named parameter syntax
torch::set_rng_state -stateTensor tensor
torch::set_rng_state -state_tensor tensor  # snake_case alternative

# CamelCase alias
torch::setRngState -stateTensor tensor
```

## Parameters

* `state_tensor` (tensor): A tensor containing the RNG state
  * Must be a valid tensor handle
  * Should be obtained from `torch::get_rng_state`
  * First element is used as a seed value

## Return Value

Returns `ok` on successful execution.

## Description

The `set_rng_state` command sets the state of the random number generator using a state tensor. This allows you to restore a previously saved RNG state, which is useful for reproducibility in random operations.

In the current implementation, since PyTorch C++ doesn't expose direct RNG state manipulation, the command uses the first element of the state tensor as a seed value for `torch::manual_seed`.

## Examples

```tcl
# Save current RNG state
set state [torch::get_rng_state]

# Generate some random numbers
set rand1 [torch::rand {5}]

# Restore RNG state
torch::set_rng_state $state

# Generate same sequence again
set rand2 [torch::rand {5}]

# Should be identical
puts [torch::equal $rand1 $rand2]  ;# Outputs: 1

# Using named parameter syntax
torch::set_rng_state -stateTensor $state

# Using camelCase alias
torch::setRngState -stateTensor $state
```

## Error Handling

The command will raise an error if:
* No state tensor is provided
* The state tensor name is invalid
* Extra arguments are provided in positional syntax

## Related Commands

* `torch::get_rng_state` - Get the current RNG state
* `torch::manual_seed` - Set a specific random seed
* `torch::seed` - Generate a new random seed 