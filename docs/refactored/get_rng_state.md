# torch::get_rng_state

## Overview
Returns the current random number generator (RNG) state as a tensor. This state can be used to reproduce random number sequences or to save and restore the RNG state for deterministic behavior.

## Syntax

### Snake_case (Original)
```tcl
torch::get_rng_state
```

### CamelCase (Alias)
```tcl
torch::getRngState
```

## Parameters
This command takes no parameters.

## Return Value
Returns a tensor handle containing the current RNG state. The returned tensor is a 1D tensor with 64 elements of type Long (int64).

## Examples

### Basic Usage
```tcl
# Get the current RNG state
set rng_state [torch::get_rng_state]
puts "RNG state: $rng_state"

# Using camelCase alias
set rng_state_camel [torch::getRngState]
puts "RNG state (camelCase): $rng_state_camel"
```

### Save and Restore RNG State
```tcl
# Save current RNG state
set saved_state [torch::get_rng_state]

# Generate some random numbers
set tensor1 [torch::randn {3 3} float32 cpu]
set tensor2 [torch::randn {3 3} float32 cpu]

# Restore the saved state
torch::set_rng_state $saved_state

# Generate the same random numbers again
set tensor3 [torch::randn {3 3} float32 cpu]
set tensor4 [torch::randn {3 3} float32 cpu]

# tensor1 should be identical to tensor3
# tensor2 should be identical to tensor4
```

### Reproducible Random Generation
```tcl
# Set a specific seed
torch::manual_seed 42

# Get the state after seeding
set seeded_state [torch::get_rng_state]

# Generate random tensor
set random_tensor [torch::randn {2 2} float32 cpu]

# Later, restore the seeded state
torch::set_rng_state $seeded_state

# Generate the same random tensor
set same_random_tensor [torch::randn {2 2} float32 cpu]

# Both tensors should be identical
```

## Properties of RNG State

### Tensor Properties
```tcl
set state [torch::get_rng_state]

# Check tensor properties
set shape [torch::tensor_shape $state]  ;# Returns: 64
set dtype [torch::tensor_dtype $state]  ;# Returns: "Long"

puts "RNG state shape: $shape"
puts "RNG state dtype: $dtype"
```

### State Consistency
```tcl
# Multiple calls create different tensor handles but with same properties
set state1 [torch::get_rng_state]
set state2 [torch::get_rng_state]

# Different handles
puts "Different handles: [expr {$state1 ne $state2}]"  ;# Returns: 1

# Same properties
set shape1 [torch::tensor_shape $state1]
set shape2 [torch::tensor_shape $state2]
puts "Same shape: [expr {$shape1 == $shape2}]"  ;# Returns: 1
```

## Use Cases

### 1. Reproducible Experiments
```tcl
# Save state before experiment
set experiment_state [torch::get_rng_state]

# Run experiment with random components
proc run_experiment {} {
    set data [torch::randn {100 10} float32 cpu]
    set noise [torch::randn {100 10} float32 cpu]
    return [torch::tensor_add $data $noise]
}

# First run
set result1 [run_experiment]

# Restore state and run again
torch::set_rng_state $experiment_state
set result2 [run_experiment]

# Results should be identical
```

### 2. Parallel Processing State Management
```tcl
# Save main thread state
set main_state [torch::get_rng_state]

# Create separate states for parallel workers
set worker_states {}
for {set i 0} {$i < 4} {incr i} {
    torch::manual_seed [expr {42 + $i}]
    lappend worker_states [torch::get_rng_state]
}

# Later restore main state
torch::set_rng_state $main_state
```

### 3. Debugging Random Behavior
```tcl
# Save state before problematic operation
set debug_state [torch::get_rng_state]

# Run operation that might fail
if {[catch {
    set result [some_random_operation]
} error]} {
    puts "Error occurred: $error"
    
    # Restore state to reproduce the issue
    torch::set_rng_state $debug_state
    
    # Now you can debug the exact same sequence
    set debug_result [some_random_operation]
}
```

## Error Handling
```tcl
# Command accepts no arguments
if {[catch {torch::get_rng_state extra_arg} error]} {
    puts "Error: $error"
    ;# Output: Error: wrong # args: should be "torch::get_rng_state"
}
```

## Notes

1. **No Parameters**: This command takes no parameters and always returns the current global RNG state.

2. **Tensor Handle**: Each call returns a new tensor handle, even if the RNG state is identical.

3. **Platform Independence**: The RNG state format is consistent across different platforms.

4. **Integration**: Works seamlessly with `torch::set_rng_state` for state restoration.

5. **Memory Management**: The returned tensor follows standard PyTorch tensor lifecycle management.

## Related Commands
- `torch::set_rng_state` - Restore RNG state from a saved state tensor
- `torch::manual_seed` - Set RNG seed for reproducible random generation
- `torch::initial_seed` - Get the initial seed used for RNG
- `torch::seed` - Set RNG seed and return previous seed

## Mathematical Background
The RNG state contains all information needed to continue a pseudo-random sequence from a specific point. This includes:
- Current state of the random number generator
- Internal counters and state variables
- Implementation-specific state information

The state is represented as a 64-element tensor of 64-bit integers, providing sufficient information to fully restore the RNG state.

## Performance Considerations
- Getting RNG state is a lightweight operation
- The state tensor is relatively small (64 elements)
- State operations are thread-safe in PyTorch
- Consider caching states if calling frequently in performance-critical code 