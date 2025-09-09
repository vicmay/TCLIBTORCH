# torch::set_num_threads

Sets the number of threads used for parallel CPU operations in LibTorch.

## Syntax

```tcl
# Positional syntax (backward compatible)
torch::set_num_threads num_threads

# Named parameter syntax
torch::set_num_threads -numThreads value
torch::set_num_threads -num_threads value  # snake_case alternative

# CamelCase alias
torch::setNumThreads -numThreads value
```

## Parameters

* `num_threads` (integer): The number of threads to use for parallel operations
  * Must be a positive integer (> 0)
  * Affects CPU operations only
  * Default value depends on system configuration

## Return Value

Returns `threads_set` on successful execution.

## Description

The `set_num_threads` command controls the number of threads used for parallel CPU operations in LibTorch. This affects operations that can be parallelized, such as matrix multiplication, convolutions, and other compute-intensive operations.

The setting applies globally to all subsequent operations. The actual parallelization may still be subject to the operation type and size of the data being processed.

## Examples

```tcl
# Using positional syntax
torch::set_num_threads 4

# Using named parameter syntax
torch::set_num_threads -numThreads 2
torch::set_num_threads -num_threads 3  # snake_case alternative

# Using camelCase alias
torch::setNumThreads -numThreads 4

# Get current thread count
puts [torch::get_num_threads]
```

## Error Handling

The command will raise an error if:
* No thread count is provided
* The thread count is not a valid integer
* The thread count is zero or negative
* Extra arguments are provided in positional syntax

## Related Commands

* `torch::get_num_threads` - Get the current number of threads 