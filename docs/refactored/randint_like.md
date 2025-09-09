# torch::randint_like / torch::randintLike

Create a tensor filled with random integers from a specified range using the same shape as an input tensor.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::randint_like tensor high ?low? ?dtype? ?device?
```

### Named Parameter Syntax
```tcl
torch::randint_like -input tensor -high high ?-low low? ?-dtype dtype? ?-device device? ?-requiresGrad bool?
torch::randintLike -input tensor -high high ?-low low? ?-dtype dtype? ?-device device? ?-requiresGrad bool?
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input` / `tensor` | string | Handle of the input tensor to match shape | Required |
| `high` | integer | Upper bound (exclusive) of random integers | Required |
| `low` | integer | Lower bound (inclusive) of random integers | 0 |
| `dtype` | string | Data type for the result tensor | int64 |
| `device` | string | Device for the result tensor | Same as input |
| `requiresGrad` | boolean | Whether result tensor requires gradients | false* |

**Note**: `requiresGrad=true` will fail for integer tensors as PyTorch only supports gradients for floating-point tensors.

### Supported Data Types
- `int32` (Int)
- `int64` (Long) - **default**

### Supported Devices
- `cpu`
- `cuda` (if available)

## Return Value

Returns a string handle representing the new tensor filled with random integers in the range [low, high).

## Examples

### Basic Usage
```tcl
# Create reference tensor
set input [torch::zeros {3 4}]

# Create random integer tensor with values in [0, 10)
set result [torch::randint_like $input 10]
# Result: 3x4 tensor filled with integers 0-9
```

### Positional Syntax Examples
```tcl
# Basic usage - values in [0, 10)
set result [torch::randint_like $input 10]

# With low and high - values in [5, 15)
set result [torch::randint_like $input 15 5]

# With specific dtype
set result [torch::randint_like $input 100 0 int32]

# With all parameters
set result [torch::randint_like $input 20 10 int64 cpu]
```

### Named Parameter Syntax Examples
```tcl
# Basic usage
set result [torch::randint_like -input $input -high 10]

# With low and high bounds
set result [torch::randint_like -input $input -high 50 -low 10]

# With all parameters
set result [torch::randint_like -input $input -high 1000 -low 100 -dtype int32 -device cpu]

# Parameter order doesn't matter
set result [torch::randint_like -dtype int32 -input $input -high 25 -low 5]
```

### CamelCase Alias Examples
```tcl
# Using camelCase alias
set result [torch::randintLike -input $input -high 100]

# With parameters
set result [torch::randintLike -input $input -high 255 -low 0 -dtype int32]
```

## Range Specification

### Range Semantics
- **Range**: [low, high) - includes `low`, excludes `high`
- **Default low**: 0 if not specified
- **Automatic swapping**: If `low > high`, they are automatically swapped

### Range Examples
```tcl
# Values in [0, 10): 0, 1, 2, ..., 9
set result [torch::randint_like -input $tensor -high 10]

# Values in [5, 15): 5, 6, 7, ..., 14
set result [torch::randint_like -input $tensor -high 15 -low 5]

# Single value [42, 43): only 42
set result [torch::randint_like -input $tensor -high 43 -low 42]
```

## Integration Examples

### Random Indices Generation
```tcl
set batch_size 32
set seq_length 128
set vocab_size 10000

set shape_tensor [torch::zeros [list $batch_size $seq_length]]
set token_indices [torch::randint_like -input $shape_tensor -high $vocab_size -dtype int64]
# Generate random token indices for NLP tasks
```

### Discrete Sampling
```tcl
# Generate random class labels
set prediction_shape [torch::zeros {64 1}]
set num_classes 10
set random_labels [torch::randint_like -input $prediction_shape -high $num_classes]
```

### Game Development
```tcl
# Generate random dice rolls
set dice_shape [torch::zeros {100}]
set dice_rolls [torch::randint_like -input $dice_shape -high 7 -low 1]  # 1-6
```

### Monte Carlo Simulations
```tcl
# Generate random integers for discrete event simulation
set simulation_grid [torch::zeros {1000 1000}]
set random_events [torch::randint_like -input $simulation_grid -high 4 -low 0]  # 0-3
```

## Mathematical Properties

### Distribution
- **Type**: Discrete uniform distribution
- **Range**: [low, high) - integers only
- **Probability**: Equal probability for each integer in range
- **Count**: (high - low) possible values

### Statistical Properties
```tcl
# For large samples, verify uniform distribution
set large_tensor [torch::zeros {10000}]
set samples [torch::randint_like -input $large_tensor -high 10 -low 0]

# Expected mean: (low + high - 1) / 2 = (0 + 9) / 2 = 4.5
set mean_val [torch::tensor_mean $samples]
set mean_scalar [torch::tensor_item $mean_val]
# mean_scalar should be approximately 4.5
```

## Error Handling

### Invalid Tensor
```tcl
catch {torch::randint_like invalid_tensor 10} error
# Error: "Invalid tensor name"
```

### Missing Required Parameters
```tcl
catch {torch::randint_like -input $tensor} error
# Error: "Invalid arguments for torch::randint_like"
```

### Invalid Range
```tcl
# High must be greater than low (or they'll be swapped automatically)
set result [torch::randint_like -input $tensor -high 5 -low 10]
# Automatically swapped to low=5, high=10
```

### RequiresGrad with Integer Tensors
```tcl
catch {torch::randint_like -input $tensor -high 10 -requiresGrad true} error
# Error: "Only Tensors of floating point ... can require gradients"
```

## Migration Guide

### From Positional to Named Parameters

**Old positional syntax:**
```tcl
set result [torch::randint_like $input 100 50 int32 cpu]
```

**New named parameter syntax:**
```tcl
set result [torch::randint_like -input $input -high 100 -low 50 -dtype int32 -device cpu]
```

**Or using camelCase alias:**
```tcl
set result [torch::randintLike -input $input -high 100 -low 50 -dtype int32 -device cpu]
```

### Benefits of Named Parameters

1. **Clarity**: Parameter names make code self-documenting
2. **Flexibility**: Parameters can be specified in any order
3. **Optional parameters**: Easier to specify only needed parameters
4. **Future-proof**: New parameters can be added without breaking existing code

## Performance Notes

- The command creates a new tensor with the same shape as the input
- Memory usage is proportional to the input tensor size
- Integer random generation is performed on the specified device
- CUDA integer generation may be faster for large tensors
- Default dtype is int64 for compatibility with PyTorch indexing operations

## Common Use Cases

### Natural Language Processing
```tcl
# Generate random token sequences
set sequence_shape [torch::zeros {32 512}]  # batch_size x seq_length
set vocab_size 50000
set random_tokens [torch::randint_like -input $sequence_shape -high $vocab_size]
```

### Computer Vision
```tcl
# Generate random class labels
set batch_images [torch::zeros {64 3 224 224}]
set label_shape [torch::zeros {64}]
set num_classes 1000
set random_labels [torch::randint_like -input $label_shape -high $num_classes]
```

### Data Augmentation
```tcl
# Random augmentation choices
set data_shape [torch::zeros {100 256}]
set num_augmentations 8
set aug_choices [torch::randint_like -input $data_shape -high $num_augmentations]
```

### Reinforcement Learning
```tcl
# Random action selection
set state_shape [torch::zeros {64 128}]  # batch of states
set action_shape [torch::zeros {64}]     # batch of actions
set num_actions 4
set random_actions [torch::randint_like -input $action_shape -high $num_actions]
```

### Simulation and Gaming
```tcl
# Random dice rolls simulation
set num_players 6
set num_rounds 100
set dice_shape [torch::zeros [list $num_players $num_rounds]]
set dice_results [torch::randint_like -input $dice_shape -high 7 -low 1]  # 1-6
```

## Advanced Examples

### Categorical Sampling
```tcl
# Simulate categorical distribution sampling
set batch_size 1000
set sample_shape [torch::zeros $batch_size]
set num_categories 5
set category_samples [torch::randint_like -input $sample_shape -high $num_categories]
```

### Index Permutation
```tcl
# Generate random indices for shuffling
set data_size 10000
set index_shape [torch::zeros $data_size]
set random_indices [torch::randint_like -input $index_shape -high $data_size]
```

### Discrete Event Simulation
```tcl
# Model discrete events (e.g., customer arrivals)
set time_steps 1000
set num_servers 10
set event_grid [torch::zeros [list $time_steps $num_servers]]
set event_types 4  # arrival, service, departure, idle
set events [torch::randint_like -input $event_grid -high $event_types]
```

## Data Type Considerations

### Integer Precision
```tcl
# int32 for memory efficiency (when values fit)
set efficient [torch::randint_like -input $tensor -high 1000 -dtype int32]

# int64 for large ranges or indexing compatibility
set large_range [torch::randint_like -input $tensor -high 1000000 -dtype int64]
```

### Memory Usage
- `int32`: 4 bytes per element
- `int64`: 8 bytes per element
- Choose based on value range requirements

## Comparison with Other Random Functions

| Function | Distribution | Range | Data Type | Use Case |
|----------|-------------|-------|-----------|----------|
| `torch::randint_like` | Discrete uniform | [low,high) | Integer | Indices, labels, discrete events |
| `torch::rand_like` | Continuous uniform | [0,1) | Float | Dropout masks, probabilities |
| `torch::randn_like` | Normal | (-∞,+∞) | Float | Weight initialization, noise |

## See Also

- [`torch::rand_like`](rand_like.md) - Create tensor with uniform distribution
- [`torch::randn_like`](randn_like.md) - Create tensor with normal distribution
- [`torch::zeros_like`](zeros_like.md) - Create tensor filled with zeros
- [`torch::ones_like`](ones_like.md) - Create tensor filled with ones
- [`torch::randint`](randint.md) - Create random integer tensor with specified shape
- [`torch::tensor_min`](tensor_min.md) - Find minimum values
- [`torch::tensor_max`](tensor_max.md) - Find maximum values 