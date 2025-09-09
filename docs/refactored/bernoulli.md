# torch::bernoulli

Samples binary values from Bernoulli distributions, where each element is 0 or 1 based on probabilistic sampling.

## Aliases
- `torch::bernoulli`  (already in camelCase)

---

## Positional Syntax (back-compat)
```tcl
set samples [torch::bernoulli <input> ?probability? ?generator?]
```

## Named Parameter Syntax (recommended)
```tcl
set samples [torch::bernoulli \
    -input         <tensor>     ;# required: input tensor \
    -p             <float>      ;# optional: probability [0.0, 1.0] \
    -generator     <string> ]   ;# optional: random generator (future)
```
Parameter aliases:
* `-input` / `-tensor`
* `-p` / `-probability`

### Parameters
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| **-input** | tensor | — | Input tensor (probabilities if no -p given, or shape template) |
| **-p** | float | use input | Fixed probability for all elements [0.0, 1.0] |
| **-generator** | string | default | Random number generator (reserved for future use) |

### Returns
Binary tensor with same shape as input, containing 0s and 1s sampled from Bernoulli distribution.

---

## Examples
```tcl
# Use input tensor values as probabilities
set probs [torch::tensor_create -data {0.3 0.7 0.9 0.1} -shape {4}]
set samples [torch::bernoulli $probs]

# Positional with fixed probability  
set shape_tensor [torch::tensor_create -data {1.0 1.0 1.0} -shape {3}]
set samples [torch::bernoulli $shape_tensor 0.6]

# Named (preferred) - tensor probabilities
set samples [torch::bernoulli -input $probs]

# Named with fixed probability
set samples [torch::bernoulli -input $shape_tensor -p 0.4]

# Parameter aliases
set samples [torch::bernoulli -tensor $probs -probability 0.8]
```

---

## Sampling Modes

### Mode 1: Tensor Probabilities (no -p parameter)
Each element of the input tensor is treated as the probability for that position:
```tcl
set probs [torch::tensor_create -data {0.1 0.5 0.9} -shape {3}]
set samples [torch::bernoulli -input $probs]
# samples[0] has 10% chance of being 1
# samples[1] has 50% chance of being 1  
# samples[2] has 90% chance of being 1
```

### Mode 2: Fixed Probability (with -p parameter)
All elements use the same fixed probability:
```tcl
set shape [torch::tensor_create -data {1.0 1.0 1.0 1.0} -shape {4}]
set samples [torch::bernoulli -input $shape -p 0.3]
# All elements have 30% chance of being 1
```

## Mathematical Properties
- **Output range**: Binary values (0 or 1) only
- **Independence**: Each element sampled independently
- **Expected value**: E[X] = p (where p is the probability)
- **Variance**: Var[X] = p(1-p)
- **Shape preservation**: Output has same shape as input

## Use Cases
- **Dropout layers**: Random neuron deactivation in neural networks
- **Data augmentation**: Random masking or corruption
- **Monte Carlo simulation**: Binary random variable generation
- **A/B testing**: Random assignment simulation
- **Stochastic optimization**: Random exploration decisions

---

## Error Messages
| Situation | Message |
|-----------|---------|
| Missing input tensor | `Missing required parameter: -input` |
| Invalid tensor name | `Invalid input tensor name: xyz` |
| Probability out of range | `Probability p must be in range [0.0, 1.0]` |
| Unknown parameter | `Unknown parameter: -foo` |
| Missing parameter value | `Named parameters require pairs: -param value` |

---

## Migration Guide
Old:
```tcl
set samples [torch::bernoulli $probs]
set samples [torch::bernoulli $shape 0.5]
```
New (same - already modern):
```tcl
set samples [torch::bernoulli -input $probs]
set samples [torch::bernoulli -input $shape -p 0.5]
```

Enhanced usage:
```tcl
# Old (limited to positional)
set samples [torch::bernoulli $tensor 0.8]

# New (explicit and clear)  
set samples [torch::bernoulli -input $tensor -probability 0.8]
```

---

## Probability Examples
```tcl
# Certain outcomes
set always_zero [torch::bernoulli -input $shape -p 0.0]  # All 0s
set always_one [torch::bernoulli -input $shape -p 1.0]   # All 1s

# Fair coin flip
set coins [torch::bernoulli -input $shape -p 0.5]

# Biased scenarios
set rare_events [torch::bernoulli -input $shape -p 0.01]    # 1% chance
set common_events [torch::bernoulli -input $shape -p 0.95]  # 95% chance

# Variable probabilities per element
set varied_probs [torch::tensor_create -data {0.1 0.3 0.5 0.7 0.9} -shape {5}]
set varied_samples [torch::bernoulli -input $varied_probs]
```

---

## Advanced Usage
```tcl
# Dropout simulation (70% keep rate = 30% drop rate)
set dropout_mask [torch::bernoulli -input $activations -p 0.7]
set dropped_activations [torch::tensor_mul $activations $dropout_mask]

# Random data corruption (5% corruption rate)
set corruption_mask [torch::bernoulli -input $data -p 0.05]
set clean_mask [torch::tensor_sub [torch::ones_like $data] $corruption_mask]
set corrupted_data [torch::tensor_mul $data $clean_mask]

# Monte Carlo estimation
set num_trials [torch::tensor_create -data [string repeat "1.0 " 10000] -shape {10000}]
set successes [torch::bernoulli -input $num_trials -p 0.3]
set success_rate [torch::tensor_mean $successes]
```

---

## Tests
Validated by `tests/refactored/bernoulli_test.tcl` (positional, named, probability validation, mathematical properties, shape preservation).

## Compatibility
✅ Positional syntax retained • ✅ Named parameters added • ✅ Already camelCase 

## Related Commands

- **[torch::rand](rand.md)**: Uniform random values [0,1)
- **[torch::randn](randn.md)**: Normal distribution sampling
- **[torch::multinomial](multinomial.md)**: Multinomial distribution sampling
- **[torch::dropout](dropout.md)**: Neural network dropout layer
- **[torch::tensor_mul](tensor_mul.md)**: Element-wise multiplication for masking 