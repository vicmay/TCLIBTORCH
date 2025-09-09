# torch::lr_scheduler_multiplicative

Multiply the learning rate of each parameter group by a given factor at each epoch.

## 🔄 Dual Syntax Support

This command supports both legacy positional syntax and modern named parameter syntax.

### Named Parameters (Recommended)
```tcl
torch::lr_scheduler_multiplicative -optimizer $optimizer -lrLambda $lr_lambda
torch::lr_scheduler_multiplicative -optimizer $optimizer -lr_lambda $lr_lambda
torch::lrSchedulerMultiplicative -optimizer $optimizer -lrLambda $lr_lambda
```

### Positional Syntax (Legacy)
```tcl
torch::lr_scheduler_multiplicative $optimizer $lr_lambda
torch::lrSchedulerMultiplicative $optimizer $lr_lambda
```

## 📖 Description

The Multiplicative learning rate scheduler multiplies the learning rate by a constant factor `lr_lambda` at every epoch. This provides a simple exponential decay of the learning rate, which is useful for fine-tuning and gradual training adjustments.

## 🔧 Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-optimizer` | string | Yes | - | Handle to the optimizer object |
| `-lrLambda` or `-lr_lambda` | double | Yes | - | Multiplicative factor for learning rate |

### Parameter Details

- **optimizer**: Must be a valid optimizer handle created by one of the optimizer commands
- **lrLambda/lr_lambda**: The multiplicative factor. The new learning rate becomes: `lr_new = lr_old * lr_lambda`

### Parameter Name Flexibility
Both `-lrLambda` (camelCase) and `-lr_lambda` (snake_case) are accepted for consistency with different naming conventions.

## 📊 Mathematical Background

The multiplicative scheduler implements the following decay formula:

```
lr(epoch) = lr_initial * lr_lambda^epoch
```

This creates an exponential decay pattern where:
- `lr_lambda < 1.0`: Exponential decay (learning rate decreases)
- `lr_lambda = 1.0`: No change (learning rate remains constant)
- `lr_lambda > 1.0`: Exponential growth (learning rate increases)

### Example Timeline
For initial learning rate `0.1` and `lr_lambda = 0.95`:
- Epoch 0: `lr = 0.1`
- Epoch 1: `lr = 0.1 * 0.95 = 0.095`
- Epoch 2: `lr = 0.095 * 0.95 = 0.09025`
- Epoch 3: `lr = 0.09025 * 0.95 = 0.0857375`
- ...

## 💡 Use Cases

1. **Gradual Decay**: Smooth, continuous learning rate reduction
2. **Fine-tuning**: Small adjustments to pre-trained models
3. **Stable Training**: Gentle learning rate changes for sensitive models
4. **Exponential Schedules**: When you need precise exponential decay control
5. **Warm-up Phase**: Can be used with `lr_lambda > 1.0` for gradual warm-up

## 🎯 Examples

### Basic Usage with Named Parameters
```tcl
# Create optimizer
set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
set optimizer [torch::sgd $tensor 0.1]

# Create multiplicative scheduler with 5% decay per epoch
set scheduler [torch::lr_scheduler_multiplicative -optimizer $optimizer -lrLambda 0.95]

# Alternative parameter name
set scheduler [torch::lr_scheduler_multiplicative -optimizer $optimizer -lr_lambda 0.95]
```

### CamelCase Syntax
```tcl
# Using camelCase alias
set scheduler [torch::lrSchedulerMultiplicative -optimizer $optimizer -lrLambda 0.9]
```

### Legacy Positional Syntax
```tcl
# Backward compatible syntax
set scheduler [torch::lr_scheduler_multiplicative $optimizer 0.95]
set scheduler [torch::lr_scheduler_multiplicative $optimizer 0.8]
```

### Different Decay Strategies
```tcl
# Conservative decay (1% per epoch)
set scheduler [torch::lr_scheduler_multiplicative -optimizer $optimizer -lrLambda 0.99]

# Moderate decay (5% per epoch)
set scheduler [torch::lr_scheduler_multiplicative -optimizer $optimizer -lrLambda 0.95]

# Aggressive decay (20% per epoch)
set scheduler [torch::lr_scheduler_multiplicative -optimizer $optimizer -lrLambda 0.8]

# Very slow decay (0.1% per epoch)
set scheduler [torch::lr_scheduler_multiplicative -optimizer $optimizer -lrLambda 0.999]

# Warm-up phase (5% increase per epoch)
set scheduler [torch::lr_scheduler_multiplicative -optimizer $optimizer -lrLambda 1.05]
```

## 🔄 Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
set scheduler [torch::lr_scheduler_multiplicative $optimizer 0.95]
```

**New (Named Parameters):**
```tcl
set scheduler [torch::lr_scheduler_multiplicative -optimizer $optimizer -lrLambda 0.95]
```

**Benefits of Named Parameters:**
- **Clarity**: Parameter purpose is explicit
- **Flexibility**: Parameters can be specified in any order
- **Maintainability**: Code is more readable and less error-prone
- **Future-proof**: Easy to add new parameters without breaking existing code

## ⚠️ Error Handling

The command provides detailed error messages for common issues:

```tcl
# Invalid optimizer
torch::lr_scheduler_multiplicative -optimizer "invalid" -lrLambda 0.95
# Error: Invalid optimizer name

# Missing required parameters
torch::lr_scheduler_multiplicative -lrLambda 0.95
# Error: Required parameters missing: -optimizer is required

# Invalid lr_lambda value
torch::lr_scheduler_multiplicative -optimizer $optimizer -lrLambda "invalid"
# Error: Invalid lr_lambda value

# Wrong number of positional arguments
torch::lr_scheduler_multiplicative $optimizer
# Error: Usage: torch::lr_scheduler_multiplicative optimizer lr_lambda

# Unknown parameter
torch::lr_scheduler_multiplicative -optimizer $optimizer -lrLambda 0.95 -unknown param
# Error: Unknown parameter: -unknown
```

## 🎯 Best Practices

1. **Decay Selection**: Use 0.9-0.99 for gradual decay, 0.5-0.8 for aggressive decay
2. **Monitoring**: Always monitor training metrics when using multiplicative decay
3. **Combination**: Can be combined with other schedulers for complex schedules
4. **Validation**: Test different lr_lambda values for your specific task
5. **Documentation**: Use named parameters for better code documentation

## 📈 Performance Considerations

- **Scheduler Creation**: O(1) operation, very fast
- **Memory Usage**: Minimal memory overhead
- **Computational Cost**: Simple multiplication, extremely efficient
- **Precision**: Uses double precision for lr_lambda calculations

## 🔗 Related Commands

- `torch::lr_scheduler_step` - Fixed step size decay
- `torch::lr_scheduler_exponential` - Similar exponential decay with gamma parameter
- `torch::lr_scheduler_multi_step` - Step decay at specific milestones
- `torch::lr_scheduler_cosine_annealing` - Cosine annealing schedule

## 📋 Return Value

Returns a scheduler handle (string) that can be used with other scheduler operations:
- `torch::lr_scheduler_step_update` - Apply scheduling step
- `torch::get_lr` - Get current learning rate

## 🧪 Testing

The command includes comprehensive test coverage:
- ✅ Dual syntax parsing (33 test cases)
- ✅ Parameter validation and error handling
- ✅ camelCase alias functionality
- ✅ Edge cases and mathematical boundaries
- ✅ Both parameter name formats (-lrLambda and -lr_lambda)

## 📚 Technical Notes

- **Implementation**: Uses LibTorch's native scheduler infrastructure
- **Thread Safety**: Scheduler operations are thread-safe
- **Precision**: Uses double precision for lr_lambda calculations
- **Compatibility**: Maintains 100% backward compatibility with existing code
- **Parameter Names**: Supports both camelCase (-lrLambda) and snake_case (-lr_lambda) for flexibility

## 🔢 Mathematical Examples

### Common lr_lambda Values and Their Effects

| lr_lambda | Effect per Epoch | After 10 Epochs | After 100 Epochs | Use Case |
|-----------|------------------|------------------|------------------|----------|
| 0.999 | -0.1% | -1% | -9.5% | Very gradual decay |
| 0.99 | -1% | -9.6% | -63.4% | Slow decay |
| 0.95 | -5% | -40.1% | -99.4% | Moderate decay |
| 0.9 | -10% | -65.1% | -99.997% | Fast decay |
| 0.8 | -20% | -89.3% | ~0% | Very fast decay |
| 1.0 | 0% | 0% | 0% | No change |
| 1.01 | +1% | +10.5% | +170% | Gradual warm-up |

### Decay Formula Examples
```
Initial LR = 0.1, lr_lambda = 0.95:
- Epoch 1: 0.1 * 0.95 = 0.095
- Epoch 2: 0.095 * 0.95 = 0.09025
- Epoch 3: 0.09025 * 0.95 = 0.0857375
- ...
- Epoch 20: 0.1 * 0.95^20 ≈ 0.0358
``` 