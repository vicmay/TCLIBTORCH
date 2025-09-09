# torch::lr_scheduler_multi_step

Multiply the learning rate of each parameter group by gamma every milestone epochs.

## 🔄 Dual Syntax Support

This command supports both legacy positional syntax and modern named parameter syntax.

### Named Parameters (Recommended)
```tcl
torch::lr_scheduler_multi_step -optimizer $optimizer -milestones $milestones ?-gamma $gamma?
torch::lrSchedulerMultiStep -optimizer $optimizer -milestones $milestones ?-gamma $gamma?
```

### Positional Syntax (Legacy)
```tcl
torch::lr_scheduler_multi_step $optimizer $milestones ?$gamma?
torch::lrSchedulerMultiStep $optimizer $milestones ?$gamma?
```

## 📖 Description

The Multi-step learning rate scheduler multiplies the learning rate by `gamma` whenever a milestone epoch is reached. This is one of the most commonly used learning rate scheduling strategies in deep learning, especially for image classification tasks.

## 🔧 Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-optimizer` | string | Yes | - | Handle to the optimizer object |
| `-milestones` | list of integers | Yes | - | List of epoch milestones when to decay the learning rate |
| `-gamma` | double | No | 0.1 | Multiplicative factor of learning rate decay |

### Parameter Details

- **optimizer**: Must be a valid optimizer handle created by one of the optimizer commands
- **milestones**: List of positive integers representing epoch numbers when learning rate should be decayed
- **gamma**: Learning rate decay factor. The new learning rate becomes: `lr_new = lr_old * gamma`

## 📊 Mathematical Background

The multi-step scheduler implements the following decay formula:

```
lr(epoch) = lr_initial * gamma^n
```

Where `n` is the number of milestones that have been reached by the current epoch.

### Example Timeline
For milestones `{30, 60, 90}` and `gamma = 0.1`:
- Epochs 1-29: `lr = lr_initial`
- Epochs 30-59: `lr = lr_initial * 0.1`
- Epochs 60-89: `lr = lr_initial * 0.01`
- Epochs 90+: `lr = lr_initial * 0.001`

## 💡 Use Cases

1. **Image Classification**: Commonly used with step decay at 30%, 60%, 80% of total training epochs
2. **ResNet Training**: Popular with ResNet architectures for ImageNet training
3. **Long Training Runs**: When you know specific points where LR reduction helps convergence
4. **Fine-tuning**: Step down learning rate after initial training phases

## 🎯 Examples

### Basic Usage with Named Parameters
```tcl
# Create optimizer
set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
set optimizer [torch::sgd $tensor 0.1]

# Create multi-step scheduler (decay at epochs 10, 20, 30)
set scheduler [torch::lr_scheduler_multi_step -optimizer $optimizer -milestones {10 20 30}]

# With custom gamma
set scheduler [torch::lr_scheduler_multi_step -optimizer $optimizer -milestones {10 20 30} -gamma 0.5]
```

### CamelCase Syntax
```tcl
# Using camelCase alias
set scheduler [torch::lrSchedulerMultiStep -optimizer $optimizer -milestones {15 30 45} -gamma 0.2]
```

### Legacy Positional Syntax
```tcl
# Backward compatible syntax
set scheduler [torch::lr_scheduler_multi_step $optimizer {10 20 30}]
set scheduler [torch::lr_scheduler_multi_step $optimizer {10 20 30} 0.1]
```

### Common Training Patterns
```tcl
# ResNet-style training (100 epochs)
set scheduler [torch::lr_scheduler_multi_step -optimizer $optimizer -milestones {30 60 80} -gamma 0.1]

# Aggressive decay
set scheduler [torch::lr_scheduler_multi_step -optimizer $optimizer -milestones {5 10 15 20} -gamma 0.5]

# Conservative decay
set scheduler [torch::lr_scheduler_multi_step -optimizer $optimizer -milestones {50 100} -gamma 0.3]

# Single milestone
set scheduler [torch::lr_scheduler_multi_step -optimizer $optimizer -milestones {50}]
```

## 🔄 Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
set scheduler [torch::lr_scheduler_multi_step $optimizer {10 20 30} 0.1]
```

**New (Named Parameters):**
```tcl
set scheduler [torch::lr_scheduler_multi_step -optimizer $optimizer -milestones {10 20 30} -gamma 0.1]
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
torch::lr_scheduler_multi_step -optimizer "invalid" -milestones {10}
# Error: Invalid optimizer name

# Missing required parameters
torch::lr_scheduler_multi_step -optimizer $optimizer
# Error: Required parameters missing: -optimizer and -milestones are required

# Invalid milestone values
torch::lr_scheduler_multi_step -optimizer $optimizer -milestones {10 invalid 20}
# Error: Invalid milestone value

# Invalid gamma
torch::lr_scheduler_multi_step -optimizer $optimizer -milestones {10} -gamma "invalid"
# Error: Invalid gamma value

# Unknown parameter
torch::lr_scheduler_multi_step -optimizer $optimizer -milestones {10} -unknown param
# Error: Unknown parameter: -unknown
```

## 🎯 Best Practices

1. **Milestone Timing**: Place milestones at 1/3, 2/3, and 4/5 of total training epochs
2. **Gamma Selection**: Use 0.1 for aggressive decay, 0.5 for moderate decay
3. **Monitoring**: Always monitor validation loss when learning rate changes
4. **Validation**: Test different milestone patterns for your specific task
5. **Documentation**: Use named parameters for better code documentation

## 📈 Performance Considerations

- **Scheduler Creation**: O(1) operation, very fast
- **Memory Usage**: Minimal memory overhead for milestone storage
- **Milestone Ordering**: Milestones don't need to be in order (internally handled)
- **Large Milestone Lists**: Efficiently handles many milestones

## 🔗 Related Commands

- `torch::lr_scheduler_step` - Fixed step size decay
- `torch::lr_scheduler_exponential` - Exponential decay
- `torch::lr_scheduler_cosine_annealing` - Cosine annealing schedule
- `torch::lr_scheduler_plateau` - Reduce on plateau

## 📋 Return Value

Returns a scheduler handle (string) that can be used with other scheduler operations:
- `torch::lr_scheduler_step_update` - Apply scheduling step
- `torch::get_lr` - Get current learning rate

## 🧪 Testing

The command includes comprehensive test coverage:
- ✅ Dual syntax parsing (26 test cases)
- ✅ Parameter validation and error handling
- ✅ camelCase alias functionality
- ✅ Edge cases and boundary conditions

## 📚 Technical Notes

- **Implementation**: Uses LibTorch's native scheduler infrastructure
- **Thread Safety**: Scheduler operations are thread-safe
- **Precision**: Uses double precision for gamma calculations
- **Compatibility**: Maintains 100% backward compatibility with existing code 