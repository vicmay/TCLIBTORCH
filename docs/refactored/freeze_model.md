# torch::freeze_model

Freezes all parameters of a neural network model, disabling gradient computation and preventing parameter updates during training.

## Syntax

### Traditional Syntax (Positional)
```tcl
torch::freeze_model model
```

### Modern Syntax (Named Parameters)
```tcl
torch::freeze_model -model model
```

### camelCase Alias
```tcl
torch::freezeModel model
torch::freezeModel -model model
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| **model** | string | Yes | The model handle to freeze |

## Return Value

Returns the string "Model parameters frozen" upon successful execution.

## Description

The `torch::freeze_model` command disables gradient computation for all parameters in the specified model by setting `requires_grad=false` for each parameter. This is commonly used in scenarios such as:

- **Transfer Learning**: Freezing pre-trained model layers while fine-tuning only specific layers
- **Feature Extraction**: Using a pre-trained model as a fixed feature extractor
- **Inference Mode**: Preventing accidental parameter updates during model evaluation
- **Selective Training**: Freezing certain layers while allowing others to be trained

When a model is frozen, its parameters will not be updated by optimizers, effectively making those layers static during training.

## Examples

### Basic Usage

```tcl
# Create a model
set model [torch::linear 128 64]

# Freeze the model (positional syntax)
torch::freeze_model $model
# Output: Model parameters frozen

# Freeze using named parameters
torch::freeze_model -model $model
# Output: Model parameters frozen

# Freeze using camelCase alias
torch::freezeModel $model
# Output: Model parameters frozen
```

### Transfer Learning Example

```tcl
# Create a base model (e.g., pre-trained feature extractor)
set feature_extractor [torch::linear 784 256]

# Create a classifier head
set classifier [torch::linear 256 10]

# Freeze the feature extractor to preserve pre-trained weights
torch::freeze_model $feature_extractor

# Train only the classifier head
# (feature_extractor parameters won't be updated)
```

### Selective Layer Freezing

```tcl
# Create multiple layers
set layer1 [torch::linear 100 50]
set layer2 [torch::linear 50 25]
set layer3 [torch::linear 25 10]

# Freeze only the first two layers
torch::freeze_model $layer1
torch::freeze_model $layer2

# layer3 remains trainable
```

### Freeze/Unfreeze Cycle

```tcl
# Create a model
set model [torch::linear 64 32]

# Freeze the model
torch::freeze_model $model

# Later, unfreeze for fine-tuning
torch::unfreeze_model $model

# Freeze again when needed
torch::freeze_model $model
```

### Multiple Model Management

```tcl
# Create several models
set encoder [torch::linear 512 256]
set decoder [torch::linear 256 512]
set classifier [torch::linear 256 10]

# Freeze encoder and decoder for inference
torch::freeze_model $encoder
torch::freeze_model $decoder

# Keep classifier trainable for adaptation
```

## Integration with Training

### Basic Training with Frozen Model

```tcl
# Set up model and optimizer
set model [torch::linear 784 10]
set optimizer [torch::sgd $model 0.01]

# Freeze the model
torch::freeze_model $model

# During training loop, frozen parameters won't be updated
# (optimizer will skip frozen parameters)
```

### Mixed Training (Some Layers Frozen)

```tcl
# Create a two-layer network
set layer1 [torch::linear 784 128]
set layer2 [torch::linear 128 10]

# Freeze first layer (feature extractor)
torch::freeze_model $layer1

# Train only the second layer
# Create optimizer for trainable parameters only
set optimizer [torch::sgd $layer2 0.001]
```

## Error Handling

The command provides comprehensive error handling for various scenarios:

```tcl
# Missing model parameter
catch {torch::freeze_model} result
puts $result
# Output: Wrong number of arguments for positional syntax. Expected: torch::freeze_model model

# Invalid model handle
catch {torch::freeze_model "nonexistent_model"} result
puts $result
# Output: Model not found

# Missing parameter value
catch {torch::freeze_model -model} result
puts $result
# Output: Missing value for parameter: -model

# Unknown parameter
catch {torch::freeze_model -unknown_param "value"} result
puts $result
# Output: Unknown parameter: -unknown_param

# Empty model name
catch {torch::freeze_model -model ""} result
puts $result
# Output: Required parameter missing: -model
```

## Performance Considerations

### Memory Usage
- Freezing a model doesn't reduce memory usage significantly
- Memory is still allocated for parameter storage
- Gradient buffers may be freed for frozen parameters

### Computational Performance
- Frozen parameters skip gradient computation, improving training speed
- Forward pass performance is largely unaffected
- Backward pass is faster for frozen layers

### Training Efficiency
```tcl
# Create a large model
set large_model [torch::linear 1000 1000]

# Freeze it to skip gradient computation
torch::freeze_model $large_model

# Training will be faster due to reduced gradient computation
```

## Best Practices

### 1. Freeze Early in Training Pipeline
```tcl
# Freeze models before creating optimizers
torch::freeze_model $pretrained_model
set optimizer [torch::sgd $trainable_model 0.01]
```

### 2. Use Descriptive Variable Names
```tcl
# Good practice
set frozen_encoder [torch::linear 512 256]
torch::freeze_model $frozen_encoder

set trainable_decoder [torch::linear 256 512]
# decoder remains trainable
```

### 3. Document Freeze/Unfreeze Operations
```tcl
# Clear documentation of model state
# Phase 1: Freeze feature extractor
torch::freeze_model $feature_extractor

# Phase 2: Fine-tune classifier only
# (feature_extractor remains frozen)

# Phase 3: Unfreeze for end-to-end training
torch::unfreeze_model $feature_extractor
```

### 4. Verify Model State
```tcl
# Always verify successful freezing
set result [torch::freeze_model $model]
if {$result eq "Model parameters frozen"} {
    puts "Model successfully frozen"
} else {
    puts "Warning: Model freeze may have failed"
}
```

## Common Use Cases

### Transfer Learning
```tcl
# Load pre-trained model (conceptual)
set pretrained_model [torch::linear 2048 1000]

# Freeze pre-trained weights
torch::freeze_model $pretrained_model

# Add new classification head
set new_classifier [torch::linear 1000 5]  # 5 new classes

# Train only the new classifier
```

### Feature Extraction
```tcl
# Create feature extractor
set feature_extractor [torch::linear 784 256]

# Freeze to use as fixed feature extractor
torch::freeze_model $feature_extractor

# Extract features without training
```

### Gradual Unfreezing
```tcl
# Start with everything frozen
torch::freeze_model $layer1
torch::freeze_model $layer2  
torch::freeze_model $layer3

# Gradually unfreeze layers
torch::unfreeze_model $layer3  # First, train top layer
# ... train for some epochs ...
torch::unfreeze_model $layer2  # Then, train middle layer
# ... train for some epochs ...
torch::unfreeze_model $layer1  # Finally, train bottom layer
```

## Related Commands

- **torch::unfreeze_model**: Unfreezes a model's parameters
- **torch::linear**: Creates linear layers
- **torch::sgd**: Creates SGD optimizer
- **torch::adam**: Creates Adam optimizer

## Technical Details

### Parameter Freezing Mechanism
The command iterates through all parameters in the model and sets `requires_grad=false` for each parameter. This prevents the computation of gradients for these parameters during backpropagation.

### Memory Management
- Frozen parameters retain their values and memory allocation
- Gradient buffers for frozen parameters may be deallocated
- The model structure remains unchanged

### Optimizer Interaction
- Most optimizers automatically skip frozen parameters
- Only parameters with `requires_grad=true` are updated
- Learning rates and other optimizer settings are unaffected

## Migration Guide

### From Legacy Syntax
```tcl
# Old positional syntax (still supported)
torch::freeze_model $model

# New named parameter syntax
torch::freeze_model -model $model

# Modern camelCase alias
torch::freezeModel -model $model
```

### Benefits of Named Parameters
1. **Clarity**: Parameter names make code self-documenting
2. **Extensibility**: Easy to add new parameters in the future
3. **Consistency**: Matches modern API design patterns
4. **Error Prevention**: Reduces positional argument mistakes

## Examples in Context

### Complete Training Script
```tcl
# Load LibTorch TCL extension
load libtorchtcl.so

# Create models
set feature_extractor [torch::linear 784 256]
set classifier [torch::linear 256 10]

# Freeze feature extractor
torch::freeze_model $feature_extractor

# Create optimizer for trainable parameters only
set optimizer [torch::sgd $classifier 0.001]

# Training loop would go here
puts "Feature extractor frozen, classifier trainable"
```

### Model Evaluation Mode
```tcl
# Freeze all models for evaluation
torch::freeze_model $model1
torch::freeze_model $model2
torch::freeze_model $model3

# Run evaluation
# (no parameter updates will occur)

# Unfreeze for continued training
torch::unfreeze_model $model1
torch::unfreeze_model $model2
torch::unfreeze_model $model3
```

## Troubleshooting

### Common Issues

1. **Model Not Found Error**
   - Ensure the model handle is valid
   - Check that the model was created successfully

2. **Parameter Syntax Errors**
   - Use either positional OR named parameters, not both
   - Ensure all parameter names are spelled correctly

3. **Unexpected Training Behavior**
   - Verify which models are frozen/unfrozen
   - Check that optimizers are configured for the correct parameters

### Debug Commands
```tcl
# Test model freezing
set test_model [torch::linear 10 5]
set result [torch::freeze_model $test_model]
puts "Freeze result: $result"

# Test different syntaxes
torch::freeze_model $test_model
torch::freeze_model -model $test_model
torch::freezeModel $test_model
torch::freezeModel -model $test_model
```

## Version Information

- **Command**: `torch::freeze_model`
- **Aliases**: `torch::freezeModel`
- **Syntax Support**: Dual (positional and named parameters)
- **Backward Compatibility**: Full
- **Added**: LibTorch TCL Extension v1.0

---

*This documentation covers the complete functionality of the `torch::freeze_model` command. For related model management commands, see the documentation for `torch::unfreeze_model` and other model manipulation functions.* 