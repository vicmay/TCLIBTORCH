# torch::load_checkpoint

Load a saved model checkpoint containing both model and optimizer state.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::load_checkpoint -filename checkpoint_file -model model_name -optimizer optimizer_name
torch::load_checkpoint -file checkpoint_file -model model_name -optimizer optimizer_name
torch::loadCheckpoint -filename checkpoint_file -model model_name -optimizer optimizer_name
torch::loadCheckpoint -file checkpoint_file -model model_name -optimizer optimizer_name
```

### Positional Parameters (Legacy)
```tcl
torch::load_checkpoint checkpoint_file model_name optimizer_name
torch::loadCheckpoint checkpoint_file model_name optimizer_name
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `-filename` / `-file` | string | Yes | Path to the checkpoint file to load |
| `-model` | string | Yes | Name of the model to load state into |
| `-optimizer` | string | Yes | Name of the optimizer to load state into |

## Returns

Returns a string containing checkpoint loading information including:
- Checkpoint filename
- Loaded epoch number
- Loaded loss value
- Loaded learning rate
- Loading success status

## Description

The `torch::load_checkpoint` command loads a previously saved checkpoint file containing both model and optimizer state. This is essential for:

- **Training resumption**: Continue training from where you left off
- **Model deployment**: Load trained models for inference
- **Experiment reproduction**: Restore exact training state
- **Transfer learning**: Load pre-trained models as starting points

The command loads:
1. **Model state**: All model parameters (weights, biases, etc.)
2. **Optimizer state**: Optimizer internal state (momentum, learning rate schedule, etc.)
3. **Training metadata**: Epoch number, loss value, learning rate
4. **Custom metrics**: Any additional metrics saved with the checkpoint

## Examples

### Basic Usage
```tcl
# Create a model and optimizer
set model [torch::linear -in_features 784 -out_features 10]
set optimizer [torch::optimizer_adam -lr 0.001]
torch::optimizer_add_parameters $optimizer $model

# Load checkpoint using named syntax (recommended)
set result [torch::load_checkpoint -filename "model_epoch_100.pt" -model $model -optimizer $optimizer]
puts $result
# Output: "Checkpoint loaded: model_epoch_100.pt (epoch=100, loss=0.234, lr=0.001)"

# Using camelCase alias
set result [torch::loadCheckpoint -file "model_epoch_100.pt" -model $model -optimizer $optimizer]

# Using legacy positional syntax
set result [torch::load_checkpoint "model_epoch_100.pt" $model $optimizer]
```

### Training Resumption
```tcl
# Setup model and optimizer
set model [torch::sequential]
torch::sequential_add $model [torch::linear -in_features 784 -out_features 128]
torch::sequential_add $model [torch::relu]
torch::sequential_add $model [torch::linear -in_features 128 -out_features 10]

set optimizer [torch::optimizer_sgd -lr 0.01 -momentum 0.9]
torch::optimizer_add_parameters $optimizer $model

# Load checkpoint to resume training
set checkpoint_info [torch::load_checkpoint -filename "training_checkpoint.pt" -model $model -optimizer $optimizer]
puts "Resumed training from: $checkpoint_info"

# Continue training loop...
# The model and optimizer now have their previous state restored
```

### Model Deployment
```tcl
# Load pre-trained model for inference
set model [torch::create_model "resnet18"]
set optimizer [torch::optimizer_adam -lr 0.001]  # Optimizer needed for loading
torch::optimizer_add_parameters $optimizer $model

# Load the trained checkpoint
torch::load_checkpoint -file "best_model.pt" -model $model -optimizer $optimizer

# Model is now ready for inference
# (optimizer state is loaded but not needed for inference)
```

### Multiple Model Loading
```tcl
# Load different models from different checkpoints
set generator [torch::create_model "generator"]
set discriminator [torch::create_model "discriminator"]

set gen_optimizer [torch::optimizer_adam -lr 0.0002]
set disc_optimizer [torch::optimizer_adam -lr 0.0002]

torch::optimizer_add_parameters $gen_optimizer $generator
torch::optimizer_add_parameters $disc_optimizer $discriminator

# Load generator checkpoint
torch::load_checkpoint -filename "generator_epoch_50.pt" -model $generator -optimizer $gen_optimizer

# Load discriminator checkpoint
torch::load_checkpoint -filename "discriminator_epoch_50.pt" -model $discriminator -optimizer $disc_optimizer
```

### Transfer Learning
```tcl
# Load pre-trained model as starting point
set pretrained_model [torch::create_model "resnet50"]
set optimizer [torch::optimizer_adam -lr 0.001]
torch::optimizer_add_parameters $optimizer $pretrained_model

# Load ImageNet pre-trained weights
torch::load_checkpoint -file "resnet50_imagenet.pt" -model $pretrained_model -optimizer $optimizer

# Fine-tune for specific task
# The model now has pre-trained weights loaded
```

### Checkpoint Metadata Inspection
```tcl
# Load checkpoint and examine metadata
set model [torch::linear -in_features 10 -out_features 1]
set optimizer [torch::optimizer_sgd -lr 0.01]
torch::optimizer_add_parameters $optimizer $model

set info [torch::load_checkpoint -filename "experiment_1.pt" -model $model -optimizer $optimizer]
puts "Checkpoint info: $info"

# The info string contains epoch, loss, learning rate, etc.
# Parse if needed for conditional logic
if {[string match "*epoch=100*" $info]} {
    puts "Model was trained for 100 epochs"
}
```

## Migration Guide

### From Legacy Positional Syntax
```tcl
# Old syntax (still supported)
torch::load_checkpoint "checkpoint.pt" $model $optimizer

# New syntax (recommended)
torch::load_checkpoint -filename "checkpoint.pt" -model $model -optimizer $optimizer

# Or using shorter alias
torch::load_checkpoint -file "checkpoint.pt" -model $model -optimizer $optimizer

# Or using camelCase
torch::loadCheckpoint -filename "checkpoint.pt" -model $model -optimizer $optimizer
```

### Benefits of Named Parameters
- **Clarity**: Makes it explicit what each parameter represents
- **Maintainability**: Code is self-documenting
- **Consistency**: Matches modern TCL conventions
- **Flexibility**: Parameter order doesn't matter
- **Error Prevention**: Reduces risk of swapping parameters

## Error Handling

The command will throw an error in the following cases:

```tcl
# Missing required parameters
catch {torch::load_checkpoint -filename "checkpoint.pt"} error
puts "Error: $error"
# Output: "Required parameters missing: filename, model, and optimizer are required"

# Invalid model name
catch {torch::load_checkpoint -file "checkpoint.pt" -model "nonexistent" -optimizer $optimizer} error
puts "Error: $error"
# Output: "Model not found"

# Invalid optimizer name
catch {torch::load_checkpoint -file "checkpoint.pt" -model $model -optimizer "nonexistent"} error
puts "Error: $error"
# Output: "Optimizer not found"

# Nonexistent checkpoint file
catch {torch::load_checkpoint -file "missing.pt" -model $model -optimizer $optimizer} error
puts "Error: $error"
# Output: File loading error

# Unknown parameters
catch {torch::load_checkpoint -invalid "checkpoint.pt" -model $model -optimizer $optimizer} error
puts "Error: $error"
# Output: "Unknown parameter: -invalid. Valid parameters are: -filename/-file, -model, -optimizer"

# Corrupted checkpoint file
catch {torch::load_checkpoint -file "corrupted.pt" -model $model -optimizer $optimizer} error
puts "Error: $error"
# Output: Serialization error from PyTorch
```

## File Format Compatibility

### Supported Checkpoint Formats
- **PyTorch native**: `.pt`, `.pth` files created by PyTorch
- **LibTorch**: Checkpoints saved by LibTorch C++ API
- **Cross-platform**: Files can be loaded across different operating systems

### Checkpoint Contents
The checkpoint file typically contains:
- **Model state dict**: All model parameters
- **Optimizer state dict**: Optimizer internal state
- **Epoch number**: Training epoch when saved
- **Loss value**: Training loss when saved
- **Learning rate**: Current learning rate when saved
- **Custom metadata**: Additional training information

## Performance Considerations

### Loading Speed
- **File size**: Larger models take longer to load
- **Storage medium**: SSD vs HDD affects loading time
- **Network storage**: Loading from network drives is slower
- **Compression**: Compressed checkpoints load slower but save space

### Memory Usage
- **Model size**: Memory usage scales with model parameters
- **Optimizer state**: Optimizers like Adam use additional memory
- **Temporary storage**: Loading requires temporary memory allocation

### Best Practices
```tcl
# Good: Check file existence before loading
if {[file exists $checkpoint_file]} {
    torch::load_checkpoint -file $checkpoint_file -model $model -optimizer $optimizer
} else {
    puts "Warning: Checkpoint file not found: $checkpoint_file"
}

# Good: Handle loading errors gracefully
if {[catch {torch::load_checkpoint -file $checkpoint_file -model $model -optimizer $optimizer} result]} {
    puts "Failed to load checkpoint: $result"
    # Initialize model with random weights instead
    torch::model_reset $model
} else {
    puts "Successfully loaded checkpoint: $result"
}

# Good: Validate checkpoint metadata
set info [torch::load_checkpoint -file $checkpoint_file -model $model -optimizer $optimizer]
if {[string match "*epoch=0*" $info]} {
    puts "Warning: Loaded checkpoint appears to be from beginning of training"
}
```

## Integration with Training Workflows

### Training Loop Integration
```tcl
# Training setup
set model [torch::create_model "my_model"]
set optimizer [torch::optimizer_adam -lr 0.001]
torch::optimizer_add_parameters $optimizer $model

# Check for existing checkpoint
set checkpoint_file "latest_checkpoint.pt"
set start_epoch 0

if {[file exists $checkpoint_file]} {
    set info [torch::load_checkpoint -file $checkpoint_file -model $model -optimizer $optimizer]
    # Extract epoch number from info string
    regexp {epoch=(\d+)} $info match start_epoch
    puts "Resuming from epoch $start_epoch"
}

# Training loop
for {set epoch [expr $start_epoch + 1]} {$epoch <= 100} {incr epoch} {
    # Training code...
    
    # Save checkpoint every 10 epochs
    if {$epoch % 10 == 0} {
        torch::save_checkpoint $checkpoint_file $model $optimizer $epoch $loss $lr
    }
}
```

### Conditional Loading
```tcl
# Load different checkpoints based on conditions
set model [torch::create_model "model"]
set optimizer [torch::optimizer_adam -lr 0.001]
torch::optimizer_add_parameters $optimizer $model

# Select checkpoint based on availability
set checkpoint_files [list "best_model.pt" "latest_model.pt" "initial_model.pt"]

foreach checkpoint $checkpoint_files {
    if {[file exists $checkpoint]} {
        torch::load_checkpoint -file $checkpoint -model $model -optimizer $optimizer
        puts "Loaded checkpoint: $checkpoint"
        break
    }
}
```

## Troubleshooting

### Common Issues

1. **"Model not found" Error**
   - Ensure the model name exists in the module storage
   - Check model creation was successful
   - Verify model name matches exactly

2. **"Optimizer not found" Error**
   - Ensure the optimizer name exists in the optimizer storage
   - Check optimizer creation was successful
   - Verify optimizer name matches exactly

3. **File Loading Errors**
   - Check file path is correct
   - Verify file permissions
   - Ensure file is not corrupted
   - Check available disk space

4. **Architecture Mismatch**
   - Checkpoint was saved with different model architecture
   - Model parameters don't match checkpoint structure
   - Use `torch::model_load_state_dict` for partial loading

### Debug Mode
```tcl
# Enable verbose error reporting
proc debug_load_checkpoint {args} {
    puts "Debug: Loading checkpoint with args: $args"
    if {[catch {torch::load_checkpoint {*}$args} result]} {
        puts "Debug: Error occurred: $result"
        return -code error $result
    } else {
        puts "Debug: Successfully loaded: $result"
        return $result
    }
}

# Use debug version
debug_load_checkpoint -file "checkpoint.pt" -model $model -optimizer $optimizer
```

## See Also

- [torch::save_checkpoint](save_checkpoint.md) - Save model checkpoints
- [torch::get_checkpoint_info](get_checkpoint_info.md) - Get checkpoint metadata
- [torch::load_state_dict](load_state_dict.md) - Load model state only
- [torch::save_state_dict](save_state_dict.md) - Save model state only
- [torch::optimizer_load_state_dict](optimizer_load_state_dict.md) - Load optimizer state only
- [torch::model_reset](model_reset.md) - Reset model to initial state

## Notes

- The checkpoint file must have been created with compatible PyTorch/LibTorch versions
- Model and optimizer must be created before loading checkpoint
- Loading overwrites current model and optimizer state completely
- Checkpoint metadata is optional and may not be present in all files
- Cross-platform compatibility is maintained for checkpoint files
- Large models may require significant memory during loading process 