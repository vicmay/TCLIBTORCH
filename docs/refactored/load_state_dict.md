# torch::load_state_dict / torch::loadStateDict

Loads a saved neural network model's state dictionary (parameters only) from a file.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::load_state_dict model_name filename
```

### Named Parameters (New Syntax)
```tcl
torch::load_state_dict -model model_name -filename filename
torch::load_state_dict -model model_name -file filename
torch::loadStateDict -model model_name -filename filename
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model_name` / `-model` | string | Name of the existing model to load state dict into | Required |
| `filename` / `-filename` / `-file` | string | Path to the saved state dict file (.pt format) | Required |

## Returns

Returns a success message "Model state dict loaded from: <filename>" on successful loading, or an error message if the operation fails.

## Examples

### Basic Usage

```tcl
# Create a neural network model
set my_model [torch::linear 784 10]

# Save the model's state dict (parameters only)
torch::save_state_dict $my_model "model_state.pt"

# Load the saved state dict (positional syntax)
torch::load_state_dict $my_model "model_state.pt"

# Load the saved state dict (named syntax)
torch::load_state_dict -model $my_model -filename "model_state.pt"

# Load the saved state dict (camelCase alias)
torch::loadStateDict -model $my_model -filename "model_state.pt"
```

### With Alternative Parameter Names

```tcl
# Create and save a model's state dict
set my_model [torch::linear 128 64]
torch::save_state_dict $my_model "trained_params.pt"

# Load using -file parameter alias
torch::load_state_dict -model $my_model -file "trained_params.pt"

# Load using camelCase with -file alias
torch::loadStateDict -model $my_model -file "trained_params.pt"
```

### Transfer Learning Example

```tcl
# Create a pre-trained model and save its state dict
set pretrained_model [torch::linear 1000 500]
torch::save_state_dict $pretrained_model "pretrained_params.pt"

# Create a new model with same architecture
set new_model [torch::linear 1000 500]

# Load the pre-trained parameters into the new model
torch::load_state_dict $new_model "pretrained_params.pt"

# Or using named syntax for clarity
torch::load_state_dict -model $new_model -filename "pretrained_params.pt"

# Continue training with loaded parameters
```

### Training Workflow with State Dict Checkpoints

```tcl
# Create a model
set model [torch::linear 784 128]

# Training loop (simplified)
for {set epoch 0} {$epoch < 100} {incr epoch} {
    # ... training code ...
    
    # Save only the parameters (state dict) every 10 epochs
    if {$epoch % 10 == 0} {
        torch::save_state_dict $model "params_epoch_${epoch}.pt"
    }
}

# Later: Load specific epoch parameters
torch::load_state_dict $model "params_epoch_50.pt"

# Or using named syntax
torch::load_state_dict -model $model -filename "params_epoch_50.pt"
```

### Error Handling Example

```tcl
# Create a model
set model [torch::linear 256 10]

# Attempt to load from non-existent file
if {[catch {torch::load_state_dict $model "nonexistent_params.pt"} error]} {
    puts "Error loading state dict: $error"
}

# Attempt to load with invalid model
if {[catch {torch::load_state_dict "invalid_model" "params.pt"} error]} {
    puts "Error: $error"
}

# Safe loading with error handling
proc safe_load_state_dict {model_name filename} {
    if {[catch {torch::load_state_dict $model_name $filename} error]} {
        puts "Failed to load state dict: $error"
        return false
    }
    puts "Successfully loaded state dict from $filename"
    return true
}

# Usage
if {[safe_load_state_dict $model "my_params.pt"]} {
    puts "Parameters loaded successfully"
} else {
    puts "Failed to load parameters"
}
```

### State Dict vs Complete Model State

```tcl
# Create a model
set model [torch::linear 100 50]

# Save complete model state (includes architecture)
torch::save_state $model "complete_model.pt"

# Save only state dict (parameters only)
torch::save_state_dict $model "parameters_only.pt"

# Load complete model state
torch::load_state $model "complete_model.pt"

# Load only parameters (state dict)
torch::load_state_dict $model "parameters_only.pt"

# Named syntax comparison
torch::load_state -module $model -filename "complete_model.pt"
torch::load_state_dict -model $model -filename "parameters_only.pt"
```

### Multiple Model Parameter Management

```tcl
# Create different models
set encoder [torch::linear 784 256]
set decoder [torch::linear 256 784]
set classifier [torch::linear 256 10]

# Save all model parameters
torch::save_state_dict $encoder "encoder_params.pt"
torch::save_state_dict $decoder "decoder_params.pt"
torch::save_state_dict $classifier "classifier_params.pt"

# Load all model parameters (positional syntax)
torch::load_state_dict $encoder "encoder_params.pt"
torch::load_state_dict $decoder "decoder_params.pt"
torch::load_state_dict $classifier "classifier_params.pt"

# Or using named syntax for better readability
torch::load_state_dict -model $encoder -filename "encoder_params.pt"
torch::load_state_dict -model $decoder -filename "decoder_params.pt"
torch::load_state_dict -model $classifier -filename "classifier_params.pt"
```

### Parameter Fine-tuning Workflow

```tcl
# Load a pre-trained model's parameters
set model [torch::linear 1000 10]
torch::load_state_dict $model "pretrained_imagenet_params.pt"

# Freeze some layers (keeping pre-trained weights)
torch::freeze_model $model

# Fine-tune on new dataset
# ... training code ...

# Save the fine-tuned parameters
torch::save_state_dict $model "fine_tuned_params.pt"

# Later: Load the fine-tuned version
torch::load_state_dict -model $model -filename "fine_tuned_params.pt"
```

### Model Ensemble Loading

```tcl
# Create multiple models for ensemble
set model1 [torch::linear 784 10]
set model2 [torch::linear 784 10]
set model3 [torch::linear 784 10]

# Load different trained parameters into each model
torch::load_state_dict $model1 "model1_params.pt"
torch::load_state_dict $model2 "model2_params.pt"
torch::load_state_dict $model3 "model3_params.pt"

# Or using named syntax
torch::load_state_dict -model $model1 -file "model1_params.pt"
torch::load_state_dict -model $model2 -file "model2_params.pt"
torch::load_state_dict -model $model3 -file "model3_params.pt"
```

### File Path Examples

```tcl
# Different file path formats
set model [torch::linear 100 50]

# Absolute path
torch::load_state_dict $model "/home/user/models/my_params.pt"

# Relative path
torch::load_state_dict $model "./checkpoints/params.pt"

# With subdirectories
torch::load_state_dict $model "experiments/run_001/best_params.pt"

# Using named syntax with paths
torch::load_state_dict -model $model -filename "/path/to/params.pt"
torch::loadStateDict -model $model -file "relative/path/params.pt"
```

## Notes

- **State Dict vs Model State**: `load_state_dict` loads only the model parameters (weights and biases), while `load_state` loads the complete model including architecture
- **Model Architecture**: The target model must have the same architecture as the saved state dict
- **In-place Loading**: The command modifies the existing model's parameters in-place
- **File Format**: Files are typically saved in PyTorch's `.pt` format
- **Aliases**: Both `torch::load_state_dict` and `torch::loadStateDict` are equivalent
- **Parameter Alias**: The `-file` parameter is an alias for `-filename`
- **Path Handling**: File paths can be absolute or relative to the current working directory

## State Dict Benefits

1. **Smaller File Size**: Only parameters are saved, not the model architecture
2. **Flexibility**: Can load parameters into models created separately
3. **Transfer Learning**: Easy to transfer parameters between compatible models
4. **Efficiency**: Faster loading when model architecture is already defined

## Error Handling

The command will return an error if:
- The specified model name doesn't exist
- The filename is empty or invalid
- The file doesn't exist or cannot be read
- The state dict is incompatible with the model architecture
- Parameter shapes don't match the model
- File permissions prevent reading
- Required parameters are missing
- Unknown parameters are provided

## Migration from Old Syntax

If you have existing code using the positional syntax, it will continue to work unchanged:

```tcl
# Old code (still works)
torch::load_state_dict $my_model "params.pt"

# New code (recommended)
torch::load_state_dict -model $my_model -filename "params.pt"
```

## Best Practices

1. **Use state dict for transfer learning** - More flexible than complete model saves
2. **Verify model architecture compatibility** before loading
3. **Handle errors gracefully** with proper error checking
4. **Use descriptive filenames** that indicate what parameters they contain
5. **Test parameter loading** on a small subset before full deployment
6. **Keep backup copies** of important parameter files
7. **Use absolute paths** in production environments

## Common Use Cases

- **Model checkpointing during training**
- **Transfer learning from pre-trained models**
- **Model ensemble creation**
- **Parameter sharing between similar models**
- **Fine-tuning workflows**
- **Model deployment with pre-trained weights**

## See Also

- `torch::save_state_dict` - Save model state dict to file
- `torch::load_state` - Load complete model state from file
- `torch::freeze_model` - Freeze model parameters
- `torch::unfreeze_model` - Unfreeze model parameters
- `torch::linear` - Create linear layer modules
- `torch::conv2d` - Create convolutional layer modules 