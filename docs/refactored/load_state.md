# torch::load_state / torch::loadState

Loads a saved neural network module state from a file.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::load_state module_name filename
```

### Named Parameters (New Syntax)
```tcl
torch::load_state -module module_name -filename filename
torch::load_state -module module_name -file filename
torch::loadState -module module_name -filename filename
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `module_name` / `-module` | string | Name of the existing module to load state into | Required |
| `filename` / `-filename` / `-file` | string | Path to the saved state file (.pt format) | Required |

## Returns

Returns "OK" on successful loading, or an error message if the operation fails.

## Examples

### Basic Usage

```tcl
# Create a neural network module
set my_model [torch::linear 784 10]

# Save the module state
torch::save_state $my_model "model.pt"

# Load the saved state (positional syntax)
torch::load_state $my_model "model.pt"

# Load the saved state (named syntax)
torch::load_state -module $my_model -filename "model.pt"

# Load the saved state (camelCase alias)
torch::loadState -module $my_model -filename "model.pt"
```

### With Alternative Parameter Names

```tcl
# Create and save a module
set my_model [torch::linear 128 64]
torch::save_state $my_model "trained_model.pt"

# Load using -file parameter alias
torch::load_state -module $my_model -file "trained_model.pt"

# Load using camelCase with -file alias
torch::loadState -module $my_model -file "trained_model.pt"
```

### Complex Neural Network Example

```tcl
# Create a more complex network
set conv_layer [torch::conv2d 3 64 3 1 1]
set linear_layer [torch::linear 64 10]

# Save both layers
torch::save_state $conv_layer "conv_layer.pt"
torch::save_state $linear_layer "linear_layer.pt"

# Load both layers back
torch::load_state $conv_layer "conv_layer.pt"
torch::load_state $linear_layer "linear_layer.pt"

# Or using named syntax
torch::load_state -module $conv_layer -filename "conv_layer.pt"
torch::load_state -module $linear_layer -filename "linear_layer.pt"
```

### Training Workflow Example

```tcl
# Create a model
set model [torch::linear 784 128]

# Training loop (simplified)
for {set epoch 0} {$epoch < 100} {incr epoch} {
    # ... training code ...
    
    # Save checkpoint every 10 epochs
    if {$epoch % 10 == 0} {
        torch::save_state $model "checkpoint_epoch_${epoch}.pt"
    }
}

# Later: Load a specific checkpoint
torch::load_state $model "checkpoint_epoch_50.pt"

# Or using named syntax
torch::load_state -module $model -filename "checkpoint_epoch_50.pt"
```

### Error Handling Example

```tcl
# Create a model
set model [torch::linear 256 10]

# Attempt to load from non-existent file
if {[catch {torch::load_state $model "nonexistent.pt"} error]} {
    puts "Error loading model: $error"
}

# Attempt to load with invalid module
if {[catch {torch::load_state "invalid_module" "model.pt"} error]} {
    puts "Error: $error"
}

# Safe loading with error handling
proc safe_load_model {module_name filename} {
    if {[catch {torch::load_state $module_name $filename} error]} {
        puts "Failed to load model: $error"
        return false
    }
    puts "Successfully loaded model from $filename"
    return true
}

# Usage
if {[safe_load_model $model "my_model.pt"]} {
    # Continue with inference
    puts "Model loaded successfully"
} else {
    puts "Failed to load model"
}
```

### Transfer Learning Example

```tcl
# Load a pre-trained model
set pretrained_model [torch::linear 1000 500]
torch::load_state $pretrained_model "pretrained_weights.pt"

# Fine-tune on new data
# ... training code ...

# Save the fine-tuned model
torch::save_state $pretrained_model "fine_tuned_model.pt"

# Later: Load the fine-tuned version
torch::load_state $pretrained_model "fine_tuned_model.pt"
```

### Multiple Model Management

```tcl
# Create different models
set encoder [torch::linear 784 256]
set decoder [torch::linear 256 784]
set classifier [torch::linear 256 10]

# Save all models
torch::save_state $encoder "encoder.pt"
torch::save_state $decoder "decoder.pt"
torch::save_state $classifier "classifier.pt"

# Load all models (positional syntax)
torch::load_state $encoder "encoder.pt"
torch::load_state $decoder "decoder.pt"
torch::load_state $classifier "classifier.pt"

# Or using named syntax for better readability
torch::load_state -module $encoder -filename "encoder.pt"
torch::load_state -module $decoder -filename "decoder.pt"
torch::load_state -module $classifier -filename "classifier.pt"
```

### File Path Examples

```tcl
# Different file path formats
set model [torch::linear 100 50]

# Absolute path
torch::load_state $model "/home/user/models/my_model.pt"

# Relative path
torch::load_state $model "./models/my_model.pt"

# With subdirectories
torch::load_state $model "checkpoints/epoch_10/model.pt"

# Using named syntax with paths
torch::load_state -module $model -filename "/path/to/model.pt"
torch::loadState -module $model -file "relative/path/model.pt"
```

## Notes

- The module must exist and be created before loading state
- The saved state file must be compatible with the module architecture
- Files are typically saved in PyTorch's `.pt` format
- The command modifies the existing module's parameters in-place
- Both `torch::load_state` and `torch::loadState` are equivalent
- The `-file` parameter is an alias for `-filename`
- File paths can be absolute or relative to the current working directory

## Error Handling

The command will return an error if:
- The specified module name doesn't exist
- The filename is empty or invalid
- The file doesn't exist or cannot be read
- The file format is incompatible with the module
- File permissions prevent reading
- Required parameters are missing
- Unknown parameters are provided

## Migration from Old Syntax

If you have existing code using the positional syntax, it will continue to work unchanged:

```tcl
# Old code (still works)
torch::load_state $my_model "model.pt"

# New code (recommended)
torch::load_state -module $my_model -filename "model.pt"
```

## Best Practices

1. **Always check if the module exists** before loading
2. **Use absolute paths** for production code to avoid path issues
3. **Handle errors gracefully** with proper error checking
4. **Validate file existence** before attempting to load
5. **Use descriptive filenames** that include version or timestamp information
6. **Keep backup copies** of important model files

## See Also

- `torch::save_state` - Save module state to file
- `torch::linear` - Create linear layer modules
- `torch::conv2d` - Create convolutional layer modules 