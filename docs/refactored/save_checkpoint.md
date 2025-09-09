# torch::save_checkpoint

Save a model checkpoint including model state, optimizer state, and training metadata.

## Syntax

```tcl
# Positional syntax
torch::save_checkpoint model optimizer filename ?epoch? ?loss? ?lr?

# Named parameter syntax
torch::save_checkpoint -model model -optimizer optimizer -filename filename ?-epoch epoch? ?-loss loss? ?-lr lr?
```

The command also supports a camelCase alias: `torch::saveCheckpoint`

## Arguments

* `model` (positional) or `-model model` (named): Name of the model to save
* `optimizer` (positional) or `-optimizer optimizer` (named): Name of the optimizer to save
* `filename` (positional) or `-filename/-file filename` (named): Path to save the checkpoint file
* `epoch` (optional, positional) or `-epoch epoch` (named): Current training epoch (default: 0)
* `loss` (optional, positional) or `-loss loss` (named): Current loss value (default: 0.0)
* `lr` (optional, positional) or `-lr lr` (named): Current learning rate (default: 0.0)

## Return Value

Returns a success message with the filename and metadata if the checkpoint was saved successfully.

## Examples

```tcl
# Create a model and optimizer
set model [torch::sequential]
torch::add_linear $model 10 5
set optimizer [torch::adam -model $model -lr 0.01]

# Using positional syntax - minimal args
torch::save_checkpoint $model $optimizer "checkpoint.pt"

# Using positional syntax - all args
torch::save_checkpoint $model $optimizer "checkpoint.pt" 5 0.123 0.001

# Using named parameter syntax - minimal args
torch::save_checkpoint -model $model -optimizer $optimizer -filename "checkpoint.pt"

# Using named parameter syntax - all args
torch::save_checkpoint -model $model -optimizer $optimizer -filename "checkpoint.pt" -epoch 5 -loss 0.123 -lr 0.001

# Using camelCase alias
torch::saveCheckpoint -model $model -optimizer $optimizer -file "checkpoint.pt" -epoch 5 -loss 0.123 -lr 0.001
```

## Error Conditions

* If required parameters (model, optimizer, filename) are missing
* If the model name is not found in the model storage
* If the optimizer name is not found in the optimizer storage
* If epoch value is not a valid integer
* If loss value is not a valid number
* If learning rate value is not a valid number
* If an unknown parameter is provided in named syntax

## See Also

* `torch::load_checkpoint` - Load a saved checkpoint
* `torch::get_checkpoint_info` - Get metadata from a checkpoint file
* `torch::save_state` - Save model state only
* `torch::save_state_dict` - Save model state dictionary 