# torch::lr_scheduler_linear_with_warmup / torch::lrSchedulerLinearWithWarmup

Creates a linear learning-rate scheduler with an initial warm-up phase.

---

## Positional Syntax (legacy)
```tcl
torch::lr_scheduler_linear_with_warmup <optimizer> <num_warmup_steps> <num_training_steps> ?<last_epoch>?
```
* `optimizer` – Handle returned by an optimizer constructor (e.g. `torch::optimizerSgd`).
* `num_warmup_steps` – Integer number of steps to linearly increase the learning rate from 0 to the base LR.
* `num_training_steps` – Total number of training steps after which the LR will be **0**.
* `last_epoch` – Optional integer index of the last epoch when resuming training (default `-1`).

## Named Parameter Syntax (recommended)
```tcl
torch::lrSchedulerLinearWithWarmup \
    -optimizer OPT \
    -numWarmupSteps N \
    -numTrainingSteps T \
    ?-lastEpoch E?
```
Parameter aliases: `-num_warmup_steps`, `-num_training_steps`, `-last_epoch`.

## Parameters
| Name | Type | Description | Required |
|------|------|-------------|----------|
| `-optimizer` | string | Optimizer handle | ✓ |
| `-numWarmupSteps` | int | Warm-up step count | ✓ |
| `-numTrainingSteps` | int | Total training steps | ✓ |
| `-lastEpoch` | int | Last processed epoch (resume) | ✗ |

## Return value
Returns a scheduler handle to be passed to `torch::get_lr` or step/update commands.

## Examples
```tcl
# Create an SGD optimizer
auto params ... ; # pseudocode
set opt [torch::optimizerSgd -parameters $params -lr 0.1]

# Scheduler with 100 warm-up steps and 1000 total steps (named syntax)
set sched [torch::lrSchedulerLinearWithWarmup \
              -optimizer $opt \
              -numWarmupSteps 100 \
              -numTrainingSteps 1000]

# Using positional syntax (legacy)
set sched2 [torch::lr_scheduler_linear_with_warmup $opt 100 1000]
```

## Error handling
* Missing required parameters raises an error.
* `numWarmupSteps` must be ≥ 0 and ≤ `numTrainingSteps`.
* Optimizer handle must exist.
* Unknown parameters trigger an error.

## Compatibility
✅ Positional API retained • ✅ Named parameters supported • ✅ camelCase alias registered

## See also
* [`torch::lr_scheduler_constant_with_warmup`](lr_scheduler_constant_with_warmup.md) 