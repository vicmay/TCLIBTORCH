# torch::lr_scheduler_step

Decay the learning rate of each parameter group by `gamma` every `stepSize` optimisation steps.

## 🔄 Dual Syntax Support

### Named Parameters (Recommended)
```tcl
torch::lr_scheduler_step -optimizer $optimizer -stepSize 5 ?-gamma 0.1?
# or camelCase alias
torch::lrSchedulerStep -optimizer $optimizer -stepSize 5 ?-gamma 0.1?
```

### Positional Syntax (Legacy)
```tcl
torch::lr_scheduler_step $optimizer step_size ?gamma?
torch::lrSchedulerStep $optimizer step_size ?gamma?
```

## 📖 Description
The **step** learning-rate scheduler multiplies the learning-rate by `gamma` every `stepSize` calls to `torch::lr_scheduler_step_update`.  This is equivalent to PyTorch's `torch.optim.lr_scheduler.StepLR`.

```
if step % stepSize == 0:
    lr = lr * gamma
```

## 🔧 Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-optimizer` | string | yes | — | Handle of the optimiser whose learning-rate will be scheduled |
| `-stepSize` (`-step_size`) | int | yes | — | Number of scheduler steps between each decay |
| `-gamma` | double | no | `0.1` | Multiplicative decay factor |

## 📝 Migration Guide
| Old (snake_case) | New (camelCase) |
|------------------|-----------------|
| `torch::lr_scheduler_step` | `torch::lrSchedulerStep` |

Example migration:
```tcl
# Before (legacy)
set sched [torch::lr_scheduler_step $optim 10 0.5]

# After (modern)
set sched [torch::lrSchedulerStep -optimizer $optim -stepSize 10 -gamma 0.5]
```

Both syntaxes are fully supported; existing scripts continue to work unchanged.

## 📚 Examples
```tcl
# Create model and optimiser
set model   [create_model]
set params  [torch::model_parameters $model]
set optim   [torch::optimizer_sgd $params 0.1]

# Scheduler: halve LR every 5 epochs
set sched [torch::lrSchedulerStep -optimizer $optim -stepSize 5 -gamma 0.5]

for {set epoch 0} {$epoch < 20} {incr epoch} {
    train_one_epoch $model $optim $data
    torch::lr_scheduler_step_update $sched
    puts "Epoch $epoch  LR: [torch::get_lr $optim]"
}
```

## ⚠️ Error Handling
* Missing required arguments ➜ "Required parameters missing or invalid …"
* Invalid optimiser handle ➜ "Invalid optimizer name"
* Non-positive `stepSize` or `gamma` ➜ descriptive error message

## ✅ Test Coverage
See `tests/refactored/lr_scheduler_step_test.tcl` (17 tests covering both syntaxes, alias, functionality, and errors). 