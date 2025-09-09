# torch::lr_scheduler_step_update

Immediately steps/updates a learning-rate scheduler by one iteration.

## Syntax

### Positional
```tcl
torch::lr_scheduler_step_update schedulerHandle
```

### Named Parameters
```tcl
torch::lr_scheduler_step_update -scheduler schedulerHandle
```

### CamelCase Alias
```tcl
torch::lrSchedulerStepUpdate schedulerHandle
```

## Parameters
| Name | Description |
|------|-------------|
| `scheduler` / `-scheduler` | Handle returned by any `torch::lr_scheduler_*` command. |

No other parameters are required.

## Return
String `"OK"` on success. On error, the command raises a TCL error explaining the issue (e.g., invalid handle).

## Examples
```tcl
set optim [torch::optimizer_sgd $params 0.01]
set sched [torch::lr_scheduler_lambda $optim 0.95]

# Training loop
for {set epoch 0} {$epoch < 10} {incr epoch} {
    # ... training code ...
    torch::lr_scheduler_step_update $sched ;# decrease LR
}
```

### Named form
```tcl
torch::lr_scheduler_step_update -scheduler $sched
```

## Migration Guide
Old positional calls continue to work; switch to the named form for clarity:
```tcl
# Old
torch::lr_scheduler_step_update $sched
# New
torch::lr_scheduler_step_update -scheduler $sched
```

## Error Handling
• Missing scheduler → error *scheduler parameter is required*  
• Invalid handle → error *Invalid scheduler name*  
• Unknown parameter → error *Unknown parameter: -foo*

## Related
* `torch::get_lr` – retrieve current learning rate
* `torch::lr_scheduler_step_advanced` – metric-aware step 