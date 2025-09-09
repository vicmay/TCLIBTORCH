# torch::lr_scheduler_step_advanced

Advances a learning-rate scheduler by one step, optionally supplying a validation metric to schedulers that support it (e.g., Reduce-on-Plateau).

## Syntax

### Current Positional Syntax
```tcl
torch::lr_scheduler_step_advanced scheduler ?metric?
```

### Named-Parameter Syntax
```tcl
torch::lr_scheduler_step_advanced \
    -scheduler handle \
    ?-metric value?
```

### CamelCase Alias
```tcl
torch::lrSchedulerStepAdvanced scheduler ?metric?
torch::lrSchedulerStepAdvanced -scheduler handle ?-metric value?
```

All syntaxes are equivalent and coexist; the original positional form remains fully supported.

## Parameters
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `scheduler` / `-scheduler` | string | Yes | Handle returned by a `torch::lr_scheduler_*` command. |
| `metric` / `-metric` | double | No | Validation metric to guide schedulers that react to performance (e.g., plateau schedulers). |

## Return Value
String `"OK"` on success, or a TCL error if the scheduler handle is invalid or parameters are malformed.

## Examples

### Basic Usage (Positional)
```tcl
set sched [torch::lr_scheduler_lambda $opt]
set status [torch::lr_scheduler_step_advanced $sched]
```

### With Metric (Positional)
```tcl
set status [torch::lr_scheduler_step_advanced $sched 0.93]
```

### Named Parameters
```tcl
set status [torch::lr_scheduler_step_advanced -scheduler $sched -metric 0.88]
```

### CamelCase Alias
```tcl
set status [torch::lrSchedulerStepAdvanced -scheduler $sched -metric 0.95]
```

## Migration Guide
Old calls:
```tcl
torch::lr_scheduler_step_advanced $sched 0.9
```
are still valid; the modern form is:
```tcl
torch::lr_scheduler_step_advanced -scheduler $sched -metric 0.9
```

## Error Handling
• Invalid scheduler handle → `"Invalid scheduler handle"`
• Unknown parameter → `"Unknown parameter: -foo"`
• Missing value after parameter → TCL error `"Missing value for parameter"`

## Related Commands
* `torch::lr_scheduler_lambda`
* `torch::lr_scheduler_reduce_on_plateau`
* `torch::get_lr_advanced`

---
This page is part of the LibTorch TCL modernisation project—supporting named parameters and camelCase while retaining full backward compatibility. 